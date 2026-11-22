"""Claude as the cell operator.

Starts the cellbot MCP server as a subprocess (stdio), hands its tools to Claude
through the Anthropic SDK's tool runner, and runs one conversation per part:
pick -> place -> mark 4 spots -> inspect -> remove -> report.

    export ANTHROPIC_API_KEY=...                         # Anthropic API
    #   or: export OPENROUTER_API_KEY=...                # OpenRouter (picked automatically
    #                                                    #  when only this key is set)
    python -m cellbot.orchestrator --parts 3             # dry run (cell.toml mode = "sim")
    python -m cellbot.orchestrator --backend openrouter --model anthropic/claude-sonnet-5.5
    CELLBOT_MODE=station STATION_HOST=192.168.1.20 \
        python -m cellbot.orchestrator --parts 1 --confirm   # real arms, approve each move

Claude never talks to Station directly: everything goes through the guarded
MCP tools, so the guardrails hold whatever the model decides.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path


from anthropic import AsyncAnthropic, BadRequestError
from anthropic.lib.tools import ToolError
from anthropic.lib.tools.mcp import async_mcp_tool
from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

from .config import ROOT, load_config

MODEL = "claude-opus-5-5"
OPENROUTER_MODEL = "anthropic/claude-opus-5.5"
OPENROUTER_BASE_URL = "https://openrouter.ai/api"  # Anthropic-compatible /v1/messages


def pick_backend(choice: str) -> str:
    """auto: Anthropic if ANTHROPIC_API_KEY is set, else OpenRouter if OPENROUTER_API_KEY is set."""
    if choice != "auto":
        return choice
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "anthropic"
    if os.environ.get("OPENROUTER_API_KEY"):
        return "openrouter"
    sys.exit("No API key: export ANTHROPIC_API_KEY=... or OPENROUTER_API_KEY=...")


def make_client(backend: str) -> AsyncAnthropic:
    if backend == "anthropic":
        return AsyncAnthropic()
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        sys.exit("--backend openrouter needs OPENROUTER_API_KEY in the environment.")
    # OpenRouter speaks the Anthropic Messages API (tools, images, thinking), so the same
    # SDK tool runner works. The key goes as a Bearer token; ANTHROPIC_API_KEY is never sent there.
    return AsyncAnthropic(
        auth_token=key,
        base_url=os.environ.get("OPENROUTER_BASE_URL", OPENROUTER_BASE_URL),
        default_headers={"X-Title": "cellbot"},
    )

from .operator_prompt import SYSTEM  # noqa: E402  (shared with the MCP server's run_parts prompt)

MOTION_TOOLS = {"go_to_pose", "go_home", "move_joints", "gripper", "mark_spot"}


def _short(obj, n=300) -> str:
    s = obj if isinstance(obj, str) else json.dumps(obj, default=str)
    return s if len(s) <= n else s[:n] + "..."


def _wrap_confirm(tool):
    """Ask a human on the terminal before every motion tool call (--confirm)."""
    orig = tool.call

    async def call(input):
        ans = await asyncio.to_thread(input_fn, f"\n>>> {tool.name}({json.dumps(input)})  run? [y/N/stop] ")
        if ans.strip().lower() == "stop":
            raise ToolError("Human operator pressed STOP. Call the stop tool and finish.")
        if ans.strip().lower() != "y":
            raise ToolError("Human operator declined this move. Choose another action or explain.")
        return await orig(input)

    tool.call = call
    return tool


def input_fn(prompt: str) -> str:
    return input(prompt)


async def run_part(client: AsyncAnthropic, session: ClientSession, tools, part_no: int, args) -> str:
    messages = [{"role": "user", "content": f"Process part #{part_no} now, following the procedure."}]
    params = dict(
        model=args.model,
        max_tokens=16000,
        system=SYSTEM,
        tools=tools,
        messages=messages,
        max_iterations=args.max_turns,
    )
    if not args.no_thinking:
        params.update(thinking={"type": "adaptive"}, output_config={"effort": args.effort})
    if args.backend == "anthropic":
        # Anthropic-only: server-side model fallback.
        params.update(betas=["server-side-fallback-2026-07-01"], fallbacks="default")
    else:
        # OpenRouter has no server-held threads: send the full history on every turn.
        params["thread"] = False
    try:
        return await _run(client.beta.messages.tool_runner(**params))
    except BadRequestError as e:
        if args.no_thinking or getattr(e, "_cellbot_started", False):
            raise
        # Some routes reject thinking/effort. Nothing has moved yet (first request failed),
        # so retry the part without them.
        print(f"!! {args.backend} rejected thinking/effort ({_short(str(e), 200)}); retrying without them")
        args.no_thinking = True
        params.pop("thinking", None)
        params.pop("output_config", None)
        return await _run(client.beta.messages.tool_runner(**params))


async def _run(runner) -> str:
    final_text = ""
    started = False
    try:
        async for message in runner:
            started = True
            final_text = _print_message(message) or final_text
    except BadRequestError as e:
        e._cellbot_started = started
        raise
    return final_text


def _print_message(message) -> str:
    final_text = ""
    if message.stop_reason == "refusal":
        print("!! model declined the request:", getattr(message, "stop_details", None))
    for block in message.content:
        if block.type == "text" and block.text.strip():
            final_text = block.text
            print(f"\n[claude] {block.text.strip()}\n")
        elif block.type == "tool_use":
            print(f"  -> {block.name}({_short(block.input, 160)})")
    return final_text


async def main_async(args):
    cfg = load_config()
    reports = ROOT / "reports"
    reports.mkdir(exist_ok=True)
    # The MCP server never needs the LLM keys.
    env = {k: v for k, v in os.environ.items() if k not in ("ANTHROPIC_API_KEY", "OPENROUTER_API_KEY")}
    env["PYTHONPATH"] = str(ROOT)
    params = StdioServerParameters(command=sys.executable, args=["-m", "cellbot.mcp_server"], env=env, cwd=str(ROOT))
    args.backend = pick_backend(args.backend)
    if args.model is None:
        args.model = os.environ.get("CELLBOT_MODEL") or (OPENROUTER_MODEL if args.backend == "openrouter" else MODEL)
    client = make_client(args.backend)
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            listed = await session.list_tools()
            tools = [async_mcp_tool(t, session) for t in listed.tools]
            if args.confirm:
                tools = [_wrap_confirm(t) if t.name in MOTION_TOOLS else t for t in tools]
            print(f"cellbot: {len(tools)} tools, mode={cfg.station['mode']}, backend={args.backend}, model={args.model}")
            session_md = reports / f"session_{time.strftime('%Y%m%d_%H%M%S')}.md"
            for i in range(1, args.parts + 1):
                print(f"\n===================== PART {i} =====================")
                text = await run_part(client, session, tools, i, args)
                with open(session_md, "a") as f:
                    f.write(f"\n\n## Part {i}\n\n{text}\n")
                status = await session.call_tool("learning_status", {})
                last = json.loads(status.content[0].text)["recent_parts"][-1:] if status.content else []
                print("learning:", _short(last, 400))
                est = json.loads(((cfg.state_dir / "estop.json").read_text()) if (cfg.state_dir / "estop.json").exists() else '{"latched": false}')
                if est.get("latched"):
                    print("E-STOP latched; stopping the session.")
                    break
            print(f"\nReport: {session_md}")


def main():
    ap = argparse.ArgumentParser(description="Claude orchestrator for the marking cell")
    ap.add_argument("--parts", type=int, default=1)
    ap.add_argument("--backend", default=os.environ.get("CELLBOT_LLM", "auto"), choices=["auto", "anthropic", "openrouter"],
                    help="LLM provider (env CELLBOT_LLM). auto = whichever API key is set")
    ap.add_argument("--model", default=None, help=f"default {MODEL} (anthropic) or {OPENROUTER_MODEL} (openrouter); env CELLBOT_MODEL")
    ap.add_argument("--no-thinking", action="store_true", help="omit adaptive thinking/effort params")
    ap.add_argument("--effort", default="medium", choices=["low", "medium", "high", "xhigh", "max"])
    ap.add_argument("--max-turns", type=int, default=80)
    ap.add_argument("--confirm", action="store_true", help="ask a human before every motion tool call")
    asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    main()
