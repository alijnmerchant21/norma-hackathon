"""No-Claude smoke test: drives the MCP server over stdio with a fixed script,
through the same SDK tool wrappers the orchestrator uses.

    python -m cellbot.smoke --parts 4       # dry run, shows misses shrinking

Use it to check the MCP server, guardrails and learning loop before spending
API tokens, and as a fallback demo if the network is bad.
"""

import argparse
import asyncio
import json
import os
import sys

from anthropic.lib.tools import ToolError
from anthropic.lib.tools.mcp import async_mcp_tool
from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

from .config import ROOT


async def main_async(parts: int):
    env = {**os.environ, "PYTHONPATH": str(ROOT)}
    params = StdioServerParameters(command=sys.executable, args=["-m", "cellbot.mcp_server"], env=env, cwd=str(ROOT))
    async with stdio_client(params) as (r, w):
        async with ClientSession(r, w) as s:
            await s.initialize()
            tools = {t.name: async_mcp_tool(t, s) for t in (await s.list_tools()).tools}
            print("tools:", sorted(tools))

            async def call(name, **kw):
                try:
                    res = await tools[name].call(kw)
                except ToolError as e:
                    return {"refused": str(e.content if hasattr(e, "content") else e)[:160]}
                texts = [b["text"] for b in res if isinstance(b, dict) and b.get("type") == "text"] if isinstance(res, list) else [res]
                try:
                    return json.loads(texts[0])
                except Exception:
                    return texts

            print("guard check:", await call("go_to_pose", arm="elrobot", pose="nope"))
            for p in range(1, parts + 1):
                await call("new_part_ready")
                for pose in ("pick_hover", "pick"):
                    await call("go_to_pose", arm="so101", pose=pose)
                for attempt in range(3):
                    g = await call("gripper", arm="so101", action="close")
                    if g.get("holding_object"):
                        break
                    await call("log_result", kind="exception", message=f"grasp failed (attempt {attempt+1}), retrying")
                    await call("gripper", arm="so101", action="open")
                for pose in ("pick_hover", "place_hover", "place"):
                    await call("go_to_pose", arm="so101", pose=pose)
                print("  elrobot into busy work area ->", await call("go_to_pose", arm="elrobot", pose="mark_hover_1"))
                await call("gripper", arm="so101", action="open")
                await call("go_to_pose", arm="so101", pose="place_hover")
                await call("go_home", arm="so101")
                await call("start_marking", part_id=f"smoke{p}")
                for spot in "1234":
                    await call("mark_spot", spot=spot)
                await call("go_home", arm="elrobot")
                insp = await call("inspect_marks")
                misses = {k: v.get("miss_dist_mm") for k, v in insp["spots"].items()}
                for pose in ("place_hover", "place"):
                    await call("go_to_pose", arm="so101", pose=pose)
                await call("gripper", arm="so101", action="close")
                for pose in ("place_hover", "drop_hover", "drop"):
                    await call("go_to_pose", arm="so101", pose=pose)
                await call("gripper", arm="so101", action="open")
                await call("go_to_pose", arm="so101", pose="drop_hover")
                await call("go_home", arm="so101")
                ok = all(v.get("pass") for v in insp["spots"].values())
                await call("finish_part", outcome="pass" if ok else "fail", notes="smoke test")
                print(f"part {p}: misses mm {misses} -> {'PASS' if ok else 'FAIL'}")
            print("learning:", json.dumps((await call("learning_status"))["corrections_mm"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", type=int, default=4)
    asyncio.run(main_async(ap.parse_args().parts))


if __name__ == "__main__":
    main()
