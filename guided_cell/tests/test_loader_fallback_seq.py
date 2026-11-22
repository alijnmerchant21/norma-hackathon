"""Loader fallback pre-sequence (sim): policy_<phase>_fallback_pre_seq taught steps, then the fallback command.

    python tests/test_loader_fallback_seq.py

Checks: fallback with a pre-seq runs the steps, then policy_<phase>_fallback_cmd, then home, lock held; with
fallback_cmd empty the primary policy_<phase>_cmd is used; primary variant unaffected (no fallback steps); neither
set -> refused; untaught pose -> refused before motion; failing step -> policy not started, logged.
"""
import asyncio
import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ["CELLBOT_MODE"] = "sim"
os.environ["CELLBOT_STATE"] = tempfile.mkdtemp(prefix="cellguide_fbseq_")

from cellguide.guided_cell import GuidedCell, GuardError  # noqa: E402
from cellguide.cli import CLI  # noqa: E402
import sim_settings  # noqa: E402


def check(name, cond, detail=""):
    print(f"  [{'ok' if cond else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    if not cond:
        check.failed += 1


check.failed = 0


def spy_moves(cell, markers, fail_at=None):
    calls, real = [], cell._move

    async def _move(arm, target, speed=None):
        calls.append({"target": dict(target), "after_cmd": any(m.exists() for m in markers),
                      "lock": cell.lock.locked(), "speed": speed})
        if fail_at is not None and len(calls) == fail_at:
            return {"ok": False, "error": "simulated: timeout, not reached", "waypoint": "1/3"}
        return await real(arm, target, speed)
    cell._move = _move
    return calls


def unspy(cell):
    cell._move = type(cell)._move.__get__(cell)


def episodes(cell):
    p = cell._episodes_path()
    return [json.loads(x) for x in p.read_text().splitlines()] if p.exists() else []


def n_events(cell):
    return len(cell.events_path.read_text().splitlines()) if cell.events_path.exists() else 0


def events(cell, since):
    return [json.loads(x) for x in cell.events_path.read_text().splitlines()[since:]]


async def refused(coro):
    try:
        await coro
    except GuardError as e:
        return str(e)
    return ""


async def main():
    cell = sim_settings.apply(GuidedCell())
    await cell.ensure_connected()
    await cell.go_home("elrobot")
    await cell.go_home("so101")
    tmp = Path(tempfile.mkdtemp(prefix="fbseq_cmd_"))
    prim, fb = tmp / "primary", tmp / "fallback"
    cmd_p = f"python3 -c \"open('{prim}', 'a').write('x\\n'); print('primary')\""
    cmd_f = f"python3 -c \"open('{fb}', 'a').write('x\\n'); print('fallback')\""
    lc = cell.gcfg.setdefault("loader", {})
    lc.update(arm="so101", mode="policy", policy_load_cmd=cmd_p, policy_unload_cmd=cmd_p, policy_cwd="",
              policy_load_fallback_cmd="", policy_unload_fallback_cmd=cmd_f, policy_timeout_s=10,
              policy_load_pre_pose="", policy_unload_pre_pose="", policy_load_pre_seq=[], policy_unload_pre_seq=[],
              policy_load_fallback_pre_seq=[], policy_unload_fallback_pre_seq=[],
              load_mode="", unload_mode="", taught_load_seq=[], taught_unload_seq=[])
    cell.poses.reload()
    cell.poses.save("so101", "sweep_a", cell.poses.get("so101", "place_hover")["joints"], "work")
    cell.poses.save("so101", "sweep_b", cell.poses.get("so101", "place")["joints"], "work")
    seq = ["gripper:open", "pose:sweep_a", "pose:sweep_b", "gripper:close"]
    markers = [prim, fb]

    def clean():
        for m in markers:
            m.unlink(missing_ok=True)

    print("1. fallback + fallback_pre_seq: steps, then policy_unload_fallback_cmd, then home; lock held")
    lc["policy_unload_fallback_pre_seq"] = list(seq)
    calls = spy_moves(cell, markers)
    e0 = n_events(cell)
    r = await cell.unload_part("fallback")
    before = [c for c in calls if not c["after_cmd"]]
    check("ok, fallback command ran, primary not", r["ok"] and fb.exists() and not prim.exists(), str(r)[:200])
    check("pre_seq: 4 steps in order, all ok", [s["step"] for s in r.get("pre_seq") or []] == seq
          and all(s["ok"] for s in r["pre_seq"]), str(r.get("pre_seq")))
    check("4 slow moves before the command, all under the lock", len(before) == 4
          and all(c["lock"] and c["speed"] == cell.g["slow_speed"] for c in before), str(before)[:200])
    check("home after the command", calls and calls[-1]["after_cmd"] and r["go_home"]["ok"])
    msgs = [e["message"] for e in events(cell, e0)]
    check("events: pre-sequence done before policy started",
          "policy unload pre-sequence done" in msgs
          and msgs.index("policy unload pre-sequence done") < msgs.index("policy unload started"), str(msgs))
    ep = episodes(cell)[-1]
    check("episode: variant fallback, fallback cmd, taught steps", ep["variant"] == "fallback"
          and str(fb) in ep["cmd"] and ep.get("sequence") == seq, str(ep))
    unspy(cell)
    clean()

    print("2. fallback_cmd empty + fallback_pre_seq: steps, then the PRIMARY command")
    lc["policy_unload_fallback_cmd"] = ""
    calls = spy_moves(cell, markers)
    r = await cell.unload_part("fallback")
    before = [c for c in calls if not c["after_cmd"]]
    check("ok, primary command ran as fallback", r["ok"] and r["variant"] == "fallback" and prim.exists()
          and not fb.exists() and r["command"] == cmd_p, str(r)[:200])
    check("4 moves before the command", len(before) == 4, str(len(before)))
    ep = episodes(cell)[-1]
    check("episode shows fallback, primary cmd, taught steps", ep["variant"] == "fallback" and str(prim) in ep["cmd"]
          and ep.get("sequence") == seq, str(ep))
    unspy(cell)
    clean()

    print("3. primary variant unaffected: no fallback steps")
    calls = spy_moves(cell, markers)
    r = await cell.unload_part()
    check("ok, primary, no pre_seq, only moves after the command", r["ok"] and r["variant"] == "primary"
          and "pre_seq" not in r and calls and all(c["after_cmd"] for c in calls), str(r)[:200])
    check("episode has no sequence", "sequence" not in episodes(cell)[-1])
    unspy(cell)
    clean()

    print("4. neither fallback key set: refused, nothing moved")
    lc["policy_unload_fallback_pre_seq"] = []
    calls = spy_moves(cell, markers)
    n0 = len(episodes(cell))
    msg = await refused(cell.unload_part("fallback"))
    check("refused, clear message", "policy_unload_fallback_cmd is empty" in msg
          and "policy_unload_fallback_pre_seq" in msg, msg)
    check("nothing moved, nothing run, nothing logged", calls == [] and not prim.exists() and not fb.exists()
          and len(episodes(cell)) == n0)
    unspy(cell)

    print("5. untaught pose in the fallback pre-seq: refused before any motion")
    lc["policy_unload_fallback_pre_seq"] = ["gripper:open", "pose:not_taught_yet"]
    calls = spy_moves(cell, markers)
    msg = await refused(cell.unload_part("fallback"))
    check("refused with teach hint and the key", "cellbot.teach save so101 not_taught_yet" in msg
          and "policy_unload_fallback_pre_seq" in msg, msg[:200])
    check("nothing moved, nothing run", calls == [] and not prim.exists() and not fb.exists())
    unspy(cell)

    print("6. failing step: policy not started, logged")
    lc["policy_unload_fallback_pre_seq"] = list(seq)
    calls = spy_moves(cell, markers, fail_at=3)
    e0 = n_events(cell)
    r = await cell.unload_part("fallback")
    check("not ok, policy_started False, nothing run", not r["ok"] and r.get("policy_started") is False
          and not prim.exists() and not fb.exists() and r["failed_step"] == "pose:sweep_b"
          and len(calls) == 3, str(r)[:200])
    check("a guided_fail event", any(e["kind"] == "guided_fail" and "pre-sequence" in e["message"]
                                     for e in events(cell, e0)))
    ep = episodes(cell)[-1]
    check("episode logged: fallback, not ok, policy_started False, failed_step, sequence",
          ep["variant"] == "fallback" and ep["ok"] is False and ep.get("policy_started") is False
          and ep.get("failed_step") == "pose:sweep_b" and ep.get("sequence") == seq, str(ep))
    unspy(cell)
    await cell.go_home("so101")

    print("7. CLI: unload fallback confirm text shows the taught steps and the command used")
    sh = CLI(cell, yes=False, input_fn=lambda *_: "n", out=lambda s="": None)
    asked = []

    async def confirm(text, cmd):
        asked.append(text)
        return False                                   # just look at the prompt, do not run
    sh.confirm = confirm
    await sh.execute("unload fallback")
    t = asked[-1] if asked else ""
    check("confirm text: steps and primary cmd as fallback", "gripper:open -> pose:sweep_a" in t
          and "policy_unload_cmd" in t and "FIRST taught steps" in t, t[:300])

    print(f"\n{'ALL OK' if check.failed == 0 else f'{check.failed} FAILED'}")
    return check.failed


if __name__ == "__main__":
    sys.exit(1 if asyncio.run(main()) else 0)
