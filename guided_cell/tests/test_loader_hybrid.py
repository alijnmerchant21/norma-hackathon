"""Loader hybrid (sim): a taught so101 pre-pose is reached with cellbot's guarded motion, then the policy runs.

    python tests/test_loader_hybrid.py

Checks: pre-pose reached -> policy command run -> go_home, all under the lock; pre-pose not taught -> refused with
the teach commands, nothing moved, command not run; pre-move failure -> command not run, zone truthful;
e-stop latched -> refused before motion; empty key -> exactly the old behaviour (no pre_pose in the result).
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
os.environ["CELLBOT_STATE"] = tempfile.mkdtemp(prefix="cellguide_hybrid_")

from cellguide.guided_cell import GuidedCell, GuardError  # noqa: E402
import sim_settings  # noqa: E402


def check(name, cond, detail=""):
    print(f"  [{'ok' if cond else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    if not cond:
        check.failed += 1


check.failed = 0


def spy_moves(cell, marker: Path, fail_first: bool = False):
    """Wrap cell._move: record (arm, ran-command-already?) per move; optionally make the first move fail."""
    calls, real = [], cell._move

    async def _move(arm, target, speed=None):
        calls.append({"arm": arm, "after_cmd": marker.exists(), "speed": speed, "lock": cell.lock.locked()})
        if fail_first and len(calls) == 1:
            return {"ok": False, "error": "simulated: timeout, not reached", "waypoint": "1/3"}
        return await real(arm, target, speed)
    cell._move = _move
    return calls


def events(cell, since: int) -> list[dict]:
    lines = cell.events_path.read_text().splitlines() if cell.events_path.exists() else []
    return [json.loads(x) for x in lines[since:]]


def n_events(cell) -> int:
    return len(cell.events_path.read_text().splitlines()) if cell.events_path.exists() else 0


async def main():
    cell = sim_settings.apply(GuidedCell())
    await cell.ensure_connected()
    await cell.go_home("elrobot")
    await cell.go_home("so101")
    tmp = Path(tempfile.mkdtemp(prefix="hybrid_cmd_"))
    marker = tmp / "ran"
    cmd = f"python3 -c \"open('{marker}', 'w').write('x'); print('ok')\""
    lc = cell.gcfg.setdefault("loader", {})
    lc.update(arm="so101", mode="policy", policy_load_cmd=cmd, policy_unload_cmd=cmd, policy_cwd="",
              policy_timeout_s=10, policy_load_pre_pose="", policy_unload_pre_pose="")

    # a human-taught pre-pose (sim pose store): near the box in the work area
    cell.poses.reload()
    hover = cell.poses.get("so101", "place_hover")
    cell.poses.save("so101", "unload_pre", hover["joints"], "work")
    slow = cell.g["slow_speed"]

    print("1. pre-pose taught: reached, then the policy, then home")
    lc["policy_unload_pre_pose"] = "unload_pre"
    calls = spy_moves(cell, marker)
    e0 = n_events(cell)
    seen = {}

    async def watch():                                     # while the command runs: lock held, zone unknown
        while not marker.exists():
            await asyncio.sleep(0.01)
        seen.update(lock=cell.lock.locked(), zone=cell.zone["so101"])
    w = asyncio.create_task(watch())
    r = await cell.unload_part()
    await asyncio.wait_for(w, 2)
    pre = r.get("pre_pose") or {}
    check("result ok, exit 0, homed, zone clear", r["ok"] and r["exit_code"] == 0 and r["go_home"]["ok"]
          and cell.zone["so101"] == "clear" and cell.last_pose["so101"] == "home", str(r)[:160])
    check("pre_pose in the result (name, ok, seconds)", pre.get("name") == "unload_pre" and pre.get("ok") is True
          and isinstance(pre.get("seconds"), float), str(pre))
    check("order: pre-pose move before the command, home after", len(calls) >= 2 and not calls[0]["after_cmd"]
          and calls[-1]["after_cmd"] and all(c["arm"] == "so101" for c in calls), str(calls))
    check("pre-pose move at slow speed, under the lock", calls[0]["speed"] == slow and calls[0]["lock"])
    check("while the policy ran: lock held, so101 zone unknown", seen == {"lock": True, "zone": "unknown"}, str(seen))
    msgs = [e["message"] for e in events(cell, e0)]
    check("events: pre-pose reached, policy started, ended", "pre-pose unload_pre reached" in msgs
          and msgs.index("pre-pose unload_pre reached") < msgs.index("policy unload started")
          < msgs.index("policy unload ended"), str(msgs))
    cell._move = type(cell)._move.__get__(cell)
    marker.unlink(missing_ok=True)

    print("2. pre-pose not taught: refused before motion, teach commands given")
    lc["policy_unload_pre_pose"] = "not_taught_yet"
    calls = spy_moves(cell, marker)
    z0, p0 = cell.zone["so101"], cell.last_pose["so101"]
    try:
        await cell.unload_part()
        check("refused", False)
    except GuardError as e:
        msg = str(e)
        check("refused with the exact teach command and env", "cellbot.teach save so101 not_taught_yet work" in msg
              and f"CELLBOT_STATE={cell.cfg.state_dir.resolve()}" in msg and "CELLBOT_CONFIG=" in msg
              and "cellbot.teach limp so101" in msg and "cellbot.teach hold so101" in msg, msg[:200])
    check("nothing moved, command not run, zone unchanged", calls == [] and not marker.exists()
          and cell.zone["so101"] == z0 and cell.last_pose["so101"] == p0)
    cell._move = type(cell)._move.__get__(cell)

    print("3. pre-move fails: policy not started, zone truthful")
    lc["policy_unload_pre_pose"] = "unload_pre"
    calls = spy_moves(cell, marker, fail_first=True)
    e0 = n_events(cell)
    r = await cell.unload_part()
    check("not ok, policy not started, command not run", not r["ok"] and r.get("policy_started") is False
          and not marker.exists() and "exit_code" not in r and len(calls) == 1, str(r)[:160])
    check("pre_pose ok False with the error", r["pre_pose"]["ok"] is False and "simulated" in r["pre_pose"]["error"])
    check("zone stays 'work' (work-zone pose, stopped part-way), last_pose None",
          cell.zone["so101"] == "work" and cell.last_pose["so101"] is None and r["zone"] == "work")
    check("a guided_fail event", any(e["kind"] == "guided_fail" and "pre-pose" in e["message"]
                                     for e in events(cell, e0)))
    cell._move = type(cell)._move.__get__(cell)
    try:
        await cell.go_to_pose("elrobot", "mark_hover_1")
        check("worker refused the work area after the failed pre-move", False)
    except GuardError as e:
        check("worker refused the work area after the failed pre-move", "not free" in str(e), str(e)[:80])
    await cell.go_home("so101")

    print("4. e-stop latched: refused before any motion")
    cell.estop_path.write_text(json.dumps({"latched": True, "reason": "test"}))
    calls = spy_moves(cell, marker)
    try:
        await cell.unload_part()
        check("refused", False)
    except Exception as e:
        check("refused (e-stop)", "E-STOP" in str(e), str(e)[:80])
    check("nothing moved, command not run, zone clear", calls == [] and not marker.exists()
          and cell.zone["so101"] == "clear")
    cell.estop_path.unlink()
    cell._move = type(cell)._move.__get__(cell)

    print("5. empty key: unchanged behaviour")
    lc.update(policy_unload_pre_pose="", policy_load_pre_pose="")
    calls = spy_moves(cell, marker)
    r = await cell.unload_part()
    check("ok, no pre_pose key, only the home move after the command", r["ok"] and "pre_pose" not in r
          and len(calls) >= 1 and all(c["after_cmd"] for c in calls), str(r)[:120])
    cell._move = type(cell)._move.__get__(cell)
    marker.unlink(missing_ok=True)

    print("6. load phase uses its own key (clear-zone pre-pose)")
    pick_hover = cell.poses.get("so101", "pick_hover")
    cell.poses.save("so101", "load_pre", pick_hover["joints"], "clear")
    lc["policy_load_pre_pose"] = "load_pre"
    r = await cell.load_part()
    check("load: pre-pose then policy then home", r["ok"] and r["pre_pose"]["name"] == "load_pre"
          and r["pre_pose"]["ok"] and marker.exists() and cell.zone["so101"] == "clear", str(r)[:120])

    print(f"\n{'ALL OK' if check.failed == 0 else f'{check.failed} FAILED'}")
    return check.failed


if __name__ == "__main__":
    sys.exit(1 if asyncio.run(main()) else 0)
