"""Loader hybrid pre-sequence (sim): taught steps with cellbot's guarded motion, then the policy carries on.

    python tests/test_loader_preseq.py

Checks: sequence -> policy command -> go_home, in that order, lock held throughout; no home inserted before the
policy; pre_pose + pre_seq both set -> refused, nothing moved; untaught pose -> refused with the teach hint;
failing step -> policy not started, zone truthful, guided_fail; e-stop latched -> refused; empty -> unchanged.
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
os.environ["CELLBOT_STATE"] = tempfile.mkdtemp(prefix="cellguide_preseq_")

from cellguide.guided_cell import GuidedCell, GuardError  # noqa: E402
import sim_settings  # noqa: E402


def check(name, cond, detail=""):
    print(f"  [{'ok' if cond else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    if not cond:
        check.failed += 1


check.failed = 0


def spy_moves(cell, marker: Path, fail_at: int | None = None):
    """Wrap cell._move: record each move (target, after the command?, lock held?); optionally fail move #fail_at."""
    calls, real = [], cell._move

    async def _move(arm, target, speed=None):
        calls.append({"arm": arm, "target": dict(target), "after_cmd": marker.exists(), "speed": speed,
                      "lock": cell.lock.locked()})
        if fail_at is not None and len(calls) == fail_at:
            return {"ok": False, "error": "simulated: timeout, not reached", "waypoint": "1/3"}
        return await real(arm, target, speed)
    cell._move = _move
    return calls


def unspy(cell):
    cell._move = type(cell)._move.__get__(cell)


def n_events(cell) -> int:
    return len(cell.events_path.read_text().splitlines()) if cell.events_path.exists() else 0


def events(cell, since: int) -> list[dict]:
    lines = cell.events_path.read_text().splitlines() if cell.events_path.exists() else []
    return [json.loads(x) for x in lines[since:]]


def is_home(cell, call) -> bool:
    home = cell.poses.get("so101", cell.cfg.arms["so101"].home_pose)["joints"]
    return all(abs(call["target"].get(j, 1e9) - v) < 1e-6 for j, v in home.items()) and len(call["target"]) > 1


async def main():
    cell = sim_settings.apply(GuidedCell())
    await cell.ensure_connected()
    await cell.go_home("elrobot")
    await cell.go_home("so101")
    tmp = Path(tempfile.mkdtemp(prefix="preseq_cmd_"))
    marker = tmp / "ran"
    cmd = f"python3 -c \"open('{marker}', 'w').write('x'); print('ok')\""
    lc = cell.gcfg.setdefault("loader", {})
    lc.update(arm="so101", mode="policy", policy_load_cmd=cmd, policy_unload_cmd=cmd, policy_cwd="",
              policy_timeout_s=10, policy_load_pre_pose="", policy_unload_pre_pose="",
              policy_load_pre_seq=[], policy_unload_pre_seq=[], load_mode="", unload_mode="")
    cell.poses.reload()
    cell.poses.save("so101", "grasp_pre", cell.poses.get("so101", "place_hover")["joints"], "work")
    cell.poses.save("so101", "grasp", cell.poses.get("so101", "place")["joints"], "work")
    gr = cell.cfg.arms["so101"].gripper
    seq = ["gripper:open", "pose:grasp_pre", "pose:grasp", "gripper:close"]

    print("1. pre-sequence, then the policy, then home; lock held")
    lc["policy_unload_pre_seq"] = list(seq)
    calls = spy_moves(cell, marker)
    e0 = n_events(cell)
    seen = {}

    async def watch():
        while not marker.exists():
            await asyncio.sleep(0.01)
        seen.update(lock=cell.lock.locked(), zone=cell.zone["so101"])
    w = asyncio.create_task(watch())
    r = await cell.unload_part()
    await asyncio.wait_for(w, 2)
    ps = r.get("pre_seq") or []
    check("result ok, exit 0, homed, zone clear", r["ok"] and r["exit_code"] == 0 and r["go_home"]["ok"]
          and cell.zone["so101"] == "clear", str(r)[:200])
    check("pre_seq: 4 steps in order, each ok with seconds", [s["step"] for s in ps] == seq
          and all(s["ok"] and isinstance(s["seconds"], float) for s in ps), str(ps))
    before = [c for c in calls if not c["after_cmd"]]
    check("4 moves before the command (grip, pose, pose, grip), all slow and under the lock", len(before) == 4
          and list(before[0]["target"]) == [gr] and list(before[3]["target"]) == [gr]
          and all(c["speed"] == cell.g["slow_speed"] and c["lock"] for c in before), str(before)[:200])
    check("no home inserted before the policy", not any(is_home(cell, c) for c in before))
    check("home after the command", calls and calls[-1]["after_cmd"] and is_home(cell, calls[-1]))
    check("while the policy ran: lock held, zone unknown", seen == {"lock": True, "zone": "unknown"}, str(seen))
    msgs = [e["message"] for e in events(cell, e0)]
    check("events: pre-sequence started/done before policy started",
          "policy unload pre-sequence started" in msgs and "policy unload pre-sequence done" in msgs
          and msgs.index("policy unload pre-sequence done") < msgs.index("policy unload started"), str(msgs))
    unspy(cell)
    marker.unlink(missing_ok=True)

    print("2. pre_pose and pre_seq both set: refused, nothing moved")
    lc["policy_unload_pre_pose"] = "grasp_pre"
    calls = spy_moves(cell, marker)
    try:
        await cell.unload_part()
        check("refused", False)
    except GuardError as e:
        check("refused with a clear message", "policy_unload_pre_pose" in str(e) and "policy_unload_pre_seq" in str(e)
              and "use one" in str(e), str(e)[:160])
    check("nothing moved, command not run", calls == [] and not marker.exists())
    lc["policy_unload_pre_pose"] = ""
    unspy(cell)

    print("3. untaught pose: refused with the teach hint, nothing moved")
    lc["policy_unload_pre_seq"] = ["gripper:open", "pose:not_taught_yet"]
    calls = spy_moves(cell, marker)
    try:
        await cell.unload_part()
        check("refused", False)
    except GuardError as e:
        check("refused with the teach command", "cellbot.teach save so101 not_taught_yet" in str(e)
              and "policy_unload_pre_seq" in str(e), str(e)[:160])
    check("nothing moved, command not run", calls == [] and not marker.exists())
    lc["policy_unload_pre_seq"] = ["pose:grasp", "bogus"]
    try:
        await cell.unload_part()
        check("bad step refused", False)
    except GuardError as e:
        check("bad step refused", "bad steps" in str(e), str(e)[:100])
    unspy(cell)

    print("4. failing step: policy not started, zone truthful")
    lc["policy_unload_pre_seq"] = list(seq)
    calls = spy_moves(cell, marker, fail_at=3)            # grip ok, grasp_pre ok, grasp fails
    e0 = n_events(cell)
    r = await cell.unload_part()
    check("not ok, policy_started False, command not run", not r["ok"] and r.get("policy_started") is False
          and not marker.exists() and "exit_code" not in r and len(calls) == 3, str(r)[:200])
    check("pre_seq lists steps up to the failure", [s["step"] for s in r["pre_seq"]] == seq[:3]
          and r["pre_seq"][-1]["ok"] is False and r["failed_step"] == "pose:grasp")
    check("zone 'work' (stopped part-way in the work area)", cell.zone["so101"] == "work" and r["zone"] == "work")
    check("a guided_fail event", any(e["kind"] == "guided_fail" and "pre-sequence" in e["message"]
                                     for e in events(cell, e0)))
    unspy(cell)
    await cell.go_home("so101")

    print("5. e-stop latched: refused before any motion")
    cell.estop_path.write_text(json.dumps({"latched": True, "reason": "test"}))
    calls = spy_moves(cell, marker)
    try:
        await cell.unload_part()
        check("refused", False)
    except Exception as e:
        check("refused (e-stop)", "E-STOP" in str(e), str(e)[:80])
    check("nothing moved, command not run", calls == [] and not marker.exists())
    cell.estop_path.unlink()
    unspy(cell)

    print("6. empty pre_seq: unchanged behaviour")
    lc["policy_unload_pre_seq"] = []
    calls = spy_moves(cell, marker)
    r = await cell.unload_part()
    check("ok, no pre_seq key, only moves after the command", r["ok"] and "pre_seq" not in r
          and "pre_pose" not in r and calls and all(c["after_cmd"] for c in calls), str(r)[:160])
    unspy(cell)

    print(f"\n{'ALL OK' if check.failed == 0 else f'{check.failed} FAILED'}")
    return check.failed


if __name__ == "__main__":
    sys.exit(1 if asyncio.run(main()) else 0)
