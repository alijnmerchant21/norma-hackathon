"""Loader taught sequences and per-phase modes (sim).

    python tests/test_loader_seq.py

Checks: unload with a 3-step sweep runs in order then home, under the lock, slow; per-phase mode (load human,
unload taught seq); validation refuses before motion (pose not taught, bad step string, wait too long);
a failing step stops the rest with a truthful zone and an event; e-stop latched refuses; empty seq = the
fixed pick/place sequence.
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
os.environ["CELLBOT_STATE"] = tempfile.mkdtemp(prefix="cellguide_seq_")

from cellguide.guided_cell import GuidedCell, GuardError  # noqa: E402
import sim_settings  # noqa: E402


def check(name, cond, detail=""):
    print(f"  [{'ok' if cond else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    if not cond:
        check.failed += 1


check.failed = 0


def spy(cell, fail_at: int | None = None):
    """Record every _move (arm, target, speed, lock held); optionally make move number fail_at fail."""
    calls, real = [], type(cell)._move.__get__(cell)

    async def _move(arm, target, speed=None):
        calls.append({"arm": arm, "target": dict(target), "speed": speed, "lock": cell.lock.locked()})
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
    return [json.loads(x) for x in cell.events_path.read_text().splitlines()[since:]]


def pose_of(cell, call) -> str | None:
    """Which taught so101 pose a recorded move went to (None for gripper moves)."""
    for name in cell.poses.names("so101"):
        if cell.poses.get("so101", name)["joints"] == call["target"]:
            return name
    return None


async def main():
    cell = sim_settings.apply(GuidedCell())
    await cell.ensure_connected()
    await cell.go_home("elrobot")
    await cell.go_home("so101")
    cell.poses.reload()
    ph, dh = cell.poses.get("so101", "place_hover")["joints"], cell.poses.get("so101", "drop_hover")["joints"]
    mid = {j: (ph[j] + dh[j]) / 2 for j in ph}
    j0 = next(iter(ph))                                   # small offsets: each pose has its own joints
    cell.poses.save("so101", "sweep_start", {**ph, j0: ph[j0] + 0.004}, "work")
    cell.poses.save("so101", "sweep_mid", {**mid, j0: mid[j0] + 0.004}, "work")
    cell.poses.save("so101", "sweep_end", {**dh, j0: dh[j0] + 0.004}, "clear")
    lc = cell.gcfg.setdefault("loader", {})
    lc.update(arm="so101", mode="taught", load_mode="", unload_mode="", taught_load_seq=[], taught_unload_seq=[],
              policy_load_pre_pose="", policy_unload_pre_pose="")
    slow = cell.g["slow_speed"]

    print("1. unload: 3-pose sweep with gripper + wait, in order, then home")
    lc["taught_unload_seq"] = ["gripper:close", "pose:sweep_start", "pose:sweep_mid", "wait:0.2", "pose:sweep_end"]
    calls = spy(cell)
    e0 = n_events(cell)
    zones = []
    real_ppm = cell._pre_pose_move

    async def ppm(arm, pose, target, label="pre-pose"):
        r = await real_ppm(arm, pose, target, label)
        zones.append((pose, cell.zone[arm]))
        return r
    cell._pre_pose_move = ppm
    r = await cell.unload_part()
    del cell._pre_pose_move
    got = [s["step"] for s in r["steps"]]
    check("ok, mode taught, phase unload", r["ok"] and r["mode"] == "taught" and r["phase"] == "unload", str(r)[:200])
    check("steps in order, home appended", got == ["gripper:close", "pose:sweep_start", "pose:sweep_mid",
                                                   "wait:0.2", "pose:sweep_end", "pose:home (auto)"], str(got))
    check("each step has ok and seconds", all(s["ok"] is True and isinstance(s["seconds"], float)
                                               for s in r["steps"]))
    moved = [pose_of(cell, c) for c in calls if pose_of(cell, c)]
    check("arm moved to sweep_start, sweep_mid, sweep_end, home", moved == ["sweep_start", "sweep_mid", "sweep_end",
                                                                         "home"], str(moved))
    check("every move slow, so101 only, under the lock", calls and all(c["speed"] == slow and c["lock"]
                                                                     and c["arm"] == "so101" for c in calls))
    check("zones: work, work, clear, clear", zones == [("sweep_start", "work"), ("sweep_mid", "work"),
                                                     ("sweep_end", "clear"), ("home", "clear")], str(zones))
    check("ends home, zone clear", cell.zone["so101"] == "clear" and cell.last_pose["so101"] == "home")
    msgs = [e["message"] for e in events(cell, e0)]
    check("events: started, done", "taught unload sequence started" in msgs and "taught unload sequence done" in msgs,
          str(msgs))
    unspy(cell)

    print("2. explicit pose:home at the end is not doubled")
    lc["taught_unload_seq"] = ["pose:sweep_start", "pose:sweep_end", "pose:home"]
    r = await cell.unload_part()
    check("3 steps, last pose:home", r["ok"] and [s["step"] for s in r["steps"]] ==
          ["pose:sweep_start", "pose:sweep_end", "pose:home"], str(r["steps"]))

    print("3. per-phase mode: load human, unload taught seq (mode = policy)")
    lc.update(mode="policy", load_mode="human", unload_mode="taught", policy_load_cmd="false",
              policy_unload_cmd="false", taught_unload_seq=["pose:sweep_start", "pose:sweep_end"])
    calls = spy(cell)
    r = await cell.load_part()
    check("load: human, no motion", r["mode"] == "human" and r["ok"] and calls == [], str(r)[:120])
    r = await cell.unload_part()
    check("unload: taught sequence, not the policy", r["mode"] == "taught" and r["ok"] and "exit_code" not in r
          and r["sequence"][-1] == "pose:home (auto)", str(r)[:160])
    check("CLI/MCP helper: _phase_mode", cell._phase_mode("load") == "human" and cell._phase_mode("unload") == "taught")
    lc["unload_mode"] = ""
    check("unload_mode '' falls back to mode", cell._phase_mode("unload") == "policy")
    lc.update(mode="taught", load_mode="", unload_mode="")
    unspy(cell)

    print("4. validation refuses before any motion")
    for seq, want in ((["pose:sweep_start", "pose:not_taught_yet", "pose:sweep_end"],
                       "cellbot.teach save so101 not_taught_yet work"),
                      (["pose:sweep_start", "push:sweep_end"], "bad steps"),
                      (["pose:sweep_start", "gripper:half"], "bad steps"),
                      (["pose:sweep_start", "wait:9"], "wait needs seconds"),
                      ("pose:sweep_start", "must be a list")):
        lc["taught_unload_seq"] = seq
        calls = spy(cell)
        z0, p0 = cell.zone["so101"], cell.last_pose["so101"]
        try:
            await cell.unload_part()
            check(f"refused {seq!r}", False)
        except GuardError as e:
            check(f"refused {seq!r}", want in str(e) and "Nothing moved" in str(e) or want == "must be a list"
                  and want in str(e), str(e)[:120])
        check("  nothing moved, zone unchanged", calls == [] and cell.zone["so101"] == z0
              and cell.last_pose["so101"] == p0)
        unspy(cell)
    lc["taught_unload_seq"] = ["pose:not_taught_yet"]
    try:
        await cell.unload_part()
    except GuardError as e:
        m = str(e)
        check("teach hint has limp / save / hold and the env", "cellbot.teach limp so101" in m
              and "cellbot.teach hold so101" in m and f"CELLBOT_STATE={cell.cfg.state_dir.resolve()}" in m, m[:150])

    print("5. a failing step stops the rest; zone truthful; event")
    lc["taught_unload_seq"] = ["pose:sweep_start", "pose:sweep_mid", "pose:sweep_end"]
    calls = spy(cell, fail_at=2)                          # sweep_mid fails
    e0 = n_events(cell)
    r = await cell.unload_part()
    check("not ok, failed_step pose:sweep_mid", not r["ok"] and r["failed_step"] == "pose:sweep_mid", str(r)[:200])
    check("only 2 steps reported, last has the error", len(r["steps"]) == 2 and r["steps"][0]["ok"]
          and not r["steps"][1]["ok"] and "simulated" in r["steps"][1]["error"])
    check("no more moves after the failure", len(calls) == 2)
    check("zone 'work' (work pose, stopped part-way), last_pose None", cell.zone["so101"] == "work"
          and r["zone"] == "work" and cell.last_pose["so101"] is None)
    check("reason names the step and go_home", "sweep_mid" in r["reason"] and "go_home so101" in r["reason"])
    check("guided_fail event", any(e["kind"] == "guided_fail" and "sweep_mid" in e["message"]
                                   for e in events(cell, e0)))
    unspy(cell)
    try:
        await cell.go_to_pose("elrobot", "mark_hover_1")
        check("worker refused the work area after the failure", False)
    except GuardError as e:
        check("worker refused the work area after the failure", "not free" in str(e))
    await cell.go_home("so101")

    print("5b. failure moving to a clear pose -> zone unknown")
    calls = spy(cell, fail_at=3)                          # sweep_end (clear) fails
    r = await cell.unload_part()
    check("failed at sweep_end, zone unknown", not r["ok"] and r["failed_step"] == "pose:sweep_end"
          and cell.zone["so101"] == "unknown", str(r.get("failed_step")))
    unspy(cell)
    await cell.go_home("so101")

    print("6. e-stop latched: refused, nothing moved")
    cell.estop_path.write_text(json.dumps({"latched": True, "reason": "test"}))
    calls = spy(cell)
    try:
        await cell.unload_part()
        check("refused", False)
    except Exception as e:
        check("refused (e-stop)", "E-STOP" in str(e), str(e)[:80])
    check("nothing moved", calls == [])
    cell.estop_path.unlink()
    unspy(cell)

    print("7. worker not clear: refused")
    await cell.go_to_pose("elrobot", "mark_hover_1")
    try:
        await cell.unload_part()
        check("refused", False)
    except GuardError as e:
        check("refused (worker not clear)", "not at a clear pose" in str(e))
    await cell.go_home("elrobot")

    print("8. empty seq: the fixed pick/place sequence, unchanged")
    lc.update(taught_unload_seq=[], taught_load_seq=[])
    r = await cell.load_part()
    check("load: fixed steps", r["mode"] == "taught" and "sequence" not in r
          and [s["step"] for s in r["steps"]][:2] == ["pick_hover", "pick"], str(r)[:160])
    r = await cell.unload_part()
    check("unload: fixed steps", "sequence" not in r and r["steps"][0]["step"] == "place_hover", str(r)[:160])

    print("9. load sequence uses its own key")
    lc["taught_load_seq"] = ["pose:pick_hover", "gripper:open"]
    r = await cell.load_part()
    check("load seq ok, home appended", r["ok"] and r["phase"] == "load" and [s["step"] for s in r["steps"]] ==
          ["pose:pick_hover", "gripper:open", "pose:home (auto)"], str(r)[:160])

    print(f"\n{'ALL OK' if check.failed == 0 else f'{check.failed} FAILED'}")
    return check.failed


if __name__ == "__main__":
    sys.exit(1 if asyncio.run(main()) else 0)
