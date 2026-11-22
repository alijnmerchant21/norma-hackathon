"""Teleop handover (variant "teleop"): a human moves the so101 with the leader arm through Station (sim).

    python tests/test_loader_teleop.py

Checks: open -> no motion, so101 zone unknown, last pose cleared, episode open; every so101 motion entry point
(loader variants, go_home, go_to_pose, move_joints, gripper, _move) refused while open, the worker still moves
(not into the work area); a second handover refused; label refused while open; finish closes with times; label
success -> dataset_ranges with source "teleop demonstration"; finish without an open one -> error; worker not
clear -> refused; e-stop -> refused; any loader mode; stats grouping; open never counted as success; CLI; MCP;
the event loop is never blocked.
"""
import asyncio
import json
import os
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ["CELLBOT_MODE"] = "sim"
os.environ["CELLBOT_STATE"] = tempfile.mkdtemp(prefix="cellguide_teleop_")

from cellguide.guided_cell import GuidedCell, GuardError  # noqa: E402
import sim_settings  # noqa: E402


def check(name, cond, detail=""):
    print(f"  [{'ok' if cond else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    if not cond:
        check.failed += 1


check.failed = 0
OPEN_MSG = "a teleop handover is open: finish it with finish_teleop first"


def episodes(cell) -> list[dict]:
    p = cell._episodes_path()
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()] if p.exists() else []


async def refused(coro) -> str:
    try:
        await coro
    except Exception as e:  # noqa: BLE001  GuardError / EStopped
        return f"{type(e).__name__}: {e}"
    return ""


async def main():
    cell = sim_settings.apply(GuidedCell())
    await cell.ensure_connected()
    await cell.go_home("elrobot")
    await cell.go_home("so101")
    tmp = Path(tempfile.mkdtemp(prefix="teleop_cmd_"))
    ran = tmp / "ran"
    cmd = f"python3 -c \"open('{ran}', 'a').write('x')\""
    lc = cell.gcfg.setdefault("loader", {})
    lc.update(arm="so101", mode="policy", policy_load_cmd=cmd, policy_unload_cmd=cmd, policy_cwd="",
              policy_load_fallback_cmd="", policy_unload_fallback_cmd=cmd, policy_timeout_s=10,
              policy_load_pre_pose="", policy_unload_pre_pose="", policy_load_pre_seq=[], policy_unload_pre_seq=[],
              policy_load_fallback_pre_seq=[], policy_unload_fallback_pre_seq=[],
              load_mode="", unload_mode="", taught_load_seq=[], taught_unload_seq=[])

    moves, real = [], type(cell)._move

    async def spy(self, arm, target, speed=None):
        res = await real(self, arm, target, speed)   # raises before any motion if refused
        moves.append(arm)
        return res
    type(cell)._move = spy                      # class-level: the mixin's guard runs inside real()

    print("1. open a teleop unload: no motion, zone unknown, episode open")
    stalls = []

    async def ticker():
        while True:
            t = time.monotonic()
            await asyncio.sleep(0.02)
            stalls.append(time.monotonic() - t)
    tk = asyncio.create_task(ticker())
    r = await cell.unload_part("teleop")
    check("result: ok, variant teleop, handover, attempt_id, instructions", r.get("ok") is True
          and r.get("variant") == "teleop" and r.get("handover") is True and r.get("attempt_id")
          and "teleoperation ON" in r.get("instructions_for_human", "")
          and "drop-off" in r["instructions_for_human"] and "OFF" in r["instructions_for_human"], str(r)[:300])
    check("no motion commanded", moves == [], str(moves))
    check("so101 zone unknown, last pose cleared", cell.zone["so101"] == "unknown"
          and cell.last_pose["so101"] is None, f"{cell.zone} {cell.last_pose}")
    check("lock not held", not cell.lock.locked())
    e = episodes(cell)[-1]
    check("episode open: phase, mode, variant, t_start, t_end null, ok null, outcome null",
          e["id"] == r["attempt_id"] and e["phase"] == "unload" and e["mode"] == "policy"
          and e["variant"] == "teleop" and e["open"] is True and e["t_end"] is None and e["iso_end"] is None
          and e["ok"] is None and e["outcome"] is None and isinstance(e["t_start"], float), str(e))
    aid = r["attempt_id"]

    print("2. while open: every so101 motion refused, nothing moved")
    for name, coro in [("unload primary", cell.unload_part()), ("unload fallback", cell.unload_part("fallback")),
                       ("load primary", cell.load_part()), ("load teleop (second)", cell.load_part("teleop")),
                       ("go_home so101", cell.go_home("so101")),
                       ("go_to_pose so101", cell.go_to_pose("so101", cell.cfg.arms["so101"].home_pose)),
                       ("move_joints so101", cell.move_joints("so101", {})),
                       ("gripper so101", cell.gripper("so101", "open")),
                       ("_move so101", cell._move("so101", {}, None))]:
        msg = await refused(coro)
        check(f"{name} refused", OPEN_MSG in msg, msg[:160])
    check("nothing moved, policy never ran", moves == [] and not ran.exists(), str(moves))
    check("so101 still unknown", cell.zone["so101"] == "unknown")
    msg = await refused(cell.go_home("elrobot"))
    check("worker may still go home (clear pose)", msg == "", msg[:160])
    check("label refused while open", "finish_teleop" in await refused(asyncio.to_thread(
        cell.label_loader_attempt, True)))
    st = cell.loader_stats()
    g = st["by_phase_variant"].get("unload/teleop", {})
    check("stats: open listed as open, not success", g.get("open") == 1 and g.get("successes") == 0
          and st["open_teleop_handovers"] == [aid] and "unload/teleop" not in st["dataset_ranges"], str(g))
    moves.clear()

    print("3. finish closes with times; label success -> dataset range, source teleop")
    time.sleep(0.05)
    f = cell.finish_teleop()
    e = episodes(cell)[-1]
    check("finish: ok, reminder mentions OFF, go_home so101, label", f["ok"] and f["attempt_id"] == aid
          and "teleoperation is OFF" in f["reminder"] and "go_home so101" in f["reminder"]
          and "label_loader_attempt" in f["reminder"], str(f)[:300])
    check("episode closed: t_end, iso_end, seconds, open false, ok true, not labelled", e["open"] is False
          and e["ok"] is True and e["t_end"] >= e["t_start"] and e["iso_end"] and isinstance(e["seconds"], float)
          and e["outcome"] is None, str(e))
    check("finish moved nothing", moves == [])
    r2 = await cell.go_home("so101")
    check("go_home so101 allowed after finish", r2.get("ok") and cell.zone["so101"] == "clear", str(r2)[:200])
    cell.label_loader_attempt(True, "human demo", aid)
    dr = cell.loader_stats()["dataset_ranges"].get("unload/teleop", [])
    check("dataset_ranges: [iso_start, iso_end], source teleop demonstration", len(dr) == 1
          and dr[0]["start"] == e["iso_start"] and dr[0]["end"] == e["iso_end"]
          and dr[0]["source"] == "teleop demonstration" and dr[0]["attempt_id"] == aid, str(dr))

    print("4. finish without an open handover -> error")
    msg = await refused(asyncio.to_thread(cell.finish_teleop))
    check("refused: no handover open", "no teleop handover is open" in msg, msg)
    msg = await refused(asyncio.to_thread(cell.finish_teleop, "nope0000"))
    check("unknown id refused", "nope0000" in msg, msg)

    print("5. worker not clear -> refused; e-stop -> refused; nothing logged")
    n = len(episodes(cell))
    cell.zone["elrobot"] = "work"
    msg = await refused(cell.unload_part("teleop"))
    check("worker in work area: refused", "not at a clear pose" in msg, msg[:160])
    cell.zone["elrobot"] = "clear"
    cell.estop_path.write_text(json.dumps({"latched": True, "reason": "test"}))
    msg = await refused(cell.load_part("teleop"))
    check("e-stop latched: refused", "E-STOP" in msg, msg[:160])
    cell.estop_path.unlink()
    check("nothing logged, so101 zone unchanged", len(episodes(cell)) == n and cell.zone["so101"] == "clear")

    print("6. any mode; stats grouping; a policy success is source policy")
    for mode in ("taught", "human"):
        lc["mode"] = mode
        r = await cell.load_part("teleop")
        check(f"{mode} mode: load teleop opens", r["ok"] and episodes(cell)[-1]["mode"] == mode
              and "pickup area" in r["instructions_for_human"])
        cell.finish_teleop(r["attempt_id"])
        cell.label_loader_attempt(mode == "human", "", r["attempt_id"])
        await cell.go_home("so101")
    lc["mode"] = "policy"
    r = await cell.unload_part()
    cell.label_loader_attempt(True, "", r["attempt_id"])
    st = cell.loader_stats()
    bp = st["by_phase_variant"]
    check("groups unload/teleop, load/teleop, unload/primary", bp["unload/teleop"]["attempts"] == 1
          and bp["load/teleop"]["attempts"] == 2 and bp["load/teleop"]["successes"] == 1
          and bp["load/teleop"]["fails"] == 1 and bp["unload/primary"]["successes"] == 1, str(bp))
    check("policy range has source policy", st["dataset_ranges"]["unload/primary"][0]["source"] == "policy")
    check("no open handovers left", st["open_teleop_handovers"] == [])

    print("7. CLI: unload teleop, refusal, teleop done")
    from cellguide.cli import CLI
    outs = []
    sh = CLI(cell, yes=True, input_fn=lambda *_: "y", out=lambda s="": outs.append(str(s)))
    await sh.execute("unload teleop")
    check("CLI unload teleop prints HUMAN instructions", any(o.startswith("HUMAN:") for o in outs)
          and episodes(cell)[-1]["open"] is True, str(outs[-2:])[:300])
    outs.clear()
    await sh.execute("home so101 --yes")
    check("CLI home so101 refused while open", any("REFUSED" in o and OPEN_MSG in o for o in outs), str(outs))
    outs.clear()
    await sh.execute("teleop done")
    check("CLI teleop done closes", episodes(cell)[-1]["open"] is False, str(outs)[:200])
    outs.clear()
    await sh.execute("teleop done")
    check("CLI teleop done again refused", any("REFUSED" in o for o in outs), str(outs))
    await cell.go_home("so101")

    print("8. event loop never blocked")
    tk.cancel()
    check("max stall < 0.2 s", max(stalls) < 0.2, f"{max(stalls):.3f}")

    print("9. MCP tool")
    try:
        import inspect
        from cellguide import server
        check("finish_teleop(attempt_id)", list(inspect.signature(server.finish_teleop).parameters)
              == ["attempt_id"])
        check("prompt mentions the teleop path", "finish_teleop" in server.SYSTEM)
    except ImportError as ex:
        print(f"  [skip] server import: {ex}")

    type(cell)._move = real
    print("FAILED" if check.failed else "ALL OK", f"({check.failed} failed)")
    return check.failed


if __name__ == "__main__":
    sys.exit(1 if asyncio.run(main()) else 0)
