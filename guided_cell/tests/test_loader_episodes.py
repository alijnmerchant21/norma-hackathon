"""Loader variants (primary / fallback) and the episode log for the data flywheel (sim).

    python tests/test_loader_episodes.py

Checks: primary runs policy_<phase>_cmd with its pre-seq, fallback runs policy_<phase>_fallback_cmd WITHOUT the
pre-seq; fallback with an empty key / variant in a non-policy mode / unknown variant -> refused, nothing moved,
nothing logged; one JSON line per attempt with the right fields (exit 0, exit 3, timeout kill); labelling the
latest and by id (atomic rewrite, other lines kept); loader_stats and dataset_ranges; a log write failure does
not break the run; human and taught modes are logged; the event loop is never blocked; CLI load/label/loader.
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
os.environ["CELLBOT_STATE"] = tempfile.mkdtemp(prefix="cellguide_episodes_")

from cellguide.guided_cell import GuidedCell, GuardError  # noqa: E402
import sim_settings  # noqa: E402


def check(name, cond, detail=""):
    print(f"  [{'ok' if cond else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    if not cond:
        check.failed += 1


check.failed = 0


def episodes(cell) -> list[dict]:
    p = cell._episodes_path()
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()] if p.exists() else []


def count_moves(cell):
    calls, real = [], cell._move

    async def _move(arm, target, speed=None):
        calls.append(arm)
        return await real(arm, target, speed)
    cell._move = _move
    return calls


def unspy(cell):
    cell._move = type(cell)._move.__get__(cell)


async def refused(coro) -> str:
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
    tmp = Path(tempfile.mkdtemp(prefix="episodes_cmd_"))
    prim, fb = tmp / "primary", tmp / "fallback"
    cmd_p = f"python3 -c \"open('{prim}', 'a').write('{{task}}\\n'); print('primary')\""
    cmd_f = f"python3 -c \"open('{fb}', 'a').write('{{task}}\\n'); print('fallback')\""
    lc = cell.gcfg.setdefault("loader", {})
    lc.update(arm="so101", mode="policy", policy_load_cmd=cmd_p, policy_unload_cmd=cmd_p, policy_cwd="",
              policy_load_fallback_cmd="", policy_unload_fallback_cmd=cmd_f, policy_timeout_s=10,
              policy_load_pre_pose="", policy_unload_pre_pose="", policy_load_pre_seq=[], policy_unload_pre_seq=[],
              load_mode="", unload_mode="", taught_load_seq=[], taught_unload_seq=[],
              policy_load_fallback_pre_seq=[], policy_unload_fallback_pre_seq=[],   # independent of guided.toml
              policy_unload_task="unload task words")
    cell.poses.reload()
    cell.poses.save("so101", "grasp_pre", cell.poses.get("so101", "place_hover")["joints"], "work")
    lc["policy_unload_pre_seq"] = ["pose:grasp_pre"]

    print("1. primary: policy_unload_cmd after its pre-seq; one episode logged")
    calls = count_moves(cell)
    r = await cell.unload_part()
    ep = episodes(cell)
    check("ok, variant primary, primary command ran with the task", r["ok"] and r["variant"] == "primary"
          and prim.read_text() == "unload task words\n" and not fb.exists(), str(r)[:200])
    check("pre-seq ran (pre_seq in result, a move before the command)", "pre_seq" in r and len(calls) >= 2,
          str(calls))
    check("one episode line", len(ep) == 1, str(ep))
    e = ep[0] if ep else {}
    want = {"id", "phase", "mode", "variant", "cmd", "t_start", "t_end", "iso_start", "iso_end", "seconds",
            "exit_code", "killed", "ok", "part", "outcome"}
    check("episode fields", want <= set(e) and e["phase"] == "unload" and e["mode"] == "policy"
          and e["variant"] == "primary" and e["exit_code"] == 0 and e["killed"] is None and e["ok"] is True
          and e["outcome"] is None and "unload task words" in e["cmd"] and e["t_end"] >= e["t_start"]
          and "T" in e["iso_start"] and len(e["id"]) == 8, str(e))
    check("result carries attempt_id = episode id", r.get("attempt_id") == e.get("id"))
    unspy(cell)

    print("2. fallback: policy_unload_fallback_cmd, NO pre-seq")
    calls = count_moves(cell)
    r = await cell.unload_part("fallback")
    ep = episodes(cell)
    check("ok, fallback command ran, primary not again", r["ok"] and fb.read_text() == "unload task words\n"
          and prim.read_text().count("\n") == 1, str(r)[:200])
    check("no pre-seq: only the go_home move after the command", "pre_seq" not in r and len(calls) == 1,
          str(calls))
    check("episode variant fallback with the fallback command", len(ep) == 2 and ep[1]["variant"] == "fallback"
          and ep[1]["cmd"].startswith("python3") and str(fb) in ep[1]["cmd"], str(ep[-1:]))
    unspy(cell)
    lc["policy_unload_pre_seq"] = []

    print("3. refusals: nothing moved, nothing logged")
    calls = count_moves(cell)
    n0 = len(episodes(cell))
    msg = await refused(cell.load_part("fallback"))
    check("fallback with empty key refused, clear message", "policy_load_fallback_cmd is empty" in msg, msg)
    msg = await refused(cell.unload_part("sideways"))
    check("unknown variant refused", "variant" in msg and "Nothing moved" in msg, msg)
    lc["unload_mode"] = "human"
    msg = await refused(cell.unload_part("fallback"))
    check("fallback in human mode refused", "only in policy mode" in msg, msg)
    lc["unload_mode"] = ""
    check("no motion, no episode", calls == [] and len(episodes(cell)) == n0, f"{calls} {len(episodes(cell))}")
    unspy(cell)

    print("4. non-zero exit (3 = jump guard) and timeout kill are logged")
    lc["policy_unload_cmd"] = "python3 -c \"import sys; print('jump'); sys.exit(3)\""
    r = await cell.unload_part()
    e = episodes(cell)[-1]
    check("exit 3: not ok, logged exit_code 3", not r["ok"] and r["exit_code"] == 3 and e["exit_code"] == 3
          and e["ok"] is False and e["killed"] is None, str(e))
    await cell.go_home("so101")
    lc["policy_unload_cmd"] = "python3 -c \"import time; time.sleep(30)\""
    lc["policy_timeout_s"] = 0.6
    ticks, stop = [], asyncio.Event()

    async def ticker():                              # the event loop must keep running while the policy runs
        last = time.monotonic()
        while not stop.is_set():
            await asyncio.sleep(0.02)
            now = time.monotonic()
            ticks.append(now - last)
            last = now
    tk = asyncio.create_task(ticker())
    r = await cell.unload_part()
    stop.set()
    await tk
    e = episodes(cell)[-1]
    check("timeout: killed reason logged, not ok", "timeout" in (r.get("killed") or "")
          and "timeout" in (e["killed"] or "") and e["ok"] is False and e["seconds"] < 10, str(e))
    check("event loop never blocked > 0.2 s", ticks and max(ticks) < 0.2, f"max {max(ticks or [0]):.3f}")
    await cell.go_home("so101")
    lc["policy_timeout_s"] = 10
    lc["policy_unload_cmd"] = cmd_p

    print("5. labelling: latest and by id, atomic rewrite keeps the other lines")
    ep = episodes(cell)
    first = ep[0]["id"]
    raw_before = cell._episodes_path().read_text().splitlines()
    lab = cell.label_loader_attempt(False, "box still in work area")
    ep2 = episodes(cell)
    check("latest labelled fail with note and time", ep2[-1]["outcome"] == "fail" and lab["ok"]
          and ep2[-1]["outcome_note"] == "box still in work area" and "outcome_iso" in ep2[-1], str(ep2[-1]))
    check("other lines unchanged", cell._episodes_path().read_text().splitlines()[:-1] == raw_before[:-1])
    cell.label_loader_attempt(True, "work area empty", attempt_id=first)
    cell.label_loader_attempt(True, "", attempt_id=ep[1]["id"])
    ep3 = episodes(cell)
    check("by id: first and second labelled success", ep3[0]["outcome"] == "success"
          and ep3[1]["outcome"] == "success" and ep3[0]["outcome_note"] == "work area empty", str(ep3[:2]))
    check("relabel keeps previous_outcome", cell.label_loader_attempt(True, "", ep3[-1]["id"])["attempt"]
          .get("previous_outcome") == "fail")
    cell.label_loader_attempt(False, "relabel back", ep3[-1]["id"])
    try:
        cell.label_loader_attempt(True, "", attempt_id="nope0000")
        check("unknown id refused", False)
    except GuardError as ex:
        check("unknown id refused", "nope0000" in str(ex), str(ex))
    check("no temp files left", not [p for p in cell._episodes_path().parent.iterdir() if ".tmp" in p.name])

    print("6. loader_stats and dataset_ranges")
    st = cell.loader_stats()
    up, uf = st["by_phase_variant"].get("unload/primary", {}), st["by_phase_variant"].get("unload/fallback", {})
    check("unload/primary: 3 attempts, 1 success, 1 fail, 1 unlabelled, rate 0.5", up.get("attempts") == 3
          and up["successes"] == 1 and up["fails"] == 1 and up["unlabelled"] == 1 and up["success_rate"] == 0.5
          and isinstance(up["mean_seconds"], float), str(up))
    check("unload/fallback: 1 attempt, 1 success, rate 1.0", uf.get("attempts") == 1 and uf["successes"] == 1
          and uf["success_rate"] == 1.0, str(uf))
    dr = st["dataset_ranges"]
    check("dataset_ranges: success ranges per phase/variant, ISO start/end + source",
          [(x["start"], x["end"], x["source"]) for x in dr.get("unload/primary", [])] ==
          [(ep[0]["iso_start"], ep[0]["iso_end"], "policy")] and
          [(x["start"], x["end"], x["source"]) for x in dr.get("unload/fallback", [])] ==
          [(ep[1]["iso_start"], ep[1]["iso_end"], "policy")], str(dr))
    check("last attempts listed (<= 10)", len(st["last"]) == 4 and st["last"][-1]["id"] == ep[-1]["id"])

    print("7. a log write failure does not break the run")
    bad = tmp / "is_a_dir"
    bad.mkdir()
    cell.loader_episodes_path = bad
    prim.unlink()
    r = await cell.unload_part()
    check("run ok without attempt_id, command ran", r["ok"] and "attempt_id" not in r and prim.exists(), str(r)[:200])
    cell.loader_episodes_path = None
    n_before = len(episodes(cell))

    print("8. human and taught modes are logged too")
    lc["unload_mode"] = "human"
    r = await cell.unload_part()
    e = episodes(cell)[-1]
    check("human: logged mode human, cmd None, ok", r["ok"] and e["mode"] == "human" and e["cmd"] is None
          and e["variant"] == "primary" and len(episodes(cell)) == n_before + 1, str(e))
    lc["unload_mode"] = "taught"
    lc["taught_unload_seq"] = ["pose:grasp_pre"]
    r = await cell.unload_part()
    e = episodes(cell)[-1]
    check("taught sequence: logged mode taught", r["ok"] and e["mode"] == "taught" and e["ok"] is True, str(e))
    lc["unload_mode"], lc["taught_unload_seq"] = "", []
    st = cell.loader_stats()
    check("stats split by phase/variant regardless of mode", st["by_phase_variant"]["unload/primary"]["attempts"]
          == 5, str(st["by_phase_variant"]))

    print("9. CLI: load fallback refused, unload fallback, label, loader")
    from cellguide.cli import CLI
    outs = []
    sh = CLI(cell, yes=True, input_fn=lambda *_: "y", out=lambda s="": outs.append(str(s)))
    await sh.execute("load fallback --yes")
    check("load fallback refused (empty key)", any("REFUSED" in o and "fallback_cmd is empty" in o for o in outs),
          str(outs[-1:]))
    outs.clear()
    await sh.execute("unload fallback --yes")
    e = episodes(cell)[-1]
    check("unload fallback ran and was logged", e["variant"] == "fallback" and e["ok"], str(outs[-1:])[:300])
    await sh.execute("label fail still there")
    check("label fail sets the latest", episodes(cell)[-1]["outcome"] == "fail"
          and episodes(cell)[-1]["outcome_note"] == "still there")
    outs.clear()
    await sh.execute("loader")
    check("loader prints the stats", any("dataset_ranges" in o for o in outs), str(outs)[:200])
    outs.clear()
    await sh.execute("label maybe")
    check("label usage refused", any("usage: label" in o for o in outs), str(outs))

    print("10. MCP tools exist with the right arguments")
    try:
        import inspect
        from cellguide import server
        sig = {n: list(inspect.signature(getattr(server, n)).parameters) for n in
               ("load_part", "unload_part", "label_loader_attempt", "loader_stats")}
        check("signatures", sig == {"load_part": ["variant"], "unload_part": ["variant"],
                                    "label_loader_attempt": ["success", "note", "attempt_id"],
                                    "loader_stats": []}, str(sig))
    except ImportError as ex:                      # mcp package not installed here: skip, not a failure
        print(f"  [skip] server import: {ex}")

    print("FAILED" if check.failed else "ALL OK", f"({check.failed} failed)")
    return check.failed


if __name__ == "__main__":
    sys.exit(1 if asyncio.run(main()) else 0)
