"""Read-only readiness checklist for the guided cell. It never sends a motion or torque command.

    python -m cellguide.preflight [--json]

Same env as the MCP server (PYTHONPATH, CELLBOT_CONFIG, CELLBOT_STATE, CELLBOT_MODE, STATION_HOST). The MCP tool
`preflight` returns the same structure. Each check: {name, status: ok|warn|fail|skip, detail, fix}.
In mode "station" it connects to Station only to read joint data (connect, read_state, _motors).
"""
from __future__ import annotations

import argparse
import asyncio
import importlib
import json
import os
import sys
import time
from pathlib import Path

from . import config as gconfig  # noqa: F401  (first: tomllib fallback on Python 3.10)

STATUSES = ("ok", "warn", "fail", "skip")
READY = "READY for a slow hover-only first step"
LOADER_POSES = ["pick_hover", "pick", "place_hover", "place", "drop_hover", "drop"]
STL = ["Gripper_Base_v1_1.stl", "Gripper_Jaw_01_v1_1.stl", "Gripper_Jaw_02_v1_1.stl"]
RANGE_TOL_TICKS = 20


class _Run:
    def __init__(self):
        self.checks: list[dict] = []
        self.ctx: dict = {}

    def add(self, name, status, detail="", fix=""):
        assert status in STATUSES
        self.checks.append({"name": name, "status": status, "detail": detail, "fix": fix})

    async def step(self, name, fn):
        """Run one check function; a crash becomes a fail and does not stop the others."""
        try:
            r = fn()
            if asyncio.iscoroutine(r):
                await r
        except BaseException as e:  # noqa: BLE001  (SystemExit from config loading included)
            if isinstance(e, KeyboardInterrupt):
                raise
            self.add(name, "fail", f"internal error: {e!r}", "see the error; the other checks still ran")


def verdict(checks: list[dict]) -> str:
    n = sum(1 for c in checks if c["status"] == "fail")
    return READY if n == 0 else f"NOT READY: {n} blocking item(s)"


def _skip(run, name, why):
    run.add(name, "skip", why)


# ============================================================== a. python
def check_python(run):
    v = sys.version_info
    run.add("python", "ok" if v >= (3, 10) else "fail", f"{sys.executable} {v.major}.{v.minor}.{v.micro}",
            "" if v >= (3, 10) else "needs Python 3.10+: run bash setup_mac.sh and use .venv/bin/python")
    missing = []
    for m in ("numpy", "cv2", "scipy", "mcp", "PIL") + (("tomli",) if v < (3, 11) else ()):
        try:
            importlib.import_module(m)
        except Exception as e:  # noqa: BLE001
            missing.append(f"{m} ({type(e).__name__})")
    run.add("python packages", "fail" if missing else "ok",
            f"missing: {', '.join(missing)}" if missing else "numpy, cv2, scipy, mcp, PIL import",
            "bash setup_mac.sh (installs cellbot's requirements.txt + scipy, and tomli on 3.10, into .venv)" if missing else "")


# ============================================================== b. files
def check_files(run):
    from . import config as gconfig
    p = Path(os.environ.get("CELLGUIDE_CONFIG", gconfig.ROOT / "guided.toml"))
    try:
        g = gconfig.load(p)
    except SystemExit as e:
        run.add("guided.toml paths", "fail", f"{p}: {e}", "fix the path in guided.toml [paths]")
        return
    run.ctx["gcfg"] = g
    paths = g["_paths"]
    lines = [f"{k}={v}" for k, v in paths.items() if v is not None]
    aim = paths.get("aim_file")
    st = "ok"
    fix = ""
    if aim is not None and not aim.exists():
        st, fix = "warn", "aim_file is set but missing: set it to \"\" or teach it with elrobot_ik/aim.py"
    run.add("guided.toml paths", st, f"{p}: " + "; ".join(lines), fix)

    d = paths["elrobot_ik_dir"]
    need = ["joint_map.json", "tool_offset.json", "elrobot_follower.urdf"] + [f"assets/{s}" for s in STL]
    miss = [n for n in need if not (d / n).exists()]
    run.add("elrobot_ik files", "fail" if miss else "ok",
            f"missing in {d}: {miss}" if miss else f"{d}: joint map, tool offset, URDF, 3 gripper STLs",
            "copy the full elrobot_ik folder (joint_map.json comes from live.py, tool_offset.json from refine.py)"
            if miss else "")

    c = json.loads(paths["calib_file"].read_text())
    run.ctx["calib"] = c
    n = len(c.get("points", []))
    st = "ok" if n >= 8 else "warn"
    run.add("calibration", st, f"{paths['calib_file'].name}: {n} markers, created {c.get('created')}, "
            f"image_size {c.get('image_size')}, follower {c.get('follower')}",
            "" if st == "ok" else "fewer than 8 markers: redo elrobot_ik/calibrate_table.py with more, spread out")


# ============================================================== c. cellbot config
def check_cellbot_config(run):
    g = run.ctx.get("gcfg")
    if g is None:
        return _skip(run, "cellbot config", "guided.toml did not load")
    from . import config as gconfig
    gconfig.use_cellbot(g)
    from cellbot import config as cc
    cfg = cc.load_config()
    run.ctx["cfg"] = cfg
    st = cfg.station
    mode, host = st["mode"], st["host"]
    src = f"CELLBOT_CONFIG={os.environ['CELLBOT_CONFIG']}" if os.environ.get("CELLBOT_CONFIG") \
        else f"default {cc.ROOT / 'cell.toml'}"
    status, fix = "ok", ""
    if mode == "station" and host not in ("127.0.0.1", "localhost"):
        status, fix = "warn", ("Station runs on this Mac: use cell.guided.toml via CELLBOT_CONFIG "
                               "(or STATION_HOST=127.0.0.1)")
    run.add("cellbot config", status, f"{src}; mode {mode}; host {host}:{st['port']}", fix)

    raw = str(st["norma_core_dir"])
    repo = Path(raw)
    if not repo.is_absolute():
        repo = (cc.ROOT / repo).resolve()
    ok = (repo / "target" / "gen_python").exists()
    note = " (cellbot does not expand ~)" if raw.startswith("~") else ""
    if ok:
        run.add("norma-core", "ok", f"{repo} has target/gen_python")
    else:
        using_ours = os.environ.get("CELLBOT_CONFIG", "").endswith("cell.guided.toml")
        run.add("norma-core", "fail" if mode == "station" else "warn",
                f"{raw}{'' if str(repo) == raw else ' -> ' + str(repo)}: no target/gen_python{note}"
                + ("" if mode == "station" else " (not needed in sim)"),
                "the norma-core clone with generated protobufs (target/gen_python) must be at the norma_core_dir of "
                "cell.guided.toml" if using_ours else "use cell.guided.toml via CELLBOT_CONFIG")
    run.add("guards", "ok", ", ".join(f"{k}={v}" for k, v in cfg.guard.items()))


# ============================================================== d. e-stop
def check_estop(run):
    cfg = run.ctx.get("cfg")
    if cfg is None:
        return _skip(run, "e-stop", "cellbot config did not load")
    p = cfg.state_dir / "estop.json"
    try:
        s = json.loads(p.read_text())
    except FileNotFoundError:
        s = {"latched": False}
    cb = run.ctx["gcfg"]["_paths"]["cellbot_dir"]
    if s.get("latched"):
        run.add("e-stop", "fail", f"LATCHED: {s.get('reason', 'no reason')} ({p})",
                f"a human checks the cell, then: cd \"{cb}\" && CELLBOT_STATE=\"{cfg.state_dir}\" "
                "python -m cellbot.estop release")
    else:
        run.add("e-stop", "ok", f"released ({p})")


# ============================================================== cell
async def make_cell(run):
    if run.ctx.get("cell") is not None or run.ctx.get("cfg") is None:
        return
    from . import guided_cell as _gc
    from .guided_cell import GuidedCell
    from .vision_source import install
    install(_gc)                          # camera:auto + self-healing camera:N
    try:
        run.ctx["cell"] = GuidedCell()
    except Exception as e:  # noqa: BLE001
        run.ctx["cell_error"] = repr(e)
        run.add("cell", "fail", f"could not build the cell: {e!r}", "fix the items above, then run preflight again")


# ============================================================== e. Station
async def check_station(run):
    cell = run.ctx.get("cell")
    if cell is None:
        return _skip(run, "Station link", "no cell")
    if cell.backend.name != "station":
        return _skip(run, "Station link", f"mode {cell.cfg.station['mode']}: Station checks skipped")
    be, cfg, margin = cell.backend, cell.cfg, float(cell.g["limit_margin"])
    want = {a.name: a.bus_serial for a in cfg.arms.values()}
    try:
        await asyncio.wait_for(cell.ensure_connected(), 15)
    except Exception as e:  # noqa: BLE001
        found = {}
        if getattr(be, "latest", None) is not None:
            found = {s: len(b.get_motors() or []) for s, b in be._buses().items()}
        run.add("Station link", "fail", f"{cfg.station['host']}:{cfg.station['port']}: {e}; buses found {found}, "
                f"configured {want}", "start Station with --tcp, connect both arms, check bus_serial in the cell config")
        return
    found = {s: len(b.get_motors() or []) for s, b in be._buses().items()}
    run.add("Station link", "ok", f"buses found {found}; configured {want}; mapping {be.bus_of}")
    run.ctx["station_ok"] = True

    worker = cell.arm
    bad, ranges = [], {}
    for a in cfg.arms.values():
        ms = be._motors(a)
        for j, mid in zip(a.joint_names, a.motor_ids):
            m = ms.get(mid)
            if m is None or (m["rmin"] == 0 and m["rmax"] == 0):
                bad.append(f"{a.name}.{j} (motor {mid})" + (" missing" if m is None else ""))
            elif a.name == worker:
                ranges[mid] = (m["rmin"], m["rmax"])
    run.add("Station calibration", "fail" if bad else "ok",
            f"not calibrated: {bad}" if bad else "every joint of both arms has a calibrated range",
            "calibrate those motors in the Station web UI" if bad else "")
    if bad:
        return

    jm = json.loads((run.ctx["gcfg"]["_paths"]["elrobot_ik_dir"] / "joint_map.json").read_text())
    diffs = []
    for i in range(1, 8):
        lo, hi = jm[f"rev_motor_0{i}"]["range_ticks"]
        smin, smax = ranges.get(i, (None, None))
        if smin is None:
            continue
        d = max(abs((smin - lo + 2048) % 4096 - 2048), abs((smax - hi + 2048) % 4096 - 2048))
        if d > RANGE_TOL_TICKS:
            diffs.append(f"j{i}: Station {smin}..{smax} vs joint_map {lo}..{hi}")
    run.add("Station ranges vs joint_map", "warn" if diffs else "ok",
            "; ".join(diffs) if diffs else f"all 7 worker joints within {RANGE_TOL_TICKS} ticks of joint_map.json",
            "the arm was recalibrated in Station after the joint map / table calibration were made: redo elrobot_ik "
            "refine + calibrate_table, and re-teach poses" if diffs else "")

    state = await be.read_state()
    pos, edge, torque = [], [], []
    for a in cfg.arms.values():
        js = state[a.name]
        pos.append(f"{a.name}: " + " ".join(f"{j}={js[j]['pos']:.3f}" for j in a.joint_names))
        edge += [f"{a.name}.{j}={js[j]['pos']:.3f}" for j in a.arm_joints
                 if not (margin <= js[j]["pos"] <= 1 - margin)]
        torque.append(f"{a.name} {sum(1 for j in a.joint_names if js[j]['torque'])}/{len(a.joint_names)}")
    run.add("joint positions", "warn" if edge else "ok",
            "; ".join(pos) + (f"; on an end stop: {edge}" if edge else ""),
            f"resting on an end stop (outside [{margin}, {1 - margin}]): taught poses and IK targets must stay "
            "inside; move it off the stop before teaching" if edge else "")
    run.add("torque", "ok", "torque on: " + ", ".join(torque))


# ============================================================== f. poses
def check_poses(run):
    cell = run.ctx.get("cell")
    if cell is None:
        return _skip(run, "taught poses", "no cell")
    cell.poses.reload()
    loader = cell.gcfg.get("loader", {}).get("arm", "so101")
    w = cell.arm
    have = {a: set(cell.poses.names(a)) for a in cell.cfg.arms}
    f = cell.poses.path
    look = cell.m.get("look_pose", "look")
    if cell.cfg.arms[loader].home_pose in have.get(loader, ()):
        run.add(f"pose {loader} home", "ok", f"{f}")
    else:
        run.add(f"pose {loader} home", "fail", f"missing in {f}",
                "a human teaches it: python -m cellbot.teach save so101 home clear (same CELLBOT_STATE); the worker "
                "cannot enter the work area until the loader is homed")
    if cell.cfg.arms[w].home_pose in have.get(w, ()):
        run.add(f"pose {w} home", "ok", f"{f}")
    else:
        run.add(f"pose {w} home", "fail", f"missing in {f}",
                f"a human teaches it: python -m cellbot.teach save {w} home clear (same CELLBOT_STATE)")
    if look in have.get(w, ()):
        run.add(f"pose {w} {look}", "ok", "")
    else:
        run.add(f"pose {w} {look}", "warn", "missing: retreat falls back to home",
                f"teach it with the arm fully out of the top camera's view: python -m cellbot.teach save {w} {look} clear "
                "(same CELLBOT_STATE)")
    if cell.gcfg.get("loader", {}).get("mode") == "taught":
        miss = [p for p in LOADER_POSES if p not in have.get(loader, ())]
        run.add("loader poses (taught mode)", "warn" if miss else "ok",
                f"missing: {miss}" if miss else ", ".join(LOADER_POSES),
                "a human teaches them with python -m cellbot.teach save so101 <pose> <zone>" if miss else "")
    bad = []
    for a in cell.cfg.arms:
        for name in cell.poses.names(a):
            for j, v in cell.poses.get(a, name)["joints"].items():
                try:
                    lo, hi = cell._limits(a, j)
                except Exception:  # noqa: BLE001
                    continue
                if not (lo <= float(v) <= hi):
                    bad.append(f"{a}.{name}.{j}={v} outside [{lo:.2f}, {hi:.2f}]")
    run.add("pose values within limits", "fail" if bad else "ok", "; ".join(bad) if bad else "all stored poses",
            "re-teach those poses off the end stops (cellbot refuses them)" if bad else "")


# ============================================================== g-j: need kinematics + vision
async def _guided(run):
    cell = run.ctx.get("cell")
    if cell is None or run.ctx.get("guided") is not None:
        return run.ctx.get("guided", False)
    if cell.backend.name == "station" and not run.ctx.get("station_ok"):
        run.ctx["guided"] = False
        run.ctx["guided_error"] = "no Station link"
        return False
    from . import guided_cell as _gc
    from .vision_source import install
    install(_gc)                          # camera:auto + self-healing camera:N (cell built elsewhere)
    try:
        await asyncio.wait_for(cell.ensure_guided(), 20)
        run.ctx["guided"] = True
    except Exception as e:  # noqa: BLE001
        run.ctx["guided"] = False
        run.ctx["guided_error"] = repr(e)
        from .vision_source import VisionError
        if isinstance(e, VisionError) and "camera" in str(e):
            run.add("vision", "fail", f"{e}", "close the other program using the camera (e.g. the standalone "
                    "vision server), check [vision] source (camera:N or camera:auto) and width/height; "
                    "python -m cellguide.vision_source probe lists the cameras")
        else:
            run.add("kinematics", "fail", f"could not set up kinematics/vision: {e!r}",
                    "fix the Station ranges / elrobot_ik files above")
    return run.ctx["guided"]


async def check_tool(run):
    if not await _guided(run):
        return _skip(run, "tool tip", run.ctx.get("guided_error", "no cell"))
    cell = run.ctx["cell"]
    src = cell.kin.tool_source
    auto = bool(cell.gcfg.get("tool", {}).get("close_gripper", False))
    if src.startswith("closed fingertips") and auto:
        run.add("tool tip", "ok", f"closed gripper fingertips are the tool (no pen); {src}. guided motion closes "
                "the gripper first ([tool] close_gripper = true)")
    elif src.startswith("closed fingertips"):
        run.add("tool tip", "warn", f"closed gripper fingertips are the tool (no pen); {src}. [tool] "
                "close_gripper = false", "close the gripper before guided motion (gripper elrobot close), or set "
                "[tool] close_gripper = true")
    else:
        run.add("tool tip", "ok", src)


async def check_part(run):
    if not await _guided(run):
        return _skip(run, "part geometry", run.ctx.get("guided_error", "no cell"))
    k = run.ctx["cell"].kin
    sh, ch = k.surface_h * 1000, k.cam_h * 1000
    d = f"surface_height_mm {sh:.0f}; parallax correction {'on (camera_height_mm %.0f)' % ch if ch > 0 else 'off'}"
    if sh > 0 and ch == 0:
        run.add("part geometry", "warn", d,
                f"if table_calib was touched on the table, a {sh:.0f} mm tall part shifts targets sideways near the "
                "image edges; make a lid-height calibration or set camera_height_mm")
    else:
        run.add("part geometry", "ok", d)


async def check_vision(run):
    cell = run.ctx.get("cell")
    cfg_src = ((cell.gcfg if cell is not None else run.ctx.get("gcfg")) or {}).get("vision", {}).get("source", "")
    if cfg_src.startswith("station:"):
        from .vision_source import ASPECT_FIX, STATION_FRAMES_NOTE
        run.add("vision source", "fail", f"[vision] source = {cfg_src!r}: {STATION_FRAMES_NOTE}", ASPECT_FIX)
    if not await _guided(run):
        return _skip(run, "vision", run.ctx.get("guided_error", "no cell"))
    from .guided_cell import GuardError
    from .vision_source import ASPECT_FIX, CameraVision, image_size_check
    cell = run.ctx["cell"]
    if isinstance(cell.vision, CameraVision):      # frames this process reads itself: size and real rate
        n0, t0 = cell.vision.n, time.monotonic()
        await asyncio.sleep(1.0)
        fps = (cell.vision.n - n0) / (time.monotonic() - t0)
        w, h = cell.vision.size
        label = getattr(cell.vision, "label", f"camera {cell.vision.index}")
        run.add("camera", "ok" if fps >= 5 else "warn", f"{label}: {w}x{h}, {fps:.1f} fps processed",
                "" if fps >= 5 else "slow: the tracker needs a steady frame rate (USB bandwidth, lighting, CPU)")
        for wmsg in getattr(cell.vision, "warnings", []):
            run.add("camera selection", "warn", wmsg, "set [vision] source to the camera actually used, or "
                    "camera:auto; refresh the reference with python -m cellguide.vision_source save-ref N")
    st, err, t_end = None, None, time.monotonic() + 1.5
    while time.monotonic() < t_end:                # a few reads: an in-process tracker needs frames to lock
        try:
            st = await cell.vision.state()
            err = None
            if cell._live(st) and st.get("map_locked"):
                break
        except Exception as e:  # noqa: BLE001
            err = e
        await asyncio.sleep(0.1)
    if st is None:
        run.add("vision", "fail", f"{err}",
                "check the camera index and that no other program holds it" if cfg_src.startswith("camera:") else
                "start Manasa's server: python vision.py --camera <top> --serve 8765 --no-window")
        return
    age = max(0.0, time.time() - float(st["timestamp"]))
    lim = float(cell.gcfg["vision"]["max_state_age_s"])
    run.add("vision", "ok" if age <= lim else "warn", f"{cell.vision.describe()}; state age {age:.2f} s",
            "" if age <= lim else f"older than max_state_age_s {lim}: vision stalled or slow")
    cs = cell.kin.calib.get("image_size")
    if not cs:
        run.add("image size", "warn", f"vision {st['image_size']}; calibration has no image_size", "")
        size_ok = True
    else:
        s, why = image_size_check(st["image_size"], cs)
        run.add("image size", s, why, ASPECT_FIX if s == "fail" else "")
        size_ok = s != "fail"
    holes = st.get("holes") or []
    locked = bool(st.get("map_locked"))
    run.add("part", "ok" if locked and cell._live(st) else "warn",
            f"status {st.get('status')}, locked {locked}, {len(holes)} holes, live {cell._live(st)}",
            "" if locked and cell._live(st) else "arm out of the view (retreat), part in the work area, then new_part")
    if not size_ok or not holes or not st.get("box"):
        return _skip(run, "hole plans", "no usable part view")
    q = await cell._q_now()
    res, bad = [], 0
    for h in holes[:8]:
        try:
            p = cell._plan(cell._aim(st, {"hole": h["id"]})["P"], q)
            res.append(f"hole {h['id']}: ok, tilt {p['tilt_deg']} deg, {'direct' if p['direct'] else 'via high point'}")
        except GuardError as e:
            bad += 1
            res.append(f"hole {h['id']}: NO ({e})")
    run.add("hole plans", "warn" if bad else "ok", "; ".join(res),
            "some holes cannot be reached from here: move the part nearer, or skip them" if bad else "")


# ============================================================== j. learning/logs
async def check_aim(run):
    if not await _guided(run):
        return _skip(run, "aim offset", run.ctx.get("guided_error", "no cell"))
    k = run.ctx["cell"].kin

    def mm(v):
        return [round(float(x) * 1000, 1) for x in v]
    n = len(k.aim_lid_samples())
    run.add("aim offset", "ok", f"aim_file {mm(k.aim_from_file)} mm + lid jog ({n} sample(s), {k.aim_lid_path.name}) "
            f"{mm(k.aim_lid)} mm = total {mm(k.aim)} mm (robot frame)")


async def check_learning(run):
    cell = run.ctx.get("cell")
    if cell is None:
        return _skip(run, "learning and logs", "no cell")
    import numpy as np
    s = cell.model.status()
    corr = ""
    calib = run.ctx.get("calib")
    if calib and run.ctx.get("guided"):
        k = cell.kin
        d = np.mean([np.array(p["tip"]) for p in calib["points"]], axis=0) - k.table.o
        c = cell.model.correction_mm([d @ k.table.e1 * 1000, d @ k.table.e2 * 1000])
        corr = f"; correction at the workspace centre {[round(float(v), 1) for v in c]} mm"
    sd = cell.cfg.state_dir
    w = os.access(sd, os.W_OK)
    run.add("learning and logs", "ok" if w else "fail",
            f"{s['marks_logged']} marks logged, model {s['model']}{corr}; state dir {sd} "
            f"({'writable' if w else 'NOT writable'})", "" if w else "make the CELLBOT_STATE folder writable")


# ============================================================== runner
async def run_checks(cell=None) -> list[dict]:
    run = _Run()
    if cell is not None:
        run.ctx["cell"] = cell
    await run.step("python", lambda: check_python(run))
    await run.step("files", lambda: check_files(run))
    await run.step("cellbot config", lambda: check_cellbot_config(run))
    await run.step("e-stop", lambda: check_estop(run))
    await run.step("cell", lambda: make_cell(run))
    await run.step("Station link", lambda: check_station(run))
    await run.step("taught poses", lambda: check_poses(run))
    await run.step("tool tip", lambda: check_tool(run))
    await run.step("part geometry", lambda: check_part(run))
    await run.step("aim offset", lambda: check_aim(run))
    await run.step("vision", lambda: check_vision(run))
    await run.step("learning and logs", lambda: check_learning(run))
    run_checks.last_cell = run.ctx.get("cell")
    return run.checks


run_checks.last_cell = None


async def report(cell=None) -> dict:
    checks = await run_checks(cell)
    if cell is None and run_checks.last_cell is not None:
        run_checks.last_cell.close_vision()        # we opened it (camera:N): release the camera
    return {"verdict": verdict(checks), "blocking": sum(c["status"] == "fail" for c in checks), "checks": checks}


def format_text(rep: dict) -> str:
    out = []
    for c in rep["checks"]:
        out.append(f"[{c['status']:>4}] {c['name']}: {c['detail']}")
        if c["status"] != "ok" and c["fix"]:
            out.append(f"       fix: {c['fix']}")
    out.append(rep["verdict"])
    return "\n".join(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Read-only readiness checklist (never moves the arms).")
    ap.add_argument("--json", action="store_true", help="print the result as JSON")
    a = ap.parse_args(argv)
    rep = asyncio.run(report())
    print(json.dumps(rep, indent=1) if a.json else format_text(rep))
    return 0 if rep["blocking"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
