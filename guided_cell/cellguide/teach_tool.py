"""Pen tip calibration for the guided cell. It NEVER moves the arm: it only reads joint positions.

    python -m cellguide.teach_tool 3 8 15 20

The numbers are calibration markers (1-based, as drawn in elrobot_ik/calib_check*.jpg) from the calib_file in
guided.toml. For each marker a human puts the PEN TIP on the marker centre (steer with the leader arm, or by hand
with torque off) and presses Enter. The tool reads the 7 arm joints from Station through cellbot's
StationBackend, then solves the pen tip t in the gripper frame by linear least squares over all touches:

    R_i @ t + p_i = P_i      (T_i = fk(q_i), P_i = marker position in the robot base frame)

Every touch alone fixes t (P_i is known); more touches average out touch errors, and the residuals show how well
the touches, the calibration and joint_map.json agree. Spread the markers over the table and vary the wrist.
The result goes to <cellbot state dir>/pen_tip.json, which cellguide/kin.py loads. Station must run with --tcp and
CELLBOT_MODE must be "station" (env or cell.toml).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time

import numpy as np

from . import config as gconfig

TICKS = 4096
MAX_MEAN_RESIDUAL_MM = 6.0
MAX_SPREAD_TICKS = 6


def load_elrobot(gcfg: dict):
    """(ElRobot, Table, calib dict) from the elrobot_ik files named in guided.toml."""
    d = gcfg["_paths"]["elrobot_ik_dir"]
    ek = gconfig.load_module("elrobot_ik_kin", d / "elrobot_kin.py")
    tb = gconfig.load_module("elrobot_ik_table", d / "table.py")
    robot = ek.ElRobot()
    if robot.map is None:
        raise SystemExit("elrobot_ik/joint_map.json missing: run elrobot_ik/live.py once")
    calib = json.loads(gcfg["_paths"]["calib_file"].read_text())
    return robot, tb.Table(calib), calib


def marker_points(table, calib: dict, markers: list[int]) -> list[np.ndarray]:
    """Marker numbers (1-based) -> marker centres in the robot base frame (m), on the calibrated plane."""
    pts = calib["points"]
    out = []
    for n in markers:
        if not 1 <= n <= len(pts):
            raise SystemExit(f"marker {n}: the calibration has markers 1..{len(pts)}")
        out.append(np.asarray(table.pixel_to_base(pts[n - 1]["px"]), float))
    return out


def solve_tip(robot, ticks_list, points) -> dict:
    """Pure maths. ticks_list: raw ticks of joints 1..7 per touch; points: touched points (m, base frame).

    Returns dict(tip_local_m, residuals_mm, residual_mean_mm). Raises ValueError with fewer than 3 touches
    (fewer give no useful residual check).
    """
    if len(ticks_list) != len(points):
        raise ValueError("one point per touch is needed")
    if len(ticks_list) < 3:
        raise ValueError("need at least 3 touches")
    Rs, ps = [], []
    for ticks in ticks_list:
        T = robot.chain.fk(robot.map.to_rad(np.asarray(ticks, float)))
        Rs.append(T[:3, :3])
        ps.append(T[:3, 3])
    A = np.vstack(Rs)
    b = np.concatenate([np.asarray(P, float) - p for P, p in zip(points, ps)])
    t, *_ = np.linalg.lstsq(A, b, rcond=None)
    res = [float(np.linalg.norm(R @ t + p - np.asarray(P, float)) * 1000) for R, p, P in zip(Rs, ps, points)]
    return {"tip_local_m": [float(v) for v in t], "residuals_mm": res,
            "residual_mean_mm": float(np.mean(res))}


def _unwrap(samples: np.ndarray) -> np.ndarray:
    """Raw ticks (n, 7) -> continuous values near the first sample (handles 4095 -> 0)."""
    ref = samples[0]
    return ref + ((samples - ref + TICKS / 2) % TICKS - TICKS / 2)


# ------------------------------------------------------------------- Station I/O (never used by the tests)
async def _read_ticks(backend, arm_cfg, n=10, gap_s=0.03) -> tuple[list[float], float]:
    """Median of n reads of motors 1..7, and the largest spread (ticks) of any joint."""
    ids = arm_cfg.motor_ids[:7]
    rows = []
    for _ in range(n):
        ms = backend._motors(arm_cfg)
        rows.append([ms[i]["present"] for i in ids])
        await asyncio.sleep(gap_s)
    u = _unwrap(np.array(rows, float))
    med = np.median(u, axis=0) % TICKS
    return [float(v) for v in med], float(np.ptp(u, axis=0).max())


async def _teach(gcfg: dict, markers: list[int]) -> int:
    robot, table, calib = load_elrobot(gcfg)
    points = marker_points(table, calib, markers)
    gconfig.use_cellbot(gcfg)
    from cellbot.config import load_config
    ccfg = load_config()
    if ccfg.station.get("mode") != "station":
        raise SystemExit("set CELLBOT_MODE=station (this tool reads the real arm)")
    from cellbot.station_backend import StationBackend
    arm = gcfg["motion"]["arm"]
    arm_cfg = ccfg.arms[arm]
    backend = StationBackend(ccfg)
    print(f"connecting to Station {ccfg.station['host']}:{ccfg.station['port']} ...")
    await backend.connect()
    print("This tool never moves the arm. Put the PEN TIP on each marker centre "
          "(leader arm, or by hand with torque off), hold still, press Enter.")
    ticks_list = []
    for n, P in zip(markers, points):
        while True:
            await asyncio.to_thread(input, f"\nmarker {n} (robot mm {np.round(P * 1000).astype(int).tolist()}): "
                                           "pen tip on the centre, then Enter ")
            ticks, spread = await _read_ticks(backend, arm_cfg)
            if spread <= MAX_SPREAD_TICKS:
                break
            print(f"  the arm moved during the reading (spread {spread:.0f} ticks > {MAX_SPREAD_TICKS}); hold still, again")
        ticks_list.append(ticks)
        print("  ticks:", [round(v) for v in ticks])
    r = solve_tip(robot, ticks_list, points)
    print("\nresiduals per touch (mm):")
    for n, e in zip(markers, r["residuals_mm"]):
        print(f"  marker {n:2d}: {e:5.1f}")
    print(f"pen tip in the gripper frame (mm): {np.round(np.array(r['tip_local_m']) * 1000, 1).tolist()}")
    print(f"mean residual {r['residual_mean_mm']:.1f} mm")
    if r["residual_mean_mm"] > MAX_MEAN_RESIDUAL_MM:
        print(f"NOT SAVED: mean residual above {MAX_MEAN_RESIDUAL_MM} mm. Touch again more carefully, "
              "check the calibration and joint_map.json.")
        return 1
    out = ccfg.state_dir / "pen_tip.json"
    out.write_text(json.dumps({"tip_local_m": r["tip_local_m"], "residual_mean_mm": round(r["residual_mean_mm"], 2),
                               "residuals_mm": [round(v, 2) for v in r["residuals_mm"]], "markers": markers,
                               "ticks": ticks_list, "created": time.strftime("%Y-%m-%d %H:%M:%S")}, indent=1))
    print(f"saved {out}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Teach the pen tip offset by touching calibration markers.")
    ap.add_argument("markers", nargs="+", type=int, help="marker numbers, 1-based (at least 3, 4-6 recommended)")
    a = ap.parse_args(argv)
    if len(a.markers) < 3:
        ap.error("give at least 3 markers")
    return asyncio.run(_teach(gconfig.load(), a.markers))


if __name__ == "__main__":
    sys.exit(main())
