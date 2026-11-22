"""Offline check of the pen tip solver (no Station):  python tests/test_teach_tool.py

Synthetic touches from a known tip and random joint angles must give that tip back to < 0.1 mm.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from cellguide import config as gconfig  # noqa: E402
from cellguide.teach_tool import solve_tip, load_elrobot, marker_points  # noqa: E402


def main() -> int:
    g = gconfig.load()
    robot, table, calib = load_elrobot(g)
    rng = np.random.default_rng(7)
    tip_true = np.array([0.004, -0.006, 0.112])            # a pen sticking out of the jaws
    lo, hi = robot.chain.lower, robot.chain.upper
    ticks_list, points = [], []
    while len(points) < 6:
        q = lo + (hi - lo) * rng.uniform(0.2, 0.8, 7)
        ticks = robot.map.to_ticks(q)                       # what Station would report (whole ticks)
        T = robot.chain.fk(robot.map.to_rad(ticks))
        P = T[:3, :3] @ tip_true + T[:3, 3]
        if table.height_above(P) < 0.0:                    # tip below the table: not a reachable touch
            continue
        ticks_list.append(ticks.tolist())
        points.append(P)
    r = solve_tip(robot, ticks_list, points)
    err = float(np.linalg.norm(np.array(r["tip_local_m"]) - tip_true) * 1000)
    fails = 0

    def check(name, ok, detail):
        nonlocal fails
        print(f"  [{'ok' if ok else 'FAIL'}] {name}  ({detail})")
        fails += not ok

    check("tip recovered", err < 0.1, f"error {err:.4f} mm")
    check("residuals ~0", r["residual_mean_mm"] < 0.01, f"{r['residual_mean_mm']:.4f} mm")

    noisy = [P + rng.normal(0, 0.002, 3) for P in points]   # 2 mm touch noise -> reported, not hidden
    rn = solve_tip(robot, ticks_list, noisy)
    check("noise shows in the residuals", 0.3 < rn["residual_mean_mm"] < 6, f"{rn['residual_mean_mm']:.2f} mm")

    try:
        solve_tip(robot, ticks_list[:2], points[:2])
        check("fewer than 3 touches refused", False, "")
    except ValueError as e:
        check("fewer than 3 touches refused", True, str(e))

    pts = marker_points(table, calib, [1, len(calib["points"])])
    check("marker numbers map to base-frame points", all(abs(table.height_above(P)) < 1e-6 for P in pts),
          str([np.round(P * 1000).astype(int).tolist() for P in pts]))
    print("ALL CHECKS PASSED" if not fails else f"{fails} CHECK(S) FAILED")
    return fails


if __name__ == "__main__":
    sys.exit(main())
