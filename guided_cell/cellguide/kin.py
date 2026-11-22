"""Geometry for the guided cell: camera pixel -> point on the part -> joint target for cellbot.

Reuses elrobot_ik (URDF kinematics, joint map, table calibration) unchanged. The IK is the one from
elrobot_ik/hover.py (tip exactly on the target, tool as vertical as the joints allow).

Units: metres and radians inside; cellbot's normalized joint values (0..1 of the calibrated arc) at the edge.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares

from .config import load_module

TICKS = 4096


class Unreachable(Exception):
    pass


class Kin:
    def __init__(self, cfg: dict, ranges: list[tuple[int, int]], limit_margin: float, state_dir: Path):
        """ranges: calibrated (range_min, range_max) raw ticks of motors 1..7, from Station."""
        d = cfg["_paths"]["elrobot_ik_dir"]
        ek = load_module("elrobot_ik_kin", d / "elrobot_kin.py")
        tb = load_module("elrobot_ik_table", d / "table.py")
        self.robot = ek.ElRobot()
        if self.robot.map is None:
            raise RuntimeError("elrobot_ik/joint_map.json missing: run elrobot_ik/live.py once")
        self.calib = json.loads(cfg["_paths"]["calib_file"].read_text())
        self.table = tb.Table(self.calib)
        self.n = self.table.n
        self.tool_source = "closed fingertips (elrobot_ik/tool_offset.json)"
        tip = cfg["tool"].get("tip_local_m") or []
        pen = state_dir / "pen_tip.json"
        if len(tip) == 3:
            self.robot.tip_local, self.tool_source = np.array(tip, float), "guided.toml [tool]"
        elif pen.exists():
            self.robot.tip_local = np.array(json.loads(pen.read_text())["tip_local_m"], float)
            self.tool_source = "state/pen_tip.json (teach_tool)"
        self.ranges = [(int(a), int(b)) for a, b in ranges]
        self.span = np.array([(b - a) % TICKS or TICKS for a, b in self.ranges], float)
        self.rmin = np.array([a for a, _ in self.ranges], float)
        self.lo, self.hi = self._limits(limit_margin + 0.004)
        p = cfg["part"]
        self.surface_h = float(p["surface_height_mm"]) / 1000
        self.cam_h = float(p.get("camera_height_mm", 0)) / 1000
        self.nadir_px = p.get("camera_nadir_px") or None
        aim = cfg["_paths"].get("aim_file")
        self.aim_from_file = np.zeros(3)          # [paths] aim_file (elrobot_ik/aim.py or hover.py)
        if aim is not None and aim.exists():
            s = json.loads(aim.read_text()).get("samples", [])
            if s:
                self.aim_from_file = np.mean([np.array(x["offset_m"]) for x in s], axis=0)
        self.aim_lid_path = Path(state_dir) / "aim_lid.json"      # taught at lid height with the CLI `jog`
        self.reload_aim()
        px = np.array([q["px"] for q in self.calib["points"]], np.float64).reshape(-1, 1, 2)
        ab = cv2.perspectiveTransform(px, self.table.H).reshape(-1, 2)
        self._hull = cv2.convexHull(ab.astype(np.float32))

    # -------------------------------------------------------------- aim offset
    def aim_lid_samples(self) -> list[dict]:
        try:
            return list(json.loads(self.aim_lid_path.read_text()).get("samples", []))
        except FileNotFoundError:
            return []

    def reload_aim(self) -> None:
        """Total aim offset = aim_file mean + mean of state/aim_lid.json samples (each sample is a full estimate)."""
        s = self.aim_lid_samples()
        self.aim_lid = np.mean([np.array(x["offset_m"], float) for x in s], axis=0) if s else np.zeros(3)
        self.aim = self.aim_from_file + self.aim_lid

    def add_aim_lid_sample(self, offset_m, at_m, height_above_part_mm: float) -> dict:
        s = self.aim_lid_samples()
        rec = {"offset_m": [float(v) for v in offset_m], "at_m": [float(v) for v in at_m],
               "height_above_part_mm": round(float(height_above_part_mm), 1), "t": time.time()}
        s.append(rec)
        self.aim_lid_path.parent.mkdir(parents=True, exist_ok=True)
        self.aim_lid_path.write_text(json.dumps({"samples": s}, indent=1))
        self.reload_aim()
        return rec

    def clear_aim_lid(self) -> int:
        n = len(self.aim_lid_samples())
        self.aim_lid_path.parent.mkdir(parents=True, exist_ok=True)
        self.aim_lid_path.write_text(json.dumps({"samples": []}, indent=1))
        self.reload_aim()
        return n

    def in_plane(self, v) -> np.ndarray:
        """Component of a vector in the table plane."""
        v = np.asarray(v, float)
        return v - (v @ self.n) * self.n

    # ------------------------------------------------------------------ joints
    def _limits(self, margin: float):
        """Usable range per joint (rad): calibrated arc minus the cellbot margin, inside the URDF limits."""
        m, lo, hi = self.robot.map, [], []
        for k in range(7):
            d0 = (m.zero[k] - self.rmin[k]) % TICKS
            if d0 <= self.span[k]:                   # the zero pose lies inside the calibrated arc (hover.py)
                d = -d0
            else:
                d = ((self.rmin[k] - m.zero[k] + TICKS / 2) % TICKS) - TICKS / 2
            a, b = d + margin * self.span[k], d + (1 - margin) * self.span[k]
            a, b = max(a, -TICKS / 2 + 2), min(b, TICKS / 2 - 2)                   # to_rad wraps at half a turn
            a, b = sorted((m.sign[k] * a, m.sign[k] * b))
            lo.append(max(a * 2 * np.pi / TICKS, self.robot.chain.lower[k]))
            hi.append(min(b * 2 * np.pi / TICKS, self.robot.chain.upper[k]))
        lo, hi = np.array(lo), np.array(hi)
        bad = [k + 1 for k in range(7) if lo[k] >= hi[k]]
        if bad:
            raise RuntimeError(f"joint(s) {bad}: no usable range (Station calibration vs joint_map.json)")
        return lo, hi

    def norm_to_rad(self, norm) -> np.ndarray:
        return self.robot.map.to_rad(self.rmin + np.asarray(norm, float) * self.span)

    def rad_to_norm(self, q) -> np.ndarray:
        """Unclamped: a value outside 0..1 means outside the calibrated arc (cellbot then refuses it)."""
        m = self.robot.map
        t = m.zero + m.sign * np.asarray(q, float) * TICKS / (2 * np.pi)
        return (((t - self.rmin + TICKS / 2 - self.span / 2) % TICKS) - TICKS / 2 + self.span / 2) / self.span

    def tip(self, q) -> np.ndarray:
        return self.robot.tip(q)[0]

    def height(self, P) -> float:
        """Height of a point above the calibrated plane (m)."""
        return self.table.height_above(P)

    # ----------------------------------------------------------------- targets
    def surface_point(self, px, image_size, height_m: float | None = None) -> np.ndarray:
        """Camera pixel of something on the part's top face -> that point in the robot base frame."""
        h = self.surface_h if height_m is None else height_m
        shape = (image_size[1], image_size[0])
        T = self.table.pixel_to_base(px, shape)
        if self.cam_h > 0 and h != 0:                # the lid is nearer the lens than the table: shift to nadir
            nadir = self.nadir_px or (image_size[0] / 2, image_size[1] / 2)
            T = T + (h / self.cam_h) * (self.table.pixel_to_base(nadir, shape) - T)
        return T + h * self.n

    def in_workspace(self, P, margin_m: float) -> bool:
        d = np.asarray(P) - self.table.o
        ab = (float(d @ self.table.e1), float(d @ self.table.e2))
        return cv2.pointPolygonTest(self._hull, ab, True) >= -margin_m

    # ---------------------------------------------------------------------- IK
    REG = np.array([0.5, 0.5, 0.5, 0.5, 8.0, 0.5, 8.0])     # hover.py: keep the two roll joints near the seed

    def ik(self, target, seed, tilt_weight=10.0, roll7_window_deg=60.0):
        """Tool tip at target (m, base frame). Returns (q, error_mm, tilt_deg from vertical)."""
        down = -self.n
        seed = np.clip(seed, self.lo + 1e-3, self.hi - 1e-3)
        lo, hi = self.lo.copy(), self.hi.copy()
        w = np.deg2rad(roll7_window_deg)
        lo[6], hi[6] = max(lo[6], seed[6] - w), min(hi[6], seed[6] + w)
        if lo[6] >= hi[6]:
            lo[6], hi[6] = self.lo[6], self.hi[6]

        def res(q):
            p, f = self.robot.tip(q)
            return np.r_[(p - target) * 1000, tilt_weight * (f - down), self.REG * (q - seed)]
        q = least_squares(res, seed, bounds=(lo, hi), x_scale=0.3, max_nfev=600).x

        def polish(q2):
            p, f = self.robot.tip(q2)
            return np.r_[(p - target) * 1000, 2.0 * (f - down), 0.2 * (q2 - q)]
        q = least_squares(polish, np.clip(q, lo + 1e-4, hi - 1e-4), bounds=(lo, hi), x_scale=0.3, max_nfev=400).x
        p, f = self.robot.tip(q)
        return q, float(np.linalg.norm(p - target) * 1000), float(np.degrees(np.arccos(np.clip(f @ down, -1, 1))))

    def ik_translate(self, target, q_seed):
        """elrobot_ik/hover.py: tip to target keeping the gripper orientation of q_seed (parallel shift)."""
        R0 = self.robot.chain.fk(q_seed)[:3, :3]
        q0 = np.clip(q_seed, self.lo + 1e-4, self.hi - 1e-4)

        def res(q):
            T = self.robot.chain.fk(q)
            p = T[:3, :3] @ self.robot.tip_local + T[:3, 3]
            return np.r_[(p - target) * 1000, 300.0 * (T[:3, :3] - R0).ravel(), 0.05 * (q - q0)]
        q = least_squares(res, q0, bounds=(self.lo, self.hi), x_scale="jac", max_nfev=2000, xtol=1e-10, ftol=1e-10).x
        return q, float(np.linalg.norm(self.tip(q) - target) * 1000)

    def path_clear(self, q0, q1, floor_m, exempt_m=0.06, steps=24) -> bool:
        """True if the tip stays at or above floor_m (height above the calibrated plane) on the straight
        joint-space path q0 -> q1. Only within exempt_m of the START may it be lower, and then not lower than
        where it already is (hover.py's rule): a low start pose must rise before it travels."""
        p0 = self.tip(q0)
        h0 = self.height(p0)
        for t in np.linspace(0, 1, steps + 1):
            p = self.tip(q0 + (q1 - q0) * t)
            h = self.height(p)
            near_start = np.linalg.norm(p - p0) < exempt_m
            if h < (min(floor_m, h0 - 0.005) if near_start else floor_m):
                return False
        return True

    def path_lowest(self, q0, q1, steps=24):
        """Lowest tip height above the calibrated plane on the straight joint-space path q0 -> q1."""
        return min(self.height(self.tip(q0 + (q1 - q0) * t)) for t in np.linspace(0, 1, steps + 1))
