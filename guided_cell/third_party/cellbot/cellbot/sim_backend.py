"""Dry-run backend: a tiny simulated cell with both arms, a part and a top camera.

It is deliberately imperfect so the guardrails and the learning loop have
something to do: grasps sometimes fail, and the marker arm has a fixed per-spot
bias plus noise (configured under [sim] in cell.toml).
"""

from __future__ import annotations

import asyncio
import io
import random

import numpy as np
from PIL import Image, ImageDraw

from .config import CellConfig, PoseStore

W, H = 480, 360


class SimBackend:
    name = "sim"

    def __init__(self, cfg: CellConfig, poses: PoseStore):
        self.cfg = cfg
        self.poses = poses
        self.rng = random.Random(cfg.sim.get("seed", 0))
        self.joints: dict[str, dict[str, float]] = {}
        for name, arm in cfg.arms.items():
            home = poses.get(name, arm.home_pose)["joints"]
            self.joints[name] = {j: home.get(j, 0.5) for j in arm.arm_joints}
            self.joints[name][arm.gripper] = arm.gripper_open
        self.goals = {a: dict(j) for a, j in self.joints.items()}
        self.current = {a: {j: 20 for j in js} for a, js in self.joints.items()}
        self.part = "pickup"            # pickup | held | work | removed | dropped
        self.marks: list[tuple[float, float]] = []   # pixel positions of ink dots
        self._in_contact: str | None = None
        self.parts_done = 0

    async def connect(self):
        return

    def new_part(self):
        """Put a fresh part at the pickup spot (sim only; a human does this for real)."""
        self.part = "pickup"
        self.marks = []

    # ------------------------------------------------------------------ state
    async def read_state(self) -> dict:
        out = {}
        for a, js in self.joints.items():
            out[a] = {
                j: {"pos": v, "goal": self.goals[a][j], "current": self.current[a][j], "torque": True, "error": 0}
                for j, v in js.items()
            }
        return out

    def _near(self, arm: str, pose: str, tol: float) -> bool:
        try:
            p = self.poses.get(arm, pose)["joints"]
        except KeyError:
            return False
        return all(abs(self.joints[arm][j] - v) <= tol for j, v in p.items())

    async def write_goals(self, arm: str, goals: dict[str, float], speed: int, accel: int):
        await asyncio.sleep(0.01)
        acfg = self.cfg.arms[arm]
        for j, v in goals.items():
            self.goals[arm][j] = v
            if j == acfg.gripper:
                self._gripper(arm, v)
            else:
                self.joints[arm][j] = v
        self._update_world(arm)

    def _gripper(self, arm: str, target: float):
        acfg = self.cfg.arms[arm]
        cur = self.joints[arm][acfg.gripper]
        closing = target < cur
        if arm == "so101" and closing and self.part == "held":
            self.joints[arm][acfg.gripper] = max(target, 0.40)     # the part stops the fingers
            self.current[arm][acfg.gripper] = 280
            return
        if arm == "so101" and not closing:
            self._grasp_tried = False
        if arm == "so101" and closing:
            at = "pick" if self.part == "pickup" and self._near(arm, "pick", 0.03) else (
                "place" if self.part == "work" and self._near(arm, "place", 0.03) else None)
            if at and target < 0.45 and not getattr(self, "_grasp_tried", False):
                self._grasp_tried = True
                if self.rng.random() < self.cfg.sim.get("grasp_fail_rate", 0.0):
                    self.joints[arm][acfg.gripper] = target          # closed on nothing
                    self.current[arm][acfg.gripper] = 20
                    return
                self.part = "held"
                self.joints[arm][acfg.gripper] = 0.40                 # stopped by the part
                self.current[arm][acfg.gripper] = 280
                return
        if arm == "so101" and not closing and self.part == "held":
            if self._near(arm, "place", 0.03):
                self.part = "work"
            elif self._near(arm, "drop", 0.03):
                self.part = "removed"
                self.parts_done += 1
            else:
                self.part = "dropped"
        self.joints[arm][acfg.gripper] = target
        self.current[arm][acfg.gripper] = 20

    def _update_world(self, arm: str):
        if arm != "elrobot":
            return
        m = self.cfg.marking
        touching = None
        for s in self.cfg.spot_ids():
            if self._near("elrobot", f"mark_touch_{s}", 0.08):
                touching = s
        if touching and self._in_contact != touching:
            # New ink dot. Where it lands = target + arm bias + effect of corrections.
            q_touch = self.poses.get("elrobot", f"mark_touch_{touching}")["joints"]
            cj = m["correction_joints"]
            dq = np.array([self.joints["elrobot"][j] - q_touch[j] for j in cj])
            J = np.array(m["jacobian_mm"]) * np.array(self.cfg.sim.get("jacobian_scale", [1, 1]))[:, None]
            bias = np.array(self.cfg.sim["bias_mm"][touching])
            noise = np.array([self.rng.gauss(0, self.cfg.sim.get("noise_mm", 0)) for _ in range(2)])
            land_mm = bias + J @ dq + noise
            tx, ty = m["spot_targets_px"][touching]
            px = (tx + land_mm[0] / m["mm_per_px"], ty + land_mm[1] / m["mm_per_px"])
            if self.part == "work":
                self.marks.append(px)
        self._in_contact = touching

    async def freeze(self, arm: str):
        self.goals[arm] = dict(self.joints[arm])

    async def set_torque(self, arm: str, on: bool):
        return

    # ----------------------------------------------------------------- camera
    def _render_top(self) -> bytes:
        img = Image.new("RGB", (W, H), (150, 150, 145))
        d = ImageDraw.Draw(img)
        d.rectangle([150, 90, 330, 270], outline=(60, 60, 200), width=2)       # work area fixture
        d.text((152, 75), "WORK AREA", fill=(40, 40, 160))
        d.rectangle([20, 130, 100, 210], outline=(40, 140, 40), width=2)       # pickup
        d.text((22, 115), "PICKUP", fill=(30, 100, 30))
        d.rectangle([380, 130, 460, 210], outline=(160, 60, 40), width=2)      # done bin
        d.text((382, 115), f"DONE x{self.parts_done}", fill=(130, 40, 30))
        part_box = {"pickup": (25, 135, 95, 205), "work": (170, 110, 310, 250)}.get(self.part)
        if self.part == "pickup":
            d.rectangle(part_box, fill=(225, 215, 190))
        if self.part == "work":
            d.rectangle(part_box, fill=(225, 215, 190))
            for s, (x, y) in self.cfg.marking["spot_targets_px"].items():
                d.ellipse([x - 7, y - 7, x + 7, y + 7], outline=(185, 175, 150), width=1)   # drilled hole rings
        for x, y in self.marks:
            d.ellipse([x - 3, y - 3, x + 3, y + 3], fill=(15, 15, 25))
        # Arms: a coloured dot at a joint-dependent spot, so you can see them move.
        for arm, col in (("so101", (230, 120, 20)), ("elrobot", (30, 160, 200))):
            js = self.joints[arm]
            first = list(js.values())
            ax = int(40 + first[0] * 400)
            ay = int(330 - first[1] * 60) if arm == "so101" else int(30 + first[1] * 60)
            d.ellipse([ax - 9, ay - 9, ax + 9, ay + 9], fill=col)
            d.text((ax + 11, ay - 6), arm, fill=col)
        if self.part == "held":
            d.text((10, 340), "so101 is holding the part", fill=(120, 60, 0))
        if self.part == "dropped":
            d.text((10, 340), "PART DROPPED ON TABLE", fill=(200, 0, 0))
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=92)
        return buf.getvalue()

    async def images(self) -> dict[str, bytes]:
        return {"top": self._render_top()}

    def describe(self) -> str:
        return "SIMULATION (dry run, no hardware)"
