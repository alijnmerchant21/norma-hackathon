"""The guarded cell: every robot action Claude can take goes through here.

Guardrails (all enforced in code, not in the prompt):
  * Joint limits: calibrated arc (via normalized units) minus a margin, plus
    optional tighter per-joint envelopes from cell.toml. Out-of-range targets
    are REJECTED, never silently clamped.
  * Max step: move_joints may move a joint at most max_delta_per_call, and may
    drift at most 2x that from the last taught pose. Big moves must use taught
    poses, which only a human can create (cellbot.teach).
  * Speed / accel caps on every servo write; moves are split into small
    waypoints with checks in between.
  * Shared work zone: an arm may only go to a "work" pose while the other arm
    is at a "clear" pose. An arm in an unknown place counts as not clear.
  * Over-current on any arm joint during a move -> freeze both arms and latch
    the e-stop (likely collision).
  * E-stop: latched in state/estop.json. Claude can trigger it (stop tool) but
    only a human can release it (python -m cellbot.estop release).
  * One motion at a time (a lock), so parallel tool calls can't race the zone check.
"""

from __future__ import annotations

import asyncio
import json
import math
import time
import uuid

from .config import CellConfig, PoseStore, load_config
from .feedback import Learner


class GuardError(Exception):
    """A command was refused by a guardrail. The message says why."""


class EStopped(GuardError):
    pass


class Cell:
    def __init__(self, cfg: CellConfig | None = None):
        self.cfg = cfg or load_config()
        self.g = self.cfg.guard
        self.poses = PoseStore(self.cfg)
        mode = self.cfg.station["mode"]
        if mode == "sim":
            from .sim_backend import SimBackend
            self.backend = SimBackend(self.cfg, self.poses)
            initial_zone = "clear"
            initial_pose = {a.name: a.home_pose for a in self.cfg.arms.values()}
        elif mode == "station":
            from .station_backend import StationBackend
            self.backend = StationBackend(self.cfg)
            initial_zone = "unknown"
            initial_pose = {a: None for a in self.cfg.arms}
        else:
            raise ValueError(f"unknown station mode {mode!r}")
        self.zone = {a: initial_zone for a in self.cfg.arms}
        self.last_pose: dict[str, str | None] = dict(initial_pose)
        self.learner = Learner(self.cfg)
        self.estop_path = self.cfg.state_dir / "estop.json"
        self.events_path = self.cfg.state_dir / "events.jsonl"
        self.lock = asyncio.Lock()
        self.part: dict | None = None
        self._connected = False

    # ================================================================ helpers
    async def ensure_connected(self):
        if not self._connected:
            await self.backend.connect()
            self._connected = True

    def estop_state(self) -> dict:
        try:
            return json.loads(self.estop_path.read_text())
        except FileNotFoundError:
            return {"latched": False}

    def _check_estop(self):
        st = self.estop_state()
        if st.get("latched"):
            raise EStopped(
                f"E-STOP is latched ({st.get('reason', 'no reason')}). No motion until a human "
                "releases it with `python -m cellbot.estop release`."
            )

    def log_event(self, kind: str, message: str, **extra):
        e = {"t": time.strftime("%Y-%m-%dT%H:%M:%S"), "kind": kind, "message": message,
             "part": (self.part or {}).get("id"), **extra}
        with open(self.events_path, "a") as f:
            f.write(json.dumps(e) + "\n")
        return e

    def _limits(self, arm: str, joint: str) -> tuple[float, float]:
        m = float(self.g["limit_margin"])
        lo, hi = m, 1.0 - m
        env = self.cfg.arms[arm].envelope.get(joint)
        if env:
            lo, hi = max(lo, env[0]), min(hi, env[1])
        return lo, hi

    def _check_limits(self, arm: str, target: dict[str, float]):
        acfg = self.cfg.arms[arm]
        bad = []
        for j, v in target.items():
            if j not in acfg.joint_names:
                raise GuardError(f"{arm} has no joint '{j}'. Joints: {acfg.joint_names}")
            if not isinstance(v, (int, float)) or math.isnan(v):
                raise GuardError(f"{arm}.{j}: target must be a number")
            lo, hi = self._limits(arm, j)
            if not (lo <= v <= hi):
                bad.append(f"{j}={v:.3f} outside [{lo:.2f}, {hi:.2f}]")
        if bad:
            raise GuardError(f"{arm}: refused, joint limits: " + "; ".join(bad))

    def _other(self, arm: str) -> str:
        return next(a for a in self.cfg.arms if a != arm)

    def _check_zone(self, arm: str, target_zone: str):
        if target_zone == "work":
            other = self._other(arm)
            if self.zone[other] != "clear":
                raise GuardError(
                    f"{arm}: refused, the work area is not free: {other} is "
                    f"{'in the work area' if self.zone[other] == 'work' else 'at an unknown position'}. "
                    f"Send {other} to a clear pose (e.g. go_home) first."
                )

    # ================================================================ motion
    async def _move(self, arm: str, target: dict[str, float], speed: int | None = None) -> dict:
        """Waypointed, checked move. Caller holds self.lock and has validated the target."""
        await self.ensure_connected()
        self._check_estop()
        acfg = self.cfg.arms[arm]
        speed = int(min(speed or self.g["max_speed"], self.g["max_speed"]))
        accel = int(self.g["max_accel"])
        state = await self.backend.read_state()
        cur = {j: state[arm][j]["pos"] for j in target}
        biggest = max((abs(target[j] - cur[j]) for j in target), default=0.0)
        n = max(1, math.ceil(biggest / float(self.g["interp_step"])))
        t0 = time.monotonic()
        for i in range(1, n + 1):
            self._check_estop()
            wp = {j: cur[j] + (target[j] - cur[j]) * i / n for j in target}
            await self.backend.write_goals(arm, wp, speed, accel)
            reached = await self._wait_reached(arm, wp, final=(i == n))
            if not reached["ok"]:
                return {"ok": False, **reached, "waypoint": f"{i}/{n}"}
        state = await self.backend.read_state()
        err = {j: round(abs(state[arm][j]["pos"] - target[j]), 3) for j in target if j != acfg.gripper}
        return {"ok": True, "waypoints": n, "seconds": round(time.monotonic() - t0, 2),
                "max_error": max(err.values(), default=0.0)}

    async def _wait_reached(self, arm: str, wp: dict[str, float], final: bool) -> dict:
        acfg = self.cfg.arms[arm]
        tol = float(self.g["reach_tolerance"]) * (1.0 if final else 2.5)
        deadline = time.monotonic() + float(self.g["move_timeout_s"])
        limit = int(self.g["current_limit"])
        while True:
            state = await self.backend.read_state()
            js = state[arm]
            over = {j: js[j]["current"] for j in wp if j != acfg.gripper and js[j]["current"] > limit}
            if over:
                await self._emergency(f"over-current on {arm} {over} (possible collision)")
                return {"ok": False, "error": f"over-current {over}; both arms frozen and e-stop latched"}
            if any(js[j].get("error") for j in wp):
                errs = {j: js[j]["error"] for j in wp if js[j].get("error")}
                await self._emergency(f"servo error status on {arm} {errs}")
                return {"ok": False, "error": f"servo error {errs}; e-stop latched"}
            if all(abs(js[j]["pos"] - wp[j]) <= tol for j in wp if j != acfg.gripper):
                return {"ok": True}
            if self.estop_state().get("latched"):
                await self._freeze_all()
                return {"ok": False, "error": "e-stop latched during move; arms frozen"}
            if time.monotonic() > deadline:
                await self.backend.freeze(arm)
                lag = {j: round(js[j]["pos"] - wp[j], 3) for j in wp if abs(js[j]["pos"] - wp[j]) > tol}
                return {"ok": False, "error": f"timeout: not reached, arm held where it is. Lagging joints: {lag}"}
            await asyncio.sleep(0.03)

    async def _freeze_all(self):
        for a in self.cfg.arms:
            try:
                await self.backend.freeze(a)
            except Exception as e:  # keep trying the other arm
                self.log_event("error", f"freeze {a} failed: {e}")

    async def _emergency(self, reason: str):
        self.estop_path.write_text(json.dumps({"latched": True, "reason": reason, "t": time.time()}))
        await self._freeze_all()
        self.log_event("estop", reason)

    # ============================================================ public API
    async def status(self) -> dict:
        await self.ensure_connected()
        state = await self.backend.read_state()
        arms = {}
        for a, acfg in self.cfg.arms.items():
            js = state[a]
            arms[a] = {
                "role": acfg.role,
                "zone": self.zone[a],
                "last_pose": self.last_pose[a],
                "joints": {j: round(js[j]["pos"], 3) for j in acfg.arm_joints},
                "gripper": round(js[acfg.gripper]["pos"], 3),
                "max_current": max(js[j]["current"] for j in acfg.joint_names),
            }
        return {
            "backend": self.backend.describe(),
            "estop": self.estop_state(),
            "arms": arms,
            "part": self.part,
            "poses": {a: self.poses.names(a) for a in self.cfg.arms},
        }

    async def go_to_pose(self, arm: str, pose: str, speed: str = "normal",
                         offsets: dict[str, float] | None = None) -> dict:
        if arm not in self.cfg.arms:
            raise GuardError(f"unknown arm {arm!r}; arms are {list(self.cfg.arms)}")
        self.poses.reload()
        try:
            p = self.poses.get(arm, pose)
        except KeyError as e:
            raise GuardError(str(e).strip('"'))
        target = dict(p["joints"])
        for j, d in (offsets or {}).items():
            if abs(d) > float(self.g["max_delta_per_call"]):
                raise GuardError(f"{arm}: correction {j}{d:+.3f} is bigger than max_delta_per_call")
            target[j] = target[j] + d
        self._check_limits(arm, target)
        async with self.lock:
            self._check_estop()
            self._check_zone(arm, p["zone"])
            if p["zone"] == "work":
                self.zone[arm] = "work"            # claim the zone before moving
            sp = self.g["slow_speed"] if speed == "slow" or self.zone[arm] == "unknown" else self.g["max_speed"]
            res = await self._move(arm, target, sp)
            if res["ok"]:
                self.zone[arm] = p["zone"]
                self.last_pose[arm] = pose
            else:
                self.zone[arm] = "unknown" if p["zone"] == "clear" else "work"
                self.last_pose[arm] = None
            return {"arm": arm, "pose": pose, **res, "zone": self.zone[arm]}

    async def go_home(self, arm: str) -> dict:
        return await self.go_to_pose(arm, self.cfg.arms[arm].home_pose, speed="slow")

    async def move_joints(self, arm: str, joints: dict[str, float], relative: bool = True) -> dict:
        """Small local adjustment near the last taught pose."""
        if arm not in self.cfg.arms:
            raise GuardError(f"unknown arm {arm!r}")
        acfg = self.cfg.arms[arm]
        if acfg.gripper in joints:
            raise GuardError("use the gripper tool for the gripper")
        base_name = self.last_pose[arm]
        if base_name is None:
            raise GuardError(f"{arm}: position unknown; go to a taught pose first (e.g. go_home)")
        await self.ensure_connected()
        state = await self.backend.read_state()
        cur = {j: state[arm][j]["pos"] for j in acfg.arm_joints}
        target = {j: (cur[j] + v if relative else v) for j, v in joints.items()}
        self._check_limits(arm, target)
        md = float(self.g["max_delta_per_call"])
        too_big = {j: round(target[j] - cur[j], 3) for j in target if abs(target[j] - cur[j]) > md}
        if too_big:
            raise GuardError(f"{arm}: refused, step too big {too_big} (max {md} per call). Use a taught pose.")
        base = self.poses.get(arm, base_name)["joints"]
        drift = {j: round(target[j] - base[j], 3) for j in target if abs(target[j] - base.get(j, cur[j])) > 2 * md}
        if drift:
            raise GuardError(f"{arm}: refused, too far from taught pose '{base_name}' {drift} (max {2*md}).")
        async with self.lock:
            self._check_estop()
            res = await self._move(arm, target, self.g["slow_speed"])
            return {"arm": arm, "relative_to_pose": base_name, **res}

    async def gripper(self, arm: str, action: str = "", position: float | None = None) -> dict:
        acfg = self.cfg.arms[arm]
        if position is None:
            if action not in ("open", "close"):
                raise GuardError("action must be 'open' or 'close', or give a position")
            position = acfg.gripper_open if action == "open" else acfg.gripper_closed
        self._check_limits(arm, {acfg.gripper: position})
        async with self.lock:
            self._check_estop()
            await self._move(arm, {acfg.gripper: position}, self.g["slow_speed"])
            # Wait until the gripper stops moving (closed, or stalled on an object).
            last, still, t_end = None, 0, time.monotonic() + 2.5
            while time.monotonic() < t_end and still < 3:
                st = (await self.backend.read_state())[arm][acfg.gripper]
                still = still + 1 if last is not None and abs(st["pos"] - last) < 0.005 else 0
                last = st["pos"]
                await asyncio.sleep(0.1)
        closing = position <= (acfg.gripper_open + acfg.gripper_closed) / 2
        gap = st["pos"] - acfg.gripper_closed
        holding = closing and gap > 0.08
        out = {"arm": arm, "gripper_pos": round(st["pos"], 3), "current": st["current"]}
        if closing:
            out["holding_object"] = holding
            out["note"] = ("gripper stopped before closing fully: something is in it"
                           if holding else "gripper closed fully: NOTHING was grasped")
        return out

    # ------------------------------------------------------------- marking
    async def new_part_ready(self) -> dict:
        if self.backend.name == "sim":
            self.backend.new_part()
            return {"ok": True, "note": "SIM: a new part was put at the pickup spot."}
        return {"ok": True, "note": "Real cell: a human loads parts. Use look to check one is at pickup."}

    async def start_marking(self, part_id: str | None = None) -> dict:
        """Take the reference image. Both arms must be clear so the camera sees the part."""
        if any(self.zone[a] != "clear" for a in self.cfg.arms):
            raise GuardError(f"both arms must be at clear poses first (zones: {self.zone})")
        await self.ensure_connected()
        imgs = await self.backend.images()
        cam = self.cfg.station["inspect_camera"]
        if cam not in imgs:
            raise GuardError(f"inspect camera '{cam}' not in feed {list(imgs)}")
        self.part = {"id": part_id or uuid.uuid4().hex[:6], "marked": {}, "learned": [], "misses": {}}
        self._ref = imgs[cam]
        self.log_event("part", "marking started")
        return {"ok": True, "part": self.part["id"], "spots": self.cfg.spot_ids()}

    async def mark_spot(self, spot: str) -> dict:
        spot = str(spot)
        if self.part is None:
            raise GuardError("call start_marking first")
        if spot not in self.cfg.marking["spot_targets_px"]:
            raise GuardError(f"unknown spot {spot}; spots are {self.cfg.spot_ids()}")
        arm = next(a for a, c in self.cfg.arms.items() if c.role == "marker")
        dq, corr = self.learner.joint_offsets(spot)
        r1 = await self.go_to_pose(arm, f"mark_hover_{spot}", offsets=dq)
        if not r1["ok"]:
            return {"ok": False, "step": "hover", **r1}
        r2 = await self.go_to_pose(arm, f"mark_touch_{spot}", speed="slow", offsets=dq)
        if not r2["ok"]:
            return {"ok": False, "step": "touch", **r2}
        await asyncio.sleep(float(self.cfg.marking["dwell_s"]))
        r3 = await self.go_to_pose(arm, f"mark_hover_{spot}", speed="slow", offsets=dq)
        self.part["marked"][spot] = {"correction_mm": corr, "dq": dq}
        return {"ok": r3["ok"], "spot": spot, "applied_correction_mm": corr,
                "applied_joint_offsets": {k: round(v, 4) for k, v in dq.items()},
                "note": "marker arm is still above the part (work zone)"}

    async def inspect_marks(self) -> tuple[dict, bytes]:
        if self.part is None:
            raise GuardError("call start_marking first")
        if any(self.zone[a] != "clear" for a in self.cfg.arms):
            raise GuardError(f"move both arms to clear poses first so the camera sees the part (zones: {self.zone})")
        from . import vision
        m = self.cfg.marking
        imgs = await self.backend.images()
        cur = imgs[self.cfg.station["inspect_camera"]]
        blobs = vision.find_marks(self._ref, cur, min_area=int(m["blob_min_area_px"]))
        found, stray = vision.assign_marks(blobs, m["spot_targets_px"], float(m["mm_per_px"]))
        results = {}
        for spot in self.cfg.spot_ids():
            if spot not in found:
                results[spot] = {"found": False, "marked_this_part": spot in self.part["marked"]}
                continue
            r = {"found": True, **found[spot], "pass": found[spot]["miss_dist_mm"] <= m["tolerance_mm"]}
            if spot in self.part["marked"] and spot not in self.part["learned"]:
                info = self.part["marked"][spot]
                entry = self.learner.record(self.part["id"], spot, found[spot], info["correction_mm"], info["dq"])
                self.part["learned"].append(spot)
                r["next_correction_mm"] = entry["next_correction_mm"]
            self.part["misses"][spot] = found[spot]["miss_dist_mm"]
            results[spot] = r
        out = {"part": self.part["id"], "tolerance_mm": m["tolerance_mm"], "spots": results,
               "stray_marks": [[round(x, 1), round(y, 1)] for x, y, _ in stray]}
        return out, self._annotate(cur, found)

    def _annotate(self, jpeg: bytes, found: dict) -> bytes:
        import io
        from PIL import Image, ImageDraw
        img = Image.open(io.BytesIO(jpeg)).convert("RGB")
        d = ImageDraw.Draw(img)
        for s, (x, y) in self.cfg.marking["spot_targets_px"].items():
            d.line([x - 6, y, x + 6, y], fill=(0, 200, 0)); d.line([x, y - 6, x, y + 6], fill=(0, 200, 0))
            if s in found:
                mx, my = found[s]["mark_px"]
                ok = found[s]["miss_dist_mm"] <= self.cfg.marking["tolerance_mm"]
                d.line([x, y, mx, my], fill=(0, 200, 0) if ok else (230, 0, 0), width=2)
                d.text((x + 8, y + 4), f"{s}: {found[s]['miss_dist_mm']}mm", fill=(0, 0, 0))
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=90)
        return buf.getvalue()

    async def finish_part(self, outcome: str, notes: str = "") -> dict:
        if self.part is None:
            raise GuardError("no part in progress")
        misses = list(self.part["misses"].values())
        self.learner.close_part(self.part["id"], outcome, misses, notes)
        self.log_event("part_done", outcome, notes=notes)
        done, self.part = self.part, None
        return {"part": done["id"], "outcome": outcome, "misses_mm": done["misses"],
                "learning": self.learner.status()}

    async def stop(self, reason: str) -> dict:
        await self.ensure_connected()
        await self._emergency(f"stop requested by orchestrator: {reason}")
        return {"ok": True, "estop": self.estop_state()}

    async def look(self) -> dict[str, bytes]:
        await self.ensure_connected()
        return await self.backend.images()
