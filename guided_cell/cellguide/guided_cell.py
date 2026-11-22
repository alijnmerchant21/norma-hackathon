"""GuidedCell: cellbot's guarded Cell plus vision-guided work on a part that may move.

It subclasses cellbot.cell.Cell and changes none of its files. Every motion still goes through the same
guardrails (joint limits, e-stop, shared work zone, waypoints with over-current checks, one motion at a time).
New here:
  * targets come from the camera, not from taught poses:
        vision (box + holes, Manasa)  ->  calibration + IK (elrobot_ik, Max)  ->  Cell._move (cellbot, Ali)
  * extra checks for computed targets: inside the calibrated area, IK error, tip clearance along the path;
  * the arm hides and distorts the part in the top camera's picture when it is over it, so (default,
    [vision] sees_under_arm = false) the arm aims from one sighting taken while it is at a clear pose and does
    not watch the part during the work; "follow while hovering" runs only with a camera that sees under the arm;
  * gravity-droop correction: every guided goal is sent as q + offset; the offset is set at each hover by a closed
    loop on the measured tip (FK of joint feedback) and forgotten at retreat / a fresh look;
  * marks are measured by the camera and fed to a correction model (learn.py); everything is logged;
  * loader step for the SO-101 (loader.py) and an operator alerts feed (alerts.py).
"""
from __future__ import annotations

import asyncio
import json
import time
import uuid

import cv2
import numpy as np

from . import config as gconfig

_G = gconfig.load()
gconfig.use_cellbot(_G)

from cellbot.cell import Cell, GuardError  # noqa: E402

from .alerts import AlertsMixin  # noqa: E402
from .kin import Kin  # noqa: E402
from .learn import GuidedLearnerView, MissModel  # noqa: E402
from . import marks_dark  # noqa: E402
from .loader import LoaderMixin  # noqa: E402
from .vision_source import (ASPECT_FIX, CameraVision, FrameVision, HttpVision, VisionError, _decode,  # noqa: E402
                            frac_to_px, image_size_check)


def camera_index(src: str) -> int:
    idx = src.split(":", 1)[1].strip()
    if not idx.isdigit():
        raise ValueError(f"[vision].source = {src!r}: camera:<index> needs a number, e.g. camera:2")
    return int(idx)


def open_camera_source(gcfg: dict, capture_factory=None) -> CameraVision:
    """[vision] source = "camera:N": this process opens camera N and runs Manasa's detector + tracker on it."""
    v = gcfg["vision"]
    V = gconfig.load_module("part_vision", gcfg["_paths"]["vision_file"])
    return CameraVision(V, camera_index(v["source"]), int(v.get("width", 1280)), int(v.get("height", 720)),
                        int(v.get("view_port", 8765)), capture_factory=capture_factory)


class _ZoneLog(dict):
    """cellbot's zone dict, plus a record of when the worker arm was in the work zone (over the part), so vision
    events from those times can be recognised as distorted by the arm."""

    def __init__(self, d, arm, spans):
        super().__init__(d)
        self._arm, self._spans = arm, spans

    def __setitem__(self, k, v):
        if k == self._arm:
            was = dict.get(self, k)
            if v == "work" and was != "work":
                self._spans.append([time.time(), None])
                del self._spans[:-200]
            elif v != "work" and was == "work" and self._spans and self._spans[-1][1] is None:
                self._spans[-1][1] = time.time()
        super().__setitem__(k, v)


class GuidedCell(LoaderMixin, AlertsMixin, Cell):
    def __init__(self, cfg=None, gcfg: dict | None = None, vision=None):
        super().__init__(cfg)
        self.gcfg = gcfg or _G
        self.m = self.gcfg["motion"]
        self.arm = self.m["arm"]
        self.joints = self.cfg.arms[self.arm].arm_joints
        if len(self.joints) != 7:
            raise ValueError(f"[motion].arm = {self.arm!r} must be the 7-joint ElRobot")
        mk = self.gcfg["marks"]
        self.model = MissModel(self.cfg.state_dir, mk["gain"], mk["max_correction_mm"], mk["window"],
                               mk.get("local_scale_mm", 8.0), mk.get("outlier_mm", 12.0), mk.get("global_shrink", 0.5))
        self.learner = GuidedLearnerView(self.learner, self.model)   # learning_status / finish_part report MissModel
        self.kin: Kin | None = None
        self.vision = vision
        self.glog = self.cfg.state_dir / "guided_log.jsonl"
        self._offset = np.zeros(7)            # droop correction (rad): SET at each hover by the tip loop (_settle),
        #                                       used by the moves after it; forgotten at retreat and at a fresh look
        self._hover_info: dict | None = None  # numbers of the last hover tip loop (step result / guided_log)
        self._seed = None                     # run_program only: pre-touch correction carried to the next hole (rad)
        self._ev_seq = 0
        self._work: dict[str, dict] = {}      # what was touched on the current part
        self._size_ok = None                  # vision image size already checked against the calibration
        self._last_aim: dict | None = None    # last successful hover/touch (for `jog`); cleared by retreat
        self._last_cmd = None                 # last joint goal sent by _guided_move/_settle (rad)
        self._jog: dict | None = None
        self._sight: dict | None = None       # last sighting taken with the arm clear: {state, states, t, part_id}
        self._mark_ref: dict | None = None    # [marks] method "diff": lid picture before the dabs {frame, state, t, part_id}
        self._dabs_since_ref = 0
        self._in_view: list = []              # [t_start, t_end|None] spans of the worker arm in the work zone
        self.zone = _ZoneLog(self.zone, self.arm, self._in_view)
        if self.zone.get(self.arm) == "work":
            self._in_view.append([time.time(), None])

    # ================================================================= set-up
    async def ensure_guided(self):
        await self.ensure_connected()
        if self.kin is None:
            self.kin = await asyncio.to_thread(Kin, self.gcfg, self._ranges(), float(self.g["limit_margin"]),
                                               self.cfg.state_dir)
        if self.vision is None:
            src = self.gcfg["vision"]["source"]
            if src.startswith("http"):
                self.vision = HttpVision(src)
            elif src.startswith("station:"):
                cam = src.split(":", 1)[1]
                V = gconfig.load_module("part_vision", self.gcfg["_paths"]["vision_file"])

                async def frame():
                    imgs = await self.backend.images()
                    if cam not in imgs:
                        raise VisionError(f"camera '{cam}' not in the Station feed {list(imgs)}")
                    return _decode(imgs[cam])
                self.vision = FrameVision(V, frame, f"Station camera {cam}")
            elif src.startswith("camera:"):
                self.vision = await asyncio.to_thread(open_camera_source, self.gcfg)
                await self._station_fresh()
            else:
                raise ValueError(f"[vision].source = {src!r}: use http://host:port, camera:<index> or "
                                 "station:<camera>")

    async def _station_fresh(self, timeout: float = 4.0) -> None:
        """Wait until Station joint data is arriving again; refuse to go on if the link has stalled."""
        b = self.backend
        if getattr(b, "name", "") != "station":
            return
        t_end = time.monotonic() + timeout
        while time.monotonic() - b.latest_t > 0.4:
            if time.monotonic() > t_end:
                raise GuardError("Station joint data is not arriving (link stalled): nothing was moved. Quit and "
                                 "restart the runner; if it happens again, restart Station.")
            await asyncio.sleep(0.05)

    def close_vision(self) -> None:
        """Release an in-process camera (camera:<index>); nothing to do for the other sources."""
        if hasattr(self.vision, "close"):
            self.vision.close()

    def _ranges(self):
        acfg = self.cfg.arms[self.arm]
        if self.backend.name == "station":
            ms = self.backend._motors(acfg)
            return [(ms[i]["rmin"], ms[i]["rmax"]) for i in acfg.motor_ids[:7]]
        jm = json.loads((self.gcfg["_paths"]["elrobot_ik_dir"] / "joint_map.json").read_text())   # simulator
        return [tuple(jm[f"rev_motor_0{i}"]["range_ticks"]) for i in range(1, 8)]

    async def _q_now(self) -> np.ndarray:
        st = await self.backend.read_state()
        return self.kin.norm_to_rad([st[self.arm][j]["pos"] for j in self.joints])

    def _above_part(self, q) -> float:
        """Tip height above the part's top face (m)."""
        return self.kin.height(self.kin.tip(q)) - self.kin.surface_h

    # ================================================================= motion
    def _with_offset(self, q) -> np.ndarray:
        """The joint goal actually sent for the nominal goal q: q + learned droop offset, clipped to the usable
        range. A nominal q that is itself out of range is left out of range, so cellbot still refuses it."""
        q = np.asarray(q, float)
        return np.clip(q + self._offset, np.minimum(self.kin.lo, q), np.maximum(self.kin.hi, q))

    async def _guided_move(self, q, speed: str = "normal", settle: bool = True, cmd=None) -> dict:
        """Every guided move: send q + the learned droop offset (carried from the last settle), through cellbot's
        guards. settle=True: closed loop on the tip at the end pose, which sets the offset (not within 8 mm of the part).
        cmd: send exactly this joint goal instead of q + offset (the touch correction); q stays the nominal goal."""
        q = np.asarray(q, float)
        cmd = self._with_offset(q) if cmd is None else np.asarray(cmd, float)
        target = dict(zip(self.joints, (float(v) for v in self.kin.rad_to_norm(cmd))))
        self._check_limits(self.arm, target)                 # cellbot guard: refuse, never clamp
        await self._station_fresh()
        async with self.lock:
            self._check_estop()
            self._check_zone(self.arm, "work")
            self.zone[self.arm] = "work"
            self.last_pose[self.arm] = None                  # not at a taught pose any more
            slow = speed == "slow" or bool(self.m.get("all_slow", False))       # all_slow: first runs on hardware
            res = await self._move(self.arm, target, self.g["slow_speed"] if slow else self.g["max_speed"])
            self._last_cmd = cmd.copy()
            if res["ok"]:
                await self._arrive()
                if settle:
                    await self._settle(q)
        return res

    async def _arrive(self, timeout: float = 4.0) -> None:
        """cellbot calls a move 'reached' within reach_tolerance (several degrees). Wait until the joints have
        really stopped, so a dwell or a droop correction starts from where the arm ends up."""
        if getattr(self.backend, "name", "") != "station":
            return
        last, still, t_end = None, 0, time.monotonic() + timeout
        while time.monotonic() < t_end and still < 3:
            st = await self.backend.read_state()
            pos = np.array([st[self.arm][j]["pos"] for j in self.joints])
            still = still + 1 if last is not None and float(np.abs(pos - last).max()) < 0.0015 else 0
            last = pos
            await asyncio.sleep(0.1)

    def _movable(self, q_goal) -> np.ndarray:
        """Joints that can follow a correction at this goal: a nominal goal at (within 0.5 deg of) a joint limit
        is pinned there (joint 6 at the touch poses), so its error must neither be integrated nor keep a
        correction loop running."""
        q, edge = np.asarray(q_goal, float), np.deg2rad(0.5)
        return (q > self.kin.lo + edge) & (q < self.kin.hi - edge)

    async def _q_still(self, tol_mm: float = 1.0, max_wait: float = 1.2) -> np.ndarray:
        """Joint feedback once the TIP has stopped: two reads settle_still_s apart (default 0.4 s) whose FK tips
        differ by less than tol_mm, at most max_wait s. _arrive (cellbot) calls a joint still below ~0.28 deg per
        0.1 s, i.e. a servo still creeping at ~2.5 deg/s passes as stopped; reading such a transient is what made
        the old per-joint settle integrate the same error several times (wind-up)."""
        win = float(self.m.get("settle_still_s", 0.4))
        q, t_end = await self._q_now(), time.monotonic() + max_wait
        while True:
            await asyncio.sleep(win)
            q2 = await self._q_now()
            moved = float(np.linalg.norm(self.kin.tip(q2) - self.kin.tip(q))) * 1000
            q = q2
            if moved < tol_mm or time.monotonic() >= t_end:
                return q

    async def _settle(self, q_goal):
        """Hover: closed loop on the TIP, not on the joints. Servos stop short under gravity and stick, so the
        tip is measured (FK of the joint feedback, joint 6 stopping short of its limit included) after it has
        stopped, and the goal of joints 1-4 is moved by the damped least-squares step towards the tip of the
        nominal goal q_goal, at most settle_rounds rounds, until the height is within settle_tol_mm.
        The carried offset is then SET (not accumulated) to the measured tracking error (command - feedback) of
        joints 1-4: whatever was carried in (another hole, a wound-up value) is replaced by what this hover
        measured, so it cannot wind up. Caps: each joint at most settle_max_deg from q_goal; the commanded tip never below
        min(clearance_mm, hover - touch_cmd_below_mm) above the part. Joints at a limit (pinned) are not moved.
        Returns (and keeps in self._hover_info) the numbers for the step result / guided_log."""
        rounds = int(self.m.get("settle_rounds", 0))
        want = self._above_part(q_goal)
        if rounds <= 0 or want < 0.008:                        # touching the part: never push harder
            return None
        m, k, t0 = self.m, self.kin, time.monotonic()
        q_goal = np.asarray(q_goal, float)
        lim = np.deg2rad(float(m.get("settle_max_deg", 4.0)))
        tol = float(m.get("settle_tol_mm", 1.5)) / 1000
        floor = min(float(m.get("clearance_mm", 25.0)) / 1000, want - float(m.get("touch_cmd_below_mm", 3.0)) / 1000)
        mask = self._movable(q_goal)
        P_goal = k.tip(q_goal)
        lo, hi = np.minimum(k.lo, q_goal), np.maximum(k.hi, q_goal)
        off_before = self._offset.copy()
        cmd = np.asarray(self._last_cmd if self._last_cmd is not None else self._with_offset(q_goal), float).copy()
        q = await self._q_still()
        err0 = self._above_part(q) - want
        n, status = 0, "ok"
        while True:
            self._check_estop()
            st = await self.backend.read_state()
            js = st[self.arm]
            if any(js[j]["current"] > int(self.g["current_limit"]) for j in self.joints):
                await self._emergency(f"over-current on {self.arm} while settling (possible collision)")
                raise GuardError("over-current while settling; both arms frozen and e-stop latched")
            if abs(self._above_part(q) - want) <= tol:
                break
            if n >= rounds:
                status = "rounds"
                break
            step = await asyncio.to_thread(self._tip_step, q, P_goal, mask)
            new = np.where(mask, np.clip(cmd + step, q_goal - lim, q_goal + lim), cmd)
            new = np.clip(new, lo, hi)
            new = await asyncio.to_thread(self._floored, q_goal, new, floor)
            if np.abs(new - cmd).max() < np.deg2rad(0.05):
                status = "cap"                                 # at a cap / the floor: nothing left that may be done
                break
            goals = dict(zip(self.joints, (float(v) for v in k.rad_to_norm(new))))
            self._check_limits(self.arm, goals)
            await self.backend.write_goals(self.arm, goals, int(self.g["slow_speed"]), int(self.g["max_accel"]))
            self._last_cmd, cmd, n = new.copy(), new, n + 1
            await asyncio.sleep(0.3)
            await self._arrive()
            q = await self._q_still()                          # measure only once the tip has stopped
        # the carried offset = the servos' MEASURED steady tracking error (command - feedback) of joints 1-4 at this
        # verified hover, i.e. how far each one stops short under its load here; that is what transfers to the
        # nearby pre-touch / touch poses (a command that only fixes the tip may share it out over the joints
        # differently from the real droop). Set, never accumulated.
        pos = mask & (np.arange(7) < 4)
        self._offset = np.where(pos, np.clip(cmd - q, -lim, lim), self._offset)
        dxy = k.in_plane(k.tip(q) - P_goal)
        self._hover_info = dict(
            hover_status=status, hover_rounds=n,
            hover_error_before_mm=round(err0 * 1000, 1) + 0.0,
            hover_error_after_mm=round((self._above_part(q) - want) * 1000, 1) + 0.0,
            hover_xy_error_after_mm=round(float(np.linalg.norm(dxy)) * 1000, 1),
            hover_offset_before_deg=[round(float(v), 2) for v in np.degrees(off_before)],
            hover_offset_after_deg=[round(float(v), 2) for v in np.degrees(self._offset)],
            hover_settle_s=round(time.monotonic() - t0, 2))
        return self._hover_info

    # ------------------------------------------------------- touch correction
    def _floored(self, q_nominal, cmd, floor_m: float) -> np.ndarray:
        """cmd, pulled back towards q_nominal (whose tip is at the touch point) just as far as needed so that the
        tip of the COMMANDED pose is not below floor_m above the part. Runs IK-free FK only (fast)."""
        q_nominal, cmd = np.asarray(q_nominal, float), np.asarray(cmd, float)
        if self._above_part(cmd) >= floor_m:
            return cmd
        if self._above_part(q_nominal) < floor_m:
            return q_nominal.copy()
        lo, hi = 0.0, 1.0                                     # fraction of (cmd - q_nominal) that is allowed
        for _ in range(20):
            t = (lo + hi) / 2
            lo, hi = (t, hi) if self._above_part(q_nominal + t * (cmd - q_nominal)) >= floor_m else (lo, t)
        return q_nominal + lo * (cmd - q_nominal)

    def _tip_step(self, q_meas, P_goal, mask) -> np.ndarray:
        """Joint change (rad) that moves the measured tip onto P_goal: damped least squares on the tip position
        Jacobian at the measured pose, only over the position joints (1-4) that can move."""
        use = [j for j in range(4) if mask[j]]
        p0, eps = self.kin.tip(q_meas), 1e-4
        J = np.zeros((3, len(use)))
        for c, j in enumerate(use):
            q2 = np.array(q_meas, float)
            q2[j] += eps
            J[:, c] = (self.kin.tip(q2) - p0) / eps
        e = np.asarray(P_goal, float) - p0
        d = J.T @ np.linalg.solve(J @ J.T + (0.01 ** 2) * np.eye(3), e)
        out = np.zeros(7)
        out[use] = d
        return out

    async def _cmd_move(self, cmd_from, cmd_to, max_step: float | None = None, max_tip_mm: float | None = 1.0) -> dict:
        """Short, slow, straight move between two COMMANDED joint goals: waypoints interpolate from cmd_from (the
        goal the arm was last sent and is holding) to cmd_to, so a servo's steady tracking error (droop, a
        stopped-short joint) stays the same along the way instead of being re-applied at the start, as a move
        interpolated from the MEASURED joints does (cellbot's _move). Used near the part: the pre-touch
        correction rounds, the final descent and the first lift leg. Same guards as _move: joint limits, e-stop,
        work zone, the lock, and cellbot's _wait_reached on every waypoint (over-current, servo error, timeout).
        max_tip_mm: also at most this much commanded tip travel per waypoint. The servos are not synchronised (each
        runs at the same speed setting), so one waypoint of a few degrees would not be a straight line; with
        ~1 mm waypoints the path stays within a fraction of a millimetre of the straight one."""
        cmd_from, cmd_to = np.asarray(cmd_from, float), np.asarray(cmd_to, float)
        target = dict(zip(self.joints, (float(v) for v in self.kin.rad_to_norm(cmd_to))))
        self._check_limits(self.arm, target)                 # cellbot guard: refuse, never clamp
        await self._station_fresh()
        async with self.lock:
            await self.ensure_connected()
            self._check_estop()
            self._check_zone(self.arm, "work")
            self.zone[self.arm] = "work"
            self.last_pose[self.arm] = None
            a = self.kin.rad_to_norm(cmd_from)
            b = np.array([target[j] for j in self.joints])
            step = float(max_step if max_step is not None else float(self.g["interp_step"]) / 2)
            n = max(1, int(np.ceil(float(np.abs(b - a).max()) / step)))
            if max_tip_mm:
                dist_mm = float(np.linalg.norm(self.kin.tip(cmd_to) - self.kin.tip(cmd_from))) * 1000
                n = max(n, int(np.ceil(dist_mm / float(max_tip_mm))))
            t0 = time.monotonic()
            for i in range(1, n + 1):
                self._check_estop()
                wp = dict(zip(self.joints, (float(v) for v in a + (b - a) * i / n)))
                await self.backend.write_goals(self.arm, wp, int(self.g["slow_speed"]), int(self.g["max_accel"]))
                self._last_cmd = self.kin.norm_to_rad([wp[j] for j in self.joints])
                reached = await self._wait_reached(self.arm, wp, final=(i == n))
                if not reached["ok"]:
                    return {"ok": False, **reached, "waypoint": f"{i}/{n}"}
            self._last_cmd = cmd_to.copy()
            await self._arrive()
        return {"ok": True, "waypoints": n, "seconds": round(time.monotonic() - t0, 2)}

    def _xy_err(self, q, P_goal) -> np.ndarray:
        """In-plane tip error FK(q) - P_goal (m) in the table frame [e1, e2]."""
        d = self.kin.tip(q) - np.asarray(P_goal, float)
        return np.array([d @ self.kin.table.e1, d @ self.kin.table.e2])

    async def _touch_correct(self, q_goal, cmd0, q_meas, base=None, floor_m: float | None = None) -> dict:
        """Closed loop at a pose ABOVE the part (the pre-touch pose; never in contact) on the full 3D tip position.
        The carried droop offset is only a guess for this pose; measure the tip by FK of the joint feedback (once
        the tip is still, _q_still), move the joint goal (joints 1-4, damped least squares) by the remaining tip
        error towards the tip of q_goal, repeat, at most touch_correct_rounds times, until BOTH the height is within
        pretouch_tol_mm (default touch_tol_mm / 2) AND the in-plane error is within pretouch_xy_tol_mm (default 1.0).
        Priority: while the height is outside pretouch_tol_mm and the full 3D step would be cut by a cap, a
        height-only step is taken instead (height first: safety and contact), then xy.
        When no further round is possible: height within touch_tol_mm is accepted ("ok"), the in-plane residual is
        never a failure, it is only reported (xy_mm, xy_vec_m; xy_status "ok" | "rounds" | "cap").
        Hard limits: each joint is at most touch_correct_max_deg away from `base` (the goal with the carried
        offset only; default cmd0), the COMMANDED tip is never below floor_m above the part (default: this
        pose's height - touch_cmd_below_mm), and a measured tip more than touch_sag_abort_mm below this pose's
        height ends it at once (status "sag"). Status "ok" | "sag" | "stuck" | "move_failed".
        The correction is local: it never changes the carried offset."""
        m, k = self.m, self.kin
        want = self._above_part(q_goal)
        tol = float(m.get("touch_tol_mm", 2.0)) / 1000
        aim = min(tol, float(m.get("pretouch_tol_mm", 0.5 * tol * 1000)) / 1000)
        xy_tol = float(m.get("pretouch_xy_tol_mm", 1.0)) / 1000
        sag_max = float(m.get("touch_sag_abort_mm", 6.0)) / 1000
        floor = want - float(m.get("touch_cmd_below_mm", 3.0)) / 1000 if floor_m is None else floor_m
        cap = np.deg2rad(float(m.get("touch_correct_max_deg", 3.0)))
        rounds = int(m.get("touch_correct_rounds", 4))
        mask = self._movable(q_goal)
        P_goal = k.tip(q_goal)
        base = np.asarray(cmd0 if base is None else base, float)
        lo, hi = np.minimum(k.lo, cmd0), np.maximum(k.hi, cmd0)
        cmd, n, move = np.asarray(cmd0, float).copy(), 0, None

        def out(status, xy_status="ok"):
            v = self._xy_err(q_meas, P_goal)
            return dict(status=status, q=q_meas, rounds=n, cmd=cmd, err_mm=(self._above_part(q_meas) - want) * 1000,
                        xy_mm=float(np.linalg.norm(v)) * 1000, xy_vec_m=v, xy_status=xy_status, move=move)

        def limited(c):
            c = np.clip(c, base - cap, base + cap)
            return np.clip(c, lo, hi)
        while True:
            err = self._above_part(q_meas) - want
            xy = float(np.linalg.norm(self._xy_err(q_meas, P_goal)))
            if -err > sag_max:
                return out("sag")
            if abs(err) <= aim and xy <= xy_tol:
                return out("ok")
            if n >= rounds:
                return out("ok" if abs(err) <= tol else "stuck", "rounds" if xy > xy_tol else "ok")
            step = await asyncio.to_thread(self._tip_step, q_meas, P_goal, mask)
            new = limited(cmd + 0.8 * step)
            if abs(err) > aim and np.abs(new - (cmd + 0.8 * step)).max() > 1e-9:
                # the full step is cut by a cap: correct the height only this round (xy after)
                P_h = k.tip(q_meas) - err * k.n
                step = await asyncio.to_thread(self._tip_step, q_meas, P_h, mask)
                new = limited(cmd + 0.8 * step)
            new = await asyncio.to_thread(self._floored, q_goal, new, floor)
            if np.abs(new - cmd).max() < np.deg2rad(0.05):   # at a cap: nothing left that may be done
                return out("ok" if abs(err) <= tol else "stuck", "cap" if xy > xy_tol else "ok")
            n += 1
            move = await self._cmd_move(cmd, new, max_tip_mm=2.0)     # out of contact: 2 mm waypoints
            if not move["ok"]:
                return out("move_failed")
            cmd = new
            q_meas = await self._q_still()                    # measure only once the tip has stopped

    def _descent_goal(self, q_t, q_p, q_meas, cmd_pre, cmd_t, base_t, cap) -> np.ndarray:
        """Joint goal of the final descent that moves the MEASURED tip straight down (table normal) from where the
        pre-touch loop left it (q_meas: its corrected in-plane position is kept) to the planned touch height.
        Prediction of where the plain goal cmd_t (touch goal + offset + pre-touch correction) puts the joints: a
        joint movable at both poses keeps its pre-touch tracking error (q_meas + cmd_t - cmd_pre); a joint pinned at
        its limit at pre-touch but not at the touch (joint 6 stopping short of its limit) follows its command; one
        pinned at both stays where it is. The remaining tip error of that prediction is removed by 3 damped
        least-squares steps on joints 1-4 (FK only, no motion). Each joint stays within touch_correct_max_deg of
        cmd_t and within its range; the floor is applied by the caller. Falls back to cmd_t if not better."""
        k = self.kin
        m_p, m_t = self._movable(q_p), self._movable(q_t)
        q_meas, cmd_t = np.asarray(q_meas, float), np.asarray(cmd_t, float)
        P0 = k.tip(q_meas)
        P_goal = P0 + k.n * float((k.tip(q_t) - P0) @ k.n)
        pred0 = np.where(m_p & m_t, q_meas + (cmd_t - np.asarray(cmd_pre, float)), np.where(m_t, cmd_t, q_meas))
        pred = pred0.copy()
        for _ in range(3):
            pred = pred + self._tip_step(pred, P_goal, m_t)
        use = m_t & (np.arange(7) < 4)
        new = np.where(use, cmd_t + (pred - pred0), cmd_t)
        new = np.clip(new, cmd_t - cap, cmd_t + cap)
        new = np.clip(new, np.minimum(k.lo, base_t), np.maximum(k.hi, base_t))
        got = np.linalg.norm(k.tip(pred0 + np.where(use, new - cmd_t, 0.0)) - P_goal)
        return new if got < np.linalg.norm(k.tip(pred0) - P_goal) else cmd_t

    def _plan(self, P_touch, q_from) -> dict:
        """Joint targets for 'above -> hover -> touch' at one point, or GuardError saying why not."""
        k, m = self.kin, self.m
        if not k.in_workspace(P_touch, float(m["workspace_margin_mm"]) / 1000):
            raise GuardError("target is outside the area covered by the camera calibration markers")
        P_hover = P_touch + k.n * float(m["hover_mm"]) / 1000
        P_high = P_touch + k.n * float(m["high_mm"]) / 1000
        q_high, e1, _ = k.ik(P_high, q_from)
        q_hover, e2, tilt = k.ik(P_hover, q_high)
        tol = float(m["ik_tol_mm"])
        floor = k.surface_h + float(m["clearance_mm"]) / 1000
        allowed = min(floor, k.height(k.tip(q_from)) - 0.005)      # may stay as low as it already is
        # Neighbouring target (tip already near this hover point, not below it): the tip is off the roll-7 axis
        # and joint 6 sits at its limit, so the tilt term keeps creeping joint 7 a little further on every
        # chained solve. Prefer a hover solved from the current joints with joint 7 held near where it is,
        # if it moves the joints less and costs at most 5 deg of tilt.
        tip, roll7 = k.tip(q_from), 60.0
        if (np.linalg.norm(tip - P_hover) < float(m.get("near_mm", 80.0)) / 1000
                and k.height(tip) >= k.height(P_hover) - 0.005):
            win = float(m.get("near_roll7_window_deg", 10.0))
            qn, en, tn = k.ik(P_hover, q_from, roll7_window_deg=win)
            if (en <= tol and tn <= tilt + 5.0 and k.path_clear(q_from, qn, floor)
                    and np.abs(qn - q_from).max() < np.abs(q_hover - q_from).max()):
                qh, eh, _ = k.ik(P_high, qn, roll7_window_deg=win)   # matching escape pose, same branch
                if eh <= tol:
                    q_high, e1, q_hover, e2, tilt, roll7 = qh, eh, qn, en, tn, win
        q_touch, e3, _ = k.ik(P_touch, q_hover, roll7_window_deg=roll7)
        # pre-touch pose (pretouch_mm above the part, straight above the touch point): the height/xy correction
        # runs there, out of contact; the lift retraces touch -> pre-touch. Without a pre-touch stage
        # (pretouch_mm <= touch_mm) a point lift_mm straight above the touch is used for the first lift leg.
        touch_mm = float(m["touch_mm"])
        pre_mm = float(m.get("pretouch_mm", 12.0))
        stage = touch_mm < pre_mm < touch_mm + float(m["hover_mm"])     # else: no pre-touch stage
        up_m = (pre_mm - touch_mm) / 1000 if stage else float(m.get("lift_straight_mm", 6.0)) / 1000
        q_pre, e4, _ = k.ik(P_touch + k.n * up_m, q_touch, roll7_window_deg=roll7)
        if max(e1, e2, e3, e4) > tol:
            raise GuardError(f"target not reachable (IK error {max(e1, e2, e3, e4):.0f} mm > {tol:.0f} mm)")
        direct = k.path_clear(q_from, q_hover, floor)
        if not direct and not k.path_clear(q_from, q_high, floor):
            raise GuardError("refused: the path to the target would pass too low over the table/part "
                             "(retreat first, or raise high_mm)")
        if (k.path_lowest(q_high, q_hover) < k.height(P_hover) - 0.010
                or k.path_lowest(q_hover, q_touch) < k.height(P_touch) - 0.004):
            raise GuardError("refused: the descent onto the target would dip below it")
        return dict(q_high=q_high, q_hover=q_hover, q_touch=q_touch, q_pre=q_pre, pretouch=stage, direct=direct,
                    tilt_deg=round(tilt, 1))

    async def retreat(self) -> dict:
        """Lift straight up, then go to the look pose (out of the camera's view), or home if none is taught."""
        await self.ensure_guided()
        self._last_aim, self._jog = None, None
        q = await self._q_now()
        lift = float(self.m["high_mm"]) / 1000 - self._above_part(q)
        if 0.005 < lift and self._above_part(q) < float(self.m["hover_mm"]) / 1000 + 0.02:
            q_up, e, _ = await asyncio.to_thread(self.kin.ik, self.kin.tip(q) + self.kin.n * lift, q)
            if e < 5:
                r = await self._guided_move(q_up, "slow", settle=False)
                if not r["ok"]:
                    return {"ok": False, "step": "lift", **r}
        self._forget_droop()                                   # relearned (tip loop) at the next hover
        self.poses.reload()
        pose = self.m["look_pose"] if self.m["look_pose"] in self.poses.names(self.arm) else \
            self.cfg.arms[self.arm].home_pose
        return await self.go_to_pose(self.arm, pose, speed="slow")

    # ================================================================= vision
    async def _vstate(self) -> dict:
        st = await self.vision.state()
        age = time.time() - float(st["timestamp"])
        if age > float(self.gcfg["vision"]["max_state_age_s"]):
            raise VisionError(f"vision data is {age:.1f} s old (vision stalled?)")
        size = st.get("image_size")
        if size and self.kin is not None and tuple(size) != self._size_ok and self.kin.calib.get("image_size"):
            status, why = image_size_check(size, self.kin.calib["image_size"])     # once per new image size
            if status == "fail":
                raise VisionError(f"{why}. Fix: {ASPECT_FIX}")
            self._size_ok = tuple(size)
        return st

    @staticmethod
    def _live(st) -> bool:
        return bool(st and st.get("safe_to_act") and st.get("status") != "STOP" and st.get("box")
                    and st["box"]["pose_source"] == "live")

    @staticmethod
    def _has(st, cond) -> bool:
        return any(c["type"] == cond for c in (st or {}).get("conditions", []))

    @staticmethod
    def _brief(st) -> dict:
        if not st:
            return {"status": "NO DATA"}
        return {"status": st["status"], "safe_to_act": st["safe_to_act"],
                "pose": st["box"]["pose_source"] if st.get("box") else None,
                "conditions": [f'{c["type"]}: {c["detail"]}' for c in st["conditions"]]}

    async def _wait_live(self, timeout: float, still: bool = False):
        """(state, last_state): state is None if no clear live view arrived in time."""
        t_end, last = time.monotonic() + timeout, None
        while True:
            try:
                last = await self._vstate()
                if self._live(last) and not (still and self._has(last, "BOX_MOVING")):
                    return last, last
            except VisionError as e:
                last = {"status": "NO DATA", "safe_to_act": False, "box": None,
                        "conditions": [{"type": "VISION", "detail": str(e)}]}
            if time.monotonic() > t_end:
                return None, last
            await asyncio.sleep(0.1)

    def _aim(self, st, spec) -> dict:
        """Where to put the tool tip for this target, from one vision state."""
        return self._aim_px(self._target_px(st, spec), st["image_size"])

    @staticmethod
    def _target_px(st, spec):
        """The target's pixel in one vision state."""
        if "hole" in spec:
            h = next((x for x in st["holes"] if str(x["id"]) == str(spec["hole"])), None)
            if h is None:
                raise GuardError(f"no hole {spec['hole']} on this part; holes: {[x['id'] for x in st['holes']]}")
            px = h["px"]
        else:
            fx, fy = (float(v) for v in spec["at"])
            if not (-0.05 <= fx <= 1.05 and -0.05 <= fy <= 1.05):
                raise GuardError("'at' is a position on the lid as fractions 0..1 of the box length and width")
            px = frac_to_px(st["box"]["corners_px"], (fx, fy))
        return px

    def _aim_px(self, px, image_size) -> dict:
        """Where to put the tool tip for a target pixel."""
        k = self.kin
        P0 = k.surface_point(px, image_size)
        d = P0 - k.table.o
        pos = np.array([d @ k.table.e1, d @ k.table.e2]) * 1000
        corr = self.model.correction_mm(pos)
        P = (P0 + k.aim + (corr[0] * k.table.e1 + corr[1] * k.table.e2) / 1000
             + k.n * float(self.m["touch_mm"]) / 1000)
        return dict(px=[round(float(px[0]), 1), round(float(px[1]), 1)], P0=P0, P=P, pos_mm=pos, corr_mm=corr)

    async def _aim_filtered(self, spec, st0=None) -> tuple[dict, dict]:
        """Aim on the per-coordinate median of the target pixel over `aim_frames` live states (~0.05 s apart).
        Gives up after 1 s and uses what it has. Returns (aim, last live state)."""
        n = max(1, int(self.m.get("aim_frames", 3)))
        pxs, last = [], None
        if st0 is not None and self._live(st0):
            pxs.append(self._target_px(st0, spec))        # raises if the hole is not on this part
            last = st0
        t_end = time.monotonic() + 1.0
        while len(pxs) < n and time.monotonic() < t_end:
            if pxs:
                await asyncio.sleep(0.05)
            try:
                st = await self._vstate()
            except VisionError:
                continue
            if self._live(st):
                try:
                    pxs.append(self._target_px(st, spec))
                    last = st
                except GuardError:
                    if not pxs:
                        raise
        if not pxs:
            raise GuardError("no clear live view of the part while aiming")
        px = np.median(np.asarray(pxs, float), axis=0)
        aim = self._aim_px(px, last["image_size"])
        aim["frames"] = len(pxs)
        return aim, last

    @staticmethod
    def _key(spec) -> str:
        return f"hole {spec['hole']}" if "hole" in spec else f"at {float(spec['at'][0]):.2f},{float(spec['at'][1]):.2f}"

    # ================================================================= sighting (arm out of the picture)
    def _sees_under_arm(self) -> bool:
        """true only with a camera that sees the part while the arm works on it (e.g. a wrist camera)."""
        return bool(self.gcfg["vision"].get("sees_under_arm", False))

    def _arm_clear(self) -> bool:
        return self.zone[self.arm] == "clear"

    def _arm_over_part(self) -> bool:
        """The worker arm is in the work zone and the camera cannot see past it: vision is not evidence now."""
        return not self._sees_under_arm() and self.zone[self.arm] == "work"

    def arm_in_view_at(self, t, grace_s: float = 1.0) -> bool:
        """Was the worker arm in the work zone at time t (time.time())? grace_s covers tracker lag after it left."""
        if t is None or self._sees_under_arm():
            return False
        t = float(t)
        return any(a - 0.2 <= t <= (b if b is not None else float("inf")) + grace_s for a, b in self._in_view)

    def _sight_ok(self) -> dict | None:
        """The stored clear sighting, or None (dropped if the part changed or the e-stop is latched)."""
        s = self._sight
        if s is not None and (s["part_id"] != (self.part or {}).get("id") or self.estop_state().get("latched")):
            self._sight = s = None
        return s

    def _store_sight(self, states: list) -> dict:
        self._sight = {"state": states[-1], "states": list(states), "t": time.time(),
                       "part_id": (self.part or {}).get("id")}
        return self._sight

    async def _look(self) -> tuple[dict | None, dict | None]:
        """Take a sighting: up to aim_frames live states (~0.05 s apart, max 1 s more), only with the arm at a
        clear pose. Returns (sighting or None, last vision state)."""
        if not self._arm_clear():
            raise GuardError("a sighting needs the worker arm at a clear pose (retreat first)")
        st, last = await self._wait_live(float(self.gcfg["vision"]["live_wait_s"]))
        if st is None:
            return None, last
        states, n, t_end = [st], max(1, int(self.m.get("aim_frames", 3))), time.monotonic() + 1.0
        while len(states) < n and time.monotonic() < t_end:
            await asyncio.sleep(0.05)
            try:
                s2 = await self._vstate()
            except VisionError:
                continue
            if self._live(s2):
                states.append(s2)
        sight = self._store_sight(states)
        self._forget_droop()                                   # new look: the next hover learns it afresh
        await self._keep_mark_ref(states[-1])
        return sight, states[-1]

    async def _clear_look(self) -> tuple[dict | None, str | None, dict | None]:
        """Retreat if the arm is not at a clear pose, then take a sighting. (sighting, failure reason, last state)."""
        if not self._arm_clear():
            r = await self.retreat()
            if not r["ok"]:
                return None, "could not retreat to get a clear view", {"move": r}
        sight, last = await self._look()
        if sight is None:
            return None, "no clear live view of the part (hidden, removed, or vision stalled)", last
        return sight, None, last

    def _aim_sight(self, spec, sight: dict) -> dict:
        """Aim on the median target pixel over the sighting's states."""
        pxs, err = [], None
        for st in sight["states"]:
            try:
                pxs.append(self._target_px(st, spec))
            except GuardError as e:
                err = e
        if not pxs:
            raise err
        aim = self._aim_px(np.median(np.asarray(pxs, float), axis=0), sight["state"]["image_size"])
        aim["frames"] = len(pxs)
        return aim

    def _diff_marks(self) -> bool:
        return str(self.gcfg["marks"].get("method", "color")).lower() == "diff"

    async def _keep_mark_ref(self, st, force: bool = False) -> None:
        """[marks] method = "diff": keep the raw picture of a clear look as the 'before' image for inspect. It is
        replaced only while no dab was made since it was taken (so it stays the look BEFORE the first dab of a run,
        also with look_each_step), when the part changed, or when forced (after an inspect)."""
        if not self._diff_marks() or st is None:
            return
        r, pid = self._mark_ref, (self.part or {}).get("id")
        if not force and r is not None and r["part_id"] == pid and self._dabs_since_ref > 0:
            return
        try:
            frame = await self.vision.raw()
        except Exception:  # noqa: BLE001  no picture: keep the old reference (inspect says if there is none)
            return
        self._mark_ref = {"frame": np.array(frame, copy=True), "state": st, "t": time.time(), "part_id": pid}
        self._dabs_since_ref = 0

    async def _emergency(self, reason: str):
        self._sight = None                    # after an e-stop nothing seen before counts
        await super()._emergency(reason)

    # ============================================================= public API
    async def new_part(self, part_id: str | None = None) -> dict:
        """A new part is in the work area: forget the old hole layout and wait until the new one is locked."""
        await self.ensure_guided()
        self._sight = None
        await self.vision.reset()
        st, last = await self._wait_live(float(self.gcfg["vision"]["live_wait_s"]) + 3.0)
        if st is None or not st.get("map_locked"):
            raise GuardError(f"vision did not lock onto a part: {self._brief(last)}. The arm must be out of the "
                             "camera's view (retreat) and the reference line and all holes visible.")
        self.part = {"id": part_id or uuid.uuid4().hex[:6], "marked": {}, "learned": [], "misses": {}}
        self._work = {}
        self._last_aim = None
        self._mark_ref, self._dabs_since_ref = None, 0
        if self._arm_clear():
            self._store_sight([st])
            await self._keep_mark_ref(st)
        self.log_event("part", "guided part started", holes=len(st["holes"]))
        return {"ok": True, "part": self.part["id"], "holes": [h["id"] for h in st["holes"]],
                "box_angle_deg": st["box"]["angle_deg"]}

    async def scene(self) -> tuple[dict, bytes]:
        """What the camera sees now: status, box pose, holes with robot coordinates and reachability."""
        await self.ensure_guided()
        try:
            st = await self._vstate()
        except VisionError as e:
            raise GuardError(str(e))
        out = self._brief(st)
        over = self._arm_over_part()
        if over:
            nr = "not reliable: arm in view"
            out = {"warning": "the worker arm is over the part: it hides and distorts the part in the top camera's "
                              "picture (dark arm parts can look like holes, the reference line may be hidden). "
                              "The box pose, holes and conditions below are NOT reliable; retreat for a reliable "
                              "view.", **out}
            out["pose"] = f"{nr} (tracker says {out.get('pose')})"
            out["conditions"] = [f"{c} ({nr})" for c in out["conditions"]]
        elif self._arm_clear() and self._live(st) and not self._sees_under_arm():
            self._store_sight([st])           # a clear look also refreshes the sighting
            await self._keep_mark_ref(st)
        out.update(vision=self.vision.describe(), map_locked=st.get("map_locked"), part=(self.part or {}).get("id"))
        if st.get("box"):
            out["box"] = {"angle_deg": st["box"]["angle_deg"], "center_px": st["box"]["center_px"],
                          "pose_age_s": st["box"].get("pose_age_s")}
            if over:
                out["box"]["reliability"] = "not reliable: arm in view"
            q = await self._q_now()
            holes = []
            for h in st["holes"][:12]:
                a = self._aim(st, {"hole": h["id"]})
                info = {"id": h["id"], "visible": h["visible"], "on_lid": [round(v, 2) for v in h["box_frac"]],
                        "robot_mm": [round(float(v)) for v in a["P0"] * 1000],
                        "learned_correction_mm": [round(float(v), 1) for v in a["corr_mm"]],
                        "done": self._key({"hole": h["id"]}) in self._work}
                if over:
                    info["reliability"] = "not reliable: arm in view"
                try:
                    await asyncio.to_thread(self._plan, a["P"], q)
                    info["reachable"] = True
                except GuardError as e:
                    info["reachable"], info["why_not"] = False, str(e)
                holes.append(info)
            out["holes"] = holes
        ev = await self.vision.events(self._ev_seq)
        if ev:
            self._ev_seq = ev[-1]["seq"]
        out["new_vision_events"] = [f'{e["type"]} {e["phase"]} {e["detail"]}'.strip()
                                    + (" (arm in view: not reliable)" if self.arm_in_view_at(e.get("t")) else "")
                                    for e in ev[-15:]]
        return out, await self.vision.annotated()

    async def work_on(self, spec: dict, action: str = "dab", dwell_s: float | None = None,
                      _program: bool = False) -> dict:
        """One target: aim, go above it, touch it, lift.
        sees_under_arm = false (top camera only): aim from a sighting taken with the arm at a clear pose (fresh
        if the arm is clear now; else the stored one if younger than sight_max_age_s; else retreat and look).
        The part is not watched while the arm is over it. sees_under_arm = true: the old follow loop."""
        await self.ensure_guided()
        if action not in ("dab", "hover"):
            raise GuardError("action must be 'dab' or 'hover'")
        if ("hole" in spec) == ("at" in spec):
            raise GuardError('a step needs exactly one of {"hole": id} or {"at": [fx, fy]}')
        k, m, key = self.kin, self.m, self._key(spec)
        t_start = time.time()
        self._last_aim, self._jog = None, None

        diag: dict = {}                                        # hover / pre-touch numbers once the arm is there

        def fail(reason, st=None, **extra):
            self.log_event("guided_fail", f"{key}: {reason}")
            out = {"ok": False, "target": key, "reason": reason, "vision": self._brief(st), **diag, **extra}
            if diag:                                           # failed at / after the hover: log it for diagnosis
                rec = dict(t=time.time(), part=(self.part or {}).get("id"), target=key, ok=False, reason=reason,
                           offset_deg=[round(float(v), 2) for v in np.degrees(self._offset)],
                           **{kk: v for kk, v in out.items() if kk.startswith(("hover_", "pretouch_", "touch_"))},
                           seconds=round(time.time() - t_start, 2))
                try:
                    with open(self.glog, "a") as f:
                        f.write(json.dumps(rec) + "\n")
                except (OSError, TypeError, ValueError):
                    pass
            return out

        follow = self._sees_under_arm()

        async def close_gripper():
            """(closed now?, failure reason or None): closed fingertips are the tool."""
            if not self.gcfg.get("tool", {}).get("close_gripper", False):
                return False, None
            acfg = self.cfg.arms[self.arm]
            g_pos = (await self.backend.read_state())[self.arm][acfg.gripper]["pos"]
            if g_pos <= acfg.gripper_closed + 0.08:
                return False, None
            try:
                await self.gripper(self.arm, "close")          # cellbot's guarded command; takes the lock itself
            except GuardError as e:
                return False, f"could not close the gripper before guided motion: {e}"
            return True, None

        sight = None
        if follow:
            st, last = await self._wait_live(float(self.gcfg["vision"]["live_wait_s"]))
            if st is None and self.zone[self.arm] == "work":       # probably our own arm hiding the part
                r = await self.retreat()
                if not r["ok"]:
                    return fail("could not retreat to get a clear view", last, move=r)
                st, last = await self._wait_live(float(self.gcfg["vision"]["live_wait_s"]))
            if st is None:
                return fail("no clear live view of the part (hidden, removed, or vision stalled)", last)
            closed_first, why = await close_gripper()
            if why:
                return fail(why, st)
            if closed_first:
                st = None                                      # that took a moment: aim on fresh frames
            aim, st = await self._aim_filtered(spec, st)
            t_seen = time.time()
        else:
            sight = self._sight_ok()
            max_age = float(m.get("sight_max_age_s", 60.0))
            usable = sight is not None and time.time() - sight["t"] <= max_age
            # a fresh look when the arm is clear anyway (a program step reuses the look it just took)
            need_look = not usable or (self._arm_clear() and not _program)
            if need_look and not self._arm_clear():            # never read vision for aiming with the arm in view
                r = await self.retreat()
                if not r["ok"]:
                    return fail("could not retreat to get a clear view", sight and sight["state"], move=r)
            closed_first, why = await close_gripper()
            if why:
                return fail(why, sight and sight["state"])
            if need_look:
                sight, last = await self._look()
                if sight is None:
                    return fail("no clear live view of the part (hidden, removed, or vision stalled)", last)
            st = sight["state"]
            aim = self._aim_sight(spec, sight)                 # raises if the hole is not on this part
            t_seen = sight["t"]

        q_from = await self._q_now()
        plan = await asyncio.to_thread(self._plan, aim["P"], q_from)          # IK solves off the event loop
        if not plan["direct"]:
            r = await self._guided_move(plan["q_high"], settle=False)
            if not r["ok"]:
                return fail("move above the target failed", st, move=r)
        self._hover_info = None
        r = await self._guided_move(plan["q_hover"])
        if not r["ok"]:
            return fail("move to hover failed", st, move=r)
        if self._hover_info:
            diag.update(self._hover_info)

        reaims, followed_mm, t_follow = 0, 0.0, time.monotonic()
        while follow:                                          # follow the part while hovering (camera sees under arm)
            try:
                st2 = await self._vstate()
            except VisionError:
                st2 = None
            if self._live(st2):
                new, t_seen = self._aim(st2, spec), time.time()
                moved = float(np.linalg.norm(new["P"] - aim["P"]) * 1000)
                if moved > float(m["reaim_mm"]):                 # confirm on a filtered aim before moving
                    new, st2 = await self._aim_filtered(spec, st2)
                    t_seen = time.time()
                    moved = float(np.linalg.norm(new["P"] - aim["P"]) * 1000)
                if moved > float(m.get("max_reaim_mm", 80.0)):
                    await self._guided_move(plan["q_high"], "slow", settle=False)
                    return fail(f"the part jumped {moved:.0f} mm (more than max_reaim_mm); look again", st2,
                                reaims=reaims)
                if moved > float(m["reaim_mm"]) and reaims >= int(m.get("max_reaims", 6)):
                    await self._guided_move(plan["q_high"], "slow", settle=False)
                    return fail(f"re-aimed {reaims} times already (max_reaims); the part does not settle, "
                                "look again", st2, reaims=reaims, followed_mm=round(followed_mm, 1))
                if moved > float(m["reaim_mm"]) or self._has(st2, "BOX_MOVING"):
                    if time.monotonic() - t_follow > float(m["follow_s"]):
                        await self._guided_move(plan["q_high"], "slow", settle=False)
                        return fail(f"the part kept moving for {m['follow_s']} s", st2, reaims=reaims)
                if moved > float(m["reaim_mm"]):
                    try:
                        q_from = await self._q_now()
                        plan = await asyncio.to_thread(self._plan, new["P"], q_from)
                    except GuardError as e:
                        await self._guided_move(plan["q_high"], "slow", settle=False)
                        return fail(f"the part moved to where the arm cannot follow: {e}", st2, reaims=reaims)
                    aim, reaims, followed_mm = new, reaims + 1, followed_mm + moved
                    self._hover_info = None
                    r = await self._guided_move(plan["q_hover"])
                    if not r["ok"]:
                        return fail("re-aim move failed", st2, move=r)
                    if self._hover_info:
                        diag.update(self._hover_info)
                    continue
                if self._has(st2, "BOX_MOVING"):
                    await asyncio.sleep(0.1)
                    continue
                break                                          # seen, in place, still
            if time.time() - t_seen <= float(m["commit_age_s"]):
                break                                          # hidden (by our arm): use the last sighting
            await self._guided_move(plan["q_high"], "slow", settle=False)
            return fail(f"lost sight of the part for more than {m['commit_age_s']} s before touching", st2)

        aim_age = round(time.time() - t_seen, 2)
        q_at = await self._q_now()
        hover_err = float(np.linalg.norm((k.tip(q_at) - (aim["P"] + k.n * float(m["hover_mm"]) / 1000))[:2]) * 1000)
        result = {"ok": True, "target": key, "action": action, "aimed_robot_mm": [round(float(v)) for v in aim["P"] * 1000],
                  "learned_correction_mm": [round(float(v), 1) for v in aim["corr_mm"]],
                  "re_aims": reaims, "followed_mm": round(followed_mm, 1), "sighting_age_s": aim_age,
                  "tool_tilt_deg": plan["tilt_deg"], "hover_error_model_mm": round(hover_err, 1), **diag}
        if not follow:
            result["aimed_from"] = f"clear view {aim_age:.1f} s ago"
        if closed_first:
            result["gripper_closed_first"] = True
        if action == "dab":
            # 1. hover -> pre-touch (pretouch_mm above the part, straight above the touch point) with the carried
            #    offset, plus the correction the previous hole of this program found (seed);
            # 2. closed loop on the measured tip THERE, out of contact (height and xy, same caps/tolerance);
            # 3. final descent pre-touch -> touch: one short, slow, straight move carrying that correction;
            #    nothing is corrected in contact, so the tip cannot drag across the lid and push the part;
            # 4. at the touch only measure and report (sag -> lift, fail; too high -> lift, fail);
            # 5. lift: straight up first (the final descent retraced), then to hover.
            q_t, q_p = plan["q_touch"], plan["q_pre"]
            cmd_mm = self._above_part(q_t) * 1000
            below = float(m.get("touch_cmd_below_mm", 3.0))
            tol_mm = float(m.get("touch_tol_mm", 2.0))
            sag_mm = float(m.get("touch_sag_abort_mm", 6.0))
            cap = np.deg2rad(float(m.get("touch_correct_max_deg", 3.0)))
            corr, cmd_from, pc = np.zeros(7), None, None

            async def lift_from(cmd_low) -> dict:
                """Leg 1: the commanded goal moved by (lift point - touch) in joints, i.e. straight up along the
                table normal by pretouch_mm - touch_mm (or lift_straight_mm), interpolated from the commanded goal:
                no sideways motion near the lid. Leg 2: to hover. Never raises."""
                try:
                    up = await self._cmd_move(cmd_low, np.asarray(cmd_low, float) + (q_p - q_t))
                    if not up["ok"]:
                        return {"ok": False, "leg": "straight up", **up}
                    return await self._cmd_move(self._last_cmd, self._with_offset(plan["q_hover"]),
                                                max_step=float(self.g["interp_step"]), max_tip_mm=None)
                except GuardError as e:                      # e.g. the e-stop latched meanwhile
                    return {"ok": False, "error": str(e)}

            if plan["pretouch"]:
                pre_h = self._above_part(q_p)
                floor_p = pre_h - below / 1000                 # commanded tip never below pre-touch - touch_cmd_below_mm
                base_p = self._with_offset(q_p)
                seed = np.zeros(7) if self._seed is None else np.clip(self._seed, -cap, cap) * self._movable(q_p)
                cmd0 = np.clip(base_p + seed, np.minimum(k.lo, base_p), np.maximum(k.hi, base_p))
                cmd0 = await asyncio.to_thread(self._floored, q_p, cmd0, floor_p)
                r = await self._guided_move(q_p, "slow", settle=False, cmd=cmd0)
                if not r["ok"]:
                    return fail("descent to the pre-touch height failed", None, move=r)
                q_first = await self._q_still()                # the tip has stopped (no creeping transient)
                pc = await self._touch_correct(q_p, cmd0, q_first, base=base_p, floor_m=floor_p)
                pre = dict(pretouch_mm=round(pre_h * 1000, 1),
                           pretouch_error_before_mm=round((self._above_part(q_first) - pre_h) * 1000, 1) + 0.0,
                           pretouch_error_after_mm=round(pc["err_mm"], 1) + 0.0,
                           pretouch_correction_rounds=pc["rounds"],
                           pretouch_seeded_deg=[round(float(v), 2) for v in np.degrees(cmd0 - base_p)],
                           pretouch_correction_deg=[round(float(v), 2) for v in np.degrees(pc["cmd"] - base_p)],
                           pretouch_q_arrived=[round(float(v), 5) for v in q_first],
                           pretouch_q_sent=[round(float(v), 5) for v in cmd0],
                           pretouch_xy_error_before_mm=round(float(np.linalg.norm(self._xy_err(q_first, k.tip(q_p)))) * 1000, 1),
                           pretouch_xy_error_after_mm=round(pc["xy_mm"], 1),
                           pretouch_xy_error_vec_mm=[round(float(v) * 1000, 1) + 0.0 for v in pc["xy_vec_m"]],
                           pretouch_xy_status=pc["xy_status"])
                result.update(pre)
                diag.update(pre)
                if pc["status"] != "ok":
                    self._seed = None if self._seed is None else np.zeros(7)
                    try:                                       # above the part, out of contact: back to hover
                        lift = await self._cmd_move(pc["cmd"], self._with_offset(plan["q_hover"]),
                                                    max_step=float(self.g["interp_step"]), max_tip_mm=None)
                    except GuardError as e:
                        lift = {"ok": False, "error": str(e)}
                    info = dict(pre, lift=lift, tip_above_part_mm=round(self._above_part(await self._q_now()) * 1000, 1))
                    if pc["status"] == "move_failed":
                        return fail("pre-touch correction move failed", None, move=pc["move"], **info)
                    if pc["status"] == "sag":
                        return fail(f"the arm sagged {-pre['pretouch_error_after_mm']:.1f} mm below the commanded "
                                    "pre-touch height; droop correction is not good enough here; lifted before "
                                    "touching", st, **info)
                    return fail(f"pre-touch height not reached within {tol_mm:g} mm after {pc['rounds']} correction "
                                f"round(s) (error {pre['pretouch_error_after_mm']:+.1f} mm; joint stuck or a "
                                "correction cap reached); lifted without touching, step not done", st, **info)
                corr, cmd_from = pc["cmd"] - base_p, pc["cmd"]
            # the final descent carries the pre-touch correction; its commanded tip never below touch - touch_cmd_below_mm
            base_t = self._with_offset(q_t)
            cmd_t = np.clip(base_t + corr, np.minimum(k.lo, base_t), np.maximum(k.hi, base_t))
            if cmd_from is not None:
                cmd_t = await asyncio.to_thread(self._descent_goal, q_t, q_p, pc["q"], cmd_from, cmd_t, base_t, cap)
            cmd_t = await asyncio.to_thread(self._floored, q_t, cmd_t, (cmd_mm - below) / 1000)
            if cmd_from is not None:
                r = await self._cmd_move(cmd_from, cmd_t)      # short, slow, straight
            else:                                              # no pre-touch stage: straight from hover, as before
                r = await self._guided_move(q_t, "slow", settle=False, cmd=cmd_t)
            if not r["ok"]:
                return fail("descent failed", None, move=r)
            q_reached = await self._q_now()
            got_mm = self._above_part(q_reached) * 1000
            err = round(got_mm - cmd_mm, 1) + 0.0                # no "-0.0"
            txy = self._xy_err(q_reached, k.tip(q_t))              # arm's own in-plane tracking error at the touch
            result.update(touch_xy_error_mm=round(float(np.linalg.norm(txy)) * 1000, 1),
                          touch_xy_error_vec_mm=[round(float(v) * 1000, 1) + 0.0 for v in txy])
            result.update(touch_error_before_mm=err, touch_error_after_mm=err, touch_error_mm=err,
                          touch_correction_rounds=0,           # nothing is corrected in contact any more
                          touch_correction_deg=[round(float(v), 2) for v in np.degrees(cmd_t - base_t)])
            if -err > sag_mm or err > tol_mm:                  # measure only: no motion at the touch, lift at once
                self._seed = None if self._seed is None else np.zeros(7)
                lift = await lift_from(cmd_t)
                info = {kk: result[kk] for kk in result if kk.startswith(("touch_", "pretouch_"))}
                info.update(lowest_tip_above_part_mm=round(got_mm, 1), lift=lift,
                            tip_above_part_mm=round(self._above_part(await self._q_now()) * 1000, 1))
                if -err > sag_mm:
                    return fail(f"the arm sagged {-err:.1f} mm below the commanded touch height; droop "
                                "correction is not good enough here", st, **info)
                return fail(f"touch height not reached within {tol_mm:g} mm after the final descent (error "
                            f"{err:+.1f} mm: blocked, or the droop changed between pre-touch and touch); lifted, "
                            "step not done", st, **info)
            await asyncio.sleep(float(self.m["dwell_s"] if dwell_s is None else dwell_s))
            q_low = await self._q_now()
            result["lowest_tip_above_part_mm"] = round(min(got_mm, self._above_part(q_low) * 1000), 1)
            r = await lift_from(cmd_t)                         # straight up first, then hover
            result["ok"] = r["ok"]
            if not r["ok"]:
                result["lift"] = r
            if self._seed is not None and plan["pretouch"]:
                self._seed = corr.copy() if r["ok"] else np.zeros(7)   # starting correction for the next hole
            self._work[key] = dict(spec=spec, corr_mm=[float(v) for v in aim["corr_mm"]], t=time.time())
            self._dabs_since_ref += 1
            if self.part is not None:
                self.part["marked"][key] = {"correction_mm": self._work[key]["corr_mm"], "dq": {}}
            rec = dict(t=time.time(), part=(self.part or {}).get("id"), target=key, px=aim["px"],
                       box_angle_deg=st["box"]["angle_deg"], aimed_m=aim["P"].tolist(), corr_mm=self._work[key]["corr_mm"],
                       re_aims=reaims, followed_mm=round(followed_mm, 1), sighting_age_s=aim_age,
                       q_cmd=plan["q_touch"].tolist(), q_reached=q_reached.tolist(), q_sent=cmd_t.tolist(),
                       offset_deg=[round(float(v), 2) for v in np.degrees(self._offset)],
                       touch_error_mm=err, touch_xy_error_mm=result["touch_xy_error_mm"],
                       touch_xy_error_vec_mm=result["touch_xy_error_vec_mm"], **{kk: result[kk] for kk in result if kk.startswith(("pretouch_", "hover_"))},
                       tip_reached_m=k.tip(q_reached).tolist(), seconds=round(time.time() - t_start, 2))
            with open(self.glog, "a") as f:
                f.write(json.dumps(rec) + "\n")
        q_end = await self._q_now()
        result["tip_above_part_mm"] = round(self._above_part(q_end) * 1000, 1)
        if result["ok"]:
            self._last_aim = dict(spec=dict(spec), key=key, action=action, P=aim["P"].copy(),
                                  P_tip=aim["P"] + k.n * float(m["hover_mm"]) / 1000, q=np.asarray(plan["q_hover"]).copy(),
                                  px=list(aim["px"]), image_size=list(st["image_size"]), vision=self._brief(st),
                                  t=time.time())
        result["seconds"] = round(time.time() - t_start, 1)
        if follow:
            result["note"] = "arm is hovering above the part; it hides the part from the top camera until it retreats"
        else:
            result["note"] = (f"aimed from one clear view of the part taken {aim_age:.1f} s ago with the arm out of the "
                              "picture. With the top camera the part is NOT watched while the arm is over it: if "
                              "it moved since that view, this step did not notice. Look again between steps "
                              "(run --look) or retreat before the next run. The arm is now hovering above the part.")
        return result

    async def run_program(self, steps: list[dict], retreat_after: bool = True, look_each_step: bool = False) -> dict:
        """Run a program written in the PART's frame. Stops at the first step that fails and says why.
        sees_under_arm = false: one clear look at the start (retreat first if needed), every step aims from it.
        look_each_step (or "look": true on a step): retreat and look again before that step."""
        if not isinstance(steps, list) or not steps:
            raise GuardError('steps must be a list like [{"hole": 0}, {"at": [0.5, 0.5], "action": "hover"}]')
        await self.ensure_guided()
        follow = self._sees_under_arm()
        done = []
        self._seed = np.zeros(7)                       # the pre-touch correction of a hole seeds the next one
        try:
            return await self._run_steps(steps, follow, done, retreat_after, look_each_step)
        finally:
            self._seed = None

    async def _run_steps(self, steps, follow, done, retreat_after, look_each_step) -> dict:
        for i, s in enumerate(steps):
            spec = {kk: s[kk] for kk in ("hole", "at") if kk in s}
            try:
                r = None
                if not follow and (i == 0 or look_each_step or s.get("look") or s.get("look_each_step")):
                    sight, why, last = await self._clear_look()
                    if sight is None:
                        key = self._key(spec) if len(spec) == 1 else str(spec)
                        self.log_event("guided_fail", f"{key}: {why}")
                        r = {"ok": False, "target": key, "reason": why,
                             "vision": self._brief(last if last and "status" in last else None)}
                if r is None:
                    r = await self.work_on(spec, s.get("action", "dab"), s.get("dwell_s"), _program=not follow)
            except GuardError as e:
                r = {"ok": False, "target": str(spec), "reason": f"refused: {e}"}
                self.log_event("guided_fail", f"{spec}: refused: {e}")
            done.append(r)
            if not r["ok"]:
                return {"ok": False, "completed_steps": i, "failed_step": i, "results": done,
                        "remaining_steps": steps[i:], "note": "decide: look again (see_part), retry the "
                        "remaining steps, skip this one, ask a human, or stop"}
        out = {"ok": True, "completed_steps": len(steps), "results": done}
        if retreat_after:
            out["retreat"] = await self.retreat()
        return out

    async def plan_program(self, steps: list[dict]) -> dict:
        """Check a program without moving: per step the target, reachability (or why not), tool tilt, path and
        joint travel, planned from the current vision state and arm pose, each step from the previous hover pose.
        Read-only: no motion, no zone change, no lock."""
        if not isinstance(steps, list) or not steps:
            raise GuardError('steps must be a list like [{"hole": 0}, {"at": [0.5, 0.5], "action": "hover"}]')
        await self.ensure_guided()
        st, last = await self._wait_live(float(self.gcfg["vision"]["live_wait_s"]))
        if st is None:
            return {"ok": False, "reason": "no clear live view of the part (hidden by the arm? retreat first)",
                    "vision": self._brief(last), "steps": []}
        q_from = await self._q_now()

        def travel(a, b) -> float:
            return round(float(np.degrees(np.abs(np.asarray(b) - np.asarray(a)).max())), 1)

        out = []
        for i, s in enumerate(steps):
            s = s if isinstance(s, dict) else {}
            spec = {kk: s[kk] for kk in ("hole", "at") if kk in s}
            action = s.get("action", "dab")
            r = {"step": i, "target": str(spec), "action": action, "reachable": False}
            try:
                if action not in ("dab", "hover"):
                    raise GuardError("action must be 'dab' or 'hover'")
                if len(spec) != 1:
                    raise GuardError('a step needs exactly one of {"hole": id} or {"at": [fx, fy]}')
                r["target"] = self._key(spec)
                a = self._aim(st, spec)
                r.update(pixel=a["px"], robot_mm=[round(float(v)) for v in a["P0"] * 1000],
                         learned_correction_mm=[round(float(v), 1) for v in a["corr_mm"]])
                p = await asyncio.to_thread(self._plan, a["P"], q_from)
                legs = {} if p["direct"] else {"to_high_point": travel(q_from, p["q_high"])}
                legs["to_hover"] = travel(q_from if p["direct"] else p["q_high"], p["q_hover"])
                if action == "dab":
                    legs["touch_and_lift"] = travel(p["q_hover"], p["q_touch"])
                r.update(reachable=True, tool_tilt_deg=p["tilt_deg"],
                         path="direct" if p["direct"] else "via the high point", joint_travel_max_deg=legs)
                q_from = p["q_hover"]                      # the next step starts from this step's hover pose
            except GuardError as e:
                r["why_not"] = str(e)
            out.append(r)
        ok = all(r["reachable"] for r in out)
        return {"ok": ok, "steps": out, "vision": self._brief(st),
                "note": "planned from the current vision state and arm pose; nothing moved. run_program takes "
                        "its own clear look before it moves" + ("" if ok else
                        ". Fix or drop the unreachable steps before run_program")}

    async def inspect(self) -> tuple[dict, bytes]:
        """Measure every mark made on this part against its target and feed the correction model."""
        await self.ensure_guided()
        if not self._work:
            raise GuardError("nothing was marked on this part yet")
        st, last = await self._wait_live(float(self.gcfg["vision"]["live_wait_s"]), still=True)
        if st is None:
            raise GuardError(f"need a clear, still view of the part (retreat first): {self._brief(last)}")
        k, mk = self.kin, self.gcfg["marks"]
        frame = await self.vision.raw()
        if self._diff_marks():
            return await self._inspect_diff(st, frame)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array(mk["hsv_lo"], np.uint8), np.array(mk["hsv_hi"], np.uint8))
        n, _, stats, cents = cv2.connectedComponentsWithStats(mask)
        blobs = [(cents[i], int(stats[i, cv2.CC_STAT_AREA])) for i in range(1, n)
                 if stats[i, cv2.CC_STAT_AREA] >= int(mk["min_area_px"])]
        size = st["image_size"]
        sx, sy = frame.shape[1] / size[0], frame.shape[0] / size[1]       # frame px per state px
        results, used = {}, set()
        for key, w in self._work.items():
            a = self._aim(st, w["spec"])
            px = np.array(a["px"])
            mm_px = float(np.linalg.norm(k.surface_point(px + [20, 0], size) - a["P0"]) / 20 * 1000)
            best = None
            for j, (c, area) in enumerate(blobs):
                d = float(np.linalg.norm(np.array([c[0] / sx, c[1] / sy]) - px)) * mm_px
                if j not in used and d <= float(mk["search_radius_mm"]) and (best is None or d < best[1]):
                    best = (j, d)
            if best is None:
                results[key] = {"found": False}
                continue
            used.add(best[0])
            c = blobs[best[0]][0]
            dm = k.surface_point((c[0] / sx, c[1] / sy), size) - a["P0"]
            miss = np.array([dm @ k.table.e1, dm @ k.table.e2]) * 1000
            dist = float(np.linalg.norm(miss))
            results[key] = {"found": True, "miss_mm": [round(float(v), 1) for v in miss], "miss_dist_mm": round(dist, 1),
                            "pass": dist <= float(mk["tolerance_mm"])}
            if not w.get("learned"):
                self.model.record((self.part or {}).get("id") or "-", key, a["pos_mm"], miss, w["corr_mm"])
                w["learned"] = True
            if self.part is not None:
                self.part["misses"][key] = round(dist, 2)
            cv2.line(frame, (int(px[0] * sx), int(px[1] * sy)), (int(c[0]), int(c[1])),
                     (0, 200, 0) if results[key]["pass"] else (0, 0, 230), 2)
            cv2.putText(frame, f"{key}: {dist:.1f}mm", (int(c[0]) + 8, int(c[1]) - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        stray = len(blobs) - len(used)
        out = {"part": (self.part or {}).get("id"), "tolerance_mm": mk["tolerance_mm"], "marks": results,
               "stray_marks": stray, "learning": self.model.status()}
        if results and not any(r["found"] for r in results.values()):
            out["note"] = ("no marks found at any target: without ink on the tool there is nothing to measure "
                           "(the closed fingertips leave no mark unless they carry ink). This is not a miss; the "
                           "correction model got no data.")
        return out, cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tobytes()

    async def _inspect_diff(self, st, frame) -> tuple[dict, bytes]:
        """[marks] method = "diff": new dark marks = the inspect picture compared with the clear look taken before
        the dabs (marks_dark.py), registered on the lid, so a black marker on black rings works. Same output as the
        colour path plus "method": "diff", the registration and the stray marks' pixels. Never invents a mark: with
        no reference or a failed registration every target is reported not found, with a note."""
        k, mk = self.kin, self.gcfg["marks"]
        if self._arm_over_part():
            raise GuardError("inspect with method 'diff' needs the worker arm out of the camera's view (retreat first)")
        pid = (self.part or {}).get("id")
        size = st["image_size"]
        sx, sy = frame.shape[1] / size[0], frame.shape[0] / size[1]       # frame px per state px
        ref = self._mark_ref if self._mark_ref is not None and self._mark_ref["part_id"] == pid else None
        results, targets, notes = {}, [], []
        for key, w in self._work.items():
            if w.get("learned") and "inspect" in w:
                results[key] = dict(w["inspect"], measured_earlier=True)     # its ink is in the reference now
                continue
            if ref is not None and w.get("t", 0.0) < ref["t"]:
                results[key] = {"found": False, "why": "dabbed before the reference picture was taken"}
                continue
            a = self._aim(st, w["spec"])
            px = np.array(a["px"])
            mm_px = float(np.linalg.norm(k.surface_point(px + [20, 0], size) - a["P0"]) / 20 * 1000)
            targets.append((key, w, a, px, mm_px))
        det = None
        if ref is None:
            notes.append("no reference picture of this part from before the dabs (method 'diff' needs a clear look "
                         "first: new_part / see_part / run with the arm clear); nothing was measured")
        elif targets:
            det = await asyncio.to_thread(marks_dark.detect, ref["frame"], frame, ref["state"], st, mk)
            if not det["ok"]:
                notes.append(f"could not register the lid between the reference and now: {det['why']}; nothing "
                             "was measured (no marks are guessed)")
        ok = det is not None and det["ok"]
        marks = det["marks"] if ok else []
        mpx = [np.array(m["px"], float) / [sx, sy] for m in marks]          # state px
        asg = marks_dark.assign([t[3] for t in targets], mpx, [t[4] for t in targets], float(mk["search_radius_mm"]))
        drawn = []
        for i, (key, w, a, px, mm_px) in enumerate(targets):
            j = asg.get(i)
            if j is None:
                results[key] = {"found": False}
                continue
            c = mpx[j]
            dm = k.surface_point(c, size) - a["P0"]
            miss = np.array([dm @ k.table.e1, dm @ k.table.e2]) * 1000
            dist = float(np.linalg.norm(miss))
            results[key] = {"found": True, "miss_mm": [round(float(v), 1) for v in miss], "miss_dist_mm": round(dist, 1),
                            "pass": dist <= float(mk["tolerance_mm"]), "mark_px": marks[j]["px"],
                            "mark_area_px": marks[j]["area_px"], "mark_contrast": marks[j]["contrast"]}
            if not w.get("learned"):
                self.model.record(pid or "-", key, a["pos_mm"], miss, w["corr_mm"], extra={"method": "diff"})
                w["learned"] = True
            w["inspect"] = dict(results[key])
            if self.part is not None:
                self.part["misses"][key] = round(dist, 2)
            drawn.append((key, px * [sx, sy], marks[j]["px"], dist))
        used = set(asg.values())
        stray = [m for j, m in enumerate(marks) if j not in used]
        out = {"part": pid, "method": "diff", "tolerance_mm": mk["tolerance_mm"], "marks": results,
               "stray_marks": len(stray), "stray_marks_px": [m["px"] for m in stray],
               "learning": self.model.status()}
        if det is not None:
            out["registration"] = det["registration"]
            if ok:
                out["detector"] = {"noise": det["noise"], "threshold": det["threshold"], "band_px": det["band_px"],
                                   "ring_shift_px": det["ring_shift_px"]}
                if det["large_changes"]:
                    out["large_dark_changes"] = det["large_changes"]
                    notes.append("large dark changes on the lid were ignored (not dab-sized): shadow, arm or a "
                                 "smeared stroke?")
        if ref is not None:
            out["reference_age_s"] = round(time.time() - ref["t"], 1)
        if ok and results and not any(r["found"] for r in results.values()):
            notes.append("no new dark mark near any target: no ink on the tool, the dab too faint (< "
                         f"{marks_dark.params(mk)['dark_min_contrast']} levels), or it landed on a ring stroke. "
                         "This is not a miss; the correction model got no data.")
        if notes:
            out["note"] = " | ".join(notes)
        img = frame.copy()
        if ok:
            lid = np.asarray(st["box"]["corners_px"], float) * [sx, sy] if st.get("box") else None
            marks_dark.draw(img, drawn, [m["px"] for m in stray], float(mk["tolerance_mm"]), lid)
        else:
            cv2.putText(img, "marks not measured: " + ("no reference picture" if ref is None else
                        "lid registration failed"), (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 230), 2)
        if ref is None or det is not None:
            if mk.get("dark_save_images", True) and det is not None:
                await asyncio.to_thread(self._save_mark_images, ref["frame"], frame, img, det)
            await self._keep_mark_ref(st, force=True)      # these marks are now part of the 'before' picture
        jpeg = await asyncio.to_thread(lambda: cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tobytes())
        return out, jpeg

    def _save_mark_images(self, before, after, result, det) -> None:
        """For the log/demo: state/marks/<time>_<part>_{before,after,diff,result}.jpg."""
        d = self.cfg.state_dir / "marks"
        d.mkdir(parents=True, exist_ok=True)
        stem = time.strftime("%Y%m%d_%H%M%S") + f"_{(self.part or {}).get('id') or '-'}"
        cv2.imwrite(str(d / f"{stem}_before.jpg"), before)
        cv2.imwrite(str(d / f"{stem}_after.jpg"), after)
        cv2.imwrite(str(d / f"{stem}_result.jpg"), result)
        if det.get("vis") is not None:
            cv2.imwrite(str(d / f"{stem}_diff.png"), det["vis"])

    async def guided_status(self) -> dict:
        await self.ensure_guided()
        k = self.kin
        q = await self._q_now()
        return {"vision": self.vision.describe(), "tool_tip": k.tool_source,
                "calibration": {"file": str(self.gcfg["_paths"]["calib_file"]), "markers": len(k.calib["points"]),
                                "made": k.calib.get("created"), "camera": k.calib.get("camera")},
                "part_top_above_calibrated_plane_mm": k.surface_h * 1000,
                "parallax_correction": k.cam_h > 0, "taught_aim_offset_mm": [round(float(v), 1) for v in k.aim * 1000],
                "aim_offset_mm": self.aim_parts(),
                "tip_now_robot_mm": [round(float(v)) for v in k.tip(q) * 1000],
                "tip_above_part_mm": round(self._above_part(q) * 1000),
                "droop_correction_deg": [round(float(v), 1) for v in np.degrees(self._offset)],
                "worked_on_this_part": list(self._work), "learning": self.model.status(),
                "marks_method": self.gcfg["marks"].get("method", "color"),
                **({"mark_reference": None if self._mark_ref is None else
                    {"age_s": round(time.time() - self._mark_ref["t"], 1), "part": self._mark_ref["part_id"],
                     "dabs_since": self._dabs_since_ref}} if self._diff_marks() else {}),
                "guards": self.guards()}

    # ================================================================= aim offset at lid height ("jog and teach")
    JOG_MAX_MM = 30.0          # total sideways jog from the start point
    JOG_MIN_ABOVE_MM = 3.0     # never below this above the part top
    JOG_IK_TOL_MM = 2.0
    JOG_MAX_JOINT_DEG = 15.0   # one small step must never need a big joint change
    JOG_WARN_ABOVE_MM = 20.0   # higher than this: parallax makes the visual judgement unreliable

    def aim_parts(self) -> dict:
        k = self.kin

        def mm(v):
            return [round(float(x) * 1000, 1) for x in v]
        return {"aim_file": mm(k.aim_from_file), "lid_jog": mm(k.aim_lid), "total": mm(k.aim),
                "lid_samples": len(k.aim_lid_samples()), "lid_file": str(k.aim_lid_path)}

    async def jog_begin(self, step_mm: float = 2.0) -> dict:
        """Start jogging at the current height around the last hover/touch target. No motion here."""
        await self.ensure_guided()
        la, k = self._last_aim, self.kin
        if la is None:
            raise GuardError("jog needs a successful hover or touch on a target in this session, with no retreat "
                             "since (hover <hole> or touch <hole> first)")
        if self.zone[self.arm] != "work":
            raise GuardError(f"{self.arm} is not in the work area (zone {self.zone[self.arm]!r})")
        self._check_estop()
        q = await self._q_now()
        lim = float(self.m.get("settle_max_deg", 4.0))
        if np.degrees(np.abs(q - la["q"]).max()) > lim + 3.0:
            raise GuardError("the arm is not where the last hover/touch left it (moved since?); hover again")
        shape = (la["image_size"][1], la["image_size"][0])
        px = np.asarray(la["px"], float)
        p0 = k.table.pixel_to_base(px, shape)

        def unit(v):
            v = k.in_plane(v)
            return v / (np.linalg.norm(v) + 1e-12)
        e_right = unit(k.table.pixel_to_base(px + [10, 0], shape) - p0)
        e_up = unit(k.table.pixel_to_base(px + [0, -10], shape) - p0)
        self._jog = dict(start=la["P_tip"].copy(), target=la["P_tip"].copy(), q=la["q"].copy(),
                         e_up=e_up, e_right=e_right, step_mm=float(step_mm), moves=0, key=la["key"])
        return self.jog_info()

    def jog_info(self) -> dict:
        j, k = self._jog, self.kin
        side = k.in_plane(j["target"] - j["start"])
        return {"target": j["key"], "step_mm": j["step_mm"], "moves": j["moves"],
                "offset_image_mm": {"up": round(float(side @ j["e_up"]) * 1000, 1),
                                    "right": round(float(side @ j["e_right"]) * 1000, 1)},
                "offset_robot_mm": [round(float(v) * 1000, 1) for v in side],
                "offset_sideways_mm": round(float(np.linalg.norm(side)) * 1000, 1),
                "target_above_part_mm": round((k.height(j["target"]) - k.surface_h) * 1000, 1)}

    async def jog_key(self, key: str) -> dict:
        """One jog key: w/s/a/d (image up/down/left/right), r/f (up/down along the table normal), [ ] step size.
        Moves go through _guided_move (all guards). Raises GuardError when a step is refused."""
        j, k = self._jog, self.kin
        if j is None:
            raise GuardError("not jogging")
        await self._station_fresh()
        if key in "[]":
            j["step_mm"] = max(0.5, j["step_mm"] / 2) if key == "[" else min(8.0, j["step_mm"] * 2)
            return {"ok": True, "step_mm": j["step_mm"]}
        step = j["step_mm"] / 1000
        if key in "wasd":
            d = {"w": j["e_up"], "s": -j["e_up"], "a": -j["e_right"], "d": j["e_right"]}[key]
            new = j["target"] + d * step
            side = float(np.linalg.norm(k.in_plane(new - j["start"]))) * 1000
            if side > self.JOG_MAX_MM + 1e-6:
                raise GuardError(f"refused: the sideways offset would be {side:.1f} mm (max {self.JOG_MAX_MM:.0f} mm)")
        elif key in "rf":
            h = k.height(j["target"]) - k.surface_h
            lo, hi = self.JOG_MIN_ABOVE_MM / 1000, float(self.m["high_mm"]) / 1000
            hn = min(h + step, hi) if key == "r" else max(h - step, lo)
            if abs(hn - h) < 1e-5:
                raise GuardError(f"refused: already at the {'upper' if key == 'r' else 'lower'} jog limit "
                                 f"({hn * 1000:.1f} mm above the part top)")
            new = j["target"] + k.n * (hn - h)
        else:
            raise GuardError(f"unknown jog key {key!r}")
        if not k.in_workspace(new, float(self.m["workspace_margin_mm"]) / 1000):
            raise GuardError("refused: outside the area covered by the camera calibration markers")
        q0 = j["q"]
        q1, err = await asyncio.to_thread(k.ik_translate, new, q0)          # IK off the event loop thread
        how = "parallel shift"
        if err > self.JOG_IK_TOL_MM or np.degrees(np.abs(q1 - q0).max()) > self.JOG_MAX_JOINT_DEG:
            q1, err, _ = await asyncio.to_thread(k.ik, new, q0, 10.0, 10.0)
            how = "ik, joint 7 within 10 deg"
        if err > self.JOG_IK_TOL_MM:
            raise GuardError(f"refused: IK error {err:.1f} mm > {self.JOG_IK_TOL_MM:.0f} mm")
        dq = float(np.degrees(np.abs(q1 - q0).max()))
        if dq > self.JOG_MAX_JOINT_DEG:
            raise GuardError(f"refused: this step needs a {dq:.0f} deg joint change (max {self.JOG_MAX_JOINT_DEG:.0f})")
        if k.path_lowest(q0, q1) < k.surface_h + self.JOG_MIN_ABOVE_MM / 1000 - 0.0005:
            raise GuardError(f"refused: the tip would pass lower than {self.JOG_MIN_ABOVE_MM:.0f} mm above the part")
        r = await self._guided_move(q1, "slow", settle=False)            # carries the learned droop offset
        if not r["ok"]:
            self._jog, self._last_aim = None, None
            return {"ok": False, "move": r}
        j.update(target=new, q=q1, moves=j["moves"] + 1)
        self._last_aim.update(P_tip=new.copy(), q=q1.copy())
        return {"ok": True, "how": how, "ik_error_mm": round(err, 2), **self.jog_info()}

    async def jog_save(self) -> dict:
        """Append the sideways jog (+ the lid offset already applied) as a new sample of state/aim_lid.json."""
        j, k = self._jog, self.kin
        side = k.in_plane(j["target"] - j["start"])
        h_mm = (k.height(j["target"]) - k.surface_h) * 1000
        rec = k.add_aim_lid_sample(k.in_plane(k.aim_lid) + side, j["target"], h_mm)
        self.log_event("aim_lid", f"lid aim sample {[round(v * 1000, 1) for v in rec['offset_m']]} mm "
                       f"at {h_mm:.0f} mm above the part")
        info = self.jog_info()
        self._jog = None
        return {"ok": True, "jog_mm": info["offset_robot_mm"], "offset_image_mm": info["offset_image_mm"],
                "sample": rec, "aim_offset_mm": self.aim_parts()}

    def jog_end(self) -> None:
        self._jog = None

    def _forget_droop(self) -> None:
        """The arm left the part (retreat) or a new look was taken: do not carry the hover offset into another
        approach. The first hover after it starts from zero and its tip loop sets the offset again (1-2 rounds)."""
        self._offset = np.zeros(7)

    def droop_clear(self) -> dict:
        """Forget the learned droop offset (it is relearned at the next hover)."""
        old = [round(float(v), 2) for v in np.degrees(self._offset)]
        self._offset = np.zeros(7)
        self.log_event("droop", f"droop correction cleared (was {old} deg)")
        return {"ok": True, "was_deg": old, "droop_correction_deg": [0.0] * 7}

    def guards(self) -> dict:
        """The limits that are active right now (cellbot's and this module's)."""
        m, g = self.m, self.g
        return {"cellbot": {k: g.get(k) for k in ("limit_margin", "max_speed", "slow_speed", "max_accel",
                                                   "interp_step", "current_limit")},
                "cellguide": {"workspace_margin_mm": m["workspace_margin_mm"], "clearance_mm": m["clearance_mm"],
                              "ik_tol_mm": m["ik_tol_mm"], "reaim_mm": m["reaim_mm"],
                              "max_reaim_mm": m.get("max_reaim_mm", 80.0), "max_reaims": m.get("max_reaims", 6),
                              "commit_age_s": m["commit_age_s"], "aim_frames": m.get("aim_frames", 3),
                              "settle_max_deg": m.get("settle_max_deg", 4.0),
                              "sees_under_arm": self._sees_under_arm(),
                              "sight_max_age_s": m.get("sight_max_age_s", 60.0),
                              "touch_sag_abort_mm": m.get("touch_sag_abort_mm", 6.0),
                              "touch_tol_mm": m.get("touch_tol_mm", 2.0),
                              "touch_correct_rounds": m.get("touch_correct_rounds", 4),
                              "touch_correct_max_deg": m.get("touch_correct_max_deg", 3.0),
                              "touch_cmd_below_mm": m.get("touch_cmd_below_mm", 3.0),
                              "pretouch_mm": m.get("pretouch_mm", 12.0),
                              "pretouch_xy_tol_mm": m.get("pretouch_xy_tol_mm", 1.0)}}
