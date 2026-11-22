"""Loader step for the SO-101: move the part pickup -> work area (load) and work area -> drop-off (unload).

Mixin used by GuidedCell. Three modes, set in guided.toml [loader]:
  * taught: cellbot's taught-pose sequence (cellbot/operator_prompt.py), run in code through Cell.go_to_pose and
            Cell.gripper, so every cellbot guard applies;
  * policy: a learned policy (e.g. SmolVLA run_policy.py) run as a subprocess. It writes to Station directly,
            OUTSIDE cellbot's guards. Only turn-taking protects the worker: the loader's zone is "unknown" from
            before the start until a go_home succeeds, so the worker cannot enter the work area meanwhile;
  * human:  no motion; a human places or removes the part.
Hybrid (policy mode + policy_<phase>_pre_pose): first a taught so101 pose is reached with cellbot's guarded, slow
motion (e-stop, limits, zones), then the policy does the rest. Both happen under self.lock, so nothing else moves.
Per phase: load_mode / unload_mode override mode ("" = use mode). Taught sequences (taught mode): taught_load_seq /
taught_unload_seq = ["pose:<name>", "gripper:open|close", "wait:<s>", ...] replace the fixed pick/place sequence
(e.g. a sweep that pushes the box out of the work area); validated before any motion, run under self.lock with
cellbot's guarded slow moves, always ending at home. Empty = the fixed sequence.
Hybrid sequence (policy mode + policy_<phase>_pre_seq, same step syntax): run with the same guards under the same
self.lock hold BEFORE the policy starts, with NO home appended (e.g. a taught grasp; the policy carries on from
there). Exclusive with policy_<phase>_pre_pose.
Variants (policy mode): variant "primary" = policy_<phase>_cmd with its pre-pose / pre-seq; "fallback" =
policy_<phase>_fallback_cmd, run the same way (task, cwd, timeout, e-stop) but never with the primary's pre-pose /
pre-seq; instead policy_<phase>_fallback_pre_seq (same step syntax, run exactly like policy_<phase>_pre_seq) if set.
fallback_cmd empty + fallback_pre_seq set = the taught steps, then the primary policy_<phase>_cmd. Episode log: every attempt that started (any mode)
appends one JSON line to <state>/loader_episodes.jsonl with its time range (the link to Station's continuous
recording, for cutting training data); label_loader_attempt sets its outcome; loader_stats summarises.
Variant "teleop" (any mode): a HANDOVER to a human who moves the so101 with its leader arm through Station's own
teleoperation. No motion is commanded; the so101 zone becomes "unknown" (the worker may not enter the work area);
an OPEN attempt is logged (open: true, t_end null). While it is open every so101 motion through cellguide/cellbot
(go_to_pose, go_home, move_joints, gripper, any _move, load/unload) is refused. finish_teleop closes it (times,
open: false); then a human confirms teleoperation is OFF, go_home so101, camera check, label_loader_attempt.
A teleop attempt labelled success is a human demonstration: a training episode for that phase's policy.
"""
from __future__ import annotations

import asyncio
import collections
import datetime
import json
import os
import signal
import sys
import time
import uuid
from pathlib import Path

from cellbot.cell import GuardError

DEFAULTS = {
    "arm": "so101", "mode": "taught",
    "policy_load_cmd": "", "policy_unload_cmd": "", "policy_cwd": "",
    "policy_load_fallback_cmd": "", "policy_unload_fallback_cmd": "",   # variant "fallback" ("" + fallback_pre_seq = *_cmd)
    "policy_load_fallback_pre_seq": [], "policy_unload_fallback_pre_seq": [],   # fallback: taught steps first, no home
    "policy_load_task": "Move the red box from the pickup area to the work area",
    "policy_unload_task": "Move the red box from the work area to the drop-off area",
    "policy_timeout_s": 60,
    "policy_load_pre_pose": "", "policy_unload_pre_pose": "",   # hybrid: taught so101 pose reached before the policy
    "policy_load_pre_seq": [], "policy_unload_pre_seq": [],     # hybrid: taught steps before the policy, no home
    "load_mode": "", "unload_mode": "",                         # per phase; "" = mode
    "taught_load_seq": [], "taught_unload_seq": [],             # taught mode; [] = fixed pick/place sequence
}
MAX_WAIT_S = 5.0
LOAD_POSES = ["pick_hover", "pick", "place_hover", "place"]
UNLOAD_POSES = ["place_hover", "place", "drop_hover", "drop"]
GRASP_RETRIES = 2
VARIANTS = ("primary", "fallback", "teleop")
EPISODES_FILE = "loader_episodes.jsonl"
TELEOP_OPEN_MSG = "a teleop handover is open: finish it with finish_teleop first"


def _iso(t: float) -> str:
    return datetime.datetime.fromtimestamp(t).astimezone().isoformat(timespec="milliseconds")


class LoaderMixin:
    def _loader_cfg(self) -> dict:
        return {**DEFAULTS, **self.gcfg.get("loader", {})}

    def _phase_mode(self, phase: str, lc: dict | None = None) -> str:
        """[loader] <phase>_mode if set, else mode."""
        lc = lc or self._loader_cfg()
        return str(lc.get(f"{phase}_mode") or "").strip() or str(lc["mode"])

    def _phase_seq(self, phase: str, lc: dict | None = None, key: str | None = None) -> list:
        lc = lc or self._loader_cfg()
        key = key or f"taught_{phase}_seq"
        seq = lc.get(key) or []
        if not isinstance(seq, list):
            raise GuardError(f"[loader] {key} must be a list of strings, e.g. "
                             '["pose:sweep_start", "pose:sweep_end"]; got ' + repr(seq))
        return seq

    def _phase_pre_seq(self, phase: str, lc: dict | None = None) -> list:
        """policy_<phase>_pre_seq (policy mode). Refuses if policy_<phase>_pre_pose is set too."""
        lc = lc or self._loader_cfg()
        seq = self._phase_seq(phase, lc, key=f"policy_{phase}_pre_seq")
        if seq and str(lc.get(f"policy_{phase}_pre_pose") or "").strip():
            raise GuardError(f"policy {phase} refused: [loader] has both policy_{phase}_pre_pose and "
                             f"policy_{phase}_pre_seq set; use one of them (put the pose in the sequence as "
                             f"\"pose:<name>\", or set policy_{phase}_pre_seq = []). Nothing moved.")
        return seq

    def _fallback_pre_seq(self, phase: str, lc: dict | None = None) -> list:
        """policy_<phase>_fallback_pre_seq: taught steps run before the fallback command ([] = none)."""
        return self._phase_seq(phase, lc, key=f"policy_{phase}_fallback_pre_seq")

    def _fallback_cmd_key(self, phase: str, lc: dict | None = None) -> str | None:
        """Key of the fallback command: policy_<phase>_fallback_cmd if set, else policy_<phase>_cmd if a fallback
        pre-seq is set (taught steps, then the same policy), else None (no fallback configured)."""
        lc = lc or self._loader_cfg()
        if str(lc.get(f"policy_{phase}_fallback_cmd") or "").strip():
            return f"policy_{phase}_fallback_cmd"
        if self._fallback_pre_seq(phase, lc):
            return f"policy_{phase}_cmd"
        return None

    async def load_part(self, variant: str = "primary") -> dict:
        """Bring a part from the pickup area to the work area."""
        return await self._loader_run("load", variant)

    async def unload_part(self, variant: str = "primary") -> dict:
        """Take the part from the work area to the drop-off area."""
        return await self._loader_run("unload", variant)

    def _check_variant(self, phase: str, variant: str, mode: str, lc: dict) -> str:
        """Refuse (nothing moved) an unknown variant, a non-primary variant outside policy mode, or a fallback whose
        command is not configured."""
        v = str(variant or "primary").strip().lower()
        if v not in VARIANTS:
            raise GuardError(f"{phase}_part refused: variant {variant!r}; use \"primary\", \"fallback\" or "
                             f"\"teleop\". Nothing moved.")
        if v == "teleop":
            return v                                           # human teleoperation: allowed in any mode
        if v != "primary" and mode != "policy":
            raise GuardError(f"{phase}_part refused: variant {v!r} exists only in policy mode ({phase} mode is "
                             f"{mode!r}); call it without variant. Nothing moved.")
        if v == "fallback" and self._fallback_cmd_key(phase, lc) is None:
            raise GuardError(f"{phase}_part refused: variant \"fallback\" but [loader] policy_{phase}_fallback_cmd is "
                             f"empty and policy_{phase}_fallback_pre_seq is [] in guided.toml; no fallback is "
                             f"configured. Nothing moved. Ask the human.")
        return v

    async def _loader_run(self, phase: str, variant: str = "primary") -> dict:
        lc = self._loader_cfg()
        arm, mode = lc["arm"], self._phase_mode(phase, lc)
        variant = self._check_variant(phase, variant, mode, lc)
        self._check_no_teleop(f"{phase}_part")
        if arm not in self.cfg.arms:
            raise GuardError(f"[loader] arm = {arm!r} is not an arm of this cell ({list(self.cfg.arms)})")
        if arm == self.arm:
            raise GuardError("[loader] arm must be the other arm, not the worker")
        await self.ensure_connected()
        if self.zone[self.arm] != "clear":
            raise GuardError(f"{phase}_part refused: the worker {self.arm} is not at a clear pose "
                             f"(zone {self.zone[self.arm]}). Retreat or go_home {self.arm} first.")
        if mode not in ("taught", "policy", "human"):
            raise GuardError(f"[loader] mode = {mode!r}: use taught, policy or human")
        if variant == "teleop":
            return await self._teleop_open_handover(arm, phase, mode)
        t0 = time.time()
        part = (getattr(self, "part", None) or {}).get("id")
        out: dict | None = None
        try:
            if mode == "taught":
                seq = self._phase_seq(phase, lc)
                if seq:
                    out = await self._taught_seq(arm, phase, seq)
                else:
                    out = await (self._taught_load(arm) if phase == "load" else self._taught_unload(arm))
            elif mode == "policy":
                out = await self._policy_run(arm, phase, lc, variant)
            else:
                what = ("place the part in the work area" if phase == "load"
                        else "remove the part from the work area to the drop-off")
                out = {"ok": True, "note": f"No motion. Ask the human to {what}, hands away from both arms when done. "
                       + ("new_part and see_part will verify that the part is there." if phase == "load" else
                          "new_part / see_part will show whether the work area is empty.")}
        except GuardError:
            raise                                              # refused before anything moved: not an attempt
        except BaseException as e:                             # link error, Ctrl+C, ...: still an attempt, log it
            self._episode_log(phase, mode, variant, t0, {"ok": False, "error": f"{type(e).__name__}: {e}"}, part)
            raise
        out = {"mode": mode, "phase": phase, "variant": variant, **out, "seconds": round(time.time() - t0, 1)}
        ep = self._episode_log(phase, mode, variant, t0, out, part)
        if ep:
            out["attempt_id"] = ep
            out["label_hint"] = ("check with the camera (see_part from a clear pose), then label_loader_attempt: "
                                 + ("success = the part is in the work area and seen" if phase == "load"
                                    else "success = the work area is empty"))
        return out

    # ------------------------------------------------------------------ episode log (data flywheel)
    def _episodes_path(self) -> Path:
        return Path(getattr(self, "loader_episodes_path", None) or self.cfg.state_dir / EPISODES_FILE)

    def _episode_log(self, phase: str, mode: str, variant: str, t0: float, out: dict, part) -> str | None:
        """Append one attempt. Never raises: a logging failure must not break the motion path."""
        try:
            t1 = time.time()
            rec = {"id": uuid.uuid4().hex[:8], "phase": phase, "mode": mode, "variant": variant,
                   "cmd": out.get("command") if mode == "policy" else None,
                   "t_start": round(t0, 3), "t_end": round(t1, 3), "iso_start": _iso(t0), "iso_end": _iso(t1),
                   "seconds": round(t1 - t0, 1), "exit_code": out.get("exit_code"), "killed": out.get("killed"),
                   "ok": bool(out.get("ok")), "part": part, "outcome": None}
            for k in ("error", "failed_step", "policy_started", "sequence"):   # sequence = taught pre-seq steps
                if k in out:
                    rec[k] = out[k]
            p = self._episodes_path()
            p.parent.mkdir(parents=True, exist_ok=True)
            with open(p, "a") as f:
                f.write(json.dumps(rec) + "\n")
            return rec["id"]
        except Exception as e:  # noqa: BLE001
            try:
                self.log_event("error", f"loader episode log failed: {type(e).__name__}: {e}")
            except Exception:  # noqa: BLE001
                pass
            return None

    def _episodes_read(self) -> list:
        """[(raw line, record or None)] in file order; unreadable lines are kept verbatim."""
        p = self._episodes_path()
        if not p.exists():
            return []
        rows = []
        for line in p.read_text().splitlines():
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
                rows.append((line, rec if isinstance(rec, dict) and "id" in rec else None))
            except ValueError:
                rows.append((line, None))
        return rows

    def _episodes_rewrite(self, rows: list, i: int, rec: dict):
        """Replace line i by rec with an atomic rewrite (other lines kept verbatim)."""
        lines = [ln for ln, _ in rows]
        lines[i] = json.dumps(rec)
        p = self._episodes_path()
        tmp = p.with_name(p.name + f".tmp{os.getpid()}")
        with open(tmp, "w") as f:
            f.write("\n".join(lines) + "\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, p)
        self._teleop_cache = None

    def label_loader_attempt(self, success: bool, note: str = "", attempt_id: str = "") -> dict:
        """Set outcome "success"/"fail" of one attempt (default: the latest) by an atomic rewrite of the log."""
        rows = self._episodes_read()
        idx = [i for i, (_, r) in enumerate(rows) if r is not None
               and (not attempt_id or r["id"] == str(attempt_id).strip())]
        if not idx:
            raise GuardError("label refused: " + (f"no loader attempt with id {attempt_id!r}" if attempt_id else
                             "no loader attempt logged yet") + f" in {self._episodes_path()}")
        i = idx[-1]
        rec = dict(rows[i][1])
        if rec.get("open"):
            raise GuardError(f"label refused: attempt {rec['id']} is an open teleop handover; "
                             + TELEOP_OPEN_MSG + ", then look with the camera and label it")
        prev = rec.get("outcome")
        t = time.time()
        rec.update(outcome="success" if success else "fail", outcome_note=str(note or ""),
                   outcome_t=round(t, 3), outcome_iso=_iso(t))
        if prev is not None:
            rec["previous_outcome"] = prev
        self._episodes_rewrite(rows, i, rec)
        self.log_event("loader", f"attempt {rec['id']} labelled {rec['outcome']}", note=rec["outcome_note"])
        return {"ok": True, "attempt": rec}

    def loader_stats(self, last: int = 10) -> dict:
        """Counts per phase/variant, the last attempts, and the time ranges of successful attempts
        (dataset_ranges) to cut training episodes from Station's recording."""
        recs = [r for _, r in self._episodes_read() if r is not None]
        groups: dict[str, dict] = {}
        ranges: dict[str, list] = {}
        for r in recs:
            key = f"{r.get('phase')}/{r.get('variant', 'primary')}"
            g = groups.setdefault(key, {"attempts": 0, "successes": 0, "fails": 0, "unlabelled": 0, "open": 0,
                                        "_s": []})
            g["attempts"] += 1
            if r.get("open"):                                  # teleop handover not finished: never a success
                g["open"] += 1
                continue
            oc = r.get("outcome")
            g["successes" if oc == "success" else "fails" if oc == "fail" else "unlabelled"] += 1
            if isinstance(r.get("seconds"), (int, float)):
                g["_s"].append(float(r["seconds"]))
            if oc == "success" and r.get("iso_end"):
                ranges.setdefault(key, []).append({
                    "start": r.get("iso_start"), "end": r.get("iso_end"), "attempt_id": r.get("id"),
                    "source": "teleop demonstration" if r.get("variant") == "teleop" else
                              "human" if r.get("mode") == "human" else
                              "taught" if r.get("mode") == "taught" else "policy"})
        for g in groups.values():
            s = g.pop("_s")
            n = g["successes"] + g["fails"]
            g["success_rate"] = round(g["successes"] / n, 3) if n else None
            g["mean_seconds"] = round(sum(s) / len(s), 1) if s else None
        keys = ("id", "phase", "variant", "mode", "iso_start", "iso_end", "seconds", "exit_code", "killed", "ok",
                "outcome", "outcome_note", "part", "open")
        open_ids = [r.get("id") for r in recs if r.get("open")]
        return {"file": str(self._episodes_path()), "total": len(recs), "by_phase_variant": groups,
                "open_teleop_handovers": open_ids,
                "last": [{k: r.get(k) for k in keys} for r in recs[-max(0, int(last)):]] if last else [],
                "dataset_ranges": ranges,
                "dataset_note": "time ranges (local ISO) of attempts labelled success, each with its source "
                                "(\"policy\", or \"teleop demonstration\" = a human moved the so101 with the leader "
                                "arm); cut them from NormaCore Station's recording (NormFS) as training episodes"}

    # ------------------------------------------------------------------ teleop handover (human demonstration)
    def _teleop_open_record(self) -> dict | None:
        """The open teleop attempt, or None. Cached on the log's mtime/size (cheap enough for every move; the
        CLI and the MCP server are separate processes, so the file is the truth)."""
        p = self._episodes_path()
        try:
            st = p.stat()
            sig = (str(p), st.st_mtime_ns, st.st_size)
        except OSError:
            return None
        c = getattr(self, "_teleop_cache", None)
        if c and c[0] == sig:
            return c[1]
        try:
            rec = next((r for _, r in reversed(self._episodes_read()) if r is not None and r.get("open")), None)
        except Exception:  # noqa: BLE001  unreadable log: do not invent a handover
            rec = None
        self._teleop_cache = (sig, rec)
        return rec

    def _check_no_teleop(self, what: str = "motion"):
        rec = self._teleop_open_record()
        if rec:
            raise GuardError(f"{what} refused: {TELEOP_OPEN_MSG} (attempt {rec['id']}, {rec.get('phase')} "
                             f"started {rec.get('iso_start')}). Nothing moved.")

    def _is_loader_arm(self, arm) -> bool:
        try:
            return arm == self._loader_cfg()["arm"]
        except Exception:  # noqa: BLE001
            return arm == "so101"

    # cellbot's generic motions: LoaderMixin is first in GuidedCell's MRO, so these wrap Cell's methods. Only the
    # loader arm is blocked; the worker keeps its own guards (the so101 zone "unknown" keeps it out of the work area).
    async def go_to_pose(self, arm, pose, *a, **k):
        if self._is_loader_arm(arm):
            self._check_no_teleop(f"go_to_pose {arm} {pose}")
        return await super().go_to_pose(arm, pose, *a, **k)

    async def move_joints(self, arm, *a, **k):
        if self._is_loader_arm(arm):
            self._check_no_teleop(f"move_joints {arm}")
        return await super().move_joints(arm, *a, **k)

    async def gripper(self, arm, *a, **k):
        if self._is_loader_arm(arm):
            self._check_no_teleop(f"gripper {arm}")
        return await super().gripper(arm, *a, **k)

    async def _move(self, arm, target, speed=None):
        if self._is_loader_arm(arm):                           # backstop for every other path
            self._check_no_teleop(f"{arm} move")
        return await super()._move(arm, target, speed)

    def _teleop_instructions(self, phase: str, arm: str) -> str:
        frm, to = (("pickup area", "work area") if phase == "load" else ("work area", "drop-off area"))
        return (f"Teleoperated {phase}, done by you with the leader arm. The cell will not move the {arm} until you "
                f"are done. 1) In NormaCore Station, switch teleoperation ON for the {arm}. 2) With the leader arm, "
                f"move the part from the {frm} to the {to}. 3) Bring the {arm} back near its home pose. 4) Let go "
                f"of the leader arm. 5) Switch teleoperation OFF in Station. 6) Tell the operator \"done\".")

    async def _teleop_open_handover(self, arm: str, phase: str, mode: str) -> dict:
        """No motion: mark the loader arm unknown and open a teleop attempt. Caller checked arm/worker/no-open."""
        self._check_estop()
        if self.lock.locked():
            raise GuardError(f"{phase}_part teleop refused: another cell motion is running; wait for it. "
                             "Nothing moved.")
        async with self.lock:                                  # momentary: nothing else may move meanwhile
            self._check_estop()
            if self.zone[self.arm] != "clear":
                raise GuardError(f"{phase}_part refused: the worker {self.arm} is not at a clear pose "
                                 f"(zone {self.zone[self.arm]}). Retreat or go_home {self.arm} first.")
            self.zone[arm] = "unknown"                         # a human is about to move it
            self.last_pose[arm] = None
            t0 = time.time()
            rec = {"id": uuid.uuid4().hex[:8], "phase": phase, "mode": mode, "variant": "teleop", "cmd": None,
                   "t_start": round(t0, 3), "t_end": None, "iso_start": _iso(t0), "iso_end": None,
                   "seconds": None, "exit_code": None, "killed": None, "ok": None,
                   "part": (getattr(self, "part", None) or {}).get("id"), "outcome": None, "open": True}
            p = self._episodes_path()
            p.parent.mkdir(parents=True, exist_ok=True)
            with open(p, "a") as f:                            # must succeed: the open line IS the motion lock
                f.write(json.dumps(rec) + "\n")
            self._teleop_cache = None
        self.log_event("loader", f"teleop {phase} handover opened", attempt=rec["id"])
        return {"ok": True, "variant": "teleop", "phase": phase, "mode": mode, "attempt_id": rec["id"],
                "handover": True, "started": rec["iso_start"], "zone": self.zone[arm],
                "instructions_for_human": self._teleop_instructions(phase, arm),
                "next": ("relay the instructions to the human and WAIT until they say they are done and "
                         "teleoperation is OFF; then finish_teleop. No " + arm + " motion until then.")}

    def finish_teleop(self, attempt_id: str = "") -> dict:
        """Close the open teleop attempt (no motion, no label)."""
        rows = self._episodes_read()
        aid = str(attempt_id or "").strip()
        idx = [i for i, (_, r) in enumerate(rows) if r is not None and r.get("open")
               and (not aid or r["id"] == aid)]
        if not idx:
            raise GuardError("finish_teleop refused: " + (f"no open teleop handover with id {aid!r}" if aid else
                             "no teleop handover is open") + f" in {self._episodes_path()}")
        i = idx[-1]
        rec = dict(rows[i][1])
        t1 = time.time()
        rec.update(t_end=round(t1, 3), iso_end=_iso(t1), seconds=round(t1 - float(rec["t_start"]), 1),
                   open=False, ok=True)
        self._episodes_rewrite(rows, i, rec)
        arm = self._loader_cfg().get("arm", "so101")
        self.log_event("loader", f"teleop {rec['phase']} handover finished", attempt=rec["id"],
                       seconds=rec["seconds"])
        return {"ok": True, "attempt_id": rec["id"], "phase": rec["phase"], "started": rec["iso_start"],
                "finished": rec["iso_end"], "seconds": rec["seconds"], "zone": self.zone.get(arm),
                "reminder": (f"Nothing moved. 1) Confirm with the human that teleoperation is OFF in Station. "
                             f"2) go_home {arm}. 3) Look with the camera (see_part from a clear pose). "
                             f"4) label_loader_attempt (attempt_id {rec['id']}): "
                             + ("success = the part is in the work area and seen" if rec["phase"] == "load"
                                else "success = the work area is empty")
                             + ". Labelled success, it is a teleop demonstration: a new training episode for the "
                             f"{rec['phase']} policy.")}

    # ------------------------------------------------------------------ taught
    def _need_poses(self, arm: str, poses: list[str]):
        self.poses.reload()
        have = set(self.poses.names(arm))
        need = poses + [self.cfg.arms[arm].home_pose]
        missing = [p for p in need if p not in have]
        if missing:
            raise GuardError(f"loader refused: taught poses missing for {arm}: {missing}. A human must teach "
                             f"them first (python -m cellbot.teach save {arm} <pose> <zone>).")

    async def _taught_load(self, arm: str) -> dict:
        self._need_poses(arm, LOAD_POSES)
        steps: list[dict] = []
        mv, grip = self._steppers(arm, steps)
        for p in ("pick_hover", "pick"):
            if not await mv(p):
                return self._taught_fail(steps)
        for attempt in range(GRASP_RETRIES + 1):
            g = await grip("close")
            if g.get("holding_object"):
                break
            await grip("open")
            if not await mv("pick_hover"):
                return self._taught_fail(steps)
            if attempt == GRASP_RETRIES:
                return self._taught_fail(steps, f"nothing grasped after {GRASP_RETRIES + 1} tries; {arm} is open at "
                                         "pick_hover. Look at the pickup area (is a part there, is it placed right?)")
            if not await mv("pick"):
                return self._taught_fail(steps)
        for p in ("pick_hover", "place_hover", "place"):
            if not await mv(p):
                return self._taught_fail(steps)
        await grip("open")
        for p in ("place_hover", "home"):
            if not await mv(p):
                return self._taught_fail(steps)
        return {"ok": True, "steps": steps}

    async def _taught_unload(self, arm: str) -> dict:
        self._need_poses(arm, UNLOAD_POSES)
        steps: list[dict] = []
        mv, grip = self._steppers(arm, steps)
        for p in ("place_hover", "place"):
            if not await mv(p):
                return self._taught_fail(steps)
        g = await grip("close")
        if not g.get("holding_object"):
            await grip("open")
            for p in ("place_hover", "home"):
                if not await mv(p):
                    return self._taught_fail(steps)
            return self._taught_fail(steps, f"nothing grasped at place; {arm} went home. Look at the work area.")
        for p in ("place_hover", "drop_hover", "drop"):
            if not await mv(p):
                return self._taught_fail(steps)
        await grip("open")
        for p in ("drop_hover", "home"):
            if not await mv(p):
                return self._taught_fail(steps)
        return {"ok": True, "steps": steps}

    def _steppers(self, arm: str, steps: list):
        async def mv(pose: str) -> bool:
            if pose == "home":
                r = await self.go_home(arm)
            else:
                r = await self.go_to_pose(arm, pose, speed="slow" if pose in ("pick", "place", "drop") else "normal")
            steps.append({"step": pose, "ok": bool(r["ok"]), **({} if r["ok"] else {"error": r.get("error")})})
            return bool(r["ok"])

        async def grip(action: str) -> dict:
            r = await self.gripper(arm, action)
            e = {"step": f"gripper {action}", "ok": True}
            if "holding_object" in r:
                e["holding"] = r["holding_object"]
            steps.append(e)
            return r
        return mv, grip

    def _taught_fail(self, steps: list, reason: str | None = None) -> dict:
        last = steps[-1] if steps else {"step": "-"}
        reason = reason or f"move to {last['step']} failed: {last.get('error')}"
        self.log_event("guided_fail", f"loader: {reason}")
        return {"ok": False, "failed_step": last["step"], "reason": reason, "steps": steps}

    # ------------------------------------------------------------------ policy
    async def _policy_run(self, arm: str, phase: str, lc: dict, variant: str = "primary") -> dict:
        fallback = variant == "fallback"
        key = (self._fallback_cmd_key(phase, lc) or f"policy_{phase}_fallback_cmd") if fallback else f"policy_{phase}_cmd"
        cmd = str(lc.get(key) or "").strip()
        if not cmd:
            raise GuardError(f"policy mode: [loader] {key} is empty; set it in guided.toml")
        cwd = None
        if lc["policy_cwd"]:
            cwd = Path(os.path.expanduser(str(lc["policy_cwd"])))
            if not cwd.is_absolute():
                cwd = (Path(self.gcfg["_file"]).parent / cwd).resolve()
            if not cwd.is_dir():
                raise GuardError(f"[loader] policy_cwd {cwd} does not exist")
        cmd = cmd.replace("{task}", str(lc[f"policy_{phase}_task"]))
        timeout = float(lc["policy_timeout_s"])
        tail: collections.deque[str] = collections.deque(maxlen=20)
        killed = None
        # fallback: never the primary's pre-pose / pre-seq; its own policy_<phase>_fallback_pre_seq if set
        pre_seq = self._fallback_pre_seq(phase, lc) if fallback else self._phase_pre_seq(phase, lc)
        seq_key = f"policy_{phase}_fallback_pre_seq" if fallback else f"policy_{phase}_pre_seq"
        plan = self._parse_seq(arm, phase, pre_seq, key=seq_key, auto_home=False) if pre_seq else []
        targets = {v: dict(self.poses.get(arm, v)["joints"]) for k, v, _ in plan if k == "pose"}
        pre_name = "" if fallback else str(lc.get(f"policy_{phase}_pre_pose") or "").strip()
        pre_target = self._pre_pose_target(arm, phase, pre_name) if pre_name else None   # refuses before motion
        pre = None
        seq_steps = None
        async with self.lock:                                   # no other cellbot motion while it runs
            self._check_estop()
            if plan:
                names = [t for _, _, t in plan]
                self.log_event("loader", f"policy {phase} pre-sequence started", arm=arm, sequence=names,
                               variant=variant)
                seq_steps = await self._run_plan_locked(arm, plan, targets, label=f"{phase} pre-seq")
                if not (len(seq_steps) == len(plan) and all(s["ok"] for s in seq_steps)):
                    last = seq_steps[-1]
                    reason = (f"pre-sequence step {len(seq_steps)}/{len(plan)} {last['step']} failed: "
                              f"{last.get('error')}. The rest of the sequence and the policy were NOT started. "
                              f"{arm} zone is {self.zone[arm]!r}"
                              + ("" if self.zone[arm] == "clear" else
                                 f"; a human should check the arm and the part, then go_home {arm} before the "
                                 "worker may move."))
                    self.log_event("guided_fail", f"loader policy {phase}: {reason}", error=last.get("error"))
                    return {"ok": False, "pre_seq": seq_steps, "sequence": names, "failed_step": last["step"],
                            "zone": self.zone[arm], "policy_started": False, "note": reason}
                self.log_event("loader", f"policy {phase} pre-sequence done", arm=arm)
                self._check_estop()                             # latched meanwhile: refused, policy not started
            if pre_name:
                # go_to_pose takes self.lock (asyncio.Lock is not re-entrant): the same guarded move, done here.
                pre = await self._pre_pose_move(arm, pre_name, pre_target)
                if not pre["ok"]:
                    self.log_event("guided_fail", f"loader policy {phase}: pre-pose {pre_name} failed; "
                                   "policy not started", error=pre.get("error"))
                    return {"ok": False, "pre_pose": pre, "zone": self.zone[arm], "policy_started": False,
                            "note": f"move to the taught pre-pose {pre_name!r} failed: {pre.get('error')}. The "
                                    f"policy was NOT started. {arm} zone is {self.zone[arm]!r}"
                                    + ("" if self.zone[arm] == "clear" else
                                       f"; a human should check it, go_home {arm} before the worker may move.")}
                self._check_estop()
            t0 = time.monotonic()
            self.zone[arm] = "unknown"                          # BEFORE start: the worker may not enter
            self.last_pose[arm] = None
            self.log_event("loader", f"policy {phase} started", cmd=cmd, cwd=str(cwd or ""), variant=variant)
            proc = await asyncio.create_subprocess_shell(
                cmd, cwd=cwd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
                start_new_session=True)                         # own process group: kill reaches the python child

            async def pump():
                async for line in proc.stdout:
                    tail.append(line.decode(errors="replace").rstrip())
            reader = asyncio.create_task(pump())
            while proc.returncode is None:
                try:
                    await asyncio.wait_for(asyncio.shield(proc.wait()), 0.2)
                except asyncio.TimeoutError:
                    pass
                if proc.returncode is not None:
                    break
                if self.estop_state().get("latched"):
                    killed = "e-stop latched while the policy ran"
                elif time.monotonic() - t0 > timeout:
                    killed = f"timeout after {timeout:g} s"
                if killed:
                    try:
                        os.killpg(proc.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    await proc.wait()
                    await self._freeze_all()                   # hold the loader where it stopped
            try:
                await asyncio.wait_for(reader, 2.0)
            except asyncio.TimeoutError:
                reader.cancel()
        code = proc.returncode
        out = {"ok": False, "exit_code": code, "output_tail": list(tail), "zone": self.zone[arm], "command": cmd}
        if pre is not None:
            out["pre_pose"] = pre
        if seq_steps is not None:
            out["pre_seq"] = seq_steps
            out["sequence"] = [t for _, _, t in plan]
        self.log_event("loader", f"policy {phase} ended", exit_code=code, killed=killed)
        if killed:
            out["killed"] = killed
        if code == 0 and not killed:
            home = await self.go_home(arm)
            out["go_home"] = home
            out["ok"] = bool(home["ok"])
            out["zone"] = self.zone[arm]
            if not home["ok"]:
                out["note"] = (f"policy finished, but go_home {arm} failed; {arm} is at an unknown position. "
                               f"A human should check it; go_home {arm} is required before the worker may move.")
        else:
            out["note"] = (f"policy did not finish cleanly; {arm} is at an unknown position (zone 'unknown'). "
                           f"A human should look at the arm and the part; go_home {arm} is required before the "
                           f"worker may move.")
            self.log_event("guided_fail", f"loader policy {phase}: " + (killed or f"exit code {code}"))
        return out

    # ------------------------------------------------------------------ hybrid pre-pose
    def _teach_hint(self, arm: str, pose: str, zone: str) -> str:
        """The exact commands a human runs to teach `pose` with the same config and state as this runner."""
        from cellbot import config as cconf
        cfg_file = Path(os.environ.get("CELLBOT_CONFIG", cconf.ROOT / "cell.toml")).resolve()
        py = sys.executable or "python3"
        return (f"  cd {cconf.ROOT}\n"
                f"  export CELLBOT_MODE=station STATION_HOST={os.environ.get('STATION_HOST', '127.0.0.1')} "
                f"CELLBOT_CONFIG={cfg_file} CELLBOT_STATE={self.cfg.state_dir.resolve()}\n"
                f"  {py} -m cellbot.teach limp {arm}            # torque off: SUPPORT the arm, move it by hand\n"
                f"  {py} -m cellbot.teach save {arm} {pose} {zone}\n"
                f"  {py} -m cellbot.teach hold {arm}            # torque on again\n"
                f"(zone 'work' if the pose is over the shared work area, else 'clear'.) Then go_home {arm} "
                f"in the runner before any motion: its idea of where {arm} is is stale after limp.")

    def _pre_pose_target(self, arm: str, phase: str, pose: str) -> dict:
        """Refuse BEFORE any motion if the pre-pose (or home, needed after the policy) is not taught or is out of
        limits. Returns the joint target."""
        self.poses.reload()
        have = set(self.poses.names(arm))
        home = self.cfg.arms[arm].home_pose
        missing = [p for p in (pose, home) if p not in have]
        if missing:
            zone = "work" if phase == "unload" else "clear"
            raise GuardError(
                f"policy {phase} refused: [loader] policy_{phase}_pre_pose = {pose!r}, but taught poses missing "
                f"for {arm}: {missing} (poses file {self.poses.path}). Nothing moved, the policy was not started. "
                f"A human must teach it on the real arm (Station running):\n"
                + "\n".join(self._teach_hint(arm, p, "clear" if p == home else zone) for p in missing))
        target = dict(self.poses.get(arm, pose)["joints"])
        self._check_limits(arm, target)
        return target

    async def _pre_pose_move(self, arm: str, pose: str, target: dict, label: str = "pre-pose") -> dict:
        """cellbot's go_to_pose body at slow speed, for a caller that ALREADY holds self.lock. Never raises after
        the zone was claimed: the result says what happened and the zone is left truthful."""
        p = self.poses.get(arm, pose)
        self._check_zone(arm, p["zone"])                       # worker not clear -> GuardError, nothing moved
        t0 = time.monotonic()
        if p["zone"] == "work":
            self.zone[arm] = "work"                            # claim the zone before moving
        try:
            res = await self._move(arm, target, self.g["slow_speed"])
        except Exception as e:                                 # e-stop latched at the start, link error, ...
            res = {"ok": False, "error": f"{type(e).__name__}: {e}"}
        if res["ok"]:
            self.zone[arm] = p["zone"]
            self.last_pose[arm] = pose
        else:                                                  # may have stopped part-way: assume the worst
            self.zone[arm] = "work" if p["zone"] == "work" else "unknown"
            self.last_pose[arm] = None
        out = {"name": pose, "ok": bool(res["ok"]), "seconds": round(time.monotonic() - t0, 1),
               "zone": self.zone[arm]}
        if not res["ok"]:
            out["error"] = res.get("error")
        self.log_event("loader", f"{label} {pose} " + ("reached" if res["ok"] else "failed"), arm=arm,
                       seconds=out["seconds"], error=out.get("error"))
        return out

    # ------------------------------------------------------------------ taught sequence
    def _parse_seq(self, arm: str, phase: str, seq: list, key: str | None = None,
                   auto_home: bool = True) -> list[tuple]:
        """Validate the whole sequence BEFORE any motion. Returns [(kind, value, text)], ending at home unless
        auto_home is False (hybrid pre-sequence: the policy carries on; home must still be taught, it is used after
        the policy). Raises GuardError (nothing moved) on a bad step, a pose not taught, or a pose out of limits."""
        key = key or f"taught_{phase}_seq"
        home = self.cfg.arms[arm].home_pose
        self.poses.reload()
        have = set(self.poses.names(arm))
        steps, bad, missing = [], [], []
        for i, raw in enumerate(seq, 1):
            text = str(raw).strip() if isinstance(raw, str) else ""
            kind, _, val = text.partition(":")
            kind, val = kind.strip(), val.strip()
            if kind == "pose" and val:
                if val not in have:
                    missing.append(val)
                steps.append(("pose", val, f"pose:{val}"))
            elif kind == "gripper" and val in ("open", "close"):
                steps.append(("gripper", val, f"gripper:{val}"))
            elif kind == "wait":
                try:
                    w = float(val)
                except ValueError:
                    w = -1.0
                if not 0 <= w <= MAX_WAIT_S:
                    bad.append(f"step {i} {raw!r}: wait needs seconds 0..{MAX_WAIT_S:g}")
                steps.append(("wait", w, f"wait:{val}"))
            else:
                bad.append(f"step {i} {raw!r}: use \"pose:<name>\", \"gripper:open\", \"gripper:close\" "
                           f"or \"wait:<seconds>\"")
        if bad:
            raise GuardError(f"{phase} refused: [loader] {key} has bad steps: " + "; ".join(bad)
                             + ". Nothing moved.")
        if auto_home and (not steps or steps[-1][:2] != ("pose", home)):
            steps.append(("pose", home, f"pose:{home} (auto)"))
        if home not in have:
            missing.append(home)
        if missing:
            missing = list(dict.fromkeys(missing))
            raise GuardError(
                f"{phase} refused: [loader] {key} needs taught poses missing for {arm}: {missing} (poses file "
                f"{self.poses.path}). Nothing moved. A human must teach them on the real arm (Station running):\n"
                + "\n".join(self._teach_hint(arm, p, "clear" if p == home else "work") for p in missing))
        for kind, val, _ in steps:
            if kind == "pose":
                self._check_limits(arm, dict(self.poses.get(arm, val)["joints"]))
        return steps

    async def _gripper_locked(self, arm: str, action: str) -> dict:
        """cellbot's gripper body for a caller that ALREADY holds self.lock."""
        acfg = self.cfg.arms[arm]
        position = acfg.gripper_open if action == "open" else acfg.gripper_closed
        self._check_limits(arm, {acfg.gripper: position})
        res = await self._move(arm, {acfg.gripper: position}, self.g["slow_speed"])
        if not res["ok"]:
            return {"ok": False, "error": res.get("error")}
        last, still, t_end, st = None, 0, time.monotonic() + 2.5, None
        while time.monotonic() < t_end and still < 3:
            st = (await self.backend.read_state())[arm][acfg.gripper]
            still = still + 1 if last is not None and abs(st["pos"] - last) < 0.005 else 0
            last = st["pos"]
            await asyncio.sleep(0.1)
        out = {"ok": True, "gripper_pos": round(st["pos"], 3) if st else None}
        if action == "close" and st:
            out["holding"] = st["pos"] - acfg.gripper_closed > 0.08
        return out

    async def _run_plan_locked(self, arm: str, plan: list, targets: dict, label: str) -> list[dict]:
        """Run parsed steps for a caller that ALREADY holds self.lock; e-stop checked before each step, a failed
        step stops the rest. Returns the step results."""
        steps: list[dict] = []
        for kind, val, text in plan:
            t0 = time.monotonic()
            try:
                self._check_estop()
                if kind == "pose":
                    r = await self._pre_pose_move(arm, val, targets[val], label=label)
                    e = {"ok": r["ok"], **({"error": r["error"]} if not r["ok"] else {})}
                elif kind == "gripper":
                    e = await self._gripper_locked(arm, val)
                else:
                    end = time.monotonic() + val
                    while time.monotonic() < end:
                        self._check_estop()
                        await asyncio.sleep(min(0.1, max(0.0, end - time.monotonic())))
                    e = {"ok": True}
            except Exception as ex:                            # e-stop / zone refusal before this step moved
                e = {"ok": False, "error": f"{type(ex).__name__}: {ex}"}   # (_pre_pose_move keeps zone truthful)
            steps.append({"step": text, **e, "seconds": round(time.monotonic() - t0, 1)})
            if not e["ok"]:
                break
        return steps

    async def _taught_seq(self, arm: str, phase: str, seq: list) -> dict:
        """Run a validated taught sequence under self.lock; a failed step stops the rest."""
        plan = self._parse_seq(arm, phase, seq)               # refuses before any motion
        targets = {v: dict(self.poses.get(arm, v)["joints"]) for k, v, _ in plan if k == "pose"}
        names = [t for _, _, t in plan]
        async with self.lock:
            self._check_estop()                                # latched: refused, nothing moved
            self.log_event("loader", f"taught {phase} sequence started", arm=arm, sequence=names)
            steps = await self._run_plan_locked(arm, plan, targets, label=f"{phase} seq")
        ok = bool(steps) and all(s["ok"] for s in steps) and len(steps) == len(plan)
        out = {"ok": ok, "sequence": names, "steps": steps, "zone": self.zone[arm]}
        if ok:
            self.log_event("loader", f"taught {phase} sequence done", arm=arm)
            return out
        last = steps[-1]
        out["failed_step"] = last["step"]
        out["reason"] = (f"step {len(steps)}/{len(plan)} {last['step']} failed: {last.get('error')}. The rest of "
                         f"the sequence was not run. {arm} zone is {self.zone[arm]!r}"
                         + ("" if self.zone[arm] == "clear" else
                            f"; a human should check the arm and the part, then go_home {arm} before the "
                            "worker may move."))
        self.log_event("guided_fail", f"loader taught {phase} sequence: {out['reason']}", error=last.get("error"))
        return out
