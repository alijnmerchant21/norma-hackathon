"""Learned aim correction from the robot's own marks (the data flywheel).

Every inspected mark gives:  miss = where the ink landed - where the hole is   (mm, in the table plane).
The aim sent to the arm was  hole + c  (c = the learned correction applied for that mark, `applied_mm`), so the
arm's own error at that place -- the miss the UNCORRECTED aim would have had -- is  e = miss - c  (`error_mm`).
Because the model learns e (not the miss), applying -gain * e again does not wind up: repeated runs converge.

Frame: `pos_mm` and every vector here are in the TABLE plane of the calibration (robot/table-fixed: origin
table.o, axes table.e1/e2, from kin.py), not the part's lid frame. The correction is added in the same frame.

The model is LOCAL: the expected error at a query position p is a Gaussian-kernel weighted mean of the recent
usable samples (length scale `local_scale_mm`, default 8 mm; holes on the parts are 17-26 mm apart), so the
marks of the same hole dominate and a hole that is fine is not pushed off by a hole that is not. Where there is
little local evidence (kernel weight W below that of one sample one length scale away) it blends towards the
global mean shrunk by `global_shrink`, so a brand-new hole position still gets part of a shared offset:
    e(p) = a * local_mean(p) + (1 - a) * global_shrink * global_mean,     a = min(1, W(p) / exp(-1/2))
    correction(p) = -gain * e(p), capped at max_correction_mm.
Samples with |error| > outlier_mm are stored (and logged) but never used (e.g. a mark assigned to the wrong hole).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np


class MissModel:
    GROUP_MM = 5.0             # status: samples within this distance belong to one target position
    FULL_W = float(np.exp(-0.5))   # one sample within one length scale counts as full local evidence

    def __init__(self, state_dir: Path, gain=0.8, max_correction_mm=12.0, window=40,
                 local_scale_mm=8.0, outlier_mm=12.0, global_shrink=0.5):
        self.path = state_dir / "guided_learning.json"
        self.log = state_dir / "guided_marks.jsonl"
        self.gain, self.max_c, self.window = float(gain), float(max_correction_mm), int(window)
        self.scale, self.outlier, self.shrink = float(local_scale_mm), float(outlier_mm), float(global_shrink)
        self.samples = json.loads(self.path.read_text())["samples"] if self.path.exists() else []

    # ----------------------------------------------------------------- model
    def _usable(self) -> list[dict]:
        return [s for s in self.samples[-self.window:] if float(np.hypot(*s["error_mm"])) <= self.outlier]

    def expected_error_mm(self, pos_mm) -> tuple[np.ndarray, float]:
        """(expected arm error at pos_mm, local weight W). Zero error if there is no usable sample."""
        s = self._usable()
        if not s:
            return np.zeros(2), 0.0
        P = np.array([x["pos_mm"] for x in s], float)
        E = np.array([x["error_mm"] for x in s], float)
        d2 = ((P - np.asarray(pos_mm, float)[:2]) ** 2).sum(1)
        w = np.exp(-d2 / (2.0 * max(self.scale, 1e-3) ** 2))
        W = float(w.sum())
        g = self.shrink * E.mean(0)
        if W < 1e-9:
            return g, 0.0
        a = min(1.0, W / self.FULL_W)
        return a * (w @ E) / W + (1.0 - a) * g, W

    def correction_mm(self, pos_mm) -> np.ndarray:
        """Shift (mm, table-plane coordinates) to add to a target at pos_mm."""
        e, _ = self.expected_error_mm(pos_mm)
        c = -self.gain * e
        n = float(np.linalg.norm(c))
        return c * (self.max_c / n) if n > self.max_c else c

    # ---------------------------------------------------------------- update
    def record(self, part, target, pos_mm, miss_mm, applied_mm, extra=None) -> dict:
        err = (np.asarray(miss_mm, float) - np.asarray(applied_mm, float))
        s = dict(t=time.time(), part=part, target=str(target), pos_mm=[round(float(v), 1) for v in pos_mm],
                 miss_mm=[round(float(v), 2) for v in miss_mm], applied_mm=[round(float(v), 2) for v in applied_mm],
                 error_mm=[round(float(v), 2) for v in err], **(extra or {}))
        if float(np.hypot(*err)) > self.outlier:
            s["outlier"] = True                # kept for the log, never used for the correction
        self.samples.append(s)
        self.path.write_text(json.dumps({"samples": self.samples[-500:]}, indent=1))
        with open(self.log, "a") as f:
            f.write(json.dumps(s) + "\n")
        return s

    # ---------------------------------------------------------------- report
    def targets(self) -> list[dict]:
        """Per known target position (recent samples within GROUP_MM of each other): samples, mean error and the
        correction that would be applied there now."""
        groups: list[dict] = []
        for s in self.samples[-self.window:]:
            p = np.asarray(s["pos_mm"], float)[:2]
            g = next((g for g in groups if np.linalg.norm(np.mean(g["P"], 0) - p) <= self.GROUP_MM), None)
            if g is None:
                g = {"P": [], "s": []}
                groups.append(g)
            g["P"].append(p)
            g["s"].append(s)
        out = []
        for g in groups:
            c = np.mean(g["P"], 0)
            used = [s for s in g["s"] if float(np.hypot(*s["error_mm"])) <= self.outlier]
            names = sorted({s["target"] for s in g["s"]})
            last = g["s"][-1]
            out.append({
                "target": ", ".join(names), "parts": sorted({str(s["part"]) for s in g["s"]}),
                "pos_mm": [round(float(v), 1) for v in c], "samples": len(g["s"]), "used": len(used),
                "mean_error_mm": None if not used else [round(float(v), 2) for v in
                                                        np.mean([s["error_mm"] for s in used], 0)],
                "last_miss_mm": round(float(np.hypot(*last["miss_mm"])), 2),
                "correction_now_mm": [round(float(v), 2) for v in self.correction_mm(c)],
            })
        return out

    def status(self) -> dict:
        used = self._usable()
        by_part: dict[str, list[float]] = {}
        for s in self.samples:
            by_part.setdefault(s["part"], []).append(float(np.hypot(*s["miss_mm"])))
        return {
            "marks_logged": len(self.samples),
            "model": None if not used else "local",
            "local_scale_mm": self.scale, "outlier_mm": self.outlier, "gain": self.gain,
            "marks_used": len(used),
            "mean_arm_error_mm": None if not used else [round(float(v), 2) for v in
                                                        np.mean([s["error_mm"] for s in used], 0)],
            "per_target": self.targets(),
            "mean_miss_mm_by_part": {p: round(float(np.mean(v)), 2) for p, v in list(by_part.items())[-10:]},
        }

    def reset(self):
        self.samples = []
        if self.path.exists():
            self.path.unlink()


class GuidedLearnerView:
    """cellbot's `learning_status` tool and `finish_part` report `cell.learner.status()`: cellbot's own per-spot
    learner, which the guided cell never feeds (it learns in MissModel), so its corrections_mm stayed {}. This
    wrapper keeps everything of that learner (close_part, the recent-parts list, ...) and fills the status from
    MissModel: corrections_mm per known target position, plus the full MissModel status under "guided"."""

    def __init__(self, base, model: MissModel):
        self._base, self._model = base, model

    def __getattr__(self, name):
        return getattr(self._base, name)

    def status(self) -> dict:
        try:
            out = dict(self._base.status())
        except Exception as e:  # noqa: BLE001  the guided part below must still be reported
            out = {"base_learner_error": repr(e)}
        g = self._model.status()
        out["corrections_mm"] = {f'{t["target"]} @ {t["pos_mm"]}': t["correction_now_mm"] for t in g["per_target"]}
        out["guided"] = g
        return out
