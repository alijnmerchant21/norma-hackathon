"""Learning feedback loop for the marking arm.

Every measured mark is logged to state/marks.jsonl. The learner keeps, per spot,
a correction in millimetres (camera x/y on the work surface):

    correction[k+1] = correction[k] - gain * miss[k]          (iterative learning control)

A spot with no history starts from the mean correction of the other spots, so a
shared offset (e.g. the part fixture being off) is learned once for all spots.

The correction is turned into a joint offset for the marker arm through a 2xN
Jacobian (mm per normalized joint unit). The Jacobian starts from the guess in
cell.toml and is refined from what actually happened (Broyden rank-1 update).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from .config import CellConfig


class Learner:
    def __init__(self, cfg: CellConfig):
        self.cfg = cfg
        m = cfg.marking
        self.joints: list[str] = list(m["correction_joints"])
        self.gain = float(m["learning_gain"])
        self.max_corr = float(m["max_correction_mm"])
        self.tol = float(m["tolerance_mm"])
        self.state_path: Path = cfg.state_dir / "learning.json"
        self.log_path: Path = cfg.state_dir / "marks.jsonl"
        if self.state_path.exists():
            self.s = json.loads(self.state_path.read_text())
        else:
            self.s = {"corrections_mm": {}, "jacobian_mm": m["jacobian_mm"], "last": {}, "parts": []}

    # -------------------------------------------------------------- queries
    @property
    def J(self) -> np.ndarray:
        return np.array(self.s["jacobian_mm"], dtype=float)

    def correction_mm(self, spot: str) -> np.ndarray:
        c = self.s["corrections_mm"]
        if spot in c:
            return np.array(c[spot], dtype=float)
        if c:
            return np.mean([np.array(v) for v in c.values()], axis=0)
        return np.zeros(2)

    def joint_offsets(self, spot: str) -> tuple[dict[str, float], list[float]]:
        """Joint deltas (normalized) that realise this spot's mm correction."""
        corr = self.correction_mm(spot)
        dq = np.linalg.pinv(self.J) @ corr
        return {j: float(v) for j, v in zip(self.joints, dq)}, [round(float(x), 2) for x in corr]

    # --------------------------------------------------------------- update
    def record(self, part_id: str, spot: str, measured: dict, applied_corr_mm, applied_dq: dict) -> dict:
        miss = np.array(measured["miss_mm"], dtype=float)
        dq = np.array([applied_dq.get(j, 0.0) for j in self.joints])
        last = self.s["last"].get(spot)
        jac_updated = False
        if last is not None:
            d_q = dq - np.array(last["dq"])
            d_p = miss - np.array(last["miss"])
            J = self.J
            # Only learn from a change big enough to stand out from measurement noise,
            # and damp it: one noisy mark must not wreck the model.
            if float(np.linalg.norm(J @ d_q)) > 1.5:
                J_new = J + 0.3 * np.outer(d_p - J @ d_q, d_q) / float(d_q @ d_q)
                # Keep the update sane: well-conditioned and same sign on the diagonal.
                if np.linalg.cond(J_new) < 50 and np.all(np.sign(np.diag(J_new)) == np.sign(np.diag(J))):
                    self.s["jacobian_mm"] = J_new.round(2).tolist()
                    jac_updated = True
        # Update from the correction that was actually used for this mark.
        prev = np.array(applied_corr_mm, dtype=float)
        new = prev - self.gain * miss
        n = float(np.linalg.norm(new))
        if n > self.max_corr:
            new = new * (self.max_corr / n)
        self.s["corrections_mm"][spot] = new.round(3).tolist()
        self.s["last"][spot] = {"dq": dq.tolist(), "miss": miss.tolist()}
        entry = {
            "t": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "part": part_id,
            "spot": spot,
            "miss_mm": measured["miss_mm"],
            "miss_dist_mm": measured["miss_dist_mm"],
            "pass": measured["miss_dist_mm"] <= self.tol,
            "applied_correction_mm": list(applied_corr_mm),
            "applied_dq": {k: round(v, 4) for k, v in applied_dq.items()},
            "next_correction_mm": self.s["corrections_mm"][spot],
            "jacobian_updated": jac_updated,
        }
        with open(self.log_path, "a") as f:
            f.write(json.dumps(entry) + "\n")
        self._save()
        return entry

    def close_part(self, part_id: str, outcome: str, misses: list[float], notes: str = ""):
        self.s["parts"].append({
            "part": part_id,
            "outcome": outcome,
            "mean_miss_mm": round(float(np.mean(misses)), 2) if misses else None,
            "max_miss_mm": round(float(np.max(misses)), 2) if misses else None,
            "notes": notes[:500],
            "t": time.strftime("%Y-%m-%dT%H:%M:%S"),
        })
        self._save()

    def status(self) -> dict:
        return {
            "corrections_mm": self.s["corrections_mm"],
            "jacobian_mm": self.s["jacobian_mm"],
            "tolerance_mm": self.tol,
            "recent_parts": self.s["parts"][-10:],
        }

    def reset(self):
        self.s = {"corrections_mm": {}, "jacobian_mm": self.cfg.marking["jacobian_mm"], "last": {}, "parts": []}
        self._save()

    def _save(self):
        self.state_path.write_text(json.dumps(self.s, indent=2))
