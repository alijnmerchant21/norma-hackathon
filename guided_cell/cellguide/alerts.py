"""Operator alerts: what happened since the last call, as a short feed Claude can act on.

Sources: cellbot's state/events.jsonl (new lines), vision events (own cursor, separate from scene()), and
rules over the logged marks and touches (pure functions below, unit-tested in tests/test_alerts.py).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

EVENT_LEVEL = {"estop": "critical", "error": "warn", "refused": "info", "guided_fail": "warn",
               "exception": "warn", "vision": "warn", "part": "info", "part_done": "info"}
VISION_START_LEVEL = {"POSE_STALE": "warn", "UNEXPECTED_HOLES": "warn", "NO_BOX": "info"}
RANK = {"info": 0, "warn": 1, "critical": 2}


def alert(level: str, source: str, message: str, needs_human: bool = False) -> dict:
    return {"level": level, "source": source, "message": message, "needs_human": bool(needs_human)}


# ------------------------------------------------------------------ file cursors
def read_new_lines(path: Path, offset: int | None) -> tuple[list[dict], int]:
    """JSON records appended since `offset`. offset None = start at the end now (do not replay history)."""
    path = Path(path)
    size = path.stat().st_size if path.exists() else 0
    if offset is None or size < offset:                      # first call, or the file was replaced
        return [], size
    if not path.exists():                                    # nothing logged yet (e.g. no marks measured)
        return [], 0
    recs = []
    with open(path, "rb") as f:
        f.seek(offset)
        data = f.read()
    end = data.rfind(b"\n") + 1                              # leave a half-written last line for next time
    for line in data[:end].splitlines():
        try:
            recs.append(json.loads(line))
        except ValueError:
            pass
    return recs, offset + end


def read_all(path: Path) -> list[dict]:
    recs, _ = read_new_lines(path, 0)
    return recs


# ------------------------------------------------------------------ event mapping
def event_alerts(events: list[dict]) -> list[dict]:
    out = []
    for e in events:
        kind = e.get("kind", "")
        lvl = EVENT_LEVEL.get(kind, "info")
        src = "guided" if kind in ("guided_fail", "loader") or str(e.get("message", "")).startswith("loader") \
            else "cellbot"
        msg = f"{kind}: {e.get('message', '')}"
        if e.get("part"):
            msg += f" (part {e['part']})"
        out.append(alert(lvl, src, msg, needs_human=(kind == "estop")))
    return out


def split_arm_in_view(events: list[dict], in_view) -> tuple[list[dict], int]:
    """Vision events raised while the worker arm was over the part are not evidence about the part (the arm
    hides and distorts it). Returns (events to report, number ignored). in_view(t) -> bool."""
    kept = [e for e in events if not in_view(e.get("t"))]
    return kept, len(events) - len(kept)


def arm_in_view_alert(n: int) -> list[dict]:
    return [alert("info", "vision", f"vision is not reliable while the arm is over the part: {n} events ignored")] \
        if n else []


def vision_alerts(events: list[dict]) -> list[dict]:
    starts: dict[str, list[str]] = {}
    ended: dict[str, int] = {}
    for e in events:
        if e.get("phase") == "end":
            ended[e["type"]] = ended.get(e["type"], 0) + 1
        else:
            starts.setdefault(e["type"], []).append(str(e.get("detail", "")))
    out = []
    for typ, details in starts.items():
        msg = f"{typ}: {details[-1]}".rstrip(": ")
        if typ == "UNEXPECTED_HOLES":
            msg += " (possible wrong part or stray ink)"
        if len(details) > 1:
            msg += f" ({len(details)}x)"
        out.append(alert(VISION_START_LEVEL.get(typ, "info"), "vision", msg))
    if ended:
        out.append(alert("info", "vision", "cleared: " + ", ".join(f"{t}" + (f" {n}x" if n > 1 else "")
                                                                for t, n in ended.items())))
    return out


# ------------------------------------------------------------------ metric rules (pure)
def _miss(rec) -> float:
    return float(np.hypot(*rec["miss_mm"]))


def miss_by_part(marks: list[dict]) -> dict[str, float]:
    """Mean miss per part, in order of first appearance."""
    by: dict[str, list[float]] = {}
    for r in marks:
        by.setdefault(str(r.get("part")), []).append(_miss(r))
    return {p: round(float(np.mean(v)), 2) for p, v in by.items()}


def rule_recent_miss(marks: list[dict], tolerance_mm: float, n: int = 4) -> list[dict]:
    if len(marks) < n:
        return []
    m = float(np.mean([_miss(r) for r in marks[-n:]]))
    if m > tolerance_mm:
        return [alert("warn", "metrics", f"mean miss of the last {n} marks is {m:.1f} mm, above the "
                                         f"{tolerance_mm} mm tolerance")]
    return []


def rule_miss_trend(marks: list[dict], n_parts: int = 3) -> list[dict]:
    per = list(miss_by_part(marks).values())[-n_parts:]
    if len(per) == n_parts and all(b > a for a, b in zip(per, per[1:])):
        return [alert("warn", "metrics", f"accuracy is getting worse (mean miss per part {per} mm): check that the "
                      "camera or the arm base was not bumped; recalibration may be needed", needs_human=True)]
    return []


def rule_droop(offset_deg, settle_max_deg: float) -> list[dict]:
    off = np.abs(np.asarray(offset_deg, float))
    near = [i + 1 for i, v in enumerate(off) if v >= 0.9 * settle_max_deg]
    if settle_max_deg > 0 and near:
        return [alert("warn", "metrics", f"droop correction on joint(s) {near} is at "
                      f"{off.max():.1f} of the {settle_max_deg} deg limit: the arm sags more than expected "
                      "(load, loose joint or a weak servo)")]
    return []


def rule_sighting_age(touches: list[dict], commit_age_s: float, n: int = 5, key: str = "commit_age_s") -> list[dict]:
    ages = [float(r["sighting_age_s"]) for r in touches[-n:] if "sighting_age_s" in r]
    if ages and float(np.mean(ages)) > 0.75 * commit_age_s:
        return [alert("info", "metrics", f"touches are aimed on old sightings (mean {np.mean(ages):.1f} s of "
                      f"{key} {commit_age_s} s): the arm hides the part for long, or vision is slow")]
    return []


def metrics(marks: list[dict], touches: list[dict], parts_done: int, recent: int = 20) -> dict:
    t = touches[-recent:]

    def mean(key):
        v = [float(r[key]) for r in t if key in r]
        return round(float(np.mean(v)), 2) if v else None
    return {"parts_done": parts_done, "marks_logged": len(marks),
            "mean_miss_mm_by_part": dict(list(miss_by_part(marks).items())[-5:]),
            "touches": len(touches), "mean_re_aims_per_touch": mean("re_aims"),
            "mean_followed_mm_per_touch": mean("followed_mm"), "mean_seconds_per_touch": mean("seconds")}


def recommend(alerts: list[dict]) -> str:
    if any(a["needs_human"] or a["level"] == "critical" for a in alerts):
        return "pause and ask a human"
    if any(a["level"] == "warn" for a in alerts):
        return "check and continue"
    return "continue"


# ------------------------------------------------------------------ cell method
class AlertsMixin:
    _alert_cursor: dict | None = None

    async def alerts(self) -> dict:
        from .vision_source import VisionError
        first = self._alert_cursor is None
        cur = self._alert_cursor or {"events": None, "vseq": None}
        out: list[dict] = []
        new_ev, cur["events"] = read_new_lines(self.events_path, cur["events"])
        out += event_alerts(new_ev)
        est = self.estop_state()
        if est.get("latched"):
            out.append(alert("critical", "cellbot", f"E-STOP is latched ({est.get('reason', 'no reason')}); only a "
                             "human can release it", needs_human=True))
        # vision: its own cursor, and is it live right now?
        try:
            await self.ensure_guided()
            ev = await self.vision.events(cur["vseq"] or 0)
            if cur["vseq"] is not None:
                kept, ignored = split_arm_in_view(ev, self.arm_in_view_at)
                out += vision_alerts(kept) + arm_in_view_alert(ignored)
            if ev:
                cur["vseq"] = ev[-1]["seq"]
            elif cur["vseq"] is None:
                cur["vseq"] = 0
            await self._vstate()
        except (VisionError, OSError) as e:
            out.append(alert("critical", "vision", f"vision unreachable or stale: {e}", needs_human=True))
        except Exception as e:  # noqa: BLE001  set-up problems are also operator information
            out.append(alert("critical", "vision", f"vision not available: {e!r}", needs_human=True))
        self._alert_cursor = cur
        # rules over the logs (every call, also the first)
        marks = read_all(self.model.log)
        touches = [r for r in read_all(self.glog) if r.get("ok", True)]   # failed-step records are diagnostics
        out += rule_recent_miss(marks, float(self.gcfg["marks"]["tolerance_mm"]))
        out += rule_miss_trend(marks)
        out += rule_droop(np.degrees(self._offset), float(self.m.get("settle_max_deg", 4.0)))
        if self._sees_under_arm():
            out += rule_sighting_age(touches, float(self.m["commit_age_s"]))
        else:
            out += rule_sighting_age(touches, float(self.m.get("sight_max_age_s", 60.0)), key="sight_max_age_s")
        parts_done = sum(1 for e in read_all(self.events_path) if e.get("kind") == "part_done")
        out.sort(key=lambda a: -RANK[a["level"]])
        return {"alerts": out, "recommendation": recommend(out), "first_call": first,
                "metrics": metrics(marks, touches, parts_done)}
