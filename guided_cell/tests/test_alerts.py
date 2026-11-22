"""Offline check of the alert rules with synthetic records (no cell, no Station):  python tests/test_alerts.py"""
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from cellguide import alerts as A  # noqa: E402


def check(name, cond, detail=""):
    print(f"  [{'ok' if cond else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    if not cond:
        check.failed += 1


check.failed = 0


def mark(part, miss):
    return {"part": part, "miss_mm": [miss, 0.0]}


def main() -> int:
    print("recent miss")
    check("4 marks above tolerance -> warn", A.rule_recent_miss([mark("a", 4)] * 4, 3.0)[0]["level"] == "warn")
    check("within tolerance -> nothing", A.rule_recent_miss([mark("a", 2)] * 6, 3.0) == [])
    check("fewer than 4 marks -> nothing", A.rule_recent_miss([mark("a", 9)] * 3, 3.0) == [])
    check("only the last 4 count", A.rule_recent_miss([mark("a", 9)] * 4 + [mark("b", 1)] * 4, 3.0) == [])

    print("trend over parts")
    rising = [mark("p1", 1), mark("p1", 1.2), mark("p2", 1.5), mark("p3", 2.5)]
    t = A.rule_miss_trend(rising)
    check("rising over 3 parts -> warn, needs a human", len(t) == 1 and t[0]["needs_human"]
          and "bumped" in t[0]["message"], t[0]["message"] if t else "")
    check("falling -> nothing", A.rule_miss_trend([mark("p1", 3), mark("p2", 2), mark("p3", 1)]) == [])
    check("flat -> nothing", A.rule_miss_trend([mark("p1", 2), mark("p2", 2), mark("p3", 2.5)]) == [])
    check("only 2 parts -> nothing", A.rule_miss_trend([mark("p1", 1), mark("p2", 2)]) == [])
    check("only the last 3 parts count", A.rule_miss_trend(rising + [mark("p4", 0.5)]) == [])

    print("droop and sighting age")
    check("droop at 3.7 of 4 deg -> warn", A.rule_droop([0, 3.7, 0, 0, 0, 0, 0], 4.0)[0]["level"] == "warn")
    check("droop at -3.8 counts too", len(A.rule_droop([0, 0, -3.8, 0, 0, 0, 0], 4.0)) == 1)
    check("droop at 3.5 of 4 deg -> nothing", A.rule_droop([3.5] * 7, 4.0) == [])
    old = [{"sighting_age_s": 7.0}] * 5
    check("old sightings -> info", A.rule_sighting_age(old, 8.0)[0]["level"] == "info")
    check("fresh sightings -> nothing", A.rule_sighting_age([{"sighting_age_s": 1.0}] * 5 + old[:1], 8.0) == [])

    print("events, vision, recommendation")
    ev = A.event_alerts([{"kind": "estop", "message": "x"}, {"kind": "refused", "message": "y"},
                         {"kind": "guided_fail", "message": "hole 1: lost"}, {"kind": "part_done", "message": "pass"}])
    check("event levels", [a["level"] for a in ev] == ["critical", "info", "warn", "info"]
          and ev[0]["needs_human"] and ev[2]["source"] == "guided")
    va = A.vision_alerts([{"type": "UNEXPECTED_HOLES", "phase": "start", "detail": "2 round marks"},
                          {"type": "POSE_STALE", "phase": "start", "detail": "no pose"},
                          {"type": "BOX_MOVING", "phase": "start", "detail": ""},
                          {"type": "BOX_MOVING", "phase": "end", "detail": ""}])
    lv = {a["message"].split(":")[0]: a["level"] for a in va}
    check("vision levels", lv.get("UNEXPECTED_HOLES") == "warn" and lv.get("POSE_STALE") == "warn"
          and lv.get("BOX_MOVING") == "info" and lv.get("cleared") == "info", str(va))
    check("stray ink hint", any("stray ink" in a["message"] for a in va))
    check("recommend continue", A.recommend([A.alert("info", "x", "m")]) == "continue")
    check("recommend check", A.recommend([A.alert("warn", "x", "m")]) == "check and continue")
    check("recommend ask on needs_human", A.recommend([A.alert("warn", "x", "m", True)]) == "pause and ask a human")
    check("recommend ask on critical", A.recommend([A.alert("critical", "x", "m")]) == "pause and ask a human")

    print("file cursor")
    p = Path(tempfile.mkdtemp()) / "events.jsonl"
    p.write_text(json.dumps({"kind": "old"}) + "\n")
    recs, off = A.read_new_lines(p, None)
    check("first read starts at the end", recs == [] and off == p.stat().st_size)
    with open(p, "a") as f:
        f.write(json.dumps({"kind": "new"}) + "\n" + '{"kind": "half')
    recs, off2 = A.read_new_lines(p, off)
    check("then only new complete lines", [r["kind"] for r in recs] == ["new"] and off2 < p.stat().st_size)
    m = A.metrics(rising, [{"re_aims": 1, "followed_mm": 4, "seconds": 2}, {"re_aims": 0, "followed_mm": 0, "seconds": 1}], 3)
    check("metrics", m["parts_done"] == 3 and m["marks_logged"] == 4 and m["mean_re_aims_per_touch"] == 0.5
          and m["mean_miss_mm_by_part"]["p3"] == 2.5, str(m))

    print(f"\n{'ALL CHECKS PASSED' if not check.failed else str(check.failed) + ' CHECK(S) FAILED'}")
    return check.failed


if __name__ == "__main__":
    sys.exit(main())
