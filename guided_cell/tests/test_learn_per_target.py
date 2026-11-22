"""Local (per-target) learned aim correction: cellguide/learn.py MissModel + how GuidedCell reports it.

    python tests/test_learn_per_target.py

A. two targets 20 mm apart, one with no error, one with a constant (-3, +4.5) mm error: after 1-2 rounds the bad
   one is corrected by ~gain x error, the good one stays < 0.5 mm;
B. simulated closed loop over 4 rounds (box shifting ~2 mm, measurement noise): the miss at the bad target shrinks
   and never flips sign (no oscillation), the good target stays good;
C. an outlier (wrong assignment, 16 mm) is stored but not used; the cap stays max_correction_mm;
D. a brand-new position gets the shrunk global offset;
E. persistence across reload (a "server restart") and the per-target status listing;
F. GuidedCell: cellbot's learning_status (cell.learner.status()) and finish_part report the guided corrections,
   also after closing the part and after a restart (they were empty before: cellbot's own learner is not fed).
G. the real marks of 2026-11-22 (if present): hole 1 corrected by ~ -gain x its error, hole 0 much less.
"""
import asyncio
import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ["CELLBOT_MODE"] = "sim"
os.environ.setdefault("CELLBOT_STATE", tempfile.mkdtemp(prefix="cellguide_learn_"))

import numpy as np  # noqa: E402

from cellguide.learn import MissModel  # noqa: E402

REAL = Path("/mnt/user-data/uploads/lerobot/guided_cell/state/guided_marks.jsonl")
GOOD, BAD = np.array([30.0, 0.0]), np.array([50.0, 0.0])          # 20 mm apart
E_BAD = np.array([-3.0, 4.5])


def check(name, cond, detail=""):
    print(f"  [{'ok' if cond else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    if not cond:
        check.failed += 1


check.failed = 0


def new_model(d=None, **kw):
    return MissModel(Path(d or tempfile.mkdtemp(prefix="cellguide_learn_")), **{"gain": 0.8, "max_correction_mm": 12.0,
                                                                                 "window": 40, **kw})


def err_at(p):
    return E_BAD.copy() if np.linalg.norm(np.asarray(p) - BAD) < 8 else np.zeros(2)


def dab(m, part, name, p, noise=None):
    """One mark: aim with the current correction, land at hole + correction + arm error (+ noise), record it."""
    c = m.correction_mm(p)
    miss = c + err_at(p) + (noise if noise is not None else 0.0)
    m.record(part, name, p, miss, c)
    return miss, c


def test_two_targets():
    print("A. two targets 20 mm apart, one fine, one off by (-3, +4.5) mm")
    m = new_model()
    dab(m, "p1", "hole 0", GOOD)
    dab(m, "p1", "hole 1", BAD)
    cb, cg = m.correction_mm(BAD), m.correction_mm(GOOD)
    check("1 round: bad target corrected by ~ -gain x error", np.linalg.norm(cb - (-0.8 * E_BAD)) < 0.4, f"{cb.round(2)}")
    check("1 round: good target stays < 0.5 mm", np.linalg.norm(cg) < 0.5, f"{cg.round(2)}")
    for name, p in (("hole 0", GOOD), ("hole 1", BAD)):
        dab(m, "p2", name, p)
    cb, cg = m.correction_mm(BAD), m.correction_mm(GOOD)
    check("2 rounds: bad target still ~ -gain x error (no wind-up)", np.linalg.norm(cb - (-0.8 * E_BAD)) < 0.4,
          f"{cb.round(2)}")
    check("2 rounds: good target stays < 0.5 mm", np.linalg.norm(cg) < 0.5, f"{cg.round(2)}")
    stored = m.samples[-1]
    check("stored error = miss - applied (the uncorrected aim's error)",
          np.allclose(np.array(stored["miss_mm"]) - stored["applied_mm"], stored["error_mm"], atol=0.02)
          and np.allclose(stored["error_mm"], E_BAD, atol=0.02), f"{stored['error_mm']}")
    one = new_model()
    one.record("p", "hole 9", (0, 0), (2.0, -1.0), (0, 0))
    c1 = one.correction_mm((0, 0))
    check("a single sample at a place: correction = gain x that error", np.allclose(c1, [-1.6, 0.8], atol=1e-6),
          f"{c1.round(3)}")


def test_closed_loop():
    print("B. closed loop, 4 rounds, box shifting ~2 mm, 0.2 mm noise")
    rng = np.random.default_rng(3)
    m = new_model()
    bad, good = [], []
    for r in range(4):
        shift = rng.normal(0, 1.2, 2)
        mg, _ = dab(m, f"p{r}", "hole 0", GOOD + shift, rng.normal(0, 0.2, 2))     # program order: hole 0 first
        mb, _ = dab(m, f"p{r}", "hole 1", BAD + shift, rng.normal(0, 0.2, 2))
        bad.append(mb)
        good.append(float(np.linalg.norm(mg)))
    nb = [float(np.linalg.norm(x)) for x in bad]
    check("miss at the bad target shrinks (non-increasing within noise) and ends < 1.5 mm",
          all(b <= a + 0.3 for a, b in zip(nb, nb[1:])) and nb[-1] < 1.5 and nb[1] < 0.4 * nb[0],
          f"{[round(v, 2) for v in nb]}")
    check("no oscillation: the bad miss never flips to the other side", all(float(x @ E_BAD) > 0 for x in bad))
    check("the good target stays good (< 1 mm every round)", max(good) < 1.0, f"{[round(v, 2) for v in good]}")
    m0 = new_model()
    nb0 = [float(np.linalg.norm(dab(m0, "p", "hole 1", BAD)[0])) for _ in range(4)]
    check("noise-free: miss 5.4 -> 1.08 and stays (gain 0.8, no ringing)",
          abs(nb0[0] - 5.41) < 0.02 and all(abs(v - 0.2 * nb0[0]) < 0.02 for v in nb0[1:]), f"{[round(v, 2) for v in nb0]}")


def test_outlier_and_cap():
    print("C. outliers and the cap")
    m = new_model()
    dab(m, "p", "hole 1", BAD)
    before = m.correction_mm(BAD)
    m.record("p", "hole 1", BAD, (14.0, -9.0), (0, 0))        # a mark assigned to the wrong hole
    after = m.correction_mm(BAD)
    check("16.6 mm sample stored (flagged) but not used", len(m.samples) == 2 and m.samples[-1].get("outlier")
          and np.allclose(before, after), f"{before.round(2)} -> {after.round(2)}")
    check("outlier counted in the listing but not used", any(t["samples"] == 2 and t["used"] == 1
                                                             for t in m.targets()), str(m.targets()))
    cap = new_model(max_correction_mm=3.0)
    cap.record("p", "hole 1", BAD, (8.0, 6.0), (0, 0))
    c = cap.correction_mm(BAD)
    check("correction capped at max_correction_mm", abs(np.linalg.norm(c) - 3.0) < 1e-6, f"{c.round(2)}")


def test_new_position():
    print("D. a brand-new position")
    m = new_model()
    for _ in range(2):
        m.record("p", "hole 0", (0, 0), (2.0, 1.0), (0, 0))
        m.record("p", "hole 1", (20, 0), (2.0, 1.0), (0, 0))
    c = m.correction_mm((10, 60))                 # far from both: shrunk global offset
    check("new position gets global_shrink x the global correction", np.allclose(c, 0.5 * -0.8 * np.array([2.0, 1.0]),
                                                                                 atol=0.02), f"{c.round(3)}")
    check("empty model: zero correction", np.allclose(new_model().correction_mm((5, 5)), 0))


def test_persistence_and_status():
    print("E. persistence across reload and the per-target listing")
    d = tempfile.mkdtemp(prefix="cellguide_learn_")
    m = new_model(d)
    for r in range(2):
        dab(m, f"p{r}", "hole 0", GOOD + [0.5 * r, 0])
        dab(m, f"p{r}", "hole 1", BAD + [0, 1.0 * r])
    m2 = new_model(d)
    check("reloaded model gives the same corrections", np.allclose(m.correction_mm(BAD), m2.correction_mm(BAD))
          and np.allclose(m.correction_mm(GOOD), m2.correction_mm(GOOD)))
    t = {x["target"]: x for x in m2.status()["per_target"]}
    check("listing has one entry per hole (samples within 5 mm grouped)", sorted(t) == ["hole 0", "hole 1"]
          and t["hole 1"]["samples"] == 2 and t["hole 0"]["samples"] == 2, json.dumps(list(t.values())))
    check("hole 1 entry: mean error, last miss and the correction applied now",
          np.allclose(t["hole 1"]["mean_error_mm"], E_BAD, atol=0.02) and t["hole 1"]["last_miss_mm"] < 1.5
          and np.linalg.norm(np.array(t["hole 1"]["correction_now_mm"]) + 0.8 * E_BAD) < 0.4, json.dumps(t["hole 1"]))


async def _cell_part():
    print("F. GuidedCell: learning_status / finish_part after closing the part and after a restart")
    from cellguide.guided_cell import GuidedCell
    cell = GuidedCell()
    cell.model.reset()
    cell.part = {"id": "t1", "marked": {}, "learned": [], "misses": {}}
    for name, p in (("hole 0", GOOD), ("hole 1", BAD)):
        c = cell.model.correction_mm(p)
        cell.model.record("t1", name, p, c + err_at(p), c, extra={"method": "diff"})
        cell.part["misses"][name] = round(float(np.linalg.norm(c + err_at(p))), 2)
    st = cell.learner.status()                        # what cellbot's learning_status tool returns
    check("learning_status lists per-target corrections", len(st["corrections_mm"]) == 2
          and "guided" in st and "recent_parts" in st, json.dumps(st["corrections_mm"]))
    done = await cell.finish_part("ok", "test")
    check("finish_part's learning is filled after closing the part",
          len(done["learning"]["corrections_mm"]) == 2 and done["learning"]["recent_parts"][-1]["part"] == "t1",
          json.dumps(done["learning"]["corrections_mm"]))
    cell2 = GuidedCell()                              # server restart: same state folder
    st2 = cell2.learner.status()
    k1 = next(k for k in st2["corrections_mm"] if k.startswith("hole 1"))
    check("after a restart the corrections are loaded from the files",
          len(st2["corrections_mm"]) == 2 and np.linalg.norm(np.array(st2["corrections_mm"][k1]) + 0.8 * E_BAD) < 0.4
          and st2["recent_parts"][-1]["part"] == "t1", json.dumps(st2["corrections_mm"]))
    check("MissModel keys from [marks] (defaults when absent)", cell2.model.scale > 0 and cell2.model.outlier > 0,
          f"scale {cell2.model.scale}, outlier {cell2.model.outlier}")


def test_real():
    print("G. the real marks of 2026-11-22")
    if not REAL.exists():
        print("  (skipped: real file not present)")
        return
    m = new_model(max_correction_mm=25.0)
    m.samples = [json.loads(x) for x in REAL.read_text().splitlines() if x.strip()]
    h0, h1 = m.correction_mm((44.2, -4.1)), m.correction_mm((62.5, -1.5))
    e1 = np.mean([s["error_mm"] for s in m.samples if s["target"] == "hole 1" and s["pos_mm"][0] > 50], 0)
    check("hole 1: ~ -gain x its mean error", np.linalg.norm(h1 + 0.8 * e1) < 1.0, f"{h1.round(2)} vs {(-0.8 * e1).round(2)}")
    check("hole 0: much smaller than hole 1, and the 14-18 mm wrong marks are not used",
          np.linalg.norm(h0) < 0.4 * np.linalg.norm(h1) and len(m._usable()) == 6, f"{h0.round(2)}")


if __name__ == "__main__":
    test_two_targets()
    test_closed_loop()
    test_outlier_and_cap()
    test_new_position()
    test_persistence_and_status()
    asyncio.run(_cell_part())
    test_real()
    print("FAILED" if check.failed else "all checks passed", f"({check.failed} failures)" if check.failed else "")
    sys.exit(1 if check.failed else 0)
