"""[marks] method = "diff": black-marker dabs found as NEW dark marks (cellguide/marks_dark.py + GuidedCell.inspect).

    python tests/test_marks_dark.py

A. synthetic lids (red quad, three dark rings, a reference line, a dark side wall) before/after, with the box
   shifted/rotated, small dark dots at known offsets, noise/blur/brightness change and JPEG:
   dots found, miss within 1 mm, no false marks on rings/line/edge, old dots not counted again, global assignment,
   stray marks, registration failure reported;
B. the real top-camera picture 20261122_105310_840_inspect.jpg (if present): "before" = the two dots inpainted,
   "after" = the raw picture, also turned 10 deg / 8 deg and shifted, brightness changed, re-encoded;
C. GuidedCell in the simulator with method = "diff": reference kept from the clear look before the first dab
   (also with look_each_step), box turned after the dabs, misses fed to MissModel once, annotated picture, the
   reference refreshed after inspect (no double counting), no-reference fallback.
"""
import asyncio
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ["CELLBOT_MODE"] = "sim"
os.environ["CELLBOT_STATE"] = tempfile.mkdtemp(prefix="cellguide_marks_")

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from cellguide import config as gconfig  # noqa: E402
from cellguide import marks_dark as md  # noqa: E402

MM_PER_PX = 1 / 2.2           # the real top camera: ~2.2 px per mm
REAL = Path("/mnt/user-data/uploads/lerobot/guided_cell/state/cli/20261122_105310_840_inspect.jpg")


def check(name, cond, detail=""):
    print(f"  [{'ok' if cond else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    if not cond:
        check.failed += 1


check.failed = 0

# ------------------------------------------------------------------ synthetic lid
CENTER = np.array([700.0, 290.0])
HOLES = {0: (-35.0, -38.0), 1: (10.0, 2.0), 2: (42.0, -45.0)}          # px from the lid centre (box frame)
LID = (170.0, 145.0)
SS = 4                                                                 # supersampling


def _T(angle_deg, shift):
    a = np.radians(angle_deg)
    R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    return lambda p: CENTER + R @ np.asarray(p, float) + np.asarray(shift, float)


def make(angle=0.0, shift=(0, 0), dots=(), gain=1.0, noise=2.0, seed=0, holes=HOLES, wall_px=None):
    """(image, state, truth): dots = [(x, y, r_px)] in the box frame (px from the lid centre)."""
    T = _T(angle, shift)
    W, H = 1280, 720
    img = np.zeros((H * SS, W * SS, 3), np.uint8)
    img[:] = (120, 150, 185)                                           # cardboard (BGR)
    hx, hy = LID[0] / 2, LID[1] / 2
    corners = [T(p) for p in [(-hx, -hy), (hx, -hy), (hx, hy), (-hx, hy)]]
    wall = 6 + 0.6 * angle if wall_px is None else wall_px              # side wall seen at an angle
    wq = [T(p) for p in [(-hx - wall, -hy), (-hx, -hy), (-hx, hy), (-hx - wall, hy)]]

    def P(p):
        return np.int32(np.round(np.asarray(p) * SS))
    cv2.fillPoly(img, [P(wq)], (25, 25, 95))
    cv2.fillPoly(img, [P(corners)], (60, 62, 208))                     # red lid
    ink = (35, 35, 70)
    for hid, (x, y) in holes.items():
        cv2.circle(img, tuple(P(T((x, y)))), int(7.5 * SS), ink, int(3.2 * SS), cv2.LINE_AA)
    cv2.line(img, tuple(P(T((-45, 50)))), tuple(P(T((45, 50)))), ink, int(2.2 * SS), cv2.LINE_AA)
    for x, y, r in dots:
        cv2.circle(img, tuple(P(T((x, y)))), int(round(r * SS)), ink, -1, cv2.LINE_AA)
    img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA).astype(np.float32)
    img = cv2.GaussianBlur(img, (0, 0), 0.6) * gain
    img += np.random.default_rng(seed).normal(0, noise, img.shape)
    img = np.clip(img, 0, 255).astype(np.uint8)
    img = cv2.imdecode(cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])[1], cv2.IMREAD_COLOR)
    st = {"image_size": [W, H], "box": {"corners_px": [list(map(float, c)) for c in corners]},
          "holes": [{"id": i, "px": list(map(float, T(p))), "visible": True} for i, p in holes.items()]}
    truth = [T((x, y)) for x, y, _ in dots]
    return img, st, truth


def nearest(marks, p):
    if not marks:
        return None, 1e9
    d = [np.hypot(m["px"][0] - p[0], m["px"][1] - p[1]) for m in marks]
    i = int(np.argmin(d))
    return marks[i], d[i]


def part_a():
    print("A. synthetic lids")
    hx0, hy0 = HOLES[0]
    hx1, hy1 = HOLES[1]
    dots = [(hx0 + 1.5, hy0 + 1, 1.6), (hx1 - 3, hy1 + 13, 1.8)]   # 3-4 px dots: inside ring 0 (a near hit), below ring 1
    before, sb, _ = make(seed=1)
    for angle, shift, gain in [(0, (0, 0), 1.0), (0, (4, -3), 1.1), (10, (8, -5), 0.9), (-10, (-6, 7), 1.12)]:
        after, sa, truth = make(angle, shift, dots, gain=gain, seed=2)
        r = md.detect(before, after, sb, sa, {})
        tag = f"{angle:+} deg, shift {shift} px, brightness x{gain}"
        check(f"{tag}: registered", r["ok"], str(r["registration"]))
        errs = [nearest(r["marks"], t)[1] * MM_PER_PX for t in truth]
        check(f"{tag}: both dots found within 1 mm", all(e < 1.0 for e in errs), f"errors {np.round(errs, 2)} mm")
        check(f"{tag}: no false marks (rings, line, side wall)", len(r["marks"]) == len(truth),
              f"{len(r['marks'])} marks: {[m['px'] for m in r['marks']]}")
        # miss measured in the AFTER picture relative to the target hole there = the drawn offset, turned with the box
        if r["marks"]:
            m, _ = nearest(r["marks"], truth[0])
            miss = (np.array(m["px"]) - np.array(sa["holes"][0]["px"])) * MM_PER_PX
            a = np.radians(angle)
            want = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]]) @ np.array([1.5, 1]) * MM_PER_PX
            check(f"{tag}: miss vector of hole 0 within 1 mm of the truth", np.linalg.norm(miss - want) < 1.0,
                  f"measured {np.round(miss, 2)} mm, truth {np.round(want, 2)} mm")

    # old dots already on the lid are not counted again; a new dot next to an old one is
    old = [(hx0 + 1.5, hy0 + 1, 1.6)]
    before, sb, _ = make(dots=old, seed=3)
    after, sa, truth = make(6, (5, 2), old + [(hx0 - 4, hy0 + 14, 1.6)], seed=4)
    r = md.detect(before, after, sb, sa, {})
    ok = len(r["marks"]) == 1 and nearest(r["marks"], truth[1])[1] < 2.2
    check("old dot in the reference is not found again, the new one is", ok, str([m["px"] for m in r["marks"]]))

    # assignment: two dots near different holes; and the case where greedy nearest-per-target would fail
    before, sb, _ = make(seed=5)
    dots2 = [(hx0 + 10, hy0 + 9, 1.7), (hx1 - 10, hy1 - 9, 1.7)]     # each leaning towards the other hole
    after, sa, truth = make(-4, (3, 3), dots2, seed=6)
    r = md.detect(before, after, sb, sa, {})
    tp = [h["px"] for h in sa["holes"][:2]]
    mp = [m["px"] for m in r["marks"]]
    asg = md.assign(tp, mp, [MM_PER_PX] * 2, 30.0)
    ok = (len(asg) == 2 and np.hypot(*(np.array(mp[asg[0]]) - truth[0])) < 2.2
          and np.hypot(*(np.array(mp[asg[1]]) - truth[1])) < 2.2)
    check("two dots near different holes: each goes to its own hole", ok, f"{asg} {mp}")
    asg = md.assign([[0, 0], [60, 0]], [[15, 0], [-20, 0]], [MM_PER_PX] * 2, 30.0)
    check("global assignment: one mark per target, all within reach matched (greedy would leave one out)",
          asg == {0: 1, 1: 0}, str(asg))
    asg = md.assign([[0, 0]], [[3, 0], [5, 1]], [MM_PER_PX], 30.0)
    check("two dots near one target: the nearer is used, the other stays a stray", asg == {0: 0}, str(asg))

    # stray mark: far from every hole
    after, sa, truth = make(3, (2, 0), [(hx0 + 12, hy0 + 6, 1.6), (-60, 25, 1.7)], seed=7)
    r = md.detect(before, after, sb, sa, {})
    mp = [m["px"] for m in r["marks"]]
    asg = md.assign([h["px"] for h in sa["holes"]], mp, [MM_PER_PX] * 3, 15.0)
    stray = [mp[j] for j in range(len(mp)) if j not in asg.values()]
    check("stray mark found and left unassigned", len(mp) == 2 and len(stray) == 1
          and np.hypot(*(np.array(stray[0]) - truth[1])) < 2.2, f"marks {mp}, stray {stray}")

    # registration failure: the lid in 'after' has a different hole layout (another part) -> reported, no marks
    other = {0: (-50.0, 10.0), 1: (0.0, -20.0), 2: (50.0, 30.0)}
    after, sa, _ = make(0, (0, 0), [(0, 0, 1.6)], seed=8, holes=other)
    r = md.detect(before, after, sb, sa, {})
    check("a different part: registration refused with a reason, no marks", not r["ok"] and not r["marks"]
          and r.get("why"), str(r.get("why")))
    after, sa, _ = make(0, (0, 0), (), seed=9)
    sa2 = dict(sa, box=None, holes=[])
    r = md.detect(before, after, sb, sa2, {})
    check("no lid pose in the inspect picture: refused, no marks", not r["ok"] and not r["marks"], str(r.get("why")))

    # limits (informational + loose checks)
    print("  limits:")
    for rad in (0.6, 0.8, 1.1, 1.4):
        after, sa, truth = make(5, (3, -2), [(hx1 + 2, hy1 + 16, rad)], seed=10)
        r = md.detect(before, after, sb, sa, {})
        print(f"    dot radius {rad} px (~{2 * rad:.1f} px across): found={nearest(r['marks'], truth[0])[1] < 2.2}")
    for gap in (-3.2, -1.6, 0.0, 1.0, 2.0):                    # dot edge this far outside the ring (<0: on the stroke)
        rr = 7.5 + 1.6 + gap + 1.6
        after, sa, truth = make(5, (3, -2), [(hx1, hy1 + rr, 1.6)], seed=11)
        r = md.detect(before, after, sb, sa, {})
        print(f"    dot edge {gap:+.1f} px from the ring stroke (dot r 1.6 px): found={nearest(r['marks'], truth[0])[1] < 2.5}")
    for ang in (15, 25, 40):
        after, sa, truth = make(ang, (12, 8), [(hx0 + 1.5, hy0 + 1, 1.6)], seed=12)
        r = md.detect(before, after, sb, sa, {})
        print(f"    box turned {ang} deg: registered={r['ok']}, dot found={nearest(r['marks'], truth[0])[1] < 2.2}, "
              f"false marks={max(0, len(r['marks']) - 1)}")
    after, sa, truth = make(20, (15, -10), [(hx0 + 1.5, hy0 + 1, 1.6)], seed=13)
    r = md.detect(before, after, sb, sa, {})
    check("box turned 20 deg and moved 8 mm: still found, no false marks",
          r["ok"] and len(r["marks"]) == 1 and nearest(r["marks"], truth[0])[1] < 2.2, str([m["px"] for m in r["marks"]]))


def part_b():
    print("B. real top-camera picture (105310 raw): dots inpainted for 'before'")
    if not REAL.exists():
        print("  (skipped: image not present)")
        return
    G = gconfig.load()
    V = gconfig.load_module("part_vision", G["_paths"]["vision_file"])
    a = cv2.imread(str(REAL))
    m = np.zeros(a.shape[:2], np.uint8)
    cv2.ellipse(m, (744, 251), (4, 8), 0, 0, 360, 255, -1)
    before = cv2.inpaint(a, m, 5, cv2.INPAINT_TELEA)
    sb = V.detect_scene(before)
    truth0 = np.array([[743.7, 248.5], [744.3, 254.5]])
    rng = np.random.default_rng(0)
    for ang, sh in [(0, (0, 0)), (10, (6, -4)), (-8, (10, 5))]:
        R = cv2.getRotationMatrix2D((720, 295), ang, 1.0)
        R[:, 2] += sh
        after = a if ang == 0 and sh == (0, 0) else cv2.warpAffine(a, R, (1280, 720), borderMode=cv2.BORDER_REPLICATE)
        if ang:
            after = np.clip(after.astype(float) * 1.08 + rng.normal(0, 2, after.shape), 0, 255).astype(np.uint8)
            after = cv2.imdecode(cv2.imencode(".jpg", after, [cv2.IMWRITE_JPEG_QUALITY, 85])[1], cv2.IMREAD_COLOR)
        sa = V.detect_scene(after)
        r = md.detect(before, after, sb, sa, {})
        truth = truth0 @ R[:, :2].T + R[:, 2]
        errs = [nearest(r["marks"], t)[1] * MM_PER_PX for t in truth]
        check(f"real lid, turned {ang:+} deg, shift {sh}: 2 dots, each within 1 mm, no other marks",
              r["ok"] and len(r["marks"]) == 2 and all(e < 1.0 for e in errs),
              f"errors {np.round(errs, 2)} mm, {len(r['marks'])} marks, ECC {r['registration'].get('ecc')}, "
              f"residual {r['registration'].get('residual_px')} px")


# ------------------------------------------------------------------ GuidedCell in the simulator
async def part_c():
    print("C. GuidedCell inspect with method = 'diff' (simulator)")
    from cellguide.guided_cell import GuidedCell
    from cellguide.vision_source import FrameVision
    sys.path.insert(0, str(ROOT / "tests"))
    import sim_settings

    G = gconfig.load()
    V = gconfig.load_module("part_vision", G["_paths"]["vision_file"])
    IMG = G["_paths"]["vision_file"].parent / "images"

    class Feed:
        def __init__(self):
            self.base = cv2.imread(str(IMG / "test_1.jpeg"))
            self.ink, self.holes_px, self.angle, self.center = [], {}, 0.0, (640, 360)

        async def __call__(self):
            f = self.base.copy()
            for hid, dx, dy in self.ink:                               # black marker dots, ~3-4 px
                x, y = self.holes_px[hid]
                cv2.circle(f, (int(round(x + dx)), int(round(y + dy))), 2, (25, 25, 40), -1, cv2.LINE_AA)
            if self.angle:
                R = cv2.getRotationMatrix2D(self.center, self.angle, 1.0)
                f = cv2.warpAffine(f, R, (f.shape[1], f.shape[0]), borderMode=cv2.BORDER_REPLICATE)
            return f

    feed = Feed()
    cell = sim_settings.apply(GuidedCell())
    cell.gcfg["marks"].update(method="diff", search_radius_mm=15.0)
    cell.vision = FrameVision(V, feed, "test frames")
    await cell.ensure_guided()
    k = cell.kin
    await cell.new_part("d1")
    st = await cell.vision.state()
    feed.holes_px = {h["id"]: h["px"] for h in st["holes"]}
    feed.center = tuple(np.mean(st["box"]["corners_px"], axis=0))
    check("reference kept at new_part", cell._mark_ref is not None and cell._mark_ref["part_id"] == "d1")
    r = await cell.run_program([{"hole": 0}, {"hole": 1}, {"hole": 2}], look_each_step=True)
    t_ref = cell._mark_ref["t"]
    first_dab = min(w["t"] for w in cell._work.values())
    check("3 dabs + retreat (look_each_step)", r["ok"] and r["retreat"]["ok"])
    check("reference is the look before the first dab (not replaced by later looks)", t_ref < first_dab)
    feed.ink = [(0, 6, 3), (1, 6, 3), (2, 6, 3)]           # the "arm" lands 6 px right, 3 px down of each hole
    feed.angle = 7.0                                       # and the dabs turned the box
    for _ in range(6):
        await cell.vision.state()
    st2 = await cell.vision.state()
    n0 = cell.model.status()["marks_logged"]
    out, jpeg = await cell.inspect()
    found = [m for m in out["marks"].values() if m["found"]]
    check("method reported", out.get("method") == "diff")
    check("3 marks measured", len(found) == 3, str(out["marks"]) + " " + str(out.get("note")))
    a1 = cell._aim(st2, {"hole": 1})
    mm_px = np.linalg.norm(k.surface_point(np.array(a1["px"]) + [20, 0], st2["image_size"]) - a1["P0"]) / 20 * 1000
    expect = float(np.hypot(6, 3) * mm_px)
    got = [m["miss_dist_mm"] for m in found]
    check("miss size right for every hole (box turned 7 deg)", found and all(abs(g - expect) < 1.0 for g in got),
          f"measured {got} mm, drawn {expect:.1f} mm; registration {out.get('registration')}")
    check("no stray marks", out["stray_marks"] == 0, str(out.get("stray_marks_px")))
    check("misses fed to the model once each", cell.model.status()["marks_logged"] == n0 + 3)
    check("annotated picture returned", len(jpeg) > 10000)
    out2, _ = await cell.inspect()
    check("second inspect: no double counting (reference refreshed), earlier results repeated",
          cell.model.status()["marks_logged"] == n0 + 3 and out2["stray_marks"] == 0
          and all(m.get("measured_earlier") for m in out2["marks"].values()), str(out2["marks"]))
    gs = await cell.guided_status()
    check("guided_status shows the method and the reference", gs.get("marks_method") == "diff"
          and gs.get("mark_reference") is not None)

    # a stray dab and a new dab on hole 0 in a second run on the same part
    r = await cell.run_program([{"hole": 0}])
    feed.ink += [(0, -4, 5), (1, 40, -30)]
    for _ in range(4):
        await cell.vision.state()
    out3, _ = await cell.inspect()
    m0 = out3["marks"]["hole 0"]
    check("second run: the new dab on hole 0 is measured, the old ones are not counted again",
          m0.get("found") and not m0.get("measured_earlier") and cell.model.status()["marks_logged"] == n0 + 4,
          str(out3["marks"]))
    check("second run: the stray dab is reported", out3["stray_marks"] == 1, str(out3.get("stray_marks_px")))

    # no reference -> clear note, nothing invented
    r = await cell.run_program([{"hole": 2}])
    cell._mark_ref = None
    feed.ink += [(2, 3, -3)]
    out4, _ = await cell.inspect()
    check("no reference: nothing measured, a clear note, model unchanged",
          not out4["marks"]["hole 2"]["found"] and "no reference" in out4.get("note", "")
          and cell.model.status()["marks_logged"] == n0 + 4, out4.get("note", "")[:90])
    check("...and a reference is kept for the next run", cell._mark_ref is not None)
    imgs = list((cell.cfg.state_dir / "marks").glob("*_result.jpg"))
    check("before/after/result pictures saved under state/marks", len(imgs) >= 2, str(len(imgs)))


def main():
    part_a()
    part_b()
    asyncio.run(part_c())
    print("FAILED" if check.failed else "all checks passed", f"({check.failed} failures)" if check.failed else "")
    sys.exit(1 if check.failed else 0)


if __name__ == "__main__":
    main()
