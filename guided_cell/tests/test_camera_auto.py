"""Offline check of [vision] source = "camera:auto" and the self-healing camera:N:  python tests/test_camera_auto.py

No real camera: a fake capture factory maps OpenCV index -> recorded frames (the real top camera, the SO-101 wrist
camera, the MacBook camera, a 320x240 camera) and can renumber them while running. Real-image checks are skipped
(with a note) when the recorded pictures are not on this machine.
"""
import asyncio
import os
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("CELLBOT_MODE", "sim")
os.environ.setdefault("CELLBOT_STATE", tempfile.mkdtemp(prefix="cellguide_camauto_"))
os.environ.setdefault("CELLBOT_CONFIG", str(ROOT / "cell.guided.toml"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from cellguide import config as gconfig  # noqa: E402
from cellguide import guided_cell as gc  # noqa: E402
from cellguide import vision_source as vs  # noqa: E402
from cellguide.vision_source import (CameraSelector, CameraVision, TopViewScorer, VisionError,  # noqa: E402
                                     camera_spec, open_camera_source)
import sim_settings  # noqa: E402

G = gconfig.load()
V = gconfig.load_module("part_vision", G["_paths"]["vision_file"])
UP = Path(os.environ.get("CELLGUIDE_TEST_IMAGES", "/mnt/user-data/uploads/lerobot/guided_cell/state"))
REF = ROOT / "top_ref.jpg"
CALIB = G["_paths"]["calib_file"]


def img(rel):
    p = UP / rel
    return cv2.imread(str(p)) if p.is_file() else None


TOP = img("marks/20261122_120728_3fe061_before.jpg")       # top camera, box elsewhere, arm partly in view
TOP2 = img("cli/20261122_105310_840_inspect.jpg")
TOP3 = img("marks/20261122_120728_3fe061_after.jpg")
WRIST = img("cli/probe_3.jpg")                              # SO-101 wrist camera, 1280x720 (sees tape too)
MAC = img("cli/probe_4.jpg")                                # MacBook camera, 1280x720
HAVE = all(x is not None for x in (TOP, TOP2, TOP3, WRIST, MAC)) and REF.is_file()
TINY = None if WRIST is None else cv2.resize(WRIST, (320, 240))   # the flaky ElRobot wrist camera: 320x240 only


class FakeCapture:
    def __init__(self, frames, fps=30.0, opened=True):
        self.frames, self.dt, self.opened = frames, 1.0 / fps, opened
        self.i, self.stopped, self.released = 0, False, False

    def isOpened(self):
        return self.opened

    def set(self, prop, val):
        return True

    def get(self, prop):
        return 0

    def read(self):
        time.sleep(self.dt)
        if self.stopped or self.released or not self.opened or not self.frames:
            return False, None
        f = self.frames[self.i % len(self.frames)]
        self.i += 1
        return True, f.copy()

    def release(self):
        self.released = True


class World:
    """index -> frames; indices past the list are out of range. Every open gets a new capture of what is there."""

    def __init__(self, cams):
        self.cams, self.opened = list(cams), []

    def __call__(self, i):
        if i >= len(self.cams) or self.cams[i] is None:
            return FakeCapture([], opened=False)
        cap = FakeCapture(self.cams[i] if isinstance(self.cams[i], list) else [self.cams[i]])
        self.opened.append((i, cap))
        return cap


def check(name, cond, detail=""):
    print(f"  [{'ok' if cond else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    if not cond:
        check.failed += 1


check.failed = 0


def selector(world, ref=REF, calib=CALIB, fallback=True):
    return CameraSelector(TopViewScorer(ref, calib), 1280, 720, auto_fallback=fallback, capture_factory=world)


def gcfg(source, **vision):
    return {**G, "vision": {**G["vision"], "source": source, "view_port": 0, "top_reference": str(REF), **vision}}


def scores():
    print("scores against the reference picture (top_ref.jpg = probe_1.jpg) and the calibration fallback")
    ref, cal = TopViewScorer(REF, CALIB), TopViewScorer(None, CALIB)
    check("reference mode is used when top_ref.jpg exists", ref.mode == "reference", ref.note)
    check("calibration fallback when the reference is missing", cal.mode == "calibration", cal.note)
    rows = {"top before": TOP, "top after": TOP3, "top inspect": TOP2, "SO-101 wrist": WRIST, "MacBook": MAC}
    s = {k: (ref.score(v), cal.score(v)) for k, v in rows.items()}
    for k, (a, b) in s.items():
        print(f"      {k:14s} reference {a:.3f}   calibration {b:.3f}")
    check("top views score above the threshold (reference)", all(s[k][0] >= ref.threshold for k in
                                                                 ("top before", "top after", "top inspect")))
    check("wrist and MacBook score far below it (reference)", max(s["SO-101 wrist"][0], s["MacBook"][0])
          < ref.threshold - ref.margin)
    check("top views score above the threshold (calibration)", all(s[k][1] >= cal.threshold for k in
                                                                   ("top before", "top after", "top inspect")))
    check("wrist and MacBook score below it (calibration)", max(s["SO-101 wrist"][1], s["MacBook"][1]) < cal.threshold)


def selection():
    print("auto selection with fake cameras")
    layouts = {"top at 0": [TOP, WRIST, MAC, TINY], "top at 2": [MAC, TINY, TOP, WRIST],
               "top at 3": [MAC, WRIST, TINY, TOP2], "top at 1, held camera at 0": [[], TOP3, WRIST]}
    for name, cams in layouts.items():
        w = World(cams)
        sel = selector(w)
        t0 = time.monotonic()
        cap, i, frame, s = sel.select()
        dt = time.monotonic() - t0
        want = next(k for k, c in enumerate(cams) if c is TOP or c is TOP2 or c is TOP3)
        check(f"{name}: picks index {want}", i == want and frame.shape[:2] == (720, 1280),
              f"got {i}, score {s:.2f}, {dt:.1f} s; {sel.table()}")
        check(f"{name}: probing stopped at the first out-of-range index",
              max(k for k, _ in w.opened) == len(cams) - 1, str(sorted({k for k, _ in w.opened})))
        check(f"{name}: probed captures are released", all(c.released for c, k in ((c, k) for k, c in w.opened)
                                                          if c is not cap))
        cap.release()
    t = selector(World([TINY])).table(selector(World([TINY])).probe())
    check("a 320x240 camera is listed with its size, not scored", "320x240, not 1280x720" in t, t)

    for name, cams, want in (("two top views", [MAC, TOP, WRIST, TOP2], "look alike"),
                             ("no top view", [MAC, TINY, WRIST], "no camera looks like the top view"),
                             ("no camera at all", [], "no camera found")):
        try:
            selector(World(cams)).select()
            msg = ""
        except VisionError as e:
            msg = str(e)
        check(f"{name}: refuses with a clear error", want in msg and "camera:N" in msg, msg)
        if cams:
            check(f"{name}: error lists every index with size/score", all(f"index {k}:" in msg for k in range(len(cams)))
                  and "score" in msg, msg)

    print("reference missing -> calibration markers")
    sel = selector(World([MAC, WRIST, TINY, TOP]), ref=ROOT / "no_such_ref.jpg")
    cap, i, _, s = sel.select()
    cap.release()
    check("calibration fallback picks the top camera", i == 3, f"index {i}, score {s:.2f}; {sel.table()}")
    try:
        selector(World([MAC, WRIST]), ref=None, calib=None).select()
        msg = ""
    except VisionError as e:
        msg = str(e)
    check("neither reference nor calibration: refuses to guess", "cannot tell which camera" in msg, msg)


async def camera_objects():
    print("CameraVision: camera:auto, self-healing camera:N")
    w = World([MAC, TINY, TOP, WRIST])
    t0 = time.monotonic()
    cam = await asyncio.to_thread(open_camera_source, gcfg("camera:auto"), w)
    dt = time.monotonic() - t0
    check("camera:auto opens the top camera", cam.index == 2 and cam.size == (1280, 720), cam.describe())
    check("label says 'camera auto -> index 2 (score ...)'", cam.describe().startswith("camera auto -> index 2 (score 0."),
          cam.describe())
    check("selection runs in well under 3 s here", dt < 3.0, f"{dt:.2f} s")
    st = await cam.state()
    check("state() serves the top view", st["image_size"] in ([1280, 720], (1280, 720)), str(st["image_size"]))
    cam.close()

    w = World([MAC, TINY, TOP, WRIST])
    cam = await asyncio.to_thread(open_camera_source, gcfg("camera:1"), w)
    check("camera:1 delivering 320x240 -> switches to index 2", cam.index == 2, cam.describe())
    check("warning says what happened", any("configured camera 1 delivers 320x240 instead of the configured 1280x720; "
                                            "using index 2 found by auto-detection" in x for x in cam.warnings),
          str(cam.warnings))
    check("label shows the switch", "camera 1 -> auto index 2" in cam.describe(), cam.describe())

    print("  preflight")
    from cellguide import preflight as P
    cell = sim_settings.apply(gc.GuidedCell())
    cell.vision = cam
    await cell.ensure_guided()
    rep = await P.report(cell)
    c = next((c for c in rep["checks"] if c["name"] == "camera"), {})
    check("preflight camera row shows the auto label", "camera 1 -> auto index 2" in c.get("detail", ""),
          c.get("detail", ""))
    c = next((c for c in rep["checks"] if c["name"] == "camera selection"), {})
    check("preflight warns about the switch", c.get("status") == "warn" and "delivers 320x240" in c.get("detail", ""),
          c.get("detail", ""))
    cam.close()

    try:
        await asyncio.to_thread(open_camera_source, gcfg("camera:1", auto_fallback=False), World([MAC, TINY, TOP]))
        msg = ""
    except VisionError as e:
        msg = str(e)
    check("auto_fallback = false: camera:1 at 320x240 fails as before", msg.startswith("camera 1:") and "320x240" in msg,
          msg)
    cam = await asyncio.to_thread(open_camera_source, gcfg("camera:0"), World([MAC, TOP, WRIST]))
    check("camera:0 is the MacBook (1280x720 but not the top view) -> switches to index 1", cam.index == 1,
          f"{cam.describe()}; {cam.warnings}")
    cam.close()
    cam = await asyncio.to_thread(open_camera_source, gcfg("camera:2"), World([MAC, WRIST, TOP]))
    check("camera:2 that is the top camera: kept, no warning", cam.index == 2 and not cam.warnings
          and cam.describe().startswith("camera 2 (score"), f"{cam.describe()}; {cam.warnings}")
    cam.close()
    try:
        await asyncio.to_thread(open_camera_source, gcfg("camera:1"), World([MAC, TINY, WRIST]))
        msg = ""
    except VisionError as e:
        msg = str(e)
    check("camera:1 wrong size and no top camera anywhere -> clear error", msg.startswith("camera 1: delivers 320x240")
          and "no replacement" in msg and "index 2:" in msg, msg)

    print("  camera lost while running")
    w = World([MAC, TOP, WRIST])
    cam = await asyncio.to_thread(open_camera_source, gcfg("camera:auto"), w)
    check("starts on index 1", cam.index == 1, cam.describe())
    running = [c for k, c in w.opened if k == 1 and not c.released][-1]
    w.cams = [MAC, WRIST, TINY, TOP2]          # top camera unplugged and back: now index 3
    running.stopped = True
    await asyncio.sleep(1.3)
    try:
        await cam.state()
        msg = ""
    except VisionError as e:
        msg = str(e)
    check("frames stop -> state() raises (stale)", msg.startswith("camera 1:"), msg)
    t0, msg2, recovered = time.monotonic(), "", False
    while time.monotonic() - t0 < 8:
        try:
            await cam.raw()
            if cam.index == 3:
                recovered = True
                break
        except VisionError as e:
            msg2 = msg2 or (str(e) if "camera lost" in str(e) else "")
        await asyncio.sleep(0.1)
    check("lost -> raw() says 'camera lost ... re-detecting'", "camera lost" in msg2 and "re-detecting" in msg2, msg2)
    check("re-detected at index 3, serving the top view again", recovered and cam.index == 3, cam.describe())
    check("re-detection is recorded as a warning", any("re-detected" in x for x in cam.warnings), str(cam.warnings))
    cam.close()

    w = World([MAC, TOP, WRIST])
    cam = await asyncio.to_thread(open_camera_source, gcfg("camera:auto"), w)
    running = [c for k, c in w.opened if k == 1 and not c.released][-1]
    w.cams = [MAC, WRIST]                       # no top camera any more
    running.frames = [TINY]                     # same capture suddenly delivers another size
    await asyncio.sleep(0.3)
    try:
        await cam.state()
        msg = ""
    except VisionError as e:
        msg = str(e)
    check("frame size changes mid-run -> state() raises at once, no frame of the other camera served",
          "camera lost" in msg and "frame size changed" in msg, msg)
    cam.close()

    fake = FakeCapture([TOP])
    cam = CameraVision(V, 4, capture_factory=lambda i: fake)       # no selector: old behaviour, but no silent swap
    fake.frames = [TINY]
    await asyncio.sleep(0.3)
    try:
        await cam.raw()
        msg = ""
    except VisionError as e:
        msg = str(e)
    check("without a selector a size change is reported, not served", "camera lost" in msg and "restart" in msg, msg)
    cam.close()


async def config_checks():
    print("configuration")
    check("camera_spec: camera:auto / camera:2", camera_spec("camera:auto") == "auto" and camera_spec("camera:2") == 2)
    try:
        camera_spec("camera:top")
        bad = False
    except ValueError:
        bad = True
    check("camera:<not a number or auto> is refused", bad)
    sel = vs.selector_from_config({**G, "vision": {**G["vision"], "top_reference": "top_ref.jpg"}})
    check("top_reference resolves relative to guided.toml's folder", sel.scorer.reference_path ==
          Path(G["_file"]).parent / "top_ref.jpg", str(sel.scorer.reference_path))
    check("auto_fallback defaults to true", vs.selector_from_config(
        {**G, "vision": {k: v for k, v in G["vision"].items() if k != "auto_fallback"}}).auto_fallback)
    vs.install(gc)
    check("install() points guided_cell at the auto-aware opener", gc.open_camera_source is open_camera_source)
    cell = sim_settings.apply(gc.GuidedCell(gcfg=gcfg("camera:auto")))
    try:
        await cell.ensure_guided()            # no camera in this sandbox: must fail cleanly, listing nothing found
        err = "opened?!"
        cell.close_vision()
    except VisionError as e:
        err = str(e)
    check("ensure_guided accepts camera:auto (fails cleanly without cameras)", err.startswith("camera auto:"), err)


async def main():
    if not HAVE:
        print(f"SKIPPED real-image checks: recorded pictures not found under {UP} or {REF} missing")
        await config_checks()
        return
    scores()
    selection()
    await camera_objects()
    await config_checks()


if __name__ == "__main__":
    asyncio.run(main())
    print("ALL CHECKS PASSED" if check.failed == 0 else f"{check.failed} CHECK(S) FAILED")
    sys.exit(1 if check.failed else 0)
