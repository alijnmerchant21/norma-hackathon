"""Offline check of [vision] source = "camera:N" (CameraVision) with a fake camera:  python tests/test_camera_vision.py

No real camera here: the capture is injected and cycles Manasa's three recorded images at ~30 fps.
"""
import asyncio
import json
import os
import socket
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("CELLBOT_MODE", "sim")
os.environ.setdefault("CELLBOT_STATE", tempfile.mkdtemp(prefix="cellguide_cam_"))
os.environ.setdefault("CELLBOT_CONFIG", str(ROOT / "cell.guided.toml"))

import cv2  # noqa: E402

from cellguide import config as gconfig  # noqa: E402
from cellguide import guided_cell as gc  # noqa: E402
from cellguide.vision_source import CameraVision, VisionError  # noqa: E402
import sim_settings  # noqa: E402  (tests/: fixed values, independent of guided.toml)

G = gconfig.load()
V = gconfig.load_module("part_vision", G["_paths"]["vision_file"])
IMG = G["_paths"]["vision_file"].parent / "images"
FRAMES = [cv2.imread(str(IMG / f"test_{i}.jpeg")) for i in (1, 2, 3)]


class FakeCapture:
    """Stands in for cv2.VideoCapture: cycles the frames at `fps`; `stopped` = the camera stops delivering."""

    def __init__(self, frames, fps=30.0, opened=True):
        self.frames, self.dt, self.opened = frames, 1.0 / fps, opened
        self.i, self.stopped, self.released, self.props = 0, False, False, {}

    def isOpened(self):
        return self.opened

    def set(self, prop, val):
        self.props[prop] = val
        return True

    def get(self, prop):
        return self.props.get(prop, 0)

    def read(self):
        time.sleep(self.dt)
        if self.stopped or self.released:
            return False, None
        f = self.frames[self.i % len(self.frames)].copy()
        self.i += 1
        return True, f

    def release(self):
        self.released = True


def check(name, cond, detail=""):
    print(f"  [{'ok' if cond else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    if not cond:
        check.failed += 1


check.failed = 0


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


async def wait_locked(cam, limit):
    t0 = time.monotonic()
    while time.monotonic() - t0 < limit:
        try:
            st = await cam.state()
            if st.get("map_locked"):
                return st, time.monotonic() - t0
        except VisionError:
            pass
        await asyncio.sleep(0.02)
    return None, time.monotonic() - t0


async def main():
    print("CameraVision with a fake 30 fps camera")
    fake = FakeCapture(FRAMES)
    port = free_port()
    cam = CameraVision(V, 2, 1280, 720, view_port=port, capture_factory=lambda i: fake)
    check("requested 1280x720 from the capture", fake.props.get(cv2.CAP_PROP_FRAME_WIDTH) == 1280
          and fake.props.get(cv2.CAP_PROP_FRAME_HEIGHT) == 720)
    check("frame size measured", cam.size == (1280, 720), str(cam.size))
    st, dt = await wait_locked(cam, 1.0)
    check("hole map locks within 1 s", st is not None, f"{dt:.2f} s, {cam.n} frames processed")
    st = await cam.state()
    age = time.time() - st["timestamp"]
    check("state() is fresh", age < 0.2, f"age {age:.3f} s")
    n0 = cam.n
    await asyncio.sleep(0.5)
    rate = (cam.n - n0) / 0.5
    check("frames are processed continuously", rate >= 10, f"{rate:.0f} fps processed")
    check("every frame read is processed by the tracker", fake.i - cam.n in (0, 1), f"read {fake.i}, processed {cam.n}")
    raw = await cam.raw()
    check("raw() is the camera frame", raw.shape == (720, 1280, 3))
    jpg = await cam.annotated()
    check("annotated() is a JPEG", jpg[:2] == b"\xff\xd8")
    ev = await cam.events(0)
    check("events() returns the tracker log", isinstance(ev, list) and len(ev) > 0, f"{len(ev)} events")
    check("describe() names the camera", "camera 2 in-process" in cam.describe(), cam.describe())

    # viewer
    def get(path):
        with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=3) as r:
            return r.status, r.headers.get("Content-Type"), r.read()
    code, ctype, body = await asyncio.to_thread(get, "/frame.jpg")
    check("viewer serves /frame.jpg", code == 200 and ctype == "image/jpeg" and body[:2] == b"\xff\xd8",
          f"{code} {ctype} {len(body)} bytes")
    code, ctype, body = await asyncio.to_thread(get, "/raw.jpg")
    check("viewer serves /raw.jpg", code == 200 and body[:2] == b"\xff\xd8")
    code, ctype, body = await asyncio.to_thread(get, "/state")
    check("viewer serves /state", code == 200 and "map_locked" in json.loads(body))

    # reset: applied by the thread before the next frame, then the map locks again
    await cam.reset()
    t0 = time.monotonic()
    unlocked = False
    while time.monotonic() - t0 < 0.5:
        if not (await cam.state()).get("map_locked"):
            unlocked = True
            break
        await asyncio.sleep(0.01)
    check("reset() forgets the hole map", unlocked)
    st, dt = await wait_locked(cam, 1.0)
    check("map locks again after reset()", st is not None, f"{dt:.2f} s")

    # GuidedCell rejects stale data from this source
    cell = sim_settings.apply(gc.GuidedCell(vision=cam))
    st = await cell._vstate()
    check("GuidedCell._vstate accepts the live camera state", st["map_locked"])
    fake.stopped = True
    await asyncio.sleep(0.85)
    try:
        await cell._vstate()
        rejected = ""
    except VisionError as e:
        rejected = str(e)
    check("camera stops -> state goes stale and _vstate rejects it", bool(rejected), rejected)
    await asyncio.sleep(0.4)
    try:
        await cam.state()
        msg = ""
    except VisionError as e:
        msg = str(e)
    check("camera stopped > 1 s -> state() raises with the camera and the hint",
          msg.startswith("camera 2:") and "another program" in msg, msg)
    fake.stopped = False
    await asyncio.sleep(0.3)
    check("camera delivers again -> state() recovers", time.time() - (await cam.state())["timestamp"] < 0.3)
    cam.close()
    check("close() releases the camera and stops the thread", fake.released and not cam._thread.is_alive())
    try:
        await cam.state()
        closed = False
    except VisionError:
        closed = True
    check("state() after close() raises", closed)

    print("preflight")
    from cellguide import preflight as P
    cell = sim_settings.apply(gc.GuidedCell())
    cell.vision = CameraVision(V, 2, capture_factory=lambda i: FakeCapture(FRAMES))
    await cell.ensure_guided()
    rep = await P.report(cell)
    c = next((c for c in rep["checks"] if c["name"] == "camera"), {})
    check("preflight reports the camera frame size and fps", c.get("status") in ("ok", "warn")
          and "1280x720" in c.get("detail", "") and "fps" in c.get("detail", ""), c.get("detail", ""))
    v = next((c for c in rep["checks"] if c["name"] == "vision"), {})
    check("preflight vision check ok on the camera source", v.get("status") == "ok", v.get("detail", ""))
    cell.close_vision()

    print("errors")
    try:
        CameraVision(V, 3, 1280, 720, capture_factory=lambda i: FakeCapture([cv2.resize(FRAMES[0], (640, 640))]))
        msg = ""
    except VisionError as e:
        msg = str(e)
    check("aspect-ratio mismatch raises", "640x640" in msg and "1280x720" in msg, msg)
    try:
        CameraVision(V, 4, capture_factory=lambda i: FakeCapture(FRAMES, opened=False))
        msg = ""
    except VisionError as e:
        msg = str(e)
    check("camera that cannot be opened raises 'camera N: ...' with the hint",
          msg.startswith("camera 4:") and "another program" in msg, msg)
    dead = FakeCapture(FRAMES)
    dead.stopped = True
    try:
        CameraVision(V, 5, capture_factory=lambda i: dead, open_timeout_s=0.3)
        msg = ""
    except VisionError as e:
        msg = str(e)
    check("camera that delivers no frames raises", msg.startswith("camera 5:"), msg)

    print("configuration")
    g = {**G, "vision": {**G["vision"], "source": "camera:2", "view_port": 0}}
    cam = gc.open_camera_source(g, capture_factory=lambda i: FakeCapture(FRAMES))
    check("open_camera_source: camera:2 with [vision] width/height, viewer off",
          cam.index == 2 and cam.req == (int(G["vision"]["width"]), int(G["vision"]["height"])) and cam._srv is None)
    cam.close()
    try:
        gc.camera_index("camera:top")
        bad = False
    except ValueError:
        bad = True
    check("camera:<not a number> is refused", bad)
    cell = sim_settings.apply(gc.GuidedCell(gcfg=g))
    try:
        await cell.ensure_guided()            # no camera 2 in this sandbox: must fail cleanly
        err = "opened?!"
        cell.close_vision()
    except VisionError as e:
        err = str(e)
    check("ensure_guided accepts camera:N (fails cleanly without a camera)", err.startswith("camera 2:"), err)
    check("guided.toml source is one of the three kinds",
          G["vision"]["source"].startswith(("camera:", "http://", "https://", "station:")), G["vision"]["source"])



def cli_checks():
    print("cli")
    from cellguide import cli
    printed = []
    src0, port = gc._G["vision"]["source"], int(gc._G["vision"].get("view_port", 8765))
    view = f"viewer http://127.0.0.1:{port}/frame.jpg" if port else "viewer off"
    rc = cli.main(["--source", "camera:9", "help"], out=printed.append)
    check("cli --source camera:9 banner", any(f"VISION: camera 9 in-process ({view})"
                                              in str(p) for p in printed), printed[0] if printed else "")
    check("cli --source does not change guided.toml's value", gc._G["vision"]["source"] == src0
          and rc == 0)


if __name__ == "__main__":
    asyncio.run(main())
    cli_checks()
    print("ALL CHECKS PASSED" if check.failed == 0 else f"{check.failed} CHECK(S) FAILED")
    sys.exit(1 if check.failed else 0)
