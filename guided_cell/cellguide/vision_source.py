"""Where the part's pose comes from. Both sources return the same dict as Manasa's vision.py /state.

  HttpVision   her server in its own process:  python vision.py --camera N --serve 8765 --no-window
  FrameVision  her detector + tracker run in this process on frames we fetch (Station cameras, tests)
  CameraVision her detector + tracker run in this process on EVERY frame of a camera we open ourselves
               (camera:N, or camera:auto = the camera that looks like the top view, see CameraSelector)

Camera helper (run on the cell's computer, nothing else may hold the cameras):
  python -m cellguide.vision_source probe          every camera index: size and how much it looks like the top view
  python -m cellguide.vision_source save-ref N     save camera N's view as the reference picture ([vision] top_reference)
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import threading
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np


class VisionError(Exception):
    pass


ASPECT_TOL = 0.02
STATION_FRAMES_NOTE = ("Station's camera frames are resized by Station (224 px square by default), so they "
                       "cannot be used with a calibration made on the 1280x720 camera image")
ASPECT_FIX = ("use Manasa's vision server on the calibration camera at the calibration resolution "
              "([vision] source = \"http://127.0.0.1:8765\"), or redo the table calibration on the frames you use")


def image_size_check(size, calib_size) -> tuple[str, str]:
    """('ok' | 'warn' | 'fail', explanation) for a vision image size against the calibration image size."""
    w, h = (int(v) for v in size)
    cw, ch = (int(v) for v in calib_size)
    a, ca = w / h, cw / ch
    if abs(a / ca - 1) > ASPECT_TOL:
        return "fail", (f"vision image is {w}x{h} (aspect {a:.3f}) but the calibration image is {cw}x{ch} "
                        f"(aspect {ca:.3f}): pixels do not map onto the calibration (cropped or squashed frames). "
                        f"{STATION_FRAMES_NOTE}")
    if (w, h) != (cw, ch):
        return "warn", (f"vision image is {w}x{h}, calibration image {cw}x{ch}: pixel coordinates are scaled, "
                        "fine if the same camera and uncropped")
    return "ok", f"{w}x{h}, same as the calibration image"


class HttpVision:
    def __init__(self, url: str, timeout: float = 0.8):
        self.url, self.timeout = url.rstrip("/"), timeout

    def _get(self, path: str) -> bytes:
        try:
            with urllib.request.urlopen(self.url + path, timeout=self.timeout) as r:
                return r.read()
        except Exception as e:
            raise VisionError(f"vision server {self.url}{path}: {e}") from e

    async def state(self) -> dict:
        s = json.loads(await asyncio.to_thread(self._get, "/state"))
        if not s:
            raise VisionError("vision server has no frame yet")
        return s

    async def events(self, since: int) -> list:
        return json.loads(await asyncio.to_thread(self._get, f"/events?since={int(since)}"))

    async def raw(self) -> np.ndarray:
        return _decode(await asyncio.to_thread(self._get, "/raw.jpg"))

    async def annotated(self) -> bytes:
        return await asyncio.to_thread(self._get, "/frame.jpg")

    async def reset(self) -> None:
        await asyncio.to_thread(self._get, "/reset")

    def describe(self) -> str:
        return f"vision server {self.url}"


class FrameVision:
    """frame_fn: async function returning the newest BGR frame. Every state() call processes one frame."""

    def __init__(self, vision_module, frame_fn, label="frames"):
        self.V, self.frame_fn, self.label = vision_module, frame_fn, label
        self.tracker = vision_module.HoleTracker(calib_path="__none__")   # robot mm come from our calibration
        self.n = 0
        self._last = None

    async def state(self) -> dict:
        frame = await self.frame_fn()
        if frame is None:
            raise VisionError(f"{self.label}: no frame")
        scene = self.V.detect_scene(frame, frame_id=self.n)
        self.n += 1
        st = self.tracker.update(scene, now=time.time())
        self._last = (frame, scene, st)
        return st

    async def events(self, since: int) -> list:
        return [e for e in self.tracker.event_log if e["seq"] > since]

    async def raw(self) -> np.ndarray:
        if self._last is None:
            await self.state()
        return self._last[0]

    async def annotated(self) -> bytes:
        if self._last is None:
            await self.state()
        frame, scene, st = self._last
        return cv2.imencode(".jpg", self.V.draw_debug(frame, scene, st), [cv2.IMWRITE_JPEG_QUALITY, 80])[1].tobytes()

    async def reset(self) -> None:
        self.tracker.reset()

    def describe(self) -> str:
        return f"in-process tracker on {self.label}"


# ================================================================ which camera is the top camera
# macOS/OpenCV renumber the cameras whenever one is plugged in or drops out, so "camera:2" can silently become
# another camera. [vision] source = "camera:auto" (and, for "camera:N", [vision] auto_fallback = true) picks the
# camera that looks like the top view: see TopViewScorer and CameraSelector.

AUTO_INDICES = range(8)
LOST_AFTER_S = 2.0          # selector present: no frames this long -> the camera is treated as gone and re-detected


class TopViewScorer:
    """How much a frame looks like the top camera's view, 0..1.

    reference  (primary, [vision] top_reference): a picture of the top view. 0.6 * correlation of 64x36 blurred
               grayscale (layout: table, tape rectangle, arm base) + 0.4 * HSV hue/saturation histogram correlation.
               The moving red box and an arm in or out of view change it little; another camera scores near 0.
    calibration (fallback, [paths] calib_file): fraction of the calibration marker pixels that sit on a bright,
               unsaturated (white tape) patch that also holds dark pixels (the X mark). Only markers on tape count,
               so the real top view scores about 0.7, not 1; a camera moved by more than ~20 px scores low.
    none       neither file: nothing can be checked, auto-detection refuses to guess."""

    REF_SIZE = (64, 36)
    THRESHOLD = {"reference": 0.40, "calibration": 0.30}
    MARGIN = {"reference": 0.20, "calibration": 0.20}   # the winner must lead the runner-up by this much

    def __init__(self, reference_path=None, calib_path=None):
        self.reference_path = Path(reference_path) if reference_path else None
        self.calib_path = Path(calib_path) if calib_path else None
        self.mode, self._ref, self._pts, self.note = "none", None, None, ""
        if self.reference_path is not None and self.reference_path.is_file():
            img = cv2.imread(str(self.reference_path))
            if img is not None:
                self.mode, self._ref = "reference", self._features(img)
                self.note = f"reference {self.reference_path.name}"
            else:
                self.note = f"reference {self.reference_path} unreadable; "
        if self.mode == "none" and self.calib_path is not None and self.calib_path.is_file():
            try:
                c = json.loads(self.calib_path.read_text())
                pts = np.array([p["px"] for p in c["points"]], float)
                self._pts, self._calib_size = pts, tuple(c.get("image_size") or (1280, 720))
                self.mode = "calibration"
                self.note += (f"no reference picture ({self.reference_path.name if self.reference_path else 'unset'}); "
                              f"calibration markers of {self.calib_path.name} (weaker check)")
            except Exception as e:  # noqa: BLE001
                self.note += f"calibration {self.calib_path.name} unusable ({e!r})"
        if self.mode == "none" and not self.note:
            self.note = "no reference picture and no calibration file"

    @property
    def threshold(self) -> float:
        return self.THRESHOLD.get(self.mode, 1.0)

    @property
    def margin(self) -> float:
        return self.MARGIN.get(self.mode, 1.0)

    @classmethod
    def _features(cls, img):
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        s = cv2.GaussianBlur(cv2.resize(g, cls.REF_SIZE, interpolation=cv2.INTER_AREA).astype(np.float32), (3, 3), 0)
        v = s - s.mean()
        v /= float(np.linalg.norm(v)) + 1e-6
        hsv = cv2.cvtColor(cv2.resize(img, (320, 180), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2HSV)
        h = cv2.calcHist([hsv], [0, 1], None, [18, 16], [0, 180, 0, 256])
        cv2.normalize(h, h, 1, 0, cv2.NORM_L1)
        return v, h

    def score(self, frame) -> float | None:
        if self.mode == "reference":
            v, h = self._features(frame)
            gray = float((v * self._ref[0]).sum())
            hist = float(cv2.compareHist(h, self._ref[1], cv2.HISTCMP_CORREL))
            return round(0.6 * max(gray, 0.0) + 0.4 * max(hist, 0.0), 3)
        if self.mode == "calibration":
            return round(self._calib_score(frame), 3)
        return None

    def _calib_score(self, frame, r=22) -> float:
        hgt, wid = frame.shape[:2]
        pts = self._pts * [wid / self._calib_size[0], hgt / self._calib_size[1]]
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        V, S = hsv[..., 2], hsv[..., 1]
        bright = max(60.0, 0.75 * float(np.percentile(V[::4, ::4], 97)))   # relative: exposure varies
        hits = 0
        for x, y in pts:
            x0, x1, y0, y1 = int(max(0, x - r)), int(min(wid, x + r + 1)), int(max(0, y - r)), int(min(hgt, y + r + 1))
            v, s = V[y0:y1, x0:x1].astype(np.int32), S[y0:y1, x0:x1].astype(np.int32)
            if v.size == 0:
                continue
            tape = (v >= bright) & (s <= 70)
            if tape.mean() < 0.25:
                continue
            if (v < 0.65 * float(np.median(v[tape]))).mean() >= 0.02:
                hits += 1
        return hits / max(1, len(pts))


class CameraSelector:
    """Finds the top camera among OpenCV indices 0..7 (see TopViewScorer). capture_factory(index) as CameraVision's."""

    def __init__(self, scorer: TopViewScorer, width=1280, height=720, auto_fallback=True, capture_factory=None,
                 indices=AUTO_INDICES, first_frame_s=1.5, settle_s=0.4, open_timeout_s=4.0):
        self.scorer, self.req, self.auto_fallback = scorer, (int(width), int(height)), bool(auto_fallback)
        self.factory = capture_factory or _open_cv_camera
        self.indices, self.first_frame_s, self.settle_s, self.open_timeout_s = indices, first_frame_s, settle_s, open_timeout_s
        self.last = []                           # candidates of the last probe (for logs / errors)

    def _open(self, index):
        if self.factory is _open_cv_camera:
            with _quiet_opencv():
                return self.factory(index)
        return self.factory(index)

    def probe(self, stop=None) -> list[dict]:
        """Every camera present: {index, size, score, note}. Stops at the first index that cannot be opened."""
        cands = []
        for i in self.indices:
            if stop is not None and stop.is_set():
                break
            try:
                cap = self._open(i)
            except Exception:  # noqa: BLE001
                cap = None
            if cap is None or not cap.isOpened():
                if cap is not None:
                    cap.release()
                break                            # past the last camera (macOS numbers them without gaps)
            try:
                frame = _read_settled(cap, self.req, self.first_frame_s, self.settle_s)
            finally:
                cap.release()
            c = {"index": i, "size": None, "score": None, "note": ""}
            if frame is None:
                c["note"] = "no frames (held by another program?)"
            else:
                c["size"] = (frame.shape[1], frame.shape[0])
                if c["size"] != self.req:
                    c["note"] = f"{c['size'][0]}x{c['size'][1]}, not {self.req[0]}x{self.req[1]}"
                else:
                    c["score"] = self.scorer.score(frame)
            cands.append(c)
        self.last = cands
        return cands

    def table(self, cands=None) -> str:
        rows = []
        for c in (self.last if cands is None else cands):
            if c["score"] is not None:
                rows.append(f"index {c['index']}: {c['size'][0]}x{c['size'][1]} score {c['score']:.2f}")
            elif c["note"]:
                rows.append(f"index {c['index']}: {c['note']}")
            else:
                rows.append(f"index {c['index']}: {c['size'][0]}x{c['size'][1]} (not scored)")
        return "; ".join(rows) or "no camera found"

    FIX = ("Fix: set [vision] source = \"camera:N\" to the top camera's index (python -m cellguide.vision_source "
           "probe lists them), or refresh the reference picture (python -m cellguide.vision_source save-ref N)")

    def choose(self, cands) -> dict:
        sc = self.scorer
        if sc.mode == "none":
            raise VisionError(f"camera auto: cannot tell which camera is the top camera: {sc.note}. Cameras: "
                              f"{self.table(cands)}. {self.FIX}")
        ok = sorted((c for c in cands if c["score"] is not None), key=lambda c: -c["score"])
        rule = f"{sc.note}; needs score >= {sc.threshold:.2f} and a lead of {sc.margin:.2f}"
        if not ok or ok[0]["score"] < sc.threshold:
            raise VisionError(f"camera auto: no camera looks like the top view ({rule}). Cameras: "
                              f"{self.table(cands)}. {self.FIX}")
        if len(ok) > 1 and ok[0]["score"] - ok[1]["score"] < sc.margin:
            raise VisionError(f"camera auto: cameras {ok[0]['index']} and {ok[1]['index']} look alike, not guessing "
                              f"({rule}). Cameras: {self.table(cands)}. {self.FIX}")
        return ok[0]

    def select(self, stop=None):
        """-> (cap, index, first frame, score). Probes, chooses, opens the winner again and re-checks it."""
        t0 = time.monotonic()
        best = self.choose(self.probe(stop))
        i = best["index"]
        cap, frame = _open_checked(self._open, i, self.req, self.open_timeout_s)
        size = (frame.shape[1], frame.shape[0])
        s = self.scorer.score(frame) if size == self.req else None
        if s is None or s < self.scorer.threshold:
            cap.release()
            got = f"{size[0]}x{size[1]}" + (f", score {s:.2f}" if s is not None else "")
            raise VisionError(f"camera auto: index {i} changed while opening it ({got}); cameras renumbered? "
                              f"Probe was: {self.table()}")
        print(f"camera auto -> index {i} (score {s:.2f}; {self.scorer.note}; {time.monotonic() - t0:.1f} s). "
              f"Candidates: {self.table()}", flush=True)
        return cap, i, frame, s


def _read_settled(cap, req, first_frame_s, settle_s):
    """Ask for req, wait for a first frame, then read a few more (auto exposure) and return the last one."""
    try:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, req[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, req[1])
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:  # noqa: BLE001
        pass
    frame, t_end = None, time.monotonic() + first_frame_s
    while frame is None and time.monotonic() < t_end:
        ok, f = cap.read()
        frame = f if ok and f is not None else None
        if frame is None:
            time.sleep(0.03)
    if frame is None:
        return None
    t_end = time.monotonic() + settle_s
    for _ in range(4):
        if time.monotonic() > t_end:
            break
        ok, f = cap.read()
        if ok and f is not None:
            frame = f
    return frame


def _open_checked(open_fn, index, req, timeout_s, hold_hint="another program may hold it (the standalone vision "
                  "server, Photo Booth, a browser tab)"):
    """Open camera `index`, ask for req, return (cap, first frame). Raises VisionError 'camera N: ...'."""
    cap = open_fn(index)
    if cap is None or not cap.isOpened():
        if cap is not None:
            cap.release()
        raise VisionError(f"camera {index}: could not be opened; {hold_hint}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, req[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, req[1])
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)     # newest frame, if the backend supports it
    frame, t_end = None, time.monotonic() + timeout_s
    while frame is None and time.monotonic() < t_end:
        ok, f = cap.read()
        frame = f if ok and f is not None else None
        if frame is None:
            time.sleep(0.05)
    if frame is None:
        cap.release()
        raise VisionError(f"camera {index}: opened but delivers no frames; {hold_hint}")
    return cap, frame


class _quiet_opencv:
    """Silence OpenCV's 'out device of bound' / 'failed to properly initialize' while probing indices."""

    def __enter__(self):
        self.level = None
        try:
            self.level = cv2.utils.logging.getLogLevel()
            cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_SILENT)
        except Exception:  # noqa: BLE001
            pass
        self.fd = None
        try:                                   # AVFoundation prints with fprintf(stderr): mute fd 2 for the open
            sys.stderr.flush()
            self.fd = os.dup(2)
            devnull = os.open(os.devnull, os.O_WRONLY)
            os.dup2(devnull, 2)
            os.close(devnull)
        except Exception:  # noqa: BLE001
            self.fd = None
        return self

    def __exit__(self, *exc):
        if self.fd is not None:
            os.dup2(self.fd, 2)
            os.close(self.fd)
        if self.level is not None:
            try:
                cv2.utils.logging.setLogLevel(self.level)
            except Exception:  # noqa: BLE001
                pass
        return False


class CameraVision:
    """Owns one camera: a daemon thread reads every frame and runs detect_scene + HoleTracker on it (the
    tracker needs every frame in order). state() returns the newest result; it goes stale by itself if frames
    stop (callers reject old timestamps). capture_factory(index) -> object with read/set/get/isOpened/release.

    index: an int (camera:N) or "auto" (camera:auto, needs a selector). selector (CameraSelector, optional): with it a
    fixed index that cannot be opened, delivers no frames, the wrong size or does not look like the top view is
    replaced by auto-detection when selector.auto_fallback; and a camera lost while running (no frames for
    LOST_AFTER_S, or a frame size change) is reported by state()/raw() and re-detected by the camera thread itself,
    never served from whatever camera now has its index. Without a selector: the old behaviour."""

    HOLD_HINT = "another program may hold it (the standalone vision server, Photo Booth, a browser tab)"

    def __init__(self, vision_module, index, width: int = 1280, height: int = 720, view_port: int = 0,
                 capture_factory=None, open_timeout_s: float = 8.0, selector: CameraSelector | None = None):
        self.V, self.req = vision_module, (int(width), int(height))
        self.auto = str(index).strip().lower() == "auto"
        self.configured = None if self.auto else int(index)
        self.factory = capture_factory or _open_cv_camera
        self.selector, self.open_timeout_s = selector, open_timeout_s
        if self.auto and selector is None:
            self.selector = CameraSelector(TopViewScorer(), *self.req, capture_factory=capture_factory)
        self.tracker = vision_module.HoleTracker(calib_path="__none__")   # robot mm come from our calibration
        self.lock = threading.Lock()
        self._last = None                     # (frame, scene, state) of the newest frame
        self._error = None                    # why the camera is not delivering (set by the thread)
        self._lost = None                     # camera gone / changed: sticky until re-detected (never stale frames)
        self._lost_detail = ""
        self._reset = False
        self._stop = threading.Event()
        self.n, self.view_port, self._srv = 0, 0, None
        self.warnings: list[str] = []         # shown by preflight
        self.score = None
        self.index = self.configured if self.configured is not None else -1
        self.label = f"camera {self.configured}" if not self.auto else "camera auto"
        self.cap, self.index, frame, self.label, self.score, warn = self._select(strict=False)
        self.warnings += warn
        for w in warn:
            print(f"WARNING: {w}", flush=True)
        self.size = (frame.shape[1], frame.shape[0])
        self._process(frame)
        self._thread = threading.Thread(target=self._run, name=f"camera{self.index}", daemon=True)
        self._thread.start()
        if view_port:
            self._srv = _start_viewer(self, int(view_port))

    # ------------------------------------------------------------ opening / choosing the camera
    def _select(self, strict: bool):
        """-> (cap, index, frame, label, score, warnings). strict (re-detection after a loss): a fixed camera that
        does not look like the top view is not kept."""
        sel = self.selector
        if self.auto:
            cap, i, frame, s = sel.select(self._stop)
            return cap, i, frame, f"camera auto -> index {i} (score {s:.2f})", s, []
        n, rw, rh = self.configured, *self.req
        fallback = sel is not None and sel.auto_fallback
        problem, soft, score = None, False, None
        try:
            cap, frame = _open_checked(self.factory, n, self.req, self.open_timeout_s, self.HOLD_HINT)
        except VisionError as e:
            if not fallback:
                raise
            cap, frame, problem = None, None, str(e).split(": ", 1)[-1]
        if cap is not None:
            h, w = frame.shape[:2]
            if abs((w / h) / (rw / rh) - 1) > ASPECT_TOL and not fallback:
                cap.release()
                raise VisionError(f"camera {n}: asked for {rw}x{rh} (aspect {rw / rh:.3f}) but it delivers "
                                  f"{w}x{h} (aspect {w / h:.3f}); pixels would not map onto the calibration. Set "
                                  "[vision] width/height to a mode this camera supports with the calibration's aspect")
            if fallback and (w, h) != self.req:
                problem = f"delivers {w}x{h} instead of the configured {rw}x{rh}"
            elif sel is not None:
                score = sel.scorer.score(frame)
                if score is not None and score < sel.scorer.threshold:
                    problem, soft = (f"does not look like the top view (score {score:.2f} < "
                                     f"{sel.scorer.threshold:.2f}, {sel.scorer.note})"), True
        label = f"camera {n}" + (f" (score {score:.2f})" if score is not None and not problem else "")
        if problem is None:
            return cap, n, frame, label, score, []
        if not fallback:                         # only a soft problem gets here
            if not strict:
                return cap, n, frame, label, score, [f"configured camera {n} {problem}; check [vision] source"]
            cap.release()
            raise VisionError(f"camera {n}: {problem}")
        if cap is not None:
            cap.release()
        try:
            cap2, i, frame2, s = sel.select(self._stop)
        except VisionError as e:
            if soft and not strict:              # explicitly configured and usable: keep it, loudly
                cap, frame = _open_checked(self.factory, n, self.req, self.open_timeout_s, self.HOLD_HINT)
                return cap, n, frame, f"camera {n} (score {score:.2f}, NOT VERIFIED)", score, [
                    f"configured camera {n} {problem}; auto-detection found nothing better: {e}"]
            raise VisionError(f"camera {n}: {problem}; auto-detection found no replacement: {e}") from e
        if i == n:
            return cap2, n, frame2, f"camera {n} (score {s:.2f})", s, []
        return cap2, i, frame2, f"camera {n} -> auto index {i} (score {s:.2f})", s, [
            f"configured camera {n} {problem}; using index {i} found by auto-detection (score {s:.2f}). "
            f"Set [vision] source = \"camera:auto\" (or \"camera:{i}\") to silence this"]

    def _relocate(self, reason: str):
        """Camera thread: the camera is gone or changed. Never serve its index blindly: re-detect it."""
        self._lost, self._lost_detail = reason, ""
        with self.lock:
            self._last = None
        print(f"WARNING: {self.label}: {reason}; re-detecting the top camera", flush=True)
        try:
            self.cap.release()
        except Exception:  # noqa: BLE001
            pass
        old = self.index
        while not self._stop.is_set():
            try:
                cap, i, frame, label, s, warn = self._select(strict=True)
            except VisionError as e:
                self._lost_detail = str(e)
                self._stop.wait(2.0)
                continue
            if self._stop.is_set():
                cap.release()
                return
            if (frame.shape[1], frame.shape[0]) != self.size:
                cap.release()
                self._lost_detail = f"re-detected camera delivers {frame.shape[1]}x{frame.shape[0]}, not {self.size}"
                self._stop.wait(2.0)
                continue
            self.cap, self.index, self.label, self.score = cap, i, label, s
            msg = f"camera re-detected after '{reason}': index {old} -> {i} ({label})"
            self.warnings += [msg] + warn
            print(msg, flush=True)
            self._reset = True                     # a new view: the tracker starts over
            self._error = None
            self._process(frame)
            self._lost = None
            return

    # ------------------------------------------------------------ frames
    def _process(self, frame):
        if self._reset:
            self._reset = False
            self.tracker.reset()
        now = time.time()
        scene = self.V.detect_scene(frame, frame_id=self.n)
        self.n += 1
        with self.lock:                  # events() reads the tracker's log under the same lock
            st = dict(self.tracker.update(scene, now=now), timestamp=now)
            self._last = (frame, scene, st)

    def _run(self):
        last_ok = time.monotonic()
        while not self._stop.is_set():
            if self._lost and self.selector is None:      # sticky: nothing to re-detect with
                time.sleep(0.2)
                continue
            try:
                ok, frame = self.cap.read()
            except Exception as e:  # noqa: BLE001
                ok, frame = False, None
                self._error = f"read failed: {e!r}"
            if not ok or frame is None:
                self._error = self._error or "stopped delivering frames"
                if self.selector is not None and time.monotonic() - last_ok > LOST_AFTER_S:
                    self._relocate(f"no frames for {LOST_AFTER_S:.0f} s ({self._error})")
                    last_ok = time.monotonic()
                    continue
                time.sleep(0.05)       # keep trying: a USB hiccup may recover; state() goes stale meanwhile
                continue
            if (frame.shape[1], frame.shape[0]) != self.size:
                reason = (f"frame size changed from {self.size[0]}x{self.size[1]} to {frame.shape[1]}x"
                          f"{frame.shape[0]} (cameras renumbered?)")
                if self.selector is not None:
                    self._relocate(reason)
                    last_ok = time.monotonic()
                else:
                    self._lost = reason
                    with self.lock:
                        self._last = None
                continue
            last_ok = time.monotonic()
            self._error = None
            try:
                self._process(frame)
            except Exception as e:  # noqa: BLE001  a detector error must not kill the thread silently
                self._error = f"vision failed on frame {self.n}: {e!r}"

    def _get(self):
        if self._stop.is_set():
            raise VisionError(f"camera {self.index}: closed")
        if self._lost:
            more = ("re-detecting the top camera" + (f"; last attempt: {self._lost_detail}" if self._lost_detail else "")
                    if self.selector is not None else "restart the runner to open the camera again")
            raise VisionError(f"{self.label}: camera lost: {self._lost}; {more}")
        with self.lock:
            last = self._last
        if last is None or (self._error and time.time() - last[2]["timestamp"] > 1.0):
            raise VisionError(f"camera {self.index}: {self._error or 'no frame yet'}; {self.HOLD_HINT}")
        return last

    async def state(self) -> dict:
        return dict(self._get()[2])

    async def events(self, since: int) -> list:
        with self.lock:
            return [e for e in self.tracker.event_log if e["seq"] > since]

    async def raw(self) -> np.ndarray:
        return self._get()[0]

    def annotated_sync(self) -> bytes:
        frame, scene, st = self._get()
        return cv2.imencode(".jpg", self.V.draw_debug(frame, scene, st), [cv2.IMWRITE_JPEG_QUALITY, 80])[1].tobytes()

    async def annotated(self) -> bytes:
        return await asyncio.to_thread(self.annotated_sync)

    async def reset(self) -> None:
        self._reset = True               # applied by the camera thread before its next frame

    def describe(self) -> str:
        w, h = self.size
        v = f", viewer http://127.0.0.1:{self.view_port}/frame.jpg" if self._srv else ""
        return f"{self.label} in-process ({w}x{h}{v})"

    def close(self) -> None:
        self._stop.set()
        if self._srv is not None:
            self._srv.shutdown()
            self._srv.server_close()
            self._srv = None
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self.cap.release()


def camera_spec(src: str):
    """'camera:N' -> N, 'camera:auto' -> 'auto'; anything else after 'camera:' is refused."""
    v = src.split(":", 1)[1].strip().lower()
    if v == "auto":
        return "auto"
    if not v.isdigit():
        raise ValueError(f"[vision].source = {src!r}: camera:<index> needs a number or auto, e.g. camera:2 or "
                         "camera:auto")
    return int(v)


def selector_from_config(gcfg: dict, capture_factory=None) -> CameraSelector:
    """[vision] top_reference (default top_ref.jpg, relative to guided.toml's folder), auto_fallback (default true),
    width/height; [paths] calib_file for the fallback check."""
    v = gcfg["vision"]
    base = Path(gcfg.get("_file") or Path(__file__).resolve().parent.parent / "guided.toml").parent
    ref = v.get("top_reference", "top_ref.jpg")
    ref = (Path(os.path.expanduser(ref)) if Path(os.path.expanduser(ref)).is_absolute() else base / ref) if ref else None
    calib = (gcfg.get("_paths") or {}).get("calib_file")
    return CameraSelector(TopViewScorer(ref, calib), int(v.get("width", 1280)), int(v.get("height", 720)),
                          auto_fallback=bool(v.get("auto_fallback", True)), capture_factory=capture_factory)


def open_camera_source(gcfg: dict, capture_factory=None) -> CameraVision:
    """[vision] source = "camera:N" or "camera:auto": this process opens the camera and runs Manasa's detector +
    tracker on it. Blocking (probing takes seconds): call through asyncio.to_thread."""
    from . import config as gconfig
    v = gcfg["vision"]
    spec = camera_spec(v["source"])
    V = gconfig.load_module("part_vision", gcfg["_paths"]["vision_file"])
    return CameraVision(V, spec, int(v.get("width", 1280)), int(v.get("height", 720)), int(v.get("view_port", 8765)),
                        capture_factory=capture_factory, selector=selector_from_config(gcfg, capture_factory))


def install(gc_module) -> None:
    """Point guided_cell's camera opener at open_camera_source above (camera:auto, self-healing camera:N).
    guided_cell.py's own opener only knows camera:N; this keeps that file unchanged."""
    if getattr(gc_module, "open_camera_source", None) is not open_camera_source:
        gc_module.open_camera_source = open_camera_source


def _open_cv_camera(index: int):
    if sys.platform == "darwin":
        cap = cv2.VideoCapture(index, cv2.CAP_AVFOUNDATION)
        if cap.isOpened():
            return cap
        cap.release()
    return cv2.VideoCapture(index)


def _start_viewer(cam: CameraVision, port: int):
    """/frame.jpg /raw.jpg /state from the camera's newest frame, for teammates. Bind failure only warns."""
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    class H(BaseHTTPRequestHandler):
        def do_GET(self):
            try:
                path = self.path.split("?")[0]
                if path == "/frame.jpg":
                    body, ctype = cam.annotated_sync(), "image/jpeg"
                elif path == "/raw.jpg":
                    body, ctype = cv2.imencode(".jpg", cam._get()[0])[1].tobytes(), "image/jpeg"
                elif path == "/state":
                    body, ctype = json.dumps(cam._get()[2], default=_jsonable).encode(), "application/json"
                else:
                    body, ctype = b"not found", "text/plain"
                code = 200 if ctype != "text/plain" else 404
            except VisionError as e:
                body, ctype, code = str(e).encode(), "text/plain", 503
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass
    try:
        srv = ThreadingHTTPServer(("0.0.0.0", port), H)
    except OSError as e:
        print(f"WARNING: camera viewer could not listen on port {port}: {e} (vision works without it)")
        return None
    srv.daemon_threads = True
    cam.view_port = srv.server_address[1]
    threading.Thread(target=srv.serve_forever, name="camera-viewer", daemon=True).start()
    return srv


def _jsonable(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def _decode(jpeg: bytes) -> np.ndarray:
    img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise VisionError("could not decode camera image")
    return img


def frac_to_px(corners, frac) -> list[float]:
    """Point on the lid given as a fraction of the box (x along the box x-axis, y along box y) -> pixel."""
    o, cx, cy = (np.array(corners[i], float) for i in (0, 1, 3))
    p = o + (cx - o) * frac[0] + (cy - o) * frac[1]
    return [float(p[0]), float(p[1])]


def _main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(prog="python -m cellguide.vision_source", description="probe cameras / save the "
                                 "top-view reference picture used by [vision] source = \"camera:auto\"")
    ap.add_argument("cmd", choices=["probe", "save-ref"])
    ap.add_argument("index", nargs="?", type=int)
    a = ap.parse_args(argv)
    from . import config as gconfig
    g = gconfig.load()
    sel = selector_from_config(g)
    if a.cmd == "probe":
        print(f"check: {sel.scorer.note}; threshold {sel.scorer.threshold:.2f}, lead {sel.scorer.margin:.2f}")
        for row in sel.table(sel.probe()).split("; "):
            print("  " + row)
        try:
            print(f"camera:auto would choose index {sel.choose(sel.last)['index']}")
        except VisionError as e:
            print(f"camera:auto would refuse: {e}")
        return 0
    if a.index is None:
        ap.error("save-ref needs the top camera's index (see: probe)")
    try:
        cap, frame = _open_checked(sel._open, a.index, sel.req, 4.0)
    except VisionError as e:
        print(e)
        return 1
    f2 = _read_settled(cap, sel.req, 1.5, 0.6)
    frame = frame if f2 is None else f2
    cap.release()
    dest = sel.scorer.reference_path
    if dest is None:
        ap.error("[vision] top_reference is empty")
    if (frame.shape[1], frame.shape[0]) != sel.req:
        print(f"camera {a.index} delivers {frame.shape[1]}x{frame.shape[0]}, not {sel.req}: not the top camera?")
        return 1
    cv2.imwrite(str(dest), frame)
    print(f"saved camera {a.index} -> {dest} (move the arm out of the view first for the best reference)")
    return 0


if __name__ == "__main__":
    sys.exit(_main())
