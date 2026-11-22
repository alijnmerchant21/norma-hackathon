"""
vision.py  -  Box + hole detection, NO ArUco tag and NO hole template needed.

PIPELINE (per frame)
  1. Resize the frame so its long side is PROC_LONG_SIDE px (any input resolution).
  2. BOX FACE: colour-segment the frame for every colour in BOX_COLORS (HSV ranges),
     keep the biggest blobs as candidate box faces.
  3. MARKINGS: inside each candidate face, pixels clearly DARKER than the face
     (Otsu threshold computed on the face pixels only) are markings. Each marking blob is
       - a HOLE if it is round (filled dot OR hollow drawn circle), or
       - the REFERENCE LINE if it is long and thin.
     The candidate face with the most holes wins (so white robot parts, tape, walls
     etc. that happen to be the same colour as the box are ignored).
  4. POSE: from the face outline (minAreaRect) -> centre, size, angle. The outline alone
     is ambiguous by 180 deg (rectangle) or 90 deg (square); the reference line resolves
     it: the box x-axis points AWAY from the line, origin is the corner on the line side.
  5. HOLES get coordinates in the box frame (fraction of box size, and mm if BOX_SIZE_MM
     is set) and stable IDs: sorted row by row (box y), then along the row (box x).

  Works for any box size and any number of holes. Only requirements:
    * box face colour is in COLOR_RANGES_HSV and differs from the table,
    * holes/line are darker than the face.

  On top of that, HoleTracker (section 5) keeps hole IDs stable when holes / the line
  are hidden, holds the last pose while the arm covers the box, and emits events.

USAGE
    pip install opencv-python numpy
    python vision.py --image a.png                          # one image -> full JSON
    python vision.py --image f1.png f2.png f3.png           # image sequence through tracker
    python vision.py --camera 0                             # live window
    python vision.py --camera 1 --serve 8765                # live + HTTP for teammates
  HTTP: /state  /events?since=N  /frame.jpg  /raw.jpg  /reset
  Live keys: q/Esc quit, s save frame+state, r reset hole map (new box).
  Pixel coordinates are in the ORIGINAL frame; robot_mm needs calib_top.json
  (made once with calibrate_top.py).
"""

import argparse
import json
import os
import threading
import time
from collections import deque
import numpy as np
import cv2

# ----------------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------------
PROC_LONG_SIDE = 1280        # working resolution (long side, px)

# Box colours to try, and their HSV ranges (OpenCV: H 0-180, S/V 0-255).
# Red wraps around H=0 so it needs two ranges. Keep red away from H~10-25 (cardboard).
BOX_COLORS = ["red"]         # only red boxes are used; add "white"/"blue"/... if that changes
COLOR_RANGES_HSV = {
    "white": [((0, 0, 160), (180, 60, 255))],
    "red":   [((0, 110, 60), (7, 255, 255)), ((165, 110, 60), (180, 255, 255))],
    "blue":  [((95, 90, 50), (130, 255, 255))],
    "green": [((40, 70, 50), (85, 255, 255))],
}
BOX_AREA_MIN_FRAC = 0.001    # face must be >= 0.1% of the frame ...
BOX_AREA_MAX_FRAC = 0.70     # ... and <= 70%
FACE_CANDIDATES   = 3        # biggest blobs per colour that get checked for holes
BOX_FILL_MIN      = 0.60     # face area / its rotated bounding-rect area (solid shape)
BOX_ASPECT_MAX    = 6.0      # long / short side of the face
BOX_MIN_HOLES     = 1        # a red blob only counts as THE box if it has >= this many holes

# Hole / reference-line classification (all relative to the box face -> scale-free)
DOT_AREA_MIN_FRAC = 0.002    # hole area / face area
DOT_AREA_MAX_FRAC = 0.10
DOT_CIRC_MIN      = 0.55     # area / enclosing-circle area (1.0 = perfect disc)
DOT_THRESH        = 0        # 0 => Otsu on face pixels; else fixed gray 0-255
REF_LINE_MIN_ASPECT   = 3.0  # long/short side of the line's rotated rect
REF_LINE_MIN_LEN_FRAC = 0.30 # line length >= 30% of the face's short side

# Real size of the box TOP FACE in mm: (long side, short side). None => no mm output
# (you still get px and box-fraction coordinates). Example: BOX_SIZE_MM = (90.0, 45.0)
BOX_SIZE_MM = None

# Colour of the dab used by check_mark(). Must differ from box colour AND hole colour,
# e.g. "red" ink on white boxes, "blue"/"green" ink on red boxes.
MARK_COLOR = "blue"

# --- Tracking (stable IDs, occlusion, events) --------------------------------
TRACK_LOCK_FRAMES = 3        # consistent frames (line visible) needed to lock the hole map
HOLE_MATCH_TOL    = 0.25     # max hole match distance, in units of the box SHORT side
POSE_HOLD_S       = 2.0      # how long the last good pose may be reused while the box /
                             # line / holes are hidden (e.g. arm over the box). Then STOP.
BOX_SIZE_TOL      = 0.15     # face size change vs. locked size => box partially hidden
BOX_MOVING_FRAC_S = 0.25     # centre speed > 25% of box length per second => BOX_MOVING

# --- Camera -> robot calibration (written by calibrate_top.py) ---------------
CALIB_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "calib_top.json")

# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def _odd(n):
    n = max(3, int(round(n)))
    return n if n % 2 else n + 1

def resize_for_processing(frame, long_side=PROC_LONG_SIDE):
    """Resize so the longer side == long_side. Returns (small, scale);
    original_px = small_px / scale."""
    h, w = frame.shape[:2]
    scale = long_side / float(max(h, w))
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    return cv2.resize(frame, (int(round(w * scale)), int(round(h * scale))),
                      interpolation=interp), scale

def color_mask(hsv, color):
    m = None
    for lo, hi in COLOR_RANGES_HSV[color]:
        r = cv2.inRange(hsv, lo, hi)
        m = r if m is None else cv2.bitwise_or(m, r)
    return m

# ----------------------------------------------------------------------------
# 1. BOX FACE CANDIDATES + MARKINGS
# ----------------------------------------------------------------------------
def _face_candidates(hsv, color):
    H, W = hsv.shape[:2]
    m = color_mask(hsv, color)
    k = _odd(max(H, W) * 0.006)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    lo, hi = BOX_AREA_MIN_FRAC * H * W, BOX_AREA_MAX_FRAC * H * W
    out = []
    for c in cnts:
        a = cv2.contourArea(c)
        if not (lo < a < hi):
            continue
        (_, _), (rw, rh), _ = cv2.minAreaRect(c)
        if min(rw, rh) < 1:
            continue
        # a box face is a solid, compact shape - rejects red wires, thin red parts etc.
        if a / (rw * rh) < BOX_FILL_MIN or max(rw, rh) / min(rw, rh) > BOX_ASPECT_MAX:
            continue
        out.append(c)
    return sorted(out, key=cv2.contourArea, reverse=True)[:FACE_CANDIDATES]

def _markings(val, face):
    """Return (holes, ref_line) inside one face, in working px.
    val = HSV V channel (brightness; black ink is dark on ANY face colour).
    holes = [(x, y, r)], ref_line = (cx, cy, length, corner_pts) or None."""
    face_area = cv2.contourArea(face)
    (_, _), (fw, fh), _ = cv2.minAreaRect(face)
    short_side = max(1.0, min(fw, fh))
    mask = np.zeros(val.shape, np.uint8)
    cv2.drawContours(mask, [cv2.convexHull(face)], -1, 255, -1)  # hull: include markings
    mask = cv2.erode(mask, np.ones((_odd(short_side * 0.08),) * 2, np.uint8))  # off the edges
    if cv2.countNonZero(mask) < 20:
        return [], None
    border = cv2.subtract(mask, cv2.erode(mask, np.ones((3, 3), np.uint8)))
    if DOT_THRESH <= 0:
        # "darker than the local background": robust to shading / uneven light
        bg = cv2.medianBlur(val, _odd(short_side * 0.35))
        diff = cv2.subtract(bg, val)
        d = diff[mask > 0].astype(np.float32)
        th = (diff > max(8.0, float(np.median(d) + 1.0 * d.std()))) & (mask > 0)
    else:
        th = (val < DOT_THRESH) & (mask > 0)
    th = th.astype(np.uint8) * 255
    ell = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (_odd(short_side * 0.06),) * 2)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, ell)              # join broken drawn rings
    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    a_min, a_max = DOT_AREA_MIN_FRAC * face_area, DOT_AREA_MAX_FRAC * face_area
    holes, line = [], None
    blob = np.zeros(val.shape, np.uint8)
    for c in cnts:
        blob[:] = 0
        cv2.drawContours(blob, [c], -1, 255, -1)
        if cv2.countNonZero(cv2.bitwise_and(blob, border)) > 0:  # touches face edge -> shading
            continue
        (rcx, rcy), (rw, rh), _ = cv2.minAreaRect(c)
        long_, short_ = max(rw, rh), max(1.0, min(rw, rh))
        if long_ / short_ >= REF_LINE_MIN_ASPECT and long_ >= REF_LINE_MIN_LEN_FRAC * short_side:
            if line is None or long_ > line[2]:
                pts = cv2.boxPoints(cv2.minAreaRect(c))
                line = (rcx, rcy, long_, pts)
            continue
        area = cv2.contourArea(c)               # external contour -> hollow circles count full
        if not (a_min < area < a_max):
            continue
        (x, y), r = cv2.minEnclosingCircle(c)
        if area / (np.pi * r * r + 1e-9) >= DOT_CIRC_MIN:
            holes.append((float(x), float(y), float(r)))
    return holes, line

def find_box(frame):
    """Best box face in the frame. Returns dict in ORIGINAL px, or None."""
    small, scale = resize_for_processing(frame)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    val = cv2.GaussianBlur(hsv[:, :, 2], (3, 3), 0)
    best, best_score = None, None
    for color in BOX_COLORS:
        for face in _face_candidates(hsv, color):
            holes, line = _markings(val, face)
            if len(holes) < BOX_MIN_HOLES:
                continue
            score = (len(holes), line is not None, cv2.contourArea(face))
            if best_score is None or score > best_score:
                best_score = score
                best = dict(color=color, face=face, holes=holes, line=line)
    if best is None:
        return None
    s = 1.0 / scale                                   # working px -> original px
    rect = cv2.minAreaRect(best["face"])
    best["rect"] = ((rect[0][0] * s, rect[0][1] * s), (rect[1][0] * s, rect[1][1] * s), rect[2])
    best["holes"] = [(x * s, y * s, r * s) for x, y, r in best["holes"]]
    if best["line"] is not None:
        cx, cy, L, pts = best["line"]
        best["line"] = (cx * s, cy * s, L * s, pts * s)
    best["face"] = (best["face"].astype(np.float64) * s)
    return best

# ----------------------------------------------------------------------------
# 2. BOX FRAME  (pose from outline, orientation from reference line)
# ----------------------------------------------------------------------------
def box_frame(box):
    (cx, cy), (w, h), _ = box["rect"]
    c = np.array([cx, cy])
    pts = cv2.boxPoints(box["rect"])
    e1, e2 = pts[1] - pts[0], pts[2] - pts[1]
    a_long, a_short = (e1, e2) if np.linalg.norm(e1) >= np.linalg.norm(e2) else (e2, e1)
    long_px, short_px = float(np.linalg.norm(a_long)), float(np.linalg.norm(a_short))
    a_long, a_short = a_long / long_px, a_short / short_px

    if box["line"] is not None:                       # x-axis points away from the line
        d = c - np.array(box["line"][:2])
        pl, ps = float(d @ a_long), float(d @ a_short)
        ex = a_long * np.sign(pl or 1) if abs(pl) >= abs(ps) else a_short * np.sign(ps or 1)
        resolved = True
    else:                                             # fallback: long side, pointing right/down
        ex = a_long if (a_long[0] > 1e-6 or (abs(a_long[0]) <= 1e-6 and a_long[1] > 0)) else -a_long
        resolved = False
    ey = np.array([-ex[1], ex[0]])                    # +90 deg in image coords (y down)
    x_is_long = abs(ex @ a_long) > 0.5
    len_x, len_y = (long_px, short_px) if x_is_long else (short_px, long_px)
    origin = c - ex * len_x / 2 - ey * len_y / 2

    mm_per_px, size_mm = None, None
    if BOX_SIZE_MM is not None:
        L_mm, S_mm = max(BOX_SIZE_MM), min(BOX_SIZE_MM)
        mm_per_px = 0.5 * (L_mm / long_px + S_mm / short_px)
        size_mm = (L_mm, S_mm) if x_is_long else (S_mm, L_mm)

    # image px -> box-frame px:  [bx, by, 1] = H @ [u, v, 1]
    H_img2box = np.array([[ex[0], ex[1], -ex @ origin],
                          [ey[0], ey[1], -ey @ origin],
                          [0, 0, 1.0]])
    return dict(center=c, ex=ex, ey=ey, origin=origin, len_x=len_x, len_y=len_y,
                resolved=resolved, mm_per_px=mm_per_px, size_mm=size_mm, H_img2box=H_img2box)

def _sort_holes(holes_box, len_y):
    """Stable IDs: group into rows by box-y, then sort each row by box-x."""
    if not holes_box:
        return []
    r_mean = np.mean([h["r"] for h in holes_box])
    tol = max(1.5 * r_mean, 0.08 * len_y)
    rows = []
    for h in sorted(holes_box, key=lambda h: h["by"]):
        if rows and abs(h["by"] - rows[-1][-1]["by"]) <= tol:
            rows[-1].append(h)
        else:
            rows.append([h])
    return [h for row in rows for h in sorted(row, key=lambda h: h["bx"])]

# ----------------------------------------------------------------------------
# 3. MAIN ENTRY
# ----------------------------------------------------------------------------
def detect_scene(frame, frame_id=0):
    h, w = frame.shape[:2]
    scene = {"timestamp": time.time(), "frame_id": int(frame_id),
             "image_size": [int(w), int(h)], "box_detected": False,
             "box": None, "holes": []}
    box = find_box(frame)
    if box is None:
        return scene
    F = box_frame(box)

    hb = []
    for x, y, r in box["holes"]:
        bx, by, _ = F["H_img2box"] @ np.array([x, y, 1.0])
        hb.append(dict(x=x, y=y, r=r, bx=float(bx), by=float(by)))
    holes = []
    for i, hh in enumerate(_sort_holes(hb, F["len_y"])):
        fx, fy = hh["bx"] / F["len_x"], hh["by"] / F["len_y"]
        holes.append({
            "id": i,
            "px": [round(hh["x"], 1), round(hh["y"], 1)],
            "radius_px": round(hh["r"], 1),
            "box_frac": [round(fx, 4), round(fy, 4)],          # 0..1 along box x / y
            "box_mm": None if F["size_mm"] is None else
                      [round(fx * F["size_mm"][0], 2), round(fy * F["size_mm"][1], 2)],
        })

    o, ex, ey = F["origin"], F["ex"], F["ey"]
    corners = [o, o + ex * F["len_x"], o + ex * F["len_x"] + ey * F["len_y"], o + ey * F["len_y"]]
    line = box["line"]
    scene.update({
        "box_detected": True,
        "box": {
            "color": box["color"],
            "center_px": [round(float(v), 1) for v in F["center"]],
            "size_px": [round(F["len_x"], 1), round(F["len_y"], 1)],      # along box x, y
            "corners_px": [[round(float(p[0]), 1), round(float(p[1]), 1)] for p in corners],
            "angle_deg": round(float(np.degrees(np.arctan2(ex[1], ex[0]))), 1),
            "orientation_resolved": F["resolved"],
            "ref_line_px": None if line is None else [round(line[0], 1), round(line[1], 1)],
            "mm_per_px": None if F["mm_per_px"] is None else round(F["mm_per_px"], 4),
            "size_mm": None if F["size_mm"] is None else list(F["size_mm"]),
        },
        "holes": holes,
    })
    scene["_H_img2box"] = F["H_img2box"]
    scene["_face_px"] = box["face"]
    scene["_ref_line_pts"] = None if line is None else line[3]
    return scene

# ----------------------------------------------------------------------------
# 4. MARK CHECK  (distance of the ink dab from the expected point, mm or px)
# ----------------------------------------------------------------------------
def check_mark(frame, expected_px, mm_per_px=None, search_radius_px=60, color=MARK_COLOR):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = color_mask(hsv, color)
    win = np.zeros(mask.shape, np.uint8)
    cv2.circle(win, (int(expected_px[0]), int(expected_px[1])), int(search_radius_px), 255, -1)
    mask = cv2.bitwise_and(mask, win)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    (mx, my), _ = cv2.minEnclosingCircle(max(cnts, key=cv2.contourArea))
    d = float(np.hypot(mx - expected_px[0], my - expected_px[1]))
    return round(d * mm_per_px, 2) if mm_per_px else round(d, 1)


# ----------------------------------------------------------------------------
# 5. TRACKER  - stable hole IDs, occlusion handling, events
# ----------------------------------------------------------------------------
# Per-frame detect_scene() knows nothing about the past. HoleTracker adds memory:
#   * HOLE MAP: once the box is seen clearly (line visible, same holes for
#     TRACK_LOCK_FRAMES frames) the hole layout is LOCKED: id -> position on the box.
#     Later frames are MATCHED to this map, so a hidden hole never renumbers the
#     others; it is reported as visible=False with its predicted pixel position.
#   * ORIENTATION when the line is hidden: (a) continuity with the last pose
#     (a box can't turn >90 deg between frames), else (b) whichever 180-deg flip
#     fits the hole map better (works when the hole layout is not symmetric).
#   * HELD POSE: if the box is hidden / partly covered / orientation unknown, the last
#     good pose is reused for up to POSE_HOLD_S (arm over the box while dabbing).
#   * EVENTS: conditions (NO_BOX, HOLES_HIDDEN, ...) raise a "start" event when they
#     appear and an "end" event when they clear; status = OK / WARN / STOP / SEARCHING.

CONDITION_LEVEL = {           # what each condition means for the robot
    "NO_BOX": "WARN",                 # becomes POSE_STALE (STOP) after POSE_HOLD_S
    "BOX_PARTIALLY_HIDDEN": "WARN",   # e.g. arm/hand over the box -> pose held
    "ORIENTATION_INFERRED": "WARN",   # line hidden, orientation from continuity/layout
    "ORIENTATION_LOST": "WARN",       # line hidden and could not infer -> pose held
    "HOLES_HIDDEN": "WARN",           # some holes not visible; positions predicted
    "UNEXPECTED_HOLES": "WARN",       # extra round marks not in the map (new box?)
    "BOX_MOVING": "WARN",             # re-target continuously
    "POSE_STALE": "STOP",             # no trustworthy pose for > POSE_HOLD_S -> stop
    "MAP_NOT_LOCKED": "SEARCHING",    # waiting for a clear view to lock the hole map
}


def _wrap180(a):
    return (a + 180.0) % 360.0 - 180.0


def _flip180(box, holes):
    """Same box seen with the opposite orientation (x-axis reversed)."""
    b = dict(box)
    c = box["corners_px"]
    b["corners_px"] = [c[2], c[3], c[0], c[1]]
    b["angle_deg"] = round(_wrap180(box["angle_deg"] + 180.0), 1)
    return b, [dict(h, box_frac=[round(1 - h["box_frac"][0], 4), round(1 - h["box_frac"][1], 4)])
               for h in holes]


def _match(map_holes, det_holes, size_px, tol=HOLE_MATCH_TOL):
    """Greedy nearest matching in box coordinates. Returns ({map_id: det_idx}, mean_err)."""
    lx, ly = size_px
    unit = max(1e-6, min(lx, ly))
    pairs = []
    for m in map_holes:
        for j, d in enumerate(det_holes):
            dist = np.hypot((m["box_frac"][0] - d["box_frac"][0]) * lx,
                            (m["box_frac"][1] - d["box_frac"][1]) * ly) / unit
            pairs.append((dist, m["id"], j))
    pairs.sort()
    used_m, used_d, out, errs = set(), set(), {}, []
    for dist, mid, j in pairs:
        if dist > tol or mid in used_m or j in used_d:
            continue
        used_m.add(mid)
        used_d.add(j)
        out[mid] = j
        errs.append(dist)
    return out, (float(np.mean(errs)) if errs else 1e9)


def _frac_to_px(corners, frac):
    o, cx, cy = (np.array(corners[i], float) for i in (0, 1, 3))
    p = o + (cx - o) * frac[0] + (cy - o) * frac[1]
    return [round(float(p[0]), 1), round(float(p[1]), 1)]


def load_calibration(path=CALIB_FILE):
    """calib_top.json: {"H": 3x3 pixel->robot-mm homography, "lid_z_mm": z}."""
    if not os.path.exists(path):
        return None
    with open(path) as f:
        c = json.load(f)
    return {"H": np.array(c["H"], dtype=np.float64), "lid_z_mm": float(c.get("lid_z_mm", 0.0))}


def px_to_robot(px, calib):
    if calib is None:
        return None
    p = cv2.perspectiveTransform(np.array([[px]], dtype=np.float64), calib["H"])[0, 0]
    return [round(float(p[0]), 1), round(float(p[1]), 1), round(calib["lid_z_mm"], 1)]


class HoleTracker:
    def __init__(self, calib_path=CALIB_FILE):
        self.calib = load_calibration(calib_path)
        self.event_log = deque(maxlen=500)       # all events, for /events?since=N
        self.seq = 0
        self.active = {}
        self.reset()

    def reset(self):
        """Forget the hole map (call when a NEW box is placed)."""
        self.map = None            # [{"id", "box_frac"}]
        self.map_size = None       # face size (px) at lock time
        self.lock_buf = []
        self.good = None           # last good pose: {"box", "t"}
        self._pending_events = [("MAP_RESET", "start", "hole map cleared, waiting for clear view")]

    # -- internals ---------------------------------------------------------
    def _try_lock(self, box, holes):
        if not holes:
            self.lock_buf = []
            return
        self.lock_buf.append((box, holes))
        self.lock_buf = self.lock_buf[-TRACK_LOCK_FRAMES:]
        if len(self.lock_buf) < TRACK_LOCK_FRAMES:
            return
        ref = self.lock_buf[0][1]
        ref_map = [{"id": h["id"], "box_frac": h["box_frac"]} for h in ref]
        for _, hs in self.lock_buf[1:]:
            m, _ = _match(ref_map, hs, box["size_px"])
            if len(hs) != len(ref) or len(m) != len(ref):
                return
        fr = np.mean([[h["box_frac"] for h in hs] for _, hs in self.lock_buf], axis=0)
        self.map = [{"id": h["id"], "box_frac": [round(float(f[0]), 4), round(float(f[1]), 4)]}
                    for h, f in zip(ref, fr)]
        self.map_size = list(box["size_px"])
        self._pending_events.append(("MAP_LOCKED", "start", f"{len(self.map)} holes locked"))

    def _resolve_orientation(self, box, holes, now):
        """Return (box, holes, how) with how in {'line', 'continuity', 'layout', None}."""
        if box["orientation_resolved"]:
            return box, holes, "line"
        fbox, fholes = _flip180(box, holes)
        if self.good is not None and now - self.good["t"] <= POSE_HOLD_S:
            d0 = abs(_wrap180(box["angle_deg"] - self.good["box"]["angle_deg"]))
            return (box, holes, "continuity") if d0 <= 90 else (fbox, fholes, "continuity")
        if self.map and holes:
            m0, e0 = _match(self.map, holes, box["size_px"])
            m1, e1 = _match(self.map, fholes, box["size_px"])
            if len(m0) != len(m1) or abs(e0 - e1) > 0.1:
                better0 = (len(m0), -e0) > (len(m1), -e1)
                return (box, holes, "layout") if better0 else (fbox, fholes, "layout")
        return box, holes, None

    def _set_conditions(self, conds):
        events = []
        for c, detail in conds.items():
            if c not in self.active:
                events.append((c, "start", detail))
            elif self.active[c] != detail and c in ("HOLES_HIDDEN", "UNEXPECTED_HOLES"):
                events.append((c, "update", detail))
        for c in self.active:
            if c not in conds:
                events.append((c, "end", ""))
        self.active = dict(conds)
        return events

    # -- main --------------------------------------------------------------
    def update(self, scene, now=None):
        now = scene["timestamp"] if now is None else now
        conds = {}
        box, holes, live = None, [], False

        if scene["box_detected"]:
            box, holes = scene["box"], scene["holes"]
            box, holes, how = self._resolve_orientation(box, holes, now)
            partial = (self.map_size is not None and
                       any(abs(a - b) / b > BOX_SIZE_TOL for a, b in zip(box["size_px"], self.map_size)))
            if partial:
                conds["BOX_PARTIALLY_HIDDEN"] = (f'size {box["size_px"][0]:.0f}x{box["size_px"][1]:.0f}px '
                                                 f'vs {self.map_size[0]:.0f}x{self.map_size[1]:.0f}px')
            if how is None:
                conds["ORIENTATION_LOST"] = "reference line hidden and orientation not inferable"
            elif how != "line":
                conds["ORIENTATION_INFERRED"] = f"reference line hidden, used {how}"
            live = how is not None and not partial
            if self.map is None and how == "line" and not partial:
                self._try_lock(box, holes)
        else:
            conds["NO_BOX"] = "no red box face found"

        # pose: live if trustworthy, else held from the last good frame
        if live:
            if self.good is not None:
                dt = max(1e-3, now - self.good["t"])
                c0, c1 = np.array(self.good["box"]["center_px"]), np.array(box["center_px"])
                speed = float(np.linalg.norm(c1 - c0) / dt)
                if speed > BOX_MOVING_FRAC_S * max(box["size_px"]):
                    conds["BOX_MOVING"] = f"{speed:.0f} px/s"
            self.good = {"box": box, "t": now}
            pose_box, pose_src = box, "live"
        elif self.good is not None and now - self.good["t"] <= POSE_HOLD_S:
            pose_box, pose_src = self.good["box"], "held"
        else:
            pose_box, pose_src = None, None
            if self.good is not None or scene["box_detected"]:
                conds["POSE_STALE"] = f"no trustworthy pose for > {POSE_HOLD_S}s"

        # holes: map-based IDs once locked, raw detection before that
        out_holes = []
        if self.map is None:
            conds["MAP_NOT_LOCKED"] = "need line + all holes visible for a few frames"
            if pose_src == "live":
                out_holes = [{"id": h["id"], "box_frac": h["box_frac"], "px": h["px"],
                              "visible": True, "robot_mm": px_to_robot(h["px"], self.calib)}
                             for h in holes]
        elif pose_box is not None:
            matched, unexpected = {}, 0
            if pose_src == "live":
                matched, _ = _match(self.map, holes, pose_box["size_px"])
                unexpected = len(holes) - len(matched)
            hidden = []
            for m in self.map:
                if m["id"] in matched:
                    px, vis = holes[matched[m["id"]]]["px"], True
                else:
                    px, vis = _frac_to_px(pose_box["corners_px"], m["box_frac"]), False
                    hidden.append(m["id"])
                out_holes.append({"id": m["id"], "box_frac": m["box_frac"], "px": px,
                                  "visible": vis, "robot_mm": px_to_robot(px, self.calib)})
            if hidden and pose_src == "live":
                conds["HOLES_HIDDEN"] = f"ids {hidden}"
            if unexpected > 0:
                conds["UNEXPECTED_HOLES"] = f"{unexpected} round marks not in hole map"

        events = self._pending_events + self._set_conditions(conds)
        self._pending_events = []
        ev_out = []
        for typ, phase, detail in events:
            self.seq += 1
            e = {"seq": self.seq, "t": round(now, 3), "frame_id": scene["frame_id"],
                 "type": typ, "phase": phase, "detail": detail}
            self.event_log.append(e)
            ev_out.append(e)

        levels = {CONDITION_LEVEL.get(c, "WARN") for c in conds}
        status = ("STOP" if "STOP" in levels else "SEARCHING" if "SEARCHING" in levels
                  else "WARN" if levels else "OK")
        safe = self.map is not None and pose_box is not None and status != "STOP"
        return {
            "frame_id": scene["frame_id"], "timestamp": round(now, 3),
            "image_size": scene["image_size"],
            "status": status, "safe_to_act": safe, "map_locked": self.map is not None,
            "box": None if pose_box is None else {
                "center_px": pose_box["center_px"], "angle_deg": pose_box["angle_deg"],
                "size_px": pose_box["size_px"], "corners_px": pose_box["corners_px"],
                "pose_source": pose_src,
                "pose_age_s": round(now - self.good["t"], 2) if self.good else None,
                "center_robot_mm": px_to_robot(pose_box["center_px"], self.calib)},
            "holes": out_holes,
            "conditions": [{"type": c, "level": CONDITION_LEVEL.get(c, "WARN"), "detail": d}
                           for c, d in conds.items()],
            "events": ev_out,
            "calibrated": self.calib is not None,
        }


# ----------------------------------------------------------------------------
# 6. DEBUG DRAW
# ----------------------------------------------------------------------------
STATUS_COLOR = {"OK": (0, 200, 0), "WARN": (0, 165, 255), "STOP": (0, 0, 255),
                "SEARCHING": (255, 200, 0)}


def draw_debug(frame, scene, state=None):
    img = frame.copy()
    h, w = img.shape[:2]
    k = max(h, w) / 1000.0
    fs, th = 0.6 * k, max(1, int(round(2 * k)))
    if scene["box_detected"]:
        cv2.polylines(img, [np.int32(scene["_face_px"])], True, (255, 255, 0), max(1, th // 2))
        if scene.get("_ref_line_pts") is not None:
            cv2.polylines(img, [np.int32(scene["_ref_line_pts"])], True, (255, 0, 255), th)
    pose = state["box"] if state else (scene["box"] if scene["box_detected"] else None)
    if pose is not None:
        c = pose["corners_px"]
        cv2.polylines(img, [np.int32(c)], True, (255, 0, 0), th)
        o, cx_, cy_ = (np.array(c[i]) for i in (0, 1, 3))
        cv2.arrowedLine(img, tuple(int(v) for v in o), tuple(int(v) for v in o + (cx_ - o) * 0.5),
                        (0, 0, 255), th)
        cv2.arrowedLine(img, tuple(int(v) for v in o), tuple(int(v) for v in o + (cy_ - o) * 0.5),
                        (0, 200, 0), th)
    holes = state["holes"] if state else scene["holes"]
    r = max(4, int(0.12 * min(pose["size_px"]))) if pose else int(10 * k)
    for hole in holes:
        u, v = int(hole["px"][0]), int(hole["px"][1])
        vis = hole.get("visible", True)
        col = (0, 200, 0) if vis else (0, 140, 255)
        cv2.circle(img, (u, v), r, col, th if vis else max(1, th // 2))
        cv2.putText(img, str(hole["id"]), (u + r + 2, v - r), cv2.FONT_HERSHEY_SIMPLEX, fs, col, th)
    if state:
        cv2.rectangle(img, (0, 0), (w, int(34 * k + 10)), (30, 30, 30), -1)
        txt = f'{state["status"]}  safe={state["safe_to_act"]}  holes={len(state["holes"])}'
        if state["box"]:
            txt += f'  angle={state["box"]["angle_deg"]:.0f}  pose={state["box"]["pose_source"]}'
        cv2.putText(img, txt, (int(10 * k), int(26 * k)), cv2.FONT_HERSHEY_SIMPLEX, fs,
                    STATUS_COLOR[state["status"]], th)
        for i, cnd in enumerate(state["conditions"]):
            cv2.putText(img, f'{cnd["type"]}: {cnd["detail"]}', (int(10 * k), int((64 + 26 * i) * k)),
                        cv2.FONT_HERSHEY_SIMPLEX, fs * 0.8,
                        STATUS_COLOR.get(cnd["level"], (0, 165, 255)), th)
    elif not scene["box_detected"]:
        cv2.putText(img, "NO BOX", (int(20 * k), int(40 * k)), cv2.FONT_HERSHEY_SIMPLEX, fs * 1.4,
                    (0, 0, 255), th)
    return img


def _public(scene):
    return {k: v for k, v in scene.items() if not k.startswith("_")}


def _json_default(o):
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


# ----------------------------------------------------------------------------
# 7. HTTP OUTPUT  - teammates read the latest state over the network
# ----------------------------------------------------------------------------
#   GET /state            latest tracker state (JSON)          <- Max, Claude
#   GET /events?since=N   events with seq > N (JSON list)      <- Claude event handler
#   GET /frame.jpg        latest annotated image               <- Claude "looks"
#   GET /raw.jpg          latest raw camera image
#   GET /reset            forget hole map (new box placed)     <- Claude / operator
class _Shared:
    def __init__(self):
        self.lock = threading.Lock()
        self.state, self.annot, self.raw = {}, b"", b""
        self.reset_requested = False


def start_server(port, shared, tracker):
    from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
    from urllib.parse import urlparse, parse_qs

    class Handler(BaseHTTPRequestHandler):
        def _send(self, code, body, ctype):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            u = urlparse(self.path)
            with shared.lock:
                if u.path == "/state":
                    body = json.dumps(shared.state, default=_json_default).encode()
                    return self._send(200, body, "application/json")
                if u.path == "/events":
                    since = int(parse_qs(u.query).get("since", ["0"])[0])
                    ev = [e for e in tracker.event_log if e["seq"] > since]
                    return self._send(200, json.dumps(ev).encode(), "application/json")
                if u.path == "/frame.jpg":
                    return self._send(200, shared.annot, "image/jpeg")
                if u.path == "/raw.jpg":
                    return self._send(200, shared.raw, "image/jpeg")
                if u.path == "/reset":
                    shared.reset_requested = True
                    return self._send(200, b'{"ok": true}', "application/json")
            self._send(404, b"not found", "text/plain")

        def log_message(self, *a):
            pass

    srv = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    print(f"[serving on http://0.0.0.0:{port}  ->  /state /events /frame.jpg /raw.jpg /reset]")
    return srv


# ----------------------------------------------------------------------------
# 8. LIVE VIDEO
# ----------------------------------------------------------------------------
def open_capture(camera, width=None, height=None):
    src = int(camera) if str(camera).isdigit() else camera      # index or URL/file
    if isinstance(src, int) and os.name == "nt":
        cap = cv2.VideoCapture(src, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise SystemExit(f"could not open camera/stream {camera!r}")
    if width:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    if height:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return cap


def run_live(camera=0, width=None, height=None, serve_port=None, show=True,
             print_every=1.0, display_max=1000):
    """Camera -> detect -> track -> show / print / serve.
    Keys: q/Esc quit, s save frame+state, r reset hole map (new box)."""
    cap = open_capture(camera, width, height)
    tracker, shared = HoleTracker(), _Shared()
    print(f"[calibration: {'loaded ' + CALIB_FILE if tracker.calib else 'none (robot_mm = null)'}]")
    if serve_port:
        start_server(serve_port, shared, tracker)
    win = "vision - live (q quit, s save, r reset map)"
    if show:
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    frame_id, last_print = 0, 0.0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("[stream ended or frame grab failed]")
                break
            if shared.reset_requested:
                tracker.reset()
                shared.reset_requested = False
            scene = detect_scene(frame, frame_id=frame_id)
            state = tracker.update(scene)
            vis = draw_debug(frame, scene, state)
            for e in state["events"]:
                print(f'  EVENT #{e["seq"]} {e["type"]} {e["phase"]} {e["detail"]}', flush=True)
            now = time.time()
            if now - last_print >= print_every:
                last_print = now
                hs = " ".join(f'{x["id"]}{"" if x["visible"] else "*"}@({x["px"][0]:.0f},{x["px"][1]:.0f})'
                              for x in state["holes"])
                print(f'[{frame_id:6d}] {state["status"]:9s} safe={state["safe_to_act"]!s:5s} '
                      f'holes: {hs or "-"}', flush=True)
            if serve_port:
                with shared.lock:
                    shared.state = state
                    shared.annot = cv2.imencode(".jpg", vis, [cv2.IMWRITE_JPEG_QUALITY, 80])[1].tobytes()
                    shared.raw = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])[1].tobytes()
            if show:
                hh, ww = vis.shape[:2]
                s = display_max / float(max(hh, ww))
                cv2.imshow(win, cv2.resize(vis, None, fx=s, fy=s) if s < 1 else vis)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == ord("r"):
                    tracker.reset()
                if key == ord("s"):
                    stamp = time.strftime("%Y%m%d_%H%M%S")
                    cv2.imwrite(f"live_{stamp}_raw.png", frame)
                    cv2.imwrite(f"live_{stamp}.png", vis)
                    with open(f"live_{stamp}.json", "w") as f:
                        json.dump(state, f, indent=2, default=_json_default)
                    print(f"[saved live_{stamp}_raw.png / .png / .json]")
                if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
                    break
            frame_id += 1
    finally:
        cap.release()
        if show:
            cv2.destroyAllWindows()


# ----------------------------------------------------------------------------
# 9. CLI
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Red box + hole detection and tracking.")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--image", nargs="+", help="one or more images (several = run as a sequence)")
    g.add_argument("--camera", help="live mode: camera index (0, 1, ...) or stream URL")
    ap.add_argument("--outdir", default="output", help="annotated images / JSON (image mode)")
    ap.add_argument("--fps", type=float, default=10.0, help="assumed fps for an image sequence")
    ap.add_argument("--width", type=int)
    ap.add_argument("--height", type=int)
    ap.add_argument("--serve", type=int, metavar="PORT", help="live: serve state over HTTP")
    ap.add_argument("--no-window", action="store_true", help="live: no display window")
    args = ap.parse_args()
    if args.camera is not None:
        run_live(args.camera, args.width, args.height, args.serve, show=not args.no_window)
        return
    os.makedirs(args.outdir, exist_ok=True)
    tracker, t0 = HoleTracker(), time.time()
    for i, path in enumerate(args.image):
        frame = cv2.imread(path)
        if frame is None:
            raise SystemExit(f"could not read {path}")
        scene = detect_scene(frame, frame_id=i)
        state = tracker.update(scene, now=t0 + i / args.fps)
        name = os.path.splitext(os.path.basename(path))[0]
        cv2.imwrite(os.path.join(args.outdir, f"{name}_annotated.png"), draw_debug(frame, scene, state))
        with open(os.path.join(args.outdir, f"{name}_state.json"), "w") as f:
            json.dump(state, f, indent=2, default=_json_default)
        if len(args.image) == 1:
            print(json.dumps(state, indent=2, default=_json_default))
        else:
            print(f'{name}: {state["status"]} safe={state["safe_to_act"]} '
                  f'holes={[(h["id"], h["visible"]) for h in state["holes"]]} '
                  f'events={[(e["type"], e["phase"]) for e in state["events"]]}')
    print(f"\n[annotated images + state JSON written to {args.outdir}/]")


if __name__ == "__main__":
    main()
