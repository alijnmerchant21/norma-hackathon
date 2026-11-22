"""Find new ink marks in the inspect camera and measure how far they missed.

Method: difference between a reference image (taken by begin_part, before any
marking) and the current image; dark new blobs are marks. Each blob is
assigned to the nearest spot target within search_radius_mm.
"""

from __future__ import annotations

import cv2
import numpy as np


def _gray(jpeg: bytes) -> np.ndarray:
    arr = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_GRAYSCALE)
    if arr is None:
        raise ValueError("could not decode camera image")
    return cv2.GaussianBlur(arr, (3, 3), 0)


def find_marks(ref_jpeg: bytes, cur_jpeg: bytes, min_area: int = 6, darken: int = 50) -> list[tuple[float, float, int]]:
    """Return [(x_px, y_px, area_px)] of new dark blobs."""
    ref, cur = _gray(ref_jpeg), _gray(cur_jpeg)
    if ref.shape != cur.shape:
        raise ValueError(f"reference {ref.shape} and current {cur.shape} images differ in size")
    diff = cv2.subtract(ref, cur)                          # positive where it got darker
    _, mask = cv2.threshold(diff, darken, 255, cv2.THRESH_BINARY)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    n, _, stats, cents = cv2.connectedComponentsWithStats(mask)
    out = []
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area >= min_area:
            out.append((float(cents[i][0]), float(cents[i][1]), area))
    return out


def assign_marks(
    blobs: list[tuple[float, float, int]],
    targets_px: dict[str, list[float]],
    mm_per_px: float,
    search_radius_mm: float = 20.0,
) -> dict[str, dict]:
    """Match each spot to its closest blob. Returns {spot: {...}} for spots found."""
    found = {}
    used = set()
    for spot, (tx, ty) in sorted(targets_px.items()):
        best, best_d = None, None
        for k, (x, y, a) in enumerate(blobs):
            if k in used:
                continue
            d = float(np.hypot(x - tx, y - ty)) * mm_per_px
            if d <= search_radius_mm and (best_d is None or d < best_d):
                best, best_d = k, d
        if best is not None:
            used.add(best)
            x, y, a = blobs[best]
            found[spot] = {
                "mark_px": [round(x, 1), round(y, 1)],
                "miss_mm": [round((x - tx) * mm_per_px, 2), round((y - ty) * mm_per_px, 2)],
                "miss_dist_mm": round(best_d, 2),
                "area_px": a,
            }
    return found, [b for k, b in enumerate(blobs) if k not in used]
