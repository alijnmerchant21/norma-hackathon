"""Find NEW dark ink marks on the lid by comparing a reference image (clear look before the dabs) with the
image taken at inspect (after them).  [marks] method = "diff".

Why: the real tool is a black fine-tip marker. Its dabs are 2-4 px dots, the same colour as the drawn hole rings
and the reference line, so a colour threshold cannot tell them apart. What is new is: a dot is there now and
was not there before.

Steps (all pure numpy / OpenCV, no asyncio; the cell calls detect() through asyncio.to_thread):
  1. registration, in the PART frame: a similarity before -> after from the hole centres both vision states see
     (or from the lid corners when fewer than 2 holes match), then refined by ECC (Euclidean) on the lid
     interior, so the rings and the reference line cancel even when the dabs pushed/rotated the box;
  2. lighting: the before image is scaled by the ratio of the two local backgrounds (median 21 px), so a
     brightness change or a soft shadow does not read as ink;
  3. new-dark = after is darker than the DARKEST before-pixel within `band` px (a tolerance band around existing
     strokes, so a residual misregistration of a ring edge does not make a false mark), by at least
     max(dark_min_contrast, 5 x noise); hysteresis (half that threshold) grows the dot; components with
     area dark_min_area_px..dark_max_area_px are marks; bigger ones are reported as large changes, never as marks;
  4. assign(): marks to targets, each mark to at most one target and vice versa, minimum total distance,
     only pairs within the search radius (scipy linear_sum_assignment).
Values are V (= max of B, G, R) levels: on a red lid V is the red channel, where black ink has the most contrast,
and bright annotation colours never read as "darker".
"""
from __future__ import annotations

import cv2
import numpy as np

DEFAULTS = dict(dark_min_contrast=25.0, dark_min_area_px=3, dark_max_area_px=150, dark_band_px=2,
                dark_min_ecc=0.6, dark_max_residual_px=3.0, dark_edge_margin_px=6,
                dark_max_local_px=3.0)


def params(mk: dict) -> dict:
    return {k: mk.get(k, v) for k, v in DEFAULTS.items()}


def _scale(st, frame_shape):
    w, h = st.get("image_size") or (frame_shape[1], frame_shape[0])
    return frame_shape[1] / float(w), frame_shape[0] / float(h)


def _pts(st, frame_shape):
    """Lid corners and visible holes {id: px} of a vision state, in frame px."""
    sx, sy = _scale(st, frame_shape)
    box = st.get("box") or {}
    corners = None if not box.get("corners_px") else np.array(box["corners_px"], float) * [sx, sy]
    holes = {str(h["id"]): np.array(h["px"], float) * [sx, sy] for h in st.get("holes", [])
             if h.get("visible", True)}
    return corners, holes


def _similarity(src, dst):
    """Least-squares similarity (Umeyama) mapping src -> dst; 2x3."""
    src, dst = np.asarray(src, float), np.asarray(dst, float)
    ms, md = src.mean(0), dst.mean(0)
    a, b = src - ms, dst - md
    C = b.T @ a / len(src)
    U, S, Vt = np.linalg.svd(C)
    d = np.sign(np.linalg.det(U @ Vt)) or 1.0
    D = np.diag([1.0, d])
    R = U @ D @ Vt
    var = (a ** 2).sum() / len(src)
    s = float(np.trace(np.diag(S) @ D) / var) if var > 1e-9 else 1.0
    t = md - s * R @ ms
    return np.c_[s * R, t]


def _apply(M, p):
    p = np.atleast_2d(np.asarray(p, float))
    return p @ M[:, :2].T + M[:, 2]


def _inv(M):
    return cv2.invertAffineTransform(np.asarray(M, np.float64))


def _vchan(img):
    return img.max(axis=2).astype(np.float32) if img.ndim == 3 else img.astype(np.float32)


def _face(img, quad, x0, y0, w, h):
    """Lid face mask in the crop (x0, y0, w, h): pixels of the lid's own colour (bright, same hue as the median
    inside the quad) -> largest blob -> convex hull (fills rings, dots, line). Leaves out the darker side walls
    the camera sees at an angle. Falls back to the quad when the colour test finds too little."""
    q = np.int32(np.round(np.asarray(quad) - [x0, y0]))
    poly = np.zeros((h, w), np.uint8)
    cv2.fillPoly(poly, [q], 255)
    hsv = cv2.cvtColor(img[y0:y0 + h, x0:x0 + w], cv2.COLOR_BGR2HSV)
    core = cv2.erode(poly, np.ones((9, 9), np.uint8)) > 0
    if core.sum() < 100:
        return poly
    H, S, Vv = (hsv[..., i].astype(np.int32) for i in range(3))
    hm, sm, vm = (float(np.median(c[core])) for c in (H, S, Vv))
    dh = np.abs(H - hm)
    dh = np.minimum(dh, 180 - dh)
    m = ((dh <= 12) & (S >= 0.5 * sm) & (Vv >= 0.7 * vm)).astype(np.uint8)
    m &= cv2.dilate(poly, np.ones((11, 11), np.uint8)) > 0
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m)
    if n < 2:
        return poly
    i = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    if stats[i, cv2.CC_STAT_AREA] < 0.5 * (poly > 0).sum():
        return poly
    ys, xs = np.nonzero(lab == i)
    hull = cv2.convexHull(np.c_[xs, ys].astype(np.int32))
    out = np.zeros((h, w), np.uint8)
    cv2.fillPoly(out, [hull], 255)
    return out


def register(before, after, st_b, st_a, p, ignore_a=None, ignore_b=None) -> dict:
    """Similarity before-px -> after-px from the vision states, refined by ECC on the lid. Returns
    {ok, M (2x3 before->after), ecc, residual_px, rotation_deg, shift_px, ecc_correction_px, why}."""
    cb, hb = _pts(st_b, before.shape)
    ca, ha = _pts(st_a, after.shape)
    common = sorted(set(hb) & set(ha))
    if len(common) >= 2:
        src, dst, how = [hb[i] for i in common], [ha[i] for i in common], f"{len(common)} holes"
    elif cb is not None and ca is not None:
        src, dst, how = list(cb), list(ca), "lid corners"
    else:
        return {"ok": False, "why": "no lid pose in one of the two images"}
    M0 = _similarity(src, dst)
    res0 = float(np.mean(np.linalg.norm(_apply(M0, src) - np.asarray(dst), axis=1)))
    if ca is None:
        return {"ok": False, "why": "no lid corners in the inspect image"}

    # ECC on a crop around the lid in the after image; before is first warped by M0
    H, W = after.shape[:2]
    x0, y0 = np.floor(ca.min(0) - 15).astype(int)
    x1, y1 = np.ceil(ca.max(0) + 15).astype(int)
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(W, x1), min(H, y1)
    if x1 - x0 < 20 or y1 - y0 < 20:
        return {"ok": False, "why": "lid too small or outside the image"}
    o = np.array([x0, y0], float)
    Va, Vb = _vchan(after), _vchan(before)
    bw = cv2.warpAffine(Vb, M0, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    ta = cv2.GaussianBlur(Va[y0:y1, x0:x1], (0, 0), 1.5)
    ib = cv2.GaussianBlur(bw[y0:y1, x0:x1], (0, 0), 1.5)
    mask = np.zeros(ta.shape, np.uint8)
    cv2.fillPoly(mask, [np.int32(np.round(ca - o))], 255)
    mask = cv2.erode(mask, np.ones((9, 9), np.uint8))
    if ignore_a is not None:
        mask[ignore_a[y0:y1, x0:x1] > 0] = 0
    if ignore_b is not None:
        ib_ign = cv2.warpAffine(ignore_b.astype(np.uint8), M0, (W, H), flags=cv2.INTER_NEAREST)
        mask[ib_ign[y0:y1, x0:x1] > 0] = 0
    Wm = np.eye(2, 3, dtype=np.float32)
    try:
        cc, Wm = cv2.findTransformECC(ta, ib, Wm, cv2.MOTION_EUCLIDEAN,
                                      (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 200, 1e-6), mask, 5)
    except cv2.error as e:
        return {"ok": False, "why": f"ECC refinement did not converge ({str(e).splitlines()[-1][:80]})",
                "residual_px": round(res0, 2), "initial_from": how}
    # after px p -> before px:  M0^-1 ( W (p - o) + o )
    Wf = np.vstack([Wm.astype(float), [0, 0, 1]])
    T = np.array([[1, 0, -o[0]], [0, 1, -o[1]], [0, 0, 1.0]])
    Ti = np.array([[1, 0, o[0]], [0, 1, o[1]], [0, 0, 1.0]])
    M0i = np.vstack([_inv(M0), [0, 0, 1]])
    A2B = (M0i @ Ti @ Wf @ T)[:2]
    M = _inv(A2B)
    corr = float(np.hypot(*Wm[:, 2]))
    res = float(np.mean(np.linalg.norm(_apply(M, src) - np.asarray(dst), axis=1)))
    ang = float(np.degrees(np.arctan2(M[1, 0], M[0, 0])))
    out = {"ok": True, "M": M, "A2B": A2B, "ecc": round(float(cc), 3), "initial_from": how,
           "residual_px": round(res, 2), "initial_residual_px": round(res0, 2),
           "ecc_correction_px": round(corr, 2), "rotation_deg": round(ang, 1),
           "lid_shift_px": [round(float(v), 1) for v in ca.mean(0) - _apply(A2B, ca.mean(0))[0]]}
    if cc < float(p["dark_min_ecc"]):
        out.update(ok=False, why=f"lid images do not match well enough (ECC {cc:.2f} < {p['dark_min_ecc']})")
    elif res > float(p["dark_max_residual_px"]):
        out.update(ok=False, why=f"hole centres disagree with the alignment by {res:.1f} px "
                                 f"(> {p['dark_max_residual_px']} px): the part changed or a hole was mis-detected")
    return out


def detect(before, after, st_b, st_a, mk: dict, ignore_a=None, ignore_b=None) -> dict:
    """New dark marks in `after` (frame px of `after`). ignore_a / ignore_b: optional masks (non-zero = ignore) of
    pixels to leave out in each image, e.g. drawn overlays. Returns {ok, why?, registration, marks: [{px, area_px,
    contrast}], large_changes: [...], noise, threshold, vis (uint8 diff picture)}."""
    p = params(mk)
    reg = register(before, after, st_b, st_a, p, ignore_a, ignore_b)
    regj = {k: v for k, v in reg.items() if k not in ("M", "A2B")}
    if not reg["ok"]:
        return {"ok": False, "why": reg.get("why"), "registration": regj, "marks": [], "large_changes": []}
    H, W = after.shape[:2]
    ca, _ = _pts(st_a, after.shape)
    x0, y0 = np.floor(ca.min(0) - 25).astype(int)
    x1, y1 = np.ceil(ca.max(0) + 25).astype(int)
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(W, x1), min(H, y1)
    Va = _vchan(after)[y0:y1, x0:x1]
    A2B = reg["A2B"].copy()
    A2B[:, 2] += A2B[:, :2] @ [x0, y0]                       # crop px -> before px
    Vb = cv2.warpAffine(_vchan(before), A2B, (x1 - x0, y1 - y0), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                        borderMode=cv2.BORDER_REPLICATE)
    lid = _face(after, ca, x0, y0, x1 - x0, y1 - y0)
    cb, _ = _pts(st_b, before.shape)
    if cb is not None:                                       # only where the lid is in BOTH images
        Hb, Wb = before.shape[:2]
        lb = np.zeros((Hb, Wb), np.uint8)
        bx0, by0 = np.maximum(np.floor(cb.min(0) - 25).astype(int), 0)
        bx1, by1 = np.minimum(np.ceil(cb.max(0) + 25).astype(int), [Wb, Hb])
        lb[by0:by1, bx0:bx1] = _face(before, cb, bx0, by0, bx1 - bx0, by1 - by0)
        lid &= cv2.warpAffine(lb, A2B, (x1 - x0, y1 - y0), flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP)
    em = int(p["dark_edge_margin_px"])
    lid = cv2.erode(lid, np.ones((2 * em + 1, 2 * em + 1), np.uint8)) if em > 0 else lid
    if ignore_a is not None:
        lid[ignore_a[y0:y1, x0:x1] > 0] = 0
    if ignore_b is not None:
        lid[cv2.warpAffine(ignore_b.astype(np.uint8), A2B, (x1 - x0, y1 - y0),
                           flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP) > 0] = 0
    # local alignment on each hole ring (the lid is not perfectly rigid/flat): a small translation per ring
    local = {}
    _, ha = _pts(st_a, after.shape)
    R = 24
    for hid, c in ha.items():
        cx, cy = int(round(c[0] - x0)), int(round(c[1] - y0))
        if not (R <= cx < Va.shape[1] - R and R <= cy < Va.shape[0] - R):
            continue
        win = np.s_[cy - R:cy + R, cx - R:cx + R]
        m = lid[win].copy()
        if (m > 0).mean() < 0.5:
            continue
        Wt = np.eye(2, 3, dtype=np.float32)
        try:
            cc, Wt = cv2.findTransformECC(cv2.GaussianBlur(Va[win], (0, 0), 1.0), cv2.GaussianBlur(Vb[win], (0, 0), 1.0),
                                          Wt, cv2.MOTION_TRANSLATION,
                                          (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1e-5), m, 1)
        except cv2.error:
            continue
        t = Wt[:, 2].astype(float)
        if cc < 0.5 or np.hypot(*t) > float(p["dark_max_local_px"]):
            continue
        sh = cv2.warpAffine(Vb, np.float32([[1, 0, t[0]], [0, 1, t[1]]]), (Vb.shape[1], Vb.shape[0]),
                            flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REPLICATE)
        disc = np.zeros(Vb.shape, np.uint8)
        cv2.circle(disc, (cx, cy), R - 2, 255, -1)
        Vb = np.where(disc > 0, sh, Vb)
        local[hid] = [round(float(t[0]), 2), round(float(t[1]), 2)]
    # lighting: ratio of local backgrounds (a 21 px median ignores dots and thin strokes)
    bga = cv2.medianBlur(np.clip(Va, 0, 255).astype(np.uint8), 21).astype(np.float32)
    bgb = cv2.medianBlur(np.clip(Vb, 0, 255).astype(np.uint8), 21).astype(np.float32)
    Vb = Vb * np.clip(bga / np.maximum(bgb, 1.0), 0.5, 2.0)
    band = max(int(p["dark_band_px"]), int(np.ceil(reg["residual_px"])) if reg["residual_px"] > 1.0 else 0)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * band + 1, 2 * band + 1))
    Vb_min = cv2.erode(Vb, k) if band > 0 else Vb
    drop = (Vb_min - Va)
    inside = lid > 0
    d = (Vb - Va)[inside]
    noise = float(1.4826 * np.median(np.abs(d - np.median(d)))) if d.size else 0.0
    thr = max(float(p["dark_min_contrast"]), 5.0 * noise)
    lo = ((drop >= 0.5 * thr) & inside).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(lo, connectivity=8)
    peaks_img = (drop >= cv2.dilate(drop, np.ones((5, 5), np.uint8))) & (drop >= thr) & inside
    marks, large = [], []
    for i in range(1, n):
        sel = lab == i
        peak = float(drop[sel].max())
        if peak < thr or int(stats[i, cv2.CC_STAT_AREA]) < int(p["dark_min_area_px"]):
            continue
        if int(stats[i, cv2.CC_STAT_AREA]) > int(p["dark_max_area_px"]):
            ys, xs = np.nonzero(sel)
            large.append({"px": [round(float(xs.mean() + x0), 1), round(float(ys.mean() + y0), 1)],
                          "area_px": int(sel.sum()), "contrast": round(peak, 1)})
            continue
        for part in _split(sel, drop, peaks_img):
            area = int(part.sum())
            if area < int(p["dark_min_area_px"]):
                continue
            w = np.clip(drop[part], 0, None)
            ys, xs = np.nonzero(part)
            c = np.array([(xs * w).sum() / w.sum() + x0, (ys * w).sum() / w.sum() + y0])
            marks.append({"px": [round(float(c[0]), 1), round(float(c[1]), 1)], "area_px": area,
                          "contrast": round(float(drop[part].max()), 1)})
    vis = np.clip(128 + 2.0 * np.where(inside, Vb - Va, 0), 0, 255).astype(np.uint8)
    return {"ok": True, "registration": regj, "marks": marks, "large_changes": large,
            "noise": round(noise, 1), "threshold": round(thr, 1), "band_px": band, "ring_shift_px": local,
            "vis": vis, "vis_origin": [x0, y0]}


def _split(sel, drop, peaks_img, min_sep=3.0, valley=0.75) -> list:
    """Split one component into separate dots when it has peaks >= min_sep px apart with a valley between them
    shallower than `valley` x the lower peak (two dabs joined by a faint smear). Returns boolean masks."""
    ys, xs = np.nonzero(sel & peaks_img)
    if len(xs) <= 1:
        return [sel]
    order = np.argsort(-drop[ys, xs])
    kept = []
    for j in order:
        q = np.array([xs[j], ys[j]], float)
        if all(np.linalg.norm(q - k) >= min_sep for k in kept):
            kept.append(q)
    if len(kept) == 1:
        return [sel]
    parent = list(range(len(kept)))

    def root(i):
        while parent[i] != i:
            i = parent[i]
        return i
    for i in range(len(kept)):
        for j in range(i + 1, len(kept)):
            a, b = kept[i], kept[j]
            n = int(np.ceil(np.linalg.norm(b - a))) + 1
            line = [a + (b - a) * t for t in np.linspace(0, 1, n)]
            v = min(float(drop[int(round(p[1])), int(round(p[0]))]) for p in line)
            ha, hb = drop[int(a[1]), int(a[0])], drop[int(b[1]), int(b[0])]
            if v >= valley * min(ha, hb):
                parent[root(j)] = root(i)
    groups = sorted({root(i) for i in range(len(kept))})
    if len(groups) == 1:
        return [sel]
    ys, xs = np.nonzero(sel)
    P = np.c_[xs, ys].astype(float)
    K = np.array(kept)
    nearest = np.argmin(np.linalg.norm(P[:, None, :] - K[None], axis=2), axis=1)
    out = []
    for g in groups:
        idx = np.isin(nearest, [i for i in range(len(kept)) if root(i) == g])
        m = np.zeros_like(sel)
        m[ys[idx], xs[idx]] = True
        out.append(m)
    return out


def assign(targets_px, marks_px, mm_per_px, radius_mm) -> dict:
    """{target index: mark index}: each mark to at most one target, minimum total distance (mm), only pairs within
    radius_mm. mm_per_px: one scale per target."""
    if not len(targets_px) or not len(marks_px):
        return {}
    from scipy.optimize import linear_sum_assignment
    T = np.asarray(targets_px, float)[:, None, :]
    Mk = np.asarray(marks_px, float)[None, :, :]
    D = np.linalg.norm(T - Mk, axis=2) * np.asarray(mm_per_px, float)[:, None]
    BIG = 1e6
    C = np.where(D <= float(radius_mm), D, BIG)
    r, c = linear_sum_assignment(C)
    return {int(i): int(j) for i, j in zip(r, c) if C[i, j] < BIG}


def draw(frame, found: list, stray: list, tol_mm: float, lid_corners=None):
    """found: [(key, target_px, mark_px, miss_dist_mm)] ; stray: [mark_px]. Draws in place (frame px): a small
    circle on the target, a line to the mark, a cross on the mark; the misses in mm are listed in a column beside
    the lid (lid_corners) with a thin leader line to each mark, so labels never overlap on a small lid."""
    f = cv2.FONT_HERSHEY_SIMPLEX
    W = frame.shape[1]
    labels = []
    for key, t, m, dist in found:
        col = (0, 200, 0) if dist <= tol_mm else (0, 0, 230)
        ti, mi = tuple(int(round(v)) for v in t), tuple(int(round(v)) for v in m)
        cv2.circle(frame, ti, 4, (255, 255, 0), 1, cv2.LINE_AA)
        cv2.line(frame, ti, mi, col, 1, cv2.LINE_AA)
        cv2.drawMarker(frame, mi, col, cv2.MARKER_CROSS, 9, 2, cv2.LINE_AA)
        labels.append((mi, f"{key}: {dist:.1f} mm", col))
    if labels:
        q = np.asarray(lid_corners, float) if lid_corners is not None else np.array([m for m, *_ in labels], float)
        tw = max(cv2.getTextSize(t, f, 0.5, 1)[0][0] for _, t, _ in labels)
        x = int(q[:, 0].max() + 25)
        right = x + tw < W - 5
        if not right:
            x = int(q[:, 0].min() - 25 - tw)
        y = int(q[:, 1].min()) + 10
        for mi, txt, col in sorted(labels, key=lambda z: z[0][1]):
            y = max(y, mi[1])
            anchor = (x - 3, y - 5) if right else (x + tw + 3, y - 5)
            cv2.line(frame, mi, anchor, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(frame, txt, (x, y), f, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(frame, txt, (x, y), f, 0.5, col if col != (0, 0, 230) else (80, 80, 255), 1, cv2.LINE_AA)
            y += 20
    for m in stray:
        cv2.circle(frame, tuple(int(round(v)) for v in m), 6, (0, 165, 255), 1, cv2.LINE_AA)
    n_pass = sum(1 for *_, d in found if d <= tol_mm)
    txt = f"ink marks (diff): {len(found)} found, {n_pass} within {tol_mm:g} mm, {len(stray)} stray (orange)"
    cv2.putText(frame, txt, (20, 30), f, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(frame, txt, (20, 30), f, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return frame
