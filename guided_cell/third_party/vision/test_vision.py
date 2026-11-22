"""
test_vision.py - simulate interruptions on the real test images and check the
tracker's behaviour (stable IDs, line hidden, arm over box, box gone).

    python test_vision.py            # prints a timeline, writes output/sim_*.png
"""
import os
import cv2
import numpy as np
import vision as V

IMG = "images/test_{}.jpeg"
os.makedirs("output/sim", exist_ok=True)


def box_color(frame, scene):
    """Median colour of the box face (to 'paint out' marks)."""
    m = np.zeros(frame.shape[:2], np.uint8)
    cv2.fillPoly(m, [np.int32(scene["_face_px"])], 255)
    return [int(v) for v in np.median(frame[m > 0], axis=0)]


def hide_hole(frame, scene, hole_id):
    f = frame.copy()
    h = next(h for h in scene["holes"] if h["id"] == hole_id)
    cv2.circle(f, (int(h["px"][0]), int(h["px"][1])), int(h["radius_px"] * 1.6), box_color(frame, scene), -1)
    return f


def hide_line(frame, scene):
    f = frame.copy()
    cv2.fillPoly(f, [np.int32(scene["_ref_line_pts"]).reshape(-1, 1, 2)], box_color(frame, scene))
    pts = np.int32(scene["_ref_line_pts"])
    c = pts.mean(0)
    big = np.int32(c + (pts - c) * 1.8)
    cv2.fillPoly(f, [big], box_color(frame, scene))
    return f


def arm_over_box(frame, scene):
    """Dark 'arm' covering half of the box (like the gripper over it while dabbing)."""
    f = frame.copy()
    c = scene["box"]["corners_px"]
    o, x, xy, y = (np.array(p) for p in c)
    poly = np.int32([o + (x - o) * 0.45 - (y - o) * 0.5, x + (x - o) * 0.3 - (y - o) * 0.5,
                     xy + (x - o) * 0.3 + (y - o) * 0.5, y + (x - o) * 0.45 + (y - o) * 0.5])
    cv2.fillPoly(f, [poly], (25, 25, 25))
    return f


def no_box(frame, scene):
    f = frame.copy()
    m = np.zeros(frame.shape[:2], np.uint8)
    cv2.fillPoly(m, [np.int32(scene["_face_px"])], 255)
    m = cv2.dilate(m, np.ones((25, 25), np.uint8))
    f[m > 0] = np.median(frame[m == 0], axis=0)
    return f


def main():
    raw = {i: cv2.imread(IMG.format(i)) for i in (1, 2, 3)}
    det = {i: V.detect_scene(raw[i]) for i in raw}

    # (label, frame, time step in s)
    seq = [("clear view", raw[1], 0.1)] * 3 + [
        ("hole 0 hidden", hide_hole(raw[1], det[1], 0), 0.1),
        ("hole 1 hidden", hide_hole(raw[1], det[1], 1), 0.1),
        ("clear (rotated)", raw[2], 0.1),
        ("line hidden (just seen)", hide_line(raw[2], det[2]), 0.1),
        ("line hidden, rotated", hide_line(raw[3], det[3]), 0.1),
        ("clear (rotated)", raw[3], 0.1),
        ("arm over box", arm_over_box(raw[3], det[3]), 0.1),
        ("box gone", no_box(raw[3], det[3]), 0.5),
        ("box gone", no_box(raw[3], det[3]), 1.0),
        ("box gone (>2 s)", no_box(raw[3], det[3]), 1.0),
        ("box back", raw[3], 0.1),
        ("line hidden after long gap", hide_line(raw[1], det[1]), 3.0),
    ]
    tracker, t = V.HoleTracker(calib_path="__none__"), 1000.0
    ok_ids = True
    for k, (label, frame, dt) in enumerate(seq):
        t += dt
        scene = V.detect_scene(frame, frame_id=k)
        st = tracker.update(scene, now=t)
        cv2.imwrite(f"output/sim/sim_{k:02d}_{label.replace(' ', '_').replace('>', 'gt')}.png",
                    V.draw_debug(frame, scene, st))
        holes = " ".join(f'{h["id"]}{"" if h["visible"] else "*"}' for h in st["holes"])
        ev = ", ".join(f'{e["type"]}:{e["phase"]}' for e in st["events"])
        src = st["box"]["pose_source"] if st["box"] else "-"
        print(f"{k:2d} {label:28s} {st['status']:9s} safe={st['safe_to_act']!s:5s} "
              f"pose={src:5s} holes=[{holes}]  {ev}")
        # ID check: whenever a hole is visible, its box_frac must match the map entry
        if st["map_locked"] and st["box"] and st["box"]["pose_source"] == "live":
            vis = {h["id"]: h for h in st["holes"] if h["visible"]}
            for d in scene["holes"]:
                pass
            for hid, h in vis.items():
                m = next(m for m in tracker.map if m["id"] == hid)
                if abs(m["box_frac"][0] - h["box_frac"][0]) > 0.05:
                    ok_ids = False
    print("\nIDs stable:", ok_ids, "   (* = hidden, position predicted)")
    print("annotated frames in output/sim/")


if __name__ == "__main__":
    main()
