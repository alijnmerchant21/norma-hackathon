"""
capture_cams.py - grab and save frames from several USB cameras at once
(works on Windows and macOS; run it on the machine the cameras are plugged into).

    python capture_cams.py --list                 # which camera indices exist?
    python capture_cams.py --cams 0 1 2           # live tiled view of 3 cameras
    python capture_cams.py --cams 0 1 2 --every 2 # also auto-save every 2 s

Keys in the window:  s = save one frame from every camera
                     r = toggle auto-save (--every seconds, default 1)
                     q / Esc = quit
Frames go to  captures/cam<idx>_<time>.png  at full camera resolution.

NOTE (macOS): the first run asks for camera permission for the Terminal app.
If the Station app already holds a camera, opening it here may fail or show
black frames; close Station's view of that camera or use Station's own export.
"""

import argparse
import os
import time
import cv2
import numpy as np


def open_cam(idx, width=None, height=None):
    if os.name == "nt":
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(idx)
    if width:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    if height:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return cap


def list_cams(max_idx=8):
    for i in range(max_idx):
        cap = open_cam(i)
        ok, f = cap.read() if cap.isOpened() else (False, None)
        print(f"camera {i}: " + (f"OK  {f.shape[1]}x{f.shape[0]}" if ok else "-"))
        cap.release()


def tile(frames, h=360):
    out = []
    for i, f in frames:
        if f is None:
            f = np.zeros((h, int(h * 4 / 3), 3), np.uint8)
        s = h / f.shape[0]
        f = cv2.resize(f, (int(f.shape[1] * s), h))
        cv2.putText(f, f"cam {i}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        out.append(f)
    return np.hstack(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--cams", type=int, nargs="+", default=[0])
    ap.add_argument("--every", type=float, default=0, help="auto-save period in s (0 = off)")
    ap.add_argument("--width", type=int)
    ap.add_argument("--height", type=int)
    ap.add_argument("--outdir", default="captures")
    args = ap.parse_args()
    if args.list:
        list_cams()
        return

    os.makedirs(args.outdir, exist_ok=True)
    caps = {i: open_cam(i, args.width, args.height) for i in args.cams}
    for i, c in caps.items():
        print(f"camera {i}: {'opened' if c.isOpened() else 'FAILED'}")
    period = args.every or 1.0
    auto, last = args.every > 0, 0.0
    win = "cameras (s save, r auto-save, q quit)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    try:
        while True:
            frames = []
            for i, c in caps.items():
                ok, f = c.read()
                frames.append((i, f if ok else None))
            now = time.time()
            key = cv2.waitKey(1) & 0xFF
            if key == ord("s") or (auto and now - last >= period):
                last = now
                stamp = time.strftime("%Y%m%d_%H%M%S") + f"_{int(now * 1000) % 1000:03d}"
                for i, f in frames:
                    if f is not None:
                        cv2.imwrite(os.path.join(args.outdir, f"cam{i}_{stamp}.png"), f)
                print(f"[saved {sum(f is not None for _, f in frames)} frames @ {stamp}]")
            if key == ord("r"):
                auto = not auto
                print(f"[auto-save {'ON' if auto else 'OFF'} every {period}s]")
            if key in (ord("q"), 27):
                break
            vis = tile(frames)
            if auto:
                cv2.circle(vis, (vis.shape[1] - 20, 20), 8, (0, 0, 255), -1)
            cv2.imshow(win, vis)
    finally:
        for c in caps.values():
            c.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
