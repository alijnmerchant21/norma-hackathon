"""
calibrate_top.py - one-time TOP camera -> robot calibration (pixel -> robot x,y in mm).
Writes calib_top.json, which vision.py loads automatically (adds "robot_mm" to holes).

PROCEDURE (with Max, ~10 min)
  1. Fix the top camera. Don't move it afterwards (if it moves: recalibrate).
  2. Max jogs the pen tip to a point AT LID HEIGHT (e.g. touching the top of a box or
     a block as tall as the box), anywhere in the work area.
  3. Click the pen tip in the window, then type the robot's x y (mm) for that pose
     (from Max's forward kinematics) into this terminal.
  4. Repeat for >= 6 points spread over the whole area where boxes can be (corners +
     middle). 4 is the minimum; more points = more accurate.
  5. Press c: computes the homography, prints the error per point, asks for the lid
     height z, saves calib_top.json.

    python calibrate_top.py --camera 1          # live top camera
    python calibrate_top.py --image top.png     # or a saved frame (static setup)

Keys: c = compute + save, u = undo last point, q = quit
"""
import argparse
import json
import cv2
import numpy as np
import vision as V

pending = []


def on_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        pending.append((x, y))


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--camera")
    g.add_argument("--image")
    ap.add_argument("--out", default=V.CALIB_FILE)
    args = ap.parse_args()

    cap = V.open_capture(args.camera) if args.camera is not None else None
    still = cv2.imread(args.image) if args.image else None
    if args.image and still is None:
        raise SystemExit(f"could not read {args.image}")

    px_pts, robot_pts = [], []
    win = "calibrate: click pen tip, then type robot x y in terminal  (c save, u undo, q quit)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win, on_mouse)
    while True:
        if cap is not None:
            ok, frame = cap.read()
            if not ok:
                break
        else:
            frame = still.copy()
        vis = frame.copy()
        for i, (p, r) in enumerate(zip(px_pts, robot_pts)):
            cv2.drawMarker(vis, (int(p[0]), int(p[1])), (0, 0, 255), cv2.MARKER_CROSS, 20, 2)
            cv2.putText(vis, f"{i}: ({r[0]:.0f},{r[1]:.0f})", (int(p[0]) + 8, int(p[1]) - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.imshow(win, vis)
        key = cv2.waitKey(1) & 0xFF

        if pending:
            x, y = pending.pop(0)
            s = input(f"pixel ({x},{y})  ->  robot x y in mm (blank = skip): ").strip()
            if s:
                try:
                    rx, ry = (float(v) for v in s.replace(",", " ").split()[:2])
                    px_pts.append((x, y))
                    robot_pts.append((rx, ry))
                    print(f"  point {len(px_pts) - 1} added ({len(px_pts)} total)")
                except ValueError:
                    print("  could not parse, expected: x y")
        if key == ord("u") and px_pts:
            px_pts.pop()
            robot_pts.pop()
            print(f"  removed last point ({len(px_pts)} left)")
        if key in (ord("q"), 27):
            break
        if key == ord("c"):
            if len(px_pts) < 4:
                print("need at least 4 points")
                continue
            P, R = np.float64(px_pts), np.float64(robot_pts)
            H, _ = cv2.findHomography(P, R, 0)
            pred = cv2.perspectiveTransform(P.reshape(-1, 1, 2), H).reshape(-1, 2)
            err = np.linalg.norm(pred - R, axis=1)
            for i, e in enumerate(err):
                print(f"  point {i}: error {e:.1f} mm")
            print(f"  mean error {err.mean():.1f} mm, max {err.max():.1f} mm "
                  f"(redo points with large error; aim for < 3 mm)")
            z = input("lid height z in robot frame (mm), e.g. from Max: ").strip()
            calib = {"H": H.tolist(), "lid_z_mm": float(z or 0.0),
                     "n_points": len(px_pts), "mean_err_mm": float(err.mean()),
                     "px_points": px_pts, "robot_points": robot_pts}
            with open(args.out, "w") as f:
                json.dump(calib, f, indent=2)
            print(f"saved {args.out}")
    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
