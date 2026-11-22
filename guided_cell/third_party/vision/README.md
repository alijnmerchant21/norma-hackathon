# Berlin Hackathon: Vision module (red box + hole detection)

This file is for whoever (person or AI coding agent) is connecting this vision module to the
arm's inverse kinematics (IK) and motion control.
Vision owner: Manasa. IK and arm owner: Max.

## What this module does

The top camera looks at the table. `vision.py` finds the **red box**, the **holes** drawn on
its lid (black hand-drawn circles), and a **reference line** drawn on one edge. It then outputs
each hole's position:

- in camera pixels, always, and
- in **robot millimetres `[x, y, z]`**, once the camera has been calibrated (see "Calibration").

Your side reads those robot-mm targets, solves IK, and moves the pen tip to each hole to dab
it. Vision decides **where** the holes are and **whether it is safe to move**. Your side
decides **how** to move. Dab order is chosen by a separate planner, owned by the Claude
teammate.

```
top camera -> vision.py ── /state (JSON over HTTP) ──> your IK / motion loop -> arm
                        └─ /events, /frame.jpg ─────> Claude planner (order, retries, stop)
```

## Files

| File | Purpose |
|---|---|
| `vision.py` | Everything for vision: detection, tracker, HTTP server, live mode, CLI. **Read `HoleTracker.update()` for the output format.** |
| `calibrate_top.py` | One-time camera-to-robot calibration. Writes `calib_top.json`. **Needs you: it uses the arm's forward-kinematics tip position.** |
| `capture_cams.py` | Lists cameras and saves frames from several cameras at once. |
| `test_vision.py` | Offline test on `images/test_*.jpeg` that simulates hidden holes, a hidden line, the arm over the box, and the box removed. |
| `images/` | Real top-camera test frames, 1280x720. |
| `output/` | Generated scratch output. Safe to delete. |

## Setup

```bash
pip install opencv-python numpy        # Python 3.9+; only these two dependencies
python test_vision.py                  # sanity check, should end with "IDs stable: True"
python capture_cams.py --list          # find the TOP camera index
python vision.py --camera <top_idx>    # live window: check that the box and holes are found
```

On macOS, the first run asks for camera permission for the Terminal. If the Station app is
already using a camera, OpenCV may fail to open it; close Station's view of that camera.

## Where it runs

**Everything runs on Max's Mac**: the Station, all cameras, the arm and this vision module.
The vision code does not need its own machine. Other machines (Manasa's laptop, the Claude
teammate) only *read* its output over Wi-Fi.

```
Max's Mac
  ├─ Station (arm + sensors)
  ├─ top camera ──> vision.py ──> localhost:8765 /state ──> IK / motion loop (same Mac)
  │                     └──────── <mac-ip>:8765 ─────────> Manasa (watch /frame.jpg),
  │                                                         Claude teammate (/state, /events)
```

### Getting frames: OpenCV directly, or through the Station

macOS normally lets two programs open the same camera, but some apps take exclusive access.
Check this first:

```bash
# with the Station running and showing the top camera:
python capture_cams.py --list
python vision.py --camera <top_idx>
```

- **If it opens:** use `--camera <top_idx>` as shown below. This is the simplest and usually
  the fastest option.
- **If it fails or shows black frames:** the Station holds the camera, so take frames from the
  Station instead. `vision.py` doesn't care where a frame comes from; it only needs a BGR numpy
  image:

```python
import vision
tracker = vision.HoleTracker()
for frame in station_top_camera_frames():   # whatever Station API / SDK gives you (convert to BGR uint8)
    state = tracker.update(vision.detect_scene(frame))
```

If the Station exposes the camera as a stream URL (MJPEG, RTSP, etc.), pass that instead of
an index: `python vision.py --camera "http://..."`.

## Integration: two options

### Option A: HTTP (recommended; keeps vision as a separate process)

Run vision on Max's Mac next to the Station:

```bash
python vision.py --camera <top_idx> --serve 8765            # add --no-window to run headless
```

The IK loop on the same Mac reads `http://localhost:8765/state`. Teammates on the same Wi-Fi
read `http://<mac-ip>:8765/...`; find the IP with `ipconfig getifaddr en0`. macOS may ask to
allow incoming connections for Python; allow it.

| Endpoint | Returns |
|---|---|
| `GET /state` | Latest tracker state (JSON, schema below). Poll at 10–30 Hz. |
| `GET /events?since=N` | Events with `seq > N`. |
| `GET /frame.jpg` / `GET /raw.jpg` | Latest annotated / raw image. |
| `GET /reset` | Forget the saved hole layout. **Call this when a new box is placed.** |

```python
import json, urllib.request
VISION = "http://localhost:8765"          # IK runs on the same Mac as vision

def get_state():
    with urllib.request.urlopen(VISION + "/state", timeout=0.5) as r:
        return json.load(r)
```

### Option B: in-process (vision inside the IK / Station script itself)

Use this if the motion code already has the camera frames, for example from the Station SDK.

```python
import cv2, vision
tracker = vision.HoleTracker()                  # loads calib_top.json if present
cap = vision.open_capture(TOP_CAMERA_INDEX)
ok, frame = cap.read()
state = tracker.update(vision.detect_scene(frame))   # same dict as GET /state
```

Call `tracker.update()` on **every** frame, in order. It keeps memory between frames.
Call `tracker.reset()` when a new box is placed.

## Output format (`/state`)

```jsonc
{
  "frame_id": 4,
  "timestamp": 1795280777.716,          // seconds (time.time() on the vision machine)
  "image_size": [1280, 720],
  "status": "OK",                       // "OK" | "WARN" | "STOP" | "SEARCHING"
  "safe_to_act": true,                  // THE flag to gate motion on (rules below)
  "map_locked": true,                   // hole layout saved; IDs are stable from here on
  "calibrated": true,                   // false => every robot_mm is null
  "box": {                              // null if there is no usable pose
    "center_px": [663.6, 198.5],
    "angle_deg": -68.7,                 // box x-axis in the image; x points AWAY from the reference line
    "size_px": [160.9, 79.4],           // along box x, box y
    "corners_px": [[..],[..],[..],[..]],// origin, +x, +x+y, +y
    "pose_source": "live",              // "live" = this frame | "held" = last good pose reused
    "pose_age_s": 0.0,                  // age of the pose; at most 2.0 s while held
    "center_robot_mm": [275.0, 16.6, 42.0]
  },
  "holes": [
    {"id": 0,                           // stable ID: numbered row by row, starting at the reference-line end
     "box_frac": [0.252, 0.491],        // position on the lid, as a 0..1 fraction of box length / width
     "px": [647.8, 235.3],              // image pixel (measured if visible, predicted if hidden)
     "visible": true,                   // false = hidden (e.g. by the pen), position predicted from the box pose
     "robot_mm": [276.1, 31.4, 42.0]}   // TARGET FOR IK: x, y from the calibration, z = lid_z_mm
  ],
  "conditions": [{"type": "HOLES_HIDDEN", "level": "WARN", "detail": "ids [0]"}],
  "events": [ /* events raised in THIS frame only; use /events?since=N for the full history */ ]
}
```

**Units and frames.** `robot_mm` uses **the frame and units of the robot points typed in
during calibration**. Use the same base frame your IK uses. `z` is the constant lid height
from `calib_top.json`; vision does not measure height. Calibration clicks the **pen tip**, so
`robot_mm` is a target for the pen tip: set the IK tool point to the pen tip.

### Conditions and events

Each frame, `conditions` lists the problems that are currently active. An event (`phase`:
`start` / `update` / `end`) is emitted when a condition appears, changes or clears. There
are also `MAP_RESET` and `MAP_LOCKED` events.

| Condition | Level | Meaning, and what motion should do |
|---|---|---|
| `MAP_NOT_LOCKED` | SEARCHING | No hole layout saved yet; it needs the line and all holes visible for 3 frames. **Don't move to holes.** |
| `HOLES_HIDDEN` | WARN | Some holes aren't visible, and their `px` / `robot_mm` are predicted. Expected while the pen is over a hole. |
| `BOX_PARTIALLY_HIDDEN` | WARN | Something (the arm, a hand) covers part of the box. The pose is **held** from the last good frame. |
| `NO_BOX` | WARN | Box not visible. The pose is held for up to 2 s. |
| `ORIENTATION_INFERRED` | WARN | Reference line hidden; orientation taken from the previous frames. |
| `ORIENTATION_LOST` | WARN | Line hidden and orientation can't be worked out. The pose is held. |
| `BOX_MOVING` | WARN | The box is moving. Re-read targets continuously, don't use a cached one. |
| `UNEXPECTED_HOLES` | WARN | Round marks not in the saved layout (new box, or a blob of ink). The planner should probably call `/reset`. |
| `POSE_STALE` | **STOP** | No trustworthy pose for more than 2 s. **Stop / don't start motion.** |

### Rules for the motion side (please follow these)

1. **Move only if `safe_to_act` is true** and the state is fresh. Reject states whose
   `timestamp` is more than about 0.5 s old (vision has crashed or stalled).
2. **`status == "STOP"` means stop now.** Safety limits (joint, speed, workspace) are still
   needed on your side. Vision is not a safety system.
3. **Approach from above:** go to `robot_mm` with `z + clearance` (e.g. 20 mm), then go
   down to `z`, then back up. While the arm is over the box, the top camera sees
   `HOLES_HIDDEN` / `BOX_PARTIALLY_HIDDEN` and `pose_source: "held"`. That is expected; keep
   going to the held target.
4. **A held pose cannot see the box moving under the arm.** After each dab, lift clear and
   wait for `pose_source == "live"` before the next hole.
5. **Address holes by `id`, never by list index.** IDs are stable after `map_locked`.
6. **When a new box is placed, call `GET /reset`** (or `tracker.reset()`), then wait for
   `map_locked: true`.

Suggested loop:

```python
for hole_id in plan:                                # order from the Claude planner
    s = wait_until(lambda s: s["safe_to_act"] and s["box"]["pose_source"] == "live")
    x, y, z = next(h for h in s["holes"] if h["id"] == hole_id)["robot_mm"]
    move_tip(x, y, z + 20)                           # your IK
    s = get_state()                                  # re-read: the box may have moved
    if s["status"] == "STOP" or not s["safe_to_act"]:
        abort_or_retry()
        continue
    x, y, z = next(h for h in s["holes"] if h["id"] == hole_id)["robot_mm"]
    move_tip(x, y, z + 20)
    move_tip(x, y, z)                                # dab
    move_tip(x, y, z + 20)
```

## Calibration (do once, together with Manasa)

`robot_mm` is `null` until `calib_top.json` exists. It holds a pixel-to-robot homography,
which is only valid at **lid height** and only while the **top camera hasn't moved**.

```bash
python calibrate_top.py --camera <top_idx>
```

1. Jog the pen tip to a point **at lid height**: touching the top of a box, or a block of
   the same height.
2. Click the pen tip in the window. In the terminal, type the robot's `x y` in mm for that
   pose, from **your forward kinematics**, in the IK base frame.
3. Repeat for **6 or more points spread over the whole work area** (corners and middle).
4. Press `c`. It prints the error for each point (aim for a mean under 3 mm; redo outliers),
   asks for the lid height `z` in the robot frame, and saves `calib_top.json`.
   `vision.py` loads it automatically on start.

If the camera gets bumped or the box height changes, recalibrate.

## Performance

- Detection takes about 20 ms per frame at 1280x720 on a laptop CPU, so the frame rate is
  limited by the camera, not by vision.
- Station's preview showed about 4 fps with about 230 ms delay. If that is the only frame
  source, the box will move between frames. Reading the camera directly with OpenCV
  (`--camera <idx>`) is preferred.

## What is not done or known yet

- **Not yet run on the live rig.** Tested on 3 real top-camera stills, plus simulated
  interruptions.
- **No calibration file exists yet.** It needs your forward-kinematics readout (see above).
- **No dedicated hand detector.** A hand shows up as `BOX_PARTIALLY_HIDDEN`, `NO_BOX` or
  `HOLES_HIDDEN`.
- **Ambiguous layouts:** if the line has been hidden for more than 2 s *and* the hole layout
  is the same both ways round (e.g. 3 evenly spaced holes), it reports `ORIENTATION_LOST`
  and eventually `POSE_STALE`. That is deliberate: it stops rather than guess.
- **Top-down view assumed.** The box outline is used as a flat rectangle, so a strongly
  tilted camera reduces accuracy.
- **Dab check:** `vision.check_mark(frame, expected_px)` measures how far a **blue** ink dab
  (the `MARK_COLOR` setting) landed from the target. Red ink is invisible on red boxes. It's
  meant for the wrist camera, but isn't connected to it yet.

## To agree with Manasa

1. IK base frame axes and origin, and confirm mm.
2. Lid height `z` in that frame (needed for calibration).
3. Pen-tip offset in the IK tool point.
4. Whether OpenCV can open the top camera while the Station is running. If not, how to get
   frames from the Station (API call, SDK or stream URL). Then choose HTTP (Option A) or
   in-process (Option B).
5. Approach clearance and dab depth (how far below `z` to press).
