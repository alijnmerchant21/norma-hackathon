# VIDEO_FACTS: guided_cell, as found on 22 Nov 2026, 09:50 (Berlin)

Sources: the files in `~/Documents/dev/lerobot/guided_cell` only. The folders `elrobot_ik`, norma-core and
the Claude app config were not readable from here; anything that depends on them is marked "not verified".
A few lines come from the project note `claude/guided-cell-test-log.md` (written by Claude during testing, last
updated 22 Nov 00:40). They are marked **[note]**. They are a record of what was reported, not a log from the code.
Times in `state/events.jsonl` are Berlin local time. The `t` values in `state/guided_log.jsonl` were converted to
Berlin time (UTC+1).

Important: `README.md` still says "Not yet run on hardware. Only the offline tests have run." That line is out
of date. `state/guided_log.jsonl`, `state/poses.station.json` and the comment at the top of `cell.guided.toml`
show real moves on 22 Nov.

---

## 1. Summary

A red box with three hand-drawn holes and a reference line on its lid sits on the table under a top camera. A
vision module (Manasa's `vision.py`) finds the box, the holes and the box's position and rotation in each 1280x720
frame. It also gives each hole an id that stays the same when the box moves. The guided cell (`cellguide`) turns a
hole's pixel position into a point on the table using Max's camera-to-table calibration. It then solves inverse
kinematics for the 7-joint ElRobot ("worker") and sends the move through Ali's `cellbot` safety layer to the
NormaCore Station, which drives the motors. The worker touches each hole with its closed gripper fingertips. No pen
or ink is used, so the touch leaves no mark. The program is just an ordered list of holes or lid points. It can be
typed in the terminal CLI (`run 0 1 2`) or written by Claude through MCP tools (`see_part`, `plan_program`,
`run_program`, `alerts`). An SO-101 ("loader") is meant to bring the box in and take it out, either from taught
poses or from two trained SmolVLA policies. On real hardware today, only the camera-guided touching of the holes by
the ElRobot has worked. Loading, unloading, Claude as the live operator and the learning loop have not been run on
the real arms.

## 2. What works on the real arms today

| Step of the cycle | Status | Evidence |
|---|---|---|
| Pickup (SO-101 takes the box) | **Not built on hardware.** Works in simulation only (taught-pose loader with sim poses) | `state/poses.station.json` has only `home` for so101: no `pick_hover`, `pick`, `place*`, `drop*`. `guided.toml` `[loader] mode = "taught"`, so on the Station `load_part` would be refused ("taught poses missing"). Sim poses exist in `state/poses.sim.json`, and `tests/test_guided_sim.py` calls `load_part`/`unload_part` |
| Move to work area (SO-101) | **Not built on hardware.** Simulation only (taught mode). Policy mode: trained checkpoint exists, running it on the arm is **not verified** | `policies/load/final` exists. `guided.toml` has `policy_load_cmd = ""`. README: "Policy loader mode is only tested offline with dummy commands, never with a real policy." No log of a `policy_run.py` run in this folder |
| Work step (ElRobot touches holes guided by the top camera) | **Working on real hardware**, at slow speed. It stops above the lid (`touch_mm = 10` in `guided.toml`) rather than touching it, with no ink | `state/guided_log.jsonl`: 7 touch records on 3 parts with real joint feedback (commanded and reached joints differ). Part `578a63` on 22 Nov 09:17–09:18: holes 0, 1, 2 all done, no failure event |
| Move to done / drop-off (SO-101) | **Not built on hardware.** Simulation only (taught mode). Policy `policies/unload/final` exists; running it on the arm is **not verified** | Same as above (`policy_unload_cmd = ""`, no drop poses taught) |

Other steps:
- Claude as operator through MCP on the real cell: **not verified.** Nothing in the logs shows it. [note] lists
  "8. Claude as operator: Not run yet" as of 00:40.
- Measuring marks and learning a correction ("data flywheel"): **not run.** There is no `state/guided_marks.jsonl`
  or `state/guided_learning.json`, and every logged `corr_mm` is `[0.0, 0.0]`.
- Lid-height aim offset (`jog`, `state/aim_lid.json`): **not saved.** The file does not exist.

## 3. Architecture

```
                        Claude (Claude desktop app, MCP over stdio)          or   human at the terminal
                                      |  cellguide.server (MCP tools)                cellguide.cli shell
                                      v                                                   |
   top camera --OpenCV 1280x720--> vision.py (Manasa)                                     |
   (camera:2)                      red box, holes, ref line,                              |
                                   HoleTracker: stable ids, OK/WARN/STOP                  |
                                      |                                                   |
                                      v                                                   v
                          GuidedCell (cellguide/guided_cell.py) <-------------------------+
                          - pixel -> table point (elrobot_ik calibration, Max)
                          - IK for the 7-joint ElRobot (kin.py, uses elrobot_ik)
                          - extra checks: calibrated area, IK error, clearance, sag, touch loop
                          - droop offset, alerts.py, learn.py (correction model), loader.py
                                      |
                                      v
                          cellbot Cell._move (Ali) = safety layer
                          joint limits, step size, waypoints, speed caps,
                          over-current freeze, e-stop latch, one arm in work zone
                                      |
                                      v
                          NormaCore Station (TCP 127.0.0.1:8888) --> ElRobot bus 5B61034679
                                                                 --> SO-101 bus 5B7B015823
                                      ^
   policy_run.py (SmolVLA) -----------+  writes goals to Station DIRECTLY (outside cellbot),
   2 Station cameras (224 px)            policy mode only, not run from the cell yet
```

Components:
- **Claude.** Operator through MCP tools in `cellguide/server.py`: `new_part`, `see_part`, `plan_program`,
  `run_program`, `inspect_part`, `retreat`, `guided_status`, `load_part`, `unload_part`, `alerts`, `preflight`, plus
  cellbot's `get_status`, `go_home`, `gripper`, `stop`, and the rest. The prompt (`cellguide/operator_prompt.py`)
  tells Claude to look at the `see_part` image, write the program (which holes, in what order, skipping unreachable
  ones), run it, read `alerts()` and decide what to do: look again, retry, skip, ask a human or stop. Claude never
  sends joint angles for the guided work. It picks targets, and the code computes the motion. A human approves
  every motion call. Ali's `third_party/cellbot/cellbot/orchestrator.py` can also drive cellbot through the Claude
  API, but its README shows it running in the simulator only.
- **Station API.** NormaCore Station on the Mac, TCP port 8888 (`cell.guided.toml`), accessed through
  `third_party/cellbot/cellbot/station_backend.py` (reads joints, writes servo goals, freeze, torque, camera images).
  Both arm buses run on one Station.
- **Cameras.** See section 5.
- **Computer vision.** `third_party/vision/vision.py`: HSV colour segmentation for the red face, Otsu threshold
  for dark markings, round blobs are holes, a long thin blob is the reference line, `HoleTracker` keeps the ids.
  This is classical OpenCV, not a learned model.
- **Kinematics and calibration.** `cellguide/kin.py`, which uses Max's `elrobot_ik` (URDF model, `joint_map.json`,
  `table_calib.json`, `aim.json`, `tool_offset.json`). Those files are outside this folder: **not verified**.
- **Trained policies.** Two SmolVLA checkpoints for the SO-101 loader (section 6), run by `policy_run.py`. Not
  connected in `guided.toml`.
- **Safety layer.**
  - cellbot guards (`cell.guided.toml [guard]`):
    - refuses targets outside joint limits (margin 0.03) and never clamps them;
    - limits any one joint to 0.15 per call;
    - splits moves into waypoints of 0.04;
    - caps speed at 900, or 300 at slow speed;
    - freezes both arms if any motor's current goes above 500;
    - times a move out after 8 s;
    - has a latching e-stop;
    - allows only one arm in the work zone.
  - Extra cellguide checks (`guided.toml [motion]`):
    - the target must be within 30 mm of the calibrated area and reachable within 3 mm (IK error);
    - the tip keeps 25 mm clearance while travelling;
    - the arm lifts at once if the tip sags more than 6 mm below the touch height;
    - a closed loop corrects the touch height to within 2 mm (up to 4 rounds);
    - the arm does not move if Station joint data is stale or the vision lock is missing.
  - In the CLI on the Station, every motion asks `Move the real arm? [y/N]`. Ctrl+C is an emergency stop.
  - In policy mode the policy bypasses cellbot. Safety there is by turns only (the SO-101 zone is `unknown`, so the
    ElRobot cannot move) plus `policy_run.py`'s own check, which stops if any goal jumps more than 300 ticks.
  - Vision is **not** a safety system: there is no hand detector (README, "Known gaps").

## 4. How the arm is controlled

- **ElRobot (worker), guided step.** The arm is moved by **IK from the camera**, not by taught poses, a policy or
  teleop replay.
  - The arm takes one clear look from home or the look pose.
  - The hole's pixel is turned into a table point (calibration plus `surface_height_mm = 40`).
  - IK finds the joints for: a high point (if needed), a hover 40 mm above the lid, then the touch height.
  - Each goal is sent as IK joints plus a learned gravity-droop offset (up to 4° per joint), through cellbot.
- **ElRobot, home and retreat.** Taught pose `home` (`state/poses.station.json`). No `look` pose is taught, so
  retreat falls back to home (code and [note]).
- **SO-101 (loader).** Designed for taught poses (`taught` mode) or a SmolVLA policy (`policy` mode). Only `home`
  is taught on the real arm. The policy has not been connected (section 2).
- **Claude's part.** Claude decides *which* holes, in *what order*, and what to do after a failure or alert. It
  does not compute or send the motion itself. That is cellguide (IK) plus cellbot (guards).
- **Teleop replay.** Not used anywhere in this folder.

## 5. How the cameras are used

| Camera | Used by | What is detected or measured | Does it change what the arm does? |
|---|---|---|---|
| Top camera, OpenCV index 2 (`guided.toml` `source = "camera:2"`), 1280x720, opened by the cellguide process | vision.py + GuidedCell | Red box face, centre, size, angle; holes (ids, pixel, position on the lid); reference line (fixes the orientation); `BOX_MOVING`, `NO_BOX`, `POSE_STALE`, `MAP_NOT_LOCKED` | **Yes.** Hole pixels become the IK targets. Without a lock (`safe_to_act: false`) the step is refused: 2 refusals logged on 21 Nov at 23:05 and 23:08 (`events.jsonl`). After the box moves, a new look sends the arm to the new place: the `box_angle_deg` of logged touches is 175.0, 145.3, 143.8 and 120.6 for different placements |
| Same top camera, mark check | `inspect_part` / `learn.py` | Blue ink dots (HSV 95–130), miss in mm per hole | Would shift the next aim, but **never run**: no ink and no marks file |
| Two Station cameras (`--cameras A,B`, 224 px frames per README) | SmolVLA policy via `policy_run.py` | Policy input images (`observation.images.cam0`, `cam1` in the checkpoint config) | Would drive the SO-101. Which indices were used: **not verified** (the docs say `A,B`). Never run from the cell |
| ElRobot wrist camera | — | Not used (README "Known gaps"). [note]: index 0, black picture | No |
| Station frames for vision (`station:` source) | — | Do not work: 224 px versus the 1280x720 calibration; refused | No |

While the arm is over the part, the top camera is **not trusted** and the part is **not watched**
(`sees_under_arm = false`).

## 6. Training

| Item | Value | Source |
|---|---|---|
| Arm | SO-101 (6 joints: `action_dim 6`, `state_dim 6`) | `policies/*/final/config.json` |
| Tasks | load: "Move the red box from the pickup area to the work area". unload: "Move the red box from the work area to the drop-off area" | `guided.toml`, `POLICY.md` |
| Number of episodes | **not verified** (not recorded in this folder). The runbook planned about 35 per task | — |
| Data format | NormaCore Station recordings exported to parquet: **not verified** here (runbook only) | — |
| Model | SmolVLA (VLM `HuggingFaceTB/SmolVLM2-500M-Video-Instruct`, frozen vision encoder, expert only trained, chunk 50, 2 cameras, 10 denoising steps). Two separate policies (load and unload have different normalisation stats) | `config.json`, `stats.safetensors` |
| Where trained | On a CUDA GPU (`"device": "cuda"` in config). Which machine or provider: **not verified** | `config.json` |
| Steps | Checkpoints at step 5,000 and 10,000 plus `final`. Step count of `final`: **not verified** | folder names in `policies/` |
| Checkpoint location | `guided_cell/policies/load/final/` and `guided_cell/policies/unload/final/` (each `model.safetensors` is 1.2 GB) | file listing |
| Training loss or eval | **not verified** (no training log in the folder) | — |

## 7. Numbers

| Number | Value | Source |
|---|---|---|
| Real-hardware touch records | 7 touches on 3 parts (`f65290`: hole 0 twice; `276098`: holes 0, 1; `578a63`: holes 0, 1, 2) | `state/guided_log.jsonl` |
| Full 3-hole programs completed on the real arm | 1 (part `578a63`, 22 Nov 09:17:35 to 09:18:18, no failure logged) | `guided_log.jsonl`, `events.jsonl` |
| Full 3-hole program in simulation | 1 logged (part `9ede28`, 21 Nov 22:58: reached joints equal commanded joints exactly, which is the simulator's behaviour) | `guided_log.jsonl` |
| Full cycles (load, work, unload) on hardware | 0 | sections 2 and 3 |
| Failed or refused guided steps on hardware | 3: hover move failed (00:10:48); refused because so101 was at an unknown position (00:14:35); "part kept moving for 8 s" (00:15:27, from the old follow mode) | `events.jsonl` |
| Station link errors | 3 × "Station joint data is stale (>1 s old)" (21 Nov 23:40, 23:40, 23:58) | `events.jsonl` |
| Time per touch step (real arm, `all_slow = true`) | 5.4–8.0 s on 22 Nov 00:05–00:15; 10.4–14.2 s on 09:17 | `guided_log.jsonl` `seconds` |
| Work-step time for 3 holes | 35.1 s (sum of the 3 steps of `578a63`); 43 s from "part started" to the last touch | `guided_log.jsonl`, `events.jsonl` |
| Sideways distance between the aimed point and the tip computed from joint readings (real arm) | 2.5–4.2 mm (7 touches) | computed from `aimed_m` and `tip_reached_m` in `guided_log.jsonl` |
| Height of that computed tip versus the aimed point | −7.4 to −6.5 mm (00:05–00:15 runs); +15.0, −2.1, +10.8 mm (09:17 run) | same |
| Accuracy on the lid in mm (physical) | **not verified.** No marks were measured. The figures above come from the arm's own encoders and model, not from a camera or a ruler. [note] reports a constant sideways offset seen on the table, size not recorded | — |
| Success rate | **not verified** (too few runs for a rate; no statistic is computed anywhere) | — |
| Policy success rate on the arm | **not verified** (never run from this folder) | — |
| Earlier fix | Joint swing between neighbouring holes cut from 54° to 10° | [note] only |
| Shoulder droop | ~4° under gravity, which is why `reach_tolerance` was raised from 0.02 to 0.04 | `cell.guided.toml` header comment |

Note: the log records are from an older code version than the files now in `cellguide/`. They lack the newer
`touch_error_*` fields, so the new closed-loop touch correction has no logged run yet.

## 8. Motion: does the object move while the arm works?

- With the current config (`sees_under_arm = false`), **the box is placed first and then worked on.** The arm takes
  one look from a clear pose, then does all steps from that look. If the box is moved while the arm is over it,
  **the arm does not notice.** Each result says how old the look was (`sighting_age_s`). In the 09:17 run this was
  6, 17 and 27 s for holes 0, 1 and 2.
- What can be shown: move the box *between* runs, or between steps with `run --look`. The arm then retreats, looks
  again and goes to the new position. [note]: after moving the box 5 cm, hover went to hole 0 in the new place.
- Continuous following (the Inbolt-style goal) exists in code only behind `sees_under_arm = true`. An earlier run
  with following on failed when the arm hid the box ("the part kept moving for 8.0 s", 22 Nov 00:15:27). It is off
  now and needs a camera that sees under the arm. The wrist camera is not used. **No follow-while-working on
  hardware.**
- During the guided step only the ElRobot moves. The SO-101 stays at home.

## 9. How to run the demo

A full cycle (pickup, work, drop-off) **cannot be run on the real arms today** (section 2). This is the shortest
real sequence that exists: the guided work step, with a human placing and removing the box. The commands come from
`TESTING.md` and `README.md`.

1. Start Station with TCP and web enabled: `./station --web --tcp` (form from `third_party/cellbot/README.md`; the
   folder to run it from is **not verified**). Turn all teleop off. Power both arms.
2. In a terminal in `guided_cell`:
   ```
   G=$(pwd)
   export CELLBOT_CONFIG=$G/cell.guided.toml CELLBOT_STATE=$G/state PYTHONPATH=$G
   PY=$G/.venv/bin/python
   env CELLBOT_MODE=station STATION_HOST=127.0.0.1 $PY -m cellguide.preflight
   ```
   Go on only if the verdict is `READY`.
3. Start the shell. Vision runs inside it on camera 2, with a viewer on port 8765. Do not run `vision.py`
   separately.
   ```
   env CELLBOT_MODE=station STATION_HOST=127.0.0.1 $PY -m cellguide.cli shell
   ```
4. In the shell, answer `y` to each `Move the real arm? [y/N]` only when the table is clear:
   ```
   home so101
   home elrobot
   gripper elrobot close
   new
   see
   plan all
   run 0 1 2
   alerts
   ```
5. Optional "the part moved" shot: after `run` (the arm has retreated), slide or rotate the box, then `see` and
   `run 0 1 2` again. It goes to the new positions. Or use `run --look 0 1 2` and move the box only while the arm
   is at home between steps.
6. Loader steps: with `[loader] mode = "human"` in `guided.toml`, `load` / `unload` only print what the human must
   do (this changes a file, so it is not done here). In the current `taught` mode they would be refused.
7. Simulation-only full cycle, no hardware:
   `env CELLBOT_MODE=sim $PY -m cellguide.cli --frames "$G/third_party/vision/images" shell`,
   then `new`, `see`, `plan all`, `run 0 2 1`, `inspect`, `alerts`.
   `load` / `unload` in sim use `state/poses.sim.json`; the exact CLI result in sim is **not verified**.
8. Claude as operator: paste the block printed by `bash setup_mac.sh` into `claude_desktop_config.json`, restart
   Claude, then ask "run the guided procedure for 1 part" and approve each motion. **Not run yet.** Do not run it
   together with the CLI shell.

## 10. What to show on screen

- **Browser, `http://127.0.0.1:8765/frame.jpg`** (served by the shell with `camera:2`). The live top view with the
  box and holes outlined, the status line (`OK`, `WARN` or `STOP`, `safe=`, hole count) and condition text. Point
  out: hole ids stay fixed when the box is turned; a hand over the box gives `NO_BOX` and `POSE_STALE`, so the arm
  refuses. Example: `state/cli/20261121_231508_544_see.jpg` shows "STOP safe=False holes=0" with a hand on the box.
- **Terminal with `cellguide.cli shell`.** The `BACKEND: Station at ...` banner; the JSON from `see` (hole ids,
  `robot_mm`, reachable); `plan all` (`"ok": true`, per-step reachability); the `Move the real arm? [y/N]` prompt;
  the `run` result (`aimed_from "clear view N s ago"`, `sighting_age_s`, `seconds`); `alerts` (recommendation
  `continue`).
- **Annotated images** in `state/cli/*_see.jpg` (saved by `see`).
- **Real camera or phone** on the table: the ElRobot fingertips going to each hole, and the arm going to the new
  place after the box is moved between runs.
- **Station web UI** (port 8889 per `cell.guided.toml`): joint positions. What it shows was **not verified** here.
- **`state/events.jsonl` / `state/guided_log.jsonl`** (e.g. `tail -f`): refusals and per-touch records, which show
  the guards working.
- A Claude app window running the procedure would show Claude's role, but this is **not run yet**.

## 11. Known limits and failures

- **Loader not on hardware.** No pick/place/drop poses taught for the SO-101. Policy commands empty in
  `guided.toml`. The trained policies have not been run on the arm from this folder. At 09:43:52 the policy flag
  `--max-seconds` was typed into the cellguide shell by mistake ("unknown command").
- **No following while working.** The top camera cannot see the part under the arm, and the wrist camera is unused.
  A moved part is only handled by looking again. Following was tried and failed (00:15:27).
- **No ink, no marks, no learning.** The touch is hidden (closed fingertips, no ink). `inspect` finds nothing, and
  the correction model has never received data. No measured accuracy exists.
- **The touch is not a contact yet.** `touch_mm = 10` stops 10 mm above the lid, and `dwell_s = 6` is a test value.
  Lowering it is still on the to-do list ([note]).
- **Calibration.** It is a table-level calibration with `surface_height_mm = 40` and no parallax numbers set
  (`camera_height_mm = 0`). A lid-height calibration or `aim_lid.json` has not been made. [note]: constant sideways
  offset.
- **Gravity droop.** The shoulder ends about 4° short (`cell.guided.toml`), corrected in software from joint
  readings only. Bending that the encoders do not see is not corrected (README).
- **Station link stalls.** 3 "joint data is stale" errors are logged. The runner now checks for fresh data before
  every move.
- **Old follow-mode vision.** It chased distorted vision while the arm was over the box (00:15:27). Fixed by
  "one clear look"; the fix has a logged run (09:17) but uses an older log format.
- **No hand or unknown-object detector.** A hand only stops the arm because skin breaks the red-box detection
  ([note]). Vision is not a safety system. A human must stay at the power switch.
- **Policy mode bypasses cellbot's guards** (by design; protected only by turn-taking and a 300-tick jump check).
- **Station camera frames (224 px)** cannot be used with the calibration.
- **README is out of date** on hardware status.
- **Done by hand.** The box is placed and removed by a human. Poses (`home`) were taught by hand. Holes and the
  reference line are drawn by hand on the box. Every motion is confirmed by a human (`y`).
- **Nothing faked in the logs** was found: the sim run (`9ede28`) and the real runs can be told apart by the joint
  feedback. But the only accuracy numbers are encoder-based (section 7). Do not present them as measured accuracy
  on the part.
