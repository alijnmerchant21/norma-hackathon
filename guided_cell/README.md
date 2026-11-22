# guided_cell: Claude as the operator of a two-arm robot cell

Built at the HackLab x BRISA Berlin Robotics & AI Hackathon (NormaCore track, 21-22 Nov 2026).

## The idea

Industrial guided-assembly cells (Inbolt and others) cost millions so that a robot can work on a part that is
never in exactly the same place. We built the same loop with two low-cost arms, one top camera and Claude as
the operator:

1. **Holder arm (SO-101)** brings the part into the work area with a trained SmolVLA policy, deliberately
   imprecise.
2. **Camera** finds the part, its holes and its rotation. Hole ids stay fixed in the part's own frame, however
   the box lies.
3. **Claude** looks at the part, writes the program (which holes, in which order) at run time and checks it
   before anything moves.
4. **Worker arm (ElRobot)** marks each hole: camera pixel -> table point -> inverse kinematics -> guarded motion
   through NormaCore Station.
5. **Camera** measures every mark. The miss goes into a correction model, so the next part is aimed better
   (data flywheel).
6. **Holder arm** takes the part out. Every loader attempt is checked by the camera and labelled; successful
   ones become new training episodes for the policy.

Claude handles exceptions like a human operator would: a refused move, a hand in the cell, a failed load, a
policy that needs a fallback or a hand-over to a human. Every motion still passes cellbot's safety guards, and
a human approves it.

## What we achieved on the real hardware (22 Nov 2026)

- **The whole cell runs on NormaCore Station** with both arms, the top camera and Claude as operator through
  MCP. 7 parts went through the cell, 5 of them through the full cycle load -> marking -> inspection -> unload,
  each with its own log and run report (`demo/run_report_*.html`).
- **Load by a trained SmolVLA policy: 5 of 6 attempts succeeded** (74-109 s each). The other attempt was
  stopped by our own jump guard in `policy_run.py`.
- **Camera-guided marking on a part placed at any angle.** A closed loop on the arm's measured tip keeps the
  touch height mostly within +-1.5 mm of the target. The best marks landed 0.4-0.5 mm from the hole centre.
- **The data flywheel is live.** Marks are found by comparing pictures before and after the dabs, the miss is
  logged per hole and fed back as a learned aim correction on the next part. Loader attempts are labelled and
  turned into time ranges of the Station recording for retraining.
- **Claude as operator** writes and checks each program, narrates every step, labels every loader attempt,
  logs exceptions for the human and latched the e-stop when it saw a hand on the box.
- **Safety by design:** joint limits, speed caps, over-current freeze, latching e-stop, one arm in the work area
  at a time, plus extra checks for computed targets (reach, clearance, sag, touch height).

## Data path

```
top camera -> vision.py (box, holes, tracker) -> table calibration + IK (elrobot_ik)
           -> cellbot guards (Cell._move) -> NormaCore Station -> ElRobot arm
marks on the lid -> top camera -> measured miss -> correction model (learn.py) -> next aim
```

## Where the code comes from

| Part | Author | Used for |
|---|---|---|
| `cellbot` | Ali | guarded Cell, Station I/O, e-stop, taught poses, MCP server |
| `Berlin_hackathon/vision.py` | Manasa | red box, holes on the lid, reference line, `HoleTracker`, HTTP server |
| `elrobot_ik` | Max | URDF kinematics, joint map, pixel <-> table calibration |
| `guided_cell` (this folder) | new | `GuidedCell(Cell)`, IK targets, clear-view aiming, droop carry, mark learning, MCP tools |

None of the three drops is modified. `guided.toml` points at them; `cellguide/config.py` loads them in place.

Files here: `guided.toml` (config), `cellguide/guided_cell.py` (the cell), `kin.py` (pixel -> target -> joints),
`vision_source.py` (HTTP or in-process vision), `learn.py` (correction model), `loader.py` (so101
loader), `alerts.py` (operator alerts), `server.py` (MCP server), `cli.py` (terminal CLI),
`operator_prompt.py` (operator instructions), `teach_tool.py` (pen tip calibration), `preflight.py`
(readiness checklist), `marks_dark.py` (mark measurement from before/after pictures), `cell.guided.toml` (our
cellbot config), `policy_run.py` (runs the SmolVLA loader policies on the SO-101, see `POLICY.md`),
`unload_hybrid.py` (taught approach + unload policy), `CLAUDE.md` (how Claude narrates a run), `demo/` (run
reports and the report generator), `setup_mac.sh`, `tests/`, `TESTING.md`.

## Setup on the Mac

Testing step by step: follow `TESTING.md` (stages 0-8, what good looks like, what to send back).
Without the Claude app: `python -m cellguide.cli shell` (same env as the server) runs every step from a
terminal on one long-lived cell (`help` lists the commands). Use `shell` on hardware: zones and the part
live in memory. Every motion prints what it will do and, on the Station, asks `Move the real arm? [y/N]`.
No camera: `CELLBOT_MODE=sim ... python -m cellguide.cli --frames <Berlin_hackathon>/images shell`.

1. Put this folder at `~/Documents/dev/lerobot/guided_cell`, next to `elrobot_ik`. Check the paths in
   `guided.toml` (`cellbot_dir`, `vision_file`, `calib_file`, `aim_file`).
2. Run `bash setup_mac.sh`. It finds Python 3.10+ (3.12/3.11/3.13 first, else `python3`, or `PYTHON=...`),
   makes `.venv` here (your conda env is not changed), installs cellbot's `requirements.txt` + `scipy`
   (+ `tomli` on 3.10), runs the offline tests and prints two things. It never touches the robot or Station.
3. Paste the printed `"cellguide"` block under `"mcpServers"` in
   `~/Library/Application Support/Claude/claude_desktop_config.json`, instead of cellbot's own server.
4. With Station running, run the printed `preflight` command and fix every `fail`.

Python 3.10: `cellguide.config` maps cellbot's `tomllib` to `tomli`; setup adds a `tomllib.py` shim to `.venv`
for cellbot's own commands. cellbot is not edited.

### cell.guided.toml

Our own copy of Ali's `cell.toml`, used through `CELLBOT_CONFIG`. Only `host = "127.0.0.1"` (Station runs on
this Mac) and `norma_core_dir` (absolute path of the norma-core clone with `target/gen_python`; cellbot does
not expand `~`) differ. Every `[guard]` value is identical; `tests/test_config_copy.py` checks this.
Ali's file stays untouched. If Ali changes his file, copy it again and change the same two lines.
State: the server uses `CELLBOT_STATE=<guided_cell>/state`. Taught poses, `pen_tip.json` and the e-stop live
there, so run `cellbot.teach`, `cellguide.teach_tool` and `cellbot.estop release` with the same `CELLBOT_STATE`.

### preflight

`python -m cellguide.preflight [--json]` (same env as the server) or the MCP tool `preflight()`. Read-only: it
never sends a motion or torque command. One line per check (`ok`, `warn`, `fail`, `skip`) with a fix:
Python and packages; paths, elrobot_ik files and calibration markers; cellbot config, norma-core and guard
values; e-stop; Station buses, calibration, ranges against `joint_map.json`, joints on an end stop, torque;
taught poses and their limits; tool tip; part height and parallax; vision source, age, image size, part lock
and a plan per hole; learning and the state folder. Verdict: `READY for a slow hover-only first step` only
if nothing fails.

### The `station:` vision source does not work today

Station resizes its camera frames (224 px square by default). The table calibration was made on the 1280x720
image, so Station frames do not map onto it. Use Manasa's vision server on the calibration camera at
1280x720. Preflight fails on a `station:` source, and at run time any vision image whose aspect ratio differs
from the calibration by more than 2 % is refused (`VisionError`).

### `camera:N`: vision in this process, no separate server

`[vision] source = "camera:2"` (or `python -m cellguide.cli --source camera:2 shell` for one run) makes the
cellguide process open camera 2 itself (`CameraVision`): a background thread reads every frame at
`[vision] width` x `height` (default 1280x720, a different aspect ratio is refused) and runs Manasa's
`detect_scene` + `HoleTracker` on each one. Teammates can still watch at `http://<this machine>:8765/frame.jpg`
(`/raw.jpg`, `/state`; `[vision] view_port`, 0 = off). The cellguide process owns the camera: do NOT run the
standalone vision server on the same camera at the same time (the second one gets no frames).
`camera:auto` (the current setting) picks the camera whose picture matches `top_ref.jpg`, because macOS
renumbers cameras when one drops out or comes back. The venv's
OpenCV is headless, which is fine because no window is opened. Default stays `http://127.0.0.1:8765`.

## First use, in this order

a. `bash setup_mac.sh` (all tests must PASS). b. Start Station with `--web --tcp`. Turn all teleop off.
c. Start Manasa's vision server on the SAME camera and resolution as the calibration image:
   `python vision.py --camera <top> --serve 8765 --no-window`, OR skip the server and use `camera:<top>`
   (section above).
d. Humans teach poses with `python -m cellbot.teach save <arm> <pose> <zone>`: `so101 home clear`,
   `elrobot home clear`, `elrobot look clear` (the arm fully out of the top camera's view).
e. Set `[part] surface_height_mm` to the lid height, or better make a calibration touched at lid height and
   set it as `calib_file` (then `surface_height_mm = 0`).
f. Tool: a black marker held in the closed ElRobot gripper (`[tool] tip_local_m`: closed fingertips + 38 mm
   stick-out). `[tool] close_gripper = true` closes the gripper before the first guided move. `inspect_part`
   finds the new dots by comparing a picture taken before the dabs with one taken after (`[marks] method =
   "diff"`) and saves `state/marks/<time>_<part>_{before,after,diff,result}`. The lid-height aim offset is
   taught from measured dots and kept in `state/aim_lid.json`.
g. Run preflight until it says READY.
h. First run: `see_part`, `plan_program([{"hole": 0, "action": "hover"}])`, then the same with
   `run_program` at slow speed, with a hand near the power switch.

## Tools

All cellbot tools stay available (`get_status`, `look`, `go_home`, `go_to_pose`, `gripper`, `finish_part`,
`log_result`, `stop`, ...). New:

- `new_part(part_id)`: forget the old hole layout, wait until the new part is locked.
- `see_part()`: holes with ids, lid position, robot mm, reachability; annotated image.
- `plan_program(steps)`: checks a program WITHOUT moving: per step target, pixel, robot mm, learned
  correction, reachable or why not, tool tilt, direct or via the high point, biggest joint change per
  leg. Each step is planned from the previous hover pose. Use it for a new kind of part or after a refusal.
- `preflight()`: the read-only checklist above.
- `run_program(steps, retreat_after, look_each_step)`: steps in the part's frame, `{"hole": id}` or
  `{"at": [fx, fy]}`, action `dab` or `hover`. One clear look first, every step aims from it;
  `look_each_step` (CLI `run --look`) looks again before every step. Stops at the first failing step.
- `inspect_part()`: miss per mark in mm, pass/fail, feeds the correction model; annotated image.
- `retreat()`: lift, then go to the `look` pose (or home).
- `guided_status()`: set-up, tip position, droop correction, learning trend, active guard limits (`guards`).
- `load_part()` / `unload_part()`: the so101 moves the part pickup -> work area / work area -> drop-off.
  Refused unless the elrobot is at a clear pose. Mode from `[loader]`, see below.
- `finish_teleop(attempt_id)`: closes a hand-over in which a human moved the SO-101 with its leader arm.
- `label_loader_attempt(success, note, attempt_id)`: the camera-checked result of a load or unload attempt.
- `loader_stats()`: attempts, successes and success rate per phase and variant, plus the time ranges of the
  successful attempts (new training episodes).
- `alerts()`: what happened since the last call (cellbot events, vision events, accuracy metrics) as
  `{level, source, message, needs_human}`, plus metrics and a recommendation: `continue`,
  `check and continue` or `pause and ask a human`.
- Prompt `run_guided(parts)` and resource `cellguide://procedure` hold the operator procedure.

## Loader modes (`[loader]` in guided.toml)

- `taught`: cellbot's taught poses (`pick_hover`, `pick`, `place_hover`, `place`, `drop_hover`, `drop`,
  `home`), the same sequence as cellbot's operator prompt, with all cellbot guards. A failed grasp at pickup
  is retried up to 2 times. Missing poses are refused with the list a human still has to teach.
- `policy`: runs `policy_load_cmd` / `policy_unload_cmd` (e.g. SmolVLA `run_policy.py`) in `policy_cwd`,
  `{task}` replaced by the task sentence. The policy writes to Station directly and BYPASSES cellbot's guards.
  Safety is by turns only: the so101 zone is set to `unknown` before the start, so the elrobot cannot move;
  the process is killed on `policy_timeout_s` or if the e-stop latches. After a clean exit the so101 goes
  home automatically; otherwise it stays `unknown` and `go_home so101` is required (after a human looks).
- `human`: no motion; a human places or removes the part, `new_part` / `see_part` verify.

Current setup: load by the SmolVLA policy (`policies/load/final`), unload by a taught sweep-and-grasp sequence
(`unload_mode = "taught"`). Variants per call: `primary`, `fallback` (taught approach and grasp, then the unload
policy: the hybrid, like the sponge_pick project) and `teleop` (a human moves the SO-101 with its leader arm;
the cell moves nothing until `finish_teleop`). Every attempt is logged in `state/loader_episodes.jsonl`. How to
run and tune the policies: `POLICY.md`. The policy weights (1.2 GB each) are not in this repo; only their
`config.json` is.

## Guard keys (`[motion]`)

`workspace_margin_mm`, `clearance_mm`, `ik_tol_mm`, `settle_max_deg`, `sight_max_age_s` (a clear look older
than this is not reused), `touch_sag_abort_mm` (lift and fail if the tip sags this far below the touch
height), `pretouch_mm` / `touch_tol_mm` / `touch_correct_rounds` / `touch_correct_max_deg` / `touch_cmd_below_mm`
(closed loop at the pre-touch height, see Droop below), `aim_frames` (aim on the median of this many live frames). Only with `[vision] sees_under_arm = true`:
`commit_age_s`, `reaim_mm`, `max_reaim_mm`, `max_reaims`, `follow_s` (the follow loop).

## Safety notes

- cellbot's guards still apply to every move: joint limits (refuse, never clamp), e-stop, one arm in the work
  area, one motion at a time, waypoints with over-current checks.
- Extra checks for computed targets: inside the calibrated marker area, IK error below `ik_tol_mm`, tip
  clearance along the path, no dip below the target on the way down, over-current while settling.
- Vision is not a safety system. Claude checks every `see_part` image for hands or foreign objects and stops
  the cell; a human stays at the power switch.
- With the top camera alone the arm takes one clear look from home/look and works from it: the part is NOT
  watched while the arm is over it (the arm hides the lid, dark arm parts look like holes). A moved part is
  handled by looking again (`run --look`, or between runs). Following needs a camera that sees under the arm
  (`sees_under_arm = true`). Vision events while the arm is over the part are ignored by `alerts()`.
- Droop: every guided goal is sent as `q + offset` (learned at hover, carried into descent and lift). The
  settle that learns it only corrects joints that can move (not one pinned at its limit, like joint 6) and
  measures after the joints have stopped. A dab first descends to the pre-touch height (`pretouch_mm` above
  the lid, default 12; must be > `touch_mm`, else no pre-touch stage) and runs the closed loop THERE, out of
  contact: it measures the tip (FK of joint feedback) and corrects the goal (height and xy), at most
  `touch_correct_rounds`, until the height is within `touch_tol_mm`; each joint is at most
  `touch_correct_max_deg` from the offset-only goal, the commanded tip never goes below pre-touch height -
  `touch_cmd_below_mm`, and the correction does not change the carried offset. Not converged, or sagged ->
  back to hover without touching, step fails. Then one short, slow, straight final descent carries that
  correction to the touch (commanded tip never below touch height - `touch_cmd_below_mm`); at the touch the
  height is only measured: more than `touch_tol_mm` too high, or more than `touch_sag_abort_mm` too low ->
  lift, step fails. Nothing moves the tip sideways while it is on the lid; the lift retraces the final
  descent straight up, then goes to hover. In one `run` program the pre-touch correction of a hole is the
  starting correction of the next (same cap). Results report `pretouch_error_before_mm`,
  `pretouch_error_after_mm`, `pretouch_correction_rounds`, `pretouch_correction_deg`, `pretouch_seeded_deg`,
  `touch_error_mm` (measured at the touch after the final descent; `touch_error_before_mm` /
  `touch_error_after_mm` are the same value, `touch_correction_rounds` is 0) and `touch_correction_deg` (the
  carried correction).
  CLI `droop clear` zeroes the offset.
- Only humans teach poses and release the e-stop. A human approves every motion call.

## Next steps

- **Unload by robot.** It is the step we are training now. The hybrid (taught approach + policy) and the taught
  sweep-and-grasp are in place, and every teleoperated hand-over becomes a new training episode.
- **Every hole within 3 mm.** A lid-height calibration and more marks per hole for the per-position learned
  correction.
- **Wrist camera** for the last centimetre, and for following a part that moves while the arm works on it.
- **Hand and unknown-object detection** in vision, next to Claude's own check of the images.
- **Tag successful attempts directly in NormFS**, instead of cutting time ranges from the recording.
