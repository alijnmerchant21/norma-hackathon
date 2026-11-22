# Testing the guided cell, stage by stage

Do the stages in order. Do not skip ahead to motion if an earlier stage fails. Each stage says what you
need, the exact commands, what "good" looks like, and what to send back if it fails.

Shorthand used below (run from the `guided_cell` folder; `setup_mac.sh` prints the same env line):

    G=$(pwd)
    export CELLBOT_CONFIG=$G/cell.guided.toml CELLBOT_STATE=$G/state PYTHONPATH=$G
    PY=$G/.venv/bin/python

(These are `export` lines on purpose: the Mac's default shell, zsh, does not split a variable holding
several settings, so the settings must be exported one by one. `$ENV` below is therefore empty and harmless.)

Add `CELLBOT_MODE=sim` or `CELLBOT_MODE=station` (and `STATION_HOST=127.0.0.1`) in front of each command.
The CLI prints `BACKEND: SIMULATION` or `BACKEND: Station at ...` first. Read it every time.
In the shell, `help` lists all commands. Images go to `state/cli/<time>_<command>.jpg`.

## Stage 0: install

Need: the Mac, this folder next to `elrobot_ik`, paths in `guided.toml` checked.

    bash setup_mac.sh

Good: six lines `PASS` (test_guided_sim, test_teach_tool, test_alerts, test_config_copy, test_preflight,
test_cli) and `Setup done.` Send back: the whole terminal output of the script.

## Stage 1: no hardware at all

Need: nothing but the recorded frames. This shows the whole worker flow on recorded images.

    env CELLBOT_MODE=sim $PY -m cellguide.cli \
        --frames "$G/third_party/vision/images" shell

Then type: `new`, `see`, `plan all`, `run 0 2 1`, `inspect`, `alerts`, `quit`.
Good: `new` lists holes `[0, 1, 2]`; `plan all` says `"ok": true`; `run` ends with `"ok": true` and the
first step shows `"gripper_closed_first": true`; `inspect` says "no marks found ... without ink"; `alerts`
recommends `continue`. Send back: the printed JSON of the failing command and `state/cli/*_see.jpg`.

## Stage 2: real camera, simulated arm

Need: the red box under the top camera, nobody's hands in the view.
1. Find the top camera index: `python ../elrobot_ik/peek.py` (press q to quit).
2. Start the vision server on the SAME camera and resolution as the calibration (1280x720):

        cd $G/third_party/vision && $PY vision.py --camera <top index> --width 1280 --height 720 --serve 8765 --no-window

   (in its own terminal window; set G and PY there too)

   Alternative without the server: skip step 2 and add `--source camera:<top index>` to the CLI in step 4.
   Then the cellguide process itself owns the camera (it prints `VISION: camera N in-process (viewer ...)`);
   the standalone vision server must NOT run at the same time on the same camera. The venv's OpenCV is
   headless; that is fine, no window is opened. The viewer in step 3 is served by cellguide on the same port.
3. Open http://127.0.0.1:8765/frame.jpg in a browser: the box and holes are outlined.
4. CLI in sim mode, WITHOUT `--frames`: `env CELLBOT_MODE=sim $PY -m cellguide.cli shell`
   (or `... cellguide.cli --source camera:<top index> shell`), then `new`, `see`, `plan all`.
   With `camera:` also run `preflight` once: the `camera` line shows the frame size and the processed fps.
5. Slide the box a few cm and rotate it a little, then `see` again.
6. Cover one hole with a finger and `see` again.

Good: after moving the box the hole ids stay the same and `robot_mm` changes the way the box moved (slide
3 cm away from the robot: about 30 mm change); the covered hole shows `"visible": false` but is still
listed. Send back: both `see` JSON outputs, the two `state/cli/*_see.jpg` files and `/frame.jpg`.

## Stage 3: real Station, read only

Need: Station running with `--web --tcp`, ALL teleop off, both arms powered.

    env CELLBOT_MODE=station STATION_HOST=127.0.0.1 $PY -m cellguide.preflight

Good: every line `ok` or `warn` and the verdict `READY`. Fix every `fail` (each line has a fix). Write down
which joints preflight reports resting on an end stop. Send back: the full preflight output.

## Stage 4: teach poses (humans only)

Need: a person holding the arm. Run from the cellbot folder with the same config and state:

    cd $G/third_party/cellbot && export CELLBOT_MODE=station STATION_HOST=127.0.0.1 \
        CELLBOT_CONFIG=$G/cell.guided.toml CELLBOT_STATE=$G/state
    $PY -m cellbot.teach limp so101          # torque off, support the arm, move it by hand
    $PY -m cellbot.teach save so101 home clear
    $PY -m cellbot.teach hold so101          # torque on again
    $PY -m cellbot.teach limp elrobot
    $PY -m cellbot.teach save elrobot home clear
    $PY -m cellbot.teach save elrobot look clear   # fully out of the top camera's view
    $PY -m cellbot.teach hold elrobot

Good: `status` in the CLI lists the poses. Send back: the teach output and `state/poses.station.json`.

## Stage 5: first motion, hover only

Need: one person at the table with a hand on the 12 V supply of the ElRobot follower. Vision running.

    env CELLBOT_MODE=station STATION_HOST=127.0.0.1 $PY -m cellguide.cli shell

After EVERY restart of the shell (zones start "unknown"): `home so101`, then `home elrobot`.
Then: `gripper elrobot close`, `new`, `see`, `plan h0`, `hover 0`.
Each motion asks `Move the real arm? [y/N]`: type `y` only when the table is clear.
The banner says `Station at host:port (connects on the first command)`; the first command prints
`connected: buses {...}`.
Good: the fingertips stop about 40 mm above hole 0, within about 1 cm sideways; the JSON shows
`tip_above_part_mm`.
How aiming works with the top camera alone: the arm takes one clear look while it is at home/look, then
works from that look. While it is over the part the camera is not trusted (the arm hides part of the lid,
dark arm parts look like holes), so the part is NOT watched during the work and a part moved then is not
noticed. "Follow while hovering" needs a camera that sees under the arm (`sees_under_arm = true`). What you
can show is "look again": `retreat`, slide the box 3 to 5 cm, `hover 0` again: it goes to the new place.
The JSON shows `aimed_from` ("clear view N s ago") and `re_aims` 0.
Anything wrong: `Ctrl+C` while a command runs = emergency stop (both arms frozen, e-stop latched), or type
`stop` (never asks), or cut the 12 V. Send back: the `hover` JSON, a phone photo of the tip over the hole,
`state/events.jsonl` and `state/cli/*_see.jpg`.

## Stage 6: first touch and the lid-height aim offset

Need: Stage 5 good. No restart needed: `set` changes `[motion]` values for this session only
(`set` alone lists them; guided.toml is not changed).
1. `set touch_mm 10`, `new`, `touch 0`. The question shows `touch_mm` and `hover_mm`: check them. The JSON
   shows `lowest_tip_above_part_mm` (at the dwell), `touch_error_mm` (measured minus commanded touch height,
   good: within about 2 mm) and `tip_above_part_mm` (after lifting, good: within about 2 mm of
   `hover_mm + touch_mm`). The droop offset learned at hover is carried into the descent and the lift; if
   the tip still sags more than `touch_sag_abort_mm` (6 mm) below the touch height the arm lifts at once and
   the step fails. The height is corrected ABOVE the lid, at `pretouch_mm` (12 mm), to within `touch_tol_mm`:
   check `pretouch_error_before_mm`, `pretouch_error_after_mm` and `pretouch_correction_rounds`; if it cannot
   get there the arm goes back to hover without touching and the step fails ("pre-touch height not
   reached"). Then one short straight descent to the touch, measured only: `touch_error_mm`; more than
   `touch_tol_mm` too high -> lift, step fails ("touch height not reached ... after the final descent").
   Watch that the tip does not slide on the lid and the box does not move. `droop` shows the offset,
   `droop clear` zeroes it. Check with a ruler before lowering.
2. Teach the aim offset at lid height: `jog` (asks once), then lower with `f` until the tip is a few mm
   above the lid (never below 3 mm; above 20 mm the offset is not trusted because of parallax). Move with
   `w a s d` (towards the top / left / bottom / right of the camera image) until the fingertips are over
   the hole; `[` `]` halve / double the step (start 2 mm), `p` prints the offset, `k` saves it
   (`state/aim_lid.json`, used at once), `q` leaves without saving. Several keys per line work (`ddw`).
   `aim` shows the aim_file part and the lid part; `aim clear` forgets the lid samples.
3. `retreat`, `touch 0` again: the tip should now be over the hole. Then lower step by step with
   `set touch_mm 5`, `set touch_mm 2`, `set touch_mm 0`, a `touch 0` after each.
Optional: blue whiteboard marker ink on the fingertip, then `run 0 2 1`, `retreat`, `inspect` for real miss
numbers. "The part moved between steps" demo: `run --look 0 1 2` and slide the box while the arm is at the
look pose between steps (never while it is over the part). Second part: `finish pass`, `new`, `see`: `learned_correction_mm` must be non-zero and the miss
should shrink. Send back: `touch` and `inspect` JSON, `aim` output, `state/cli/*_inspect.jpg`,
`state/guided_log.jsonl`, `state/guided_marks.jsonl`, `state/aim_lid.json`.

## Stage 7: loader

Start with `[loader] mode = "human"` in `guided.toml`: `load` and `unload` only print what the human must
do, then `new` checks the part is there. Next, teach the so101 poses pick_hover, pick, place_hover, place,
drop_hover, drop (Stage 4 commands) and set `mode = "taught"`, or set the policy commands and
`mode = "policy"`. Good: `load`, `new`, `run all`, `unload` in a row. Send back: the `load`/`unload` JSON
and `state/events.jsonl`.

## Stage 8: Claude as the operator

Paste the config block printed by `setup_mac.sh` into the Claude app (set `CELLBOT_MODE` to `sim` first,
then `station`), restart Claude, and ask it to "run the guided procedure for 1 part". Approve each motion call
yourself. Do not run the CLI shell at the same time (two programs would drive the same arms).
Send back: the chat transcript, `state/events.jsonl`, `state/guided_log.jsonl`.

## What to send back when something fails

| What | Where |
| --- | --- |
| the CLI's printed JSON (or REFUSED / VISION / ERROR line) | copy from the terminal |
| camera images the CLI saved | `state/cli/*.jpg` |
| cellbot events (moves, refusals, e-stop) | `state/events.jsonl` |
| every guided touch | `state/guided_log.jsonl` |
| readiness checklist | output of `python -m cellguide.preflight` |
| what the vision server sees | http://127.0.0.1:8765/frame.jpg (save the image) |
