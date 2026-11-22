# cellbot: Claude operates a two-arm marking cell on NormaCore Station

The SO-101 picks a part and puts it in the work area. The ElRobot marks 4 spots on it.
The SO-101 removes it. Claude runs the whole thing through a guarded MCP server,
measures every mark with the top camera, and learns a per-spot correction so the
next part is marked more accurately.

```
Claude (orchestrator.py, Claude API)
   │  MCP tools only (stdio)
   ▼
mcp_server.py ──► cell.py  (all guardrails live here)
                    ├── sim_backend.py      dry run, no hardware
                    └── station_backend.py  NormaCore Station TCP :8888
                    feedback.py  (miss log + learning)   vision.py (mark detection)
```

## Quick start (dry run, no arms)

```bash
pip install -r requirements.txt
python -m cellbot.smoke --parts 5          # scripted run, no API key needed
export ANTHROPIC_API_KEY=...
python -m cellbot.orchestrator --parts 3   # Claude runs the cell in the simulator
```

**OpenRouter instead of the Anthropic API:** `export OPENROUTER_API_KEY=...` (and leave
`ANTHROPIC_API_KEY` unset, or pass `--backend openrouter` / `CELLBOT_LLM=openrouter`).
The default model is `anthropic/claude-opus-5.5`; change it with `--model` or `CELLBOT_MODEL`
(e.g. `anthropic/claude-sonnet-5.5`). It uses OpenRouter's Anthropic-compatible
`/v1/messages`, so tools and camera images work the same. Server-side threads and model
fallback are Anthropic-only and are turned off; if a route rejects adaptive thinking, the
part is retried without it (or pass `--no-thinking`).

Smoke-test output from the dry run (sim has a built-in 3-4 mm bias per spot):

```
part 1: misses mm {'1': 3.81, '2': 4.03, '3': 3.2, '4': 4.27} -> FAIL
part 2: misses mm {'1': 1.41, '2': 1.0, '3': 2.12, '4': 0.5} -> FAIL
part 3: misses mm {'1': 0.5, '2': 1.12, '3': 0.71, '4': 1.41} -> PASS
part 4: misses mm {'1': 0.71, '2': 0.71, '3': 0.5, '4': 0.5} -> PASS
```

Learning state, logs and poses go to `state/` (`marks.jsonl`, `events.jsonl`,
`learning.json`, `poses.<mode>.json`). Delete `state/learning.json` to forget
what was learned. Reports from Claude go to `reports/`.

Use the same MCP server from Claude Code or Claude Desktop for interactive demos:
`claude mcp add cellbot -- python -m cellbot.mcp_server` (run from this folder).

## MCP tools

| Tool | What it does |
|---|---|
| `get_status` | Joints (normalized 0..1), zones, last pose, gripper, e-stop, part, pose list |
| `look` | Camera images + one-line status |
| `go_to_pose(arm, pose, speed)` | Move to a taught pose |
| `go_home(arm)` | Slow move to home (a clear pose) |
| `move_joints(arm, joints, relative)` | Small adjustment near the last taught pose |
| `gripper(arm, open/close)` | Reports whether something is held (gripper stopped early) |
| `new_part_ready` | Start of a part (sim: spawns a part at pickup) |
| `start_marking` | Takes the reference image; both arms must be clear |
| `mark_spot(spot)` | ElRobot hover, touch, dwell, lift, with the learned correction applied |
| `inspect_marks` | Measures misses in mm, feeds the learner, returns an annotated image |
| `learning_status` | Corrections per spot, refined Jacobian, recent parts |
| `log_result(kind, message)` | Event log for the report |
| `finish_part(outcome, notes)` | Close the part |
| `stop(reason)` | Freeze both arms and latch the e-stop |

## Guardrails (enforced in `cell.py`, not in the prompt)

- **Joint limits**: Station's calibrated arc minus `limit_margin`, plus optional
  per-joint `envelope`s. Out-of-range targets are refused, never clamped.
- **Max step**: `move_joints` moves a joint at most `max_delta_per_call` per call and
  at most 2x that from the last taught pose. Big moves need taught poses, and only a
  human can teach poses (`cellbot.teach`). So Claude can't creep the arm anywhere.
- **Speed and accel caps** on every servo write; moves are split into small
  waypoints with checks between them.
- **Shared work area**: an arm can only go to a `work` pose while the other arm is
  at a `clear` pose. After startup on real hardware both arms count as "unknown"
  until they reach a taught pose.
- **Over-current or servo error** during a move: both arms freeze and the e-stop latches.
- **Timeout**: if a waypoint isn't reached, the arm is held where it is and Claude is told.
- **E-stop**: Claude can latch it (`stop`). Only a human can release it:
  `python -m cellbot.estop release`. A human can also latch it: `python -m cellbot.estop stop`.
  Keep the Station web UI open as the hardware-level fallback.
- **One motion at a time** (lock), so parallel tool calls can't race the zone check.
- `--confirm` on the orchestrator asks a human y/N/stop before every motion call.
  Use it for the first real runs.

## Learning loop (`feedback.py`)

1. `inspect_marks` diffs the top camera against the reference image, finds the new
   dark dots, matches each to its spot and measures the miss in mm.
2. Per spot: `next_correction = applied_correction - gain * miss` (iterative learning
   control). A spot with no history starts from the mean of the others.
3. The mm correction becomes ElRobot joint offsets through a 2xN Jacobian
   (`correction_joints`, `jacobian_mm` in cell.toml). The Jacobian is refined from
   observed results with a damped Broyden update.

## Running on the real arms

1. **Station** on the teammate's Mac: quit the desktop app and run the CLI
   `./station --web --tcp` (the app binds to 127.0.0.1 only). Both buses on one Station.
2. **Camera feed**: in `station.yaml`, give the inference block a fixed bus, because
   `st3215-bus: auto` only works when exactly one bus has torque on, and here both will:
   ```yaml
   inference:
   - queue-id: inference/normvla
     shm: /tmp/normvla
     shm-size-mb: 12
     format: normvla
     st3215-bus: <SO-101 bus serial>
     update-interval: 100ms
   ```
   Also raise `usb-video.resize_target` above 224 so marks are sharper.
3. **This repo**: clone norma-core next to it (or set `norma_core_dir`), then in
   `cell.toml` set `mode = "station"`, `host`, both `bus_serial`s (or leave "auto":
   it picks the 6-motor bus for SO-101 and the 8-motor bus for ElRobot), motor ids
   and joint names, gripper open/closed values, and `camera_names`.
4. **Teach poses**: `python -m cellbot.teach limp so101`, move it by hand,
   `python -m cellbot.teach save so101 pick clear`, ..., `python -m cellbot.teach hold so101`.
   Needed: so101 `home pick_hover pick place_hover place drop_hover drop`;
   elrobot `home mark_hover_1..4 mark_touch_1..4`. Zone is `work` over the work area.
5. **Camera calibration**: set `spot_targets_px` (pixel of each spot in the top
   image with the part in the fixture) and `mm_per_px`. Set `jacobian_mm` roughly:
   nudge each correction joint by +0.02 with `move_joints` at a touch pose and see
   how far the tip moves in mm.
6. `python -m cellbot.orchestrator --parts 1 --confirm`.

## What's verified vs guessed

Verified against norma-core source: Station TCP client and port 8888, the
`st3215/inference` frame layout and register addresses, sync_write commands
(speed/accel before goal), normalized-position math including wrapped arcs, and the
`inference/normvla` frame with joints and camera JPEGs.

Guesses to check on the hardware:
- ElRobot joint names, which two joints shift the pen tip in x/y, and the starting Jacobian.
- Gripper "holding" detection (gripper stopped > 0.08 short of closed). Tune per gripper.
- `current_limit = 500` raw units for collision detection. Watch real currents in
  `get_status` first, then set it about 1.5x above normal moves.
- Camera image order in the normvla frame (`camera_names`).
- Mark detection thresholds (`darken`, `blob_min_area_px`) under venue lighting.
