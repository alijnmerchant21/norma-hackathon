# Running the SO-101 loader policies (policy_run.py)

`policy_run.py` runs a SmolVLA checkpoint on the SO-101 through Station. It fixes the CPU/MPS
dtype crash of NormaCore's `run_policy.py` and plays each predicted chunk instead of only its first action.

All commands run from the smolvla_py folder of norma-core (it has the uv env with torch):

```
cd <norma-core>/software/ai/smolvla_py
```

Replace `A,B` with the Station camera indices used in training (the same `--cameras` as `train.py`).

## 1. Dry run with timing (arm does not move)

```
uv run python /Users/MXH37/Documents/dev/lerobot/guided_cell/policy_run.py \
  --checkpoint /Users/MXH37/Documents/dev/lerobot/guided_cell/policies/load/final \
  --task "Move the red box from the pickup area to the work area" \
  --bus-serial 5B7B015823 --cameras A,B --dry-run --max-cycles 4
```

Station must run (it reads real frames), but nothing is sent. Check the first line says `DRY RUN`,
then `dtypes: all ... float32`, `loaded in N s`, `first prediction OK`. Each cycle prints the goals it
would send and one summary line. If it says MPS failed, add `--device cpu`.

## 2. First real run, in the open, short chunks

Arm in a clear pose, box in place, hand on the e-stop.

```
uv run python /Users/MXH37/Documents/dev/lerobot/guided_cell/policy_run.py \
  --checkpoint /Users/MXH37/Documents/dev/lerobot/guided_cell/policies/load/final \
  --task "Move the red box from the pickup area to the work area" \
  --bus-serial 5B7B015823 --cameras A,B --execute 5 --max-seconds 30
```

The first line must say `REAL MOTION`. Ctrl+C stops at once (nothing more is sent, torque is not changed;
the arm holds its last goal).

## 3. Normal run

Same as 2 with `--execute 20` (the default) and `--max-seconds 40` (the default).

## What the numbers mean

One line per cycle (one cycle = one fetched frame + one inference + playing part of the chunk):

- `fetch N ms`: time to get a new frame from `inference/normvla` (about 0 to 100 ms; the queue runs at about 10 Hz).
- `predict N ms`: one full model inference (image decode, VLM, 10 denoising steps). The arm stands still during it.
- `sent K/N`: actions of this chunk sent (`would send` in a dry run). Less than N means a stop limit was reached.
- `first-vs-measured D ticks (joint ...)`: how far the first goal of the chunk is from where the arm is.
- `max step D ticks`: the largest change between two goals sent one after the other (including from the
  last goal of the previous chunk).
- `t=S s`: time since the control loop started (model loading not included).

At the start: `loaded in N s` is the model load time; add it to `--max-seconds` for the loader timeout.

Exit codes: 0 clean stop (max seconds, max cycles, or `arm is still: assuming the task is finished`),
2 setup or argument error, 3 safety stop (a goal jumps more than `--max-delta-ticks`, default 300, on arm
joints 0-4, or more than `--gripper-max-delta-ticks`, default 1200, on the gripper, which jumps when it opens
after closing on the box; the line names the joint and the limit), 4 an operation is not supported on MPS
(use `--device cpu`), 5 dtype error (paste the
printed list), 130/143 Ctrl+C or SIGTERM.

## Choosing --execute and --rate

- Keep `--rate 10`. The dataset and the `inference/normvla` queue run at about 10 Hz, so one action step of the
  chunk is 0.1 s of the demonstration. A lower rate (for example 5) plays the same path slower, which is safer
  for a first try but takes longer than the 8 s demo.
- Motion per cycle is `execute / rate` seconds, then the arm pauses for `predict` time. Pick `execute` so that
  the motion part is at least about twice the predict time:
  - predict under 0.5 s: `--execute 5` to `10` (reacts often).
  - predict 0.5 to 1.5 s: `--execute 15` to `20`.
  - predict 1.5 to 3 s: `--execute 25` to `30`.
  - predict over 3 s: `--execute 40` to `50`, or try the other device (`--device cpu` vs `mps`).
- Larger `execute` means fewer pauses but longer "blind" stretches (the chunk was planned from an older frame).
  The full 8 s push is about 80 steps, so `--execute 20` needs about 4 inferences.
- `--execute 1` behaves like NormaCore's `run_policy.py --auto` (one action per inference).

## Which policy for which sentence

| Loader phase | Checkpoint | Task sentence (guided.toml) |
|---|---|---|
| load | `policies/load/final` | `policy_load_task` = "Move the red box from the pickup area to the work area" |
| unload | `policies/unload/final` | `policy_unload_task` = "Move the red box from the work area to the drop-off area" |

The sentence must be exactly the one used when recording that dataset (`SO101_TASK`). If it was different,
change the `policy_*_task` lines, not the command.

## Putting it in guided.toml

The loader runs the command in a shell from `policy_cwd`, replaces `{task}` with `policy_load_task` or
`policy_unload_task`, keeps the last 20 output lines, kills the whole process group on e-stop or after
`policy_timeout_s`, and runs `go_home` for the SO-101 only after exit code 0. TOML literal strings
(single quotes) keep the double quotes around `{task}` as they are.

```
[loader]
arm = "so101"
mode = "policy"
policy_cwd = "/Users/MXH37/Documents/dev/lerobot/robot-pizza-cup/elrobot/norma-core/software/ai/smolvla_py"
policy_load_cmd = 'uv run python /Users/MXH37/Documents/dev/lerobot/guided_cell/policy_run.py --checkpoint /Users/MXH37/Documents/dev/lerobot/guided_cell/policies/load/final --task "{task}" --bus-serial 5B7B015823 --cameras A,B --execute 20 --max-seconds 40'
policy_unload_cmd = 'uv run python /Users/MXH37/Documents/dev/lerobot/guided_cell/policy_run.py --checkpoint /Users/MXH37/Documents/dev/lerobot/guided_cell/policies/unload/final --task "{task}" --bus-serial 5B7B015823 --cameras A,B --execute 20 --max-seconds 40'
policy_load_task = "Move the red box from the pickup area to the work area"
policy_unload_task = "Move the red box from the work area to the drop-off area"
policy_timeout_s = 120   # model load time + --max-seconds + margin
```

Set `policy_cwd` to where norma-core really is on the Mac. `policy_timeout_s` counts from process start, so
it must cover loading (see `loaded in N s` from the dry run) plus `--max-seconds`; the default 60 is too short.
Add `--device cpu` to both commands if MPS failed in the dry run.

## Hybrid: taught pre-pose, then the policy

Set `policy_unload_pre_pose = "unload_pre"` (or `policy_load_pre_pose`) in `[loader]`. The loader first moves the
SO-101 to that taught pose through cellbot's guards (e-stop, limits, zones, slow speed), then starts the policy
from there; if the move fails the policy is not started. Not taught: refused before any motion, with the commands.
Teach it with the arm near the box (run from `third_party/cellbot`, same env as the runner; `work` if the pose is
over the work area, else `clear`), then `go_home so101` in the runner:
`$PY -m cellbot.teach limp so101`, `$PY -m cellbot.teach save so101 unload_pre work`, `$PY -m cellbot.teach hold so101`.

Taught grasp, then the policy: `policy_unload_pre_seq` (or `policy_load_pre_seq`) is a step list in the
`taught_*_seq` syntax, run with cellbot's guards before the policy, under the same lock, with NO home appended.
Teach `grasp_pre` (above the box, gripper open, `work`) and `grasp` (gripper around the box, `work`), then
`policy_unload_pre_seq = ["gripper:open", "pose:grasp_pre", "pose:grasp", "gripper:close"]`. The teach tool stores
arm joints only, not the gripper: open/close it with the gripper steps. Do not set `*_pre_pose` too (refused).
Risks: the box must sit where the grasp was taught, within the gripper opening; the policy was trained starting
from home with an open gripper, so it may open the gripper (drop the box) when it starts.

## Taught sweep for unload

The box moves and rotates in the work area, so instead of a grasp the SO-101 pushes it out with the closed
gripper as a paddle: load stays `policy`, unload runs taught poses with all cellbot guards (slow, zones, e-stop).
Teach (humans, Station running; `cd $G/third_party/cellbot && export CELLBOT_MODE=station STATION_HOST=127.0.0.1
CELLBOT_CONFIG=$G/cell.guided.toml CELLBOT_STATE=$G/state`), gripper closed, then `limp` / `save` / `hold`:
`$PY -m cellbot.teach limp so101`, then per pose `$PY -m cellbot.teach save so101 <name> <zone>`, then `$PY -m cellbot.teach hold so101`.
- `sweep_start` (zone `work`): behind the box (the side away from the drop-off), low, gripper a few mm above the table.
- `sweep_mid` (optional, `work`): halfway along the push line, same height.  `sweep_end` (`clear`): over the drop-off,
  fully out of the work area. Optional `sweep_pre` (`work`): high above `sweep_start`, so the descent cannot hit the box.
In `[loader]`: `unload_mode = "taught"` and
`taught_unload_seq = ["gripper:close", "pose:sweep_start", "pose:sweep_end"]` (home is appended). Validated before any
motion; a failed step stops the rest. After teaching (limp), run `go_home so101` once in the runner before anything else.

## Self-test (no robot, no torch needed)

```
python3 /Users/MXH37/Documents/dev/lerobot/guided_cell/policy_run.py --self-test
```

## Fallback and the loader data flywheel

- `[loader] policy_<phase>_fallback_cmd` (guided.toml): a second command per phase, run by `load_part` / `unload_part` with `variant="fallback"` (CLI `unload fallback`): same `{task}`, `policy_cwd`, `policy_timeout_s` and e-stop kill, never the primary's pre-pose / pre-seq. `policy_<phase>_fallback_pre_seq` (taught steps, same syntax as `policy_<phase>_pre_seq`, no home) runs first if set; with `policy_<phase>_fallback_cmd = ""` the fallback is then those steps + the primary `policy_<phase>_cmd`. Both empty = refused, nothing moved. Now: unload primary = raw `policy_run.py`, unload fallback = `unload_hybrid.py run` (taught hover + pre-grasp, then the same policy).
- Operator loop: `unload_part` -> `see_part` from a clear pose -> box still in the work area: label fail, `unload_part(variant="fallback")` once -> look -> label; still there: ask a human.
- Every attempt that started (any mode) appends a line to `<state>/loader_episodes.jsonl`: id, phase, mode, variant, cmd, t_start/t_end (unix and local ISO), seconds, exit_code, killed, ok, part, outcome, and `sequence` (the taught pre-seq steps) when one ran. Refusals (nothing moved) are not logged.
- `label_loader_attempt(success, note, attempt_id="")` (CLI `label ok|fail [note]`) sets outcome on the latest (or given) attempt; load success = part in the work area and seen, unload success = work area empty.
- `loader_stats()` (CLI `loader`): attempts / successes / fails / unlabelled / success rate / mean seconds per phase+variant, the last 10, and `dataset_ranges`.
- Station records everything continuously into NormFS. To grow the training set, cut the `dataset_ranges` (successful attempts) out of that recording, the same way as ranges marked with TAG in Station's web UI, and fine-tune the policy on them. The cellbot Station client has no tag call, so the log's time ranges are the only link to the recording.
- Time ranges cover the whole call (pre-pose/pre-seq and the model load included): trim the start to where the arm starts moving.
- A `policy_run.py` exit code 3 (jump guard) or a timeout leaves the so101 at an unknown position: a human looks, then `go_home so101`, before the fallback.
