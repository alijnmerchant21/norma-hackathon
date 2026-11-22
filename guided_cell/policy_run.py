#!/usr/bin/env python3
"""Chunked SmolVLA policy runner for the SO-101 loader (NormaCore Station).

Run from norma-core/software/ai/smolvla_py (its uv env has torch/transformers):

    uv run python /path/to/policy_run.py --checkpoint <dir> --task "<sentence>" \
        --bus-serial 5B7B015823 --cameras 0,1 [--dry-run] [--execute 20] [--rate 10]

Differences from NormaCore scripts/run_policy.py (the reference):
  * dtype fix: the model is converted to float32 BEFORE the checkpoint is loaded, then every
    parameter and buffer is checked (see load_policy). The reference crashes on CPU/MPS because
    the transformers-built backbone and expert are bfloat16 while the plain nn.Linear k_proj/v_proj
    that smolvlm_with_expert.py builds for the cross-attention layers are float32; on CUDA
    training this is hidden by torch.autocast (train.py line 212).
  * chunked execution: one inference gives chunk_size (50) actions; the first --execute N are sent
    at --rate Hz, one sync-write each, then a fresh frame is fetched and the policy runs again.
  * safety: delta guard on every planned goal before any of the chunk is sent, exit code 3.

Exit codes: 0 clean stop (max seconds / max cycles / arm still), 2 bad arguments or setup,
3 safety stop (delta guard), 4 op not supported on the device, 5 dtype error, 130/143 Ctrl+C/SIGTERM.

The scheduling, guard, clipping and stop logic (section CORE) needs no torch, so --self-test runs
anywhere with plain Python 3.10+.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import re
import signal
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")   # must be set before torch is imported

# Inference/normvla updates at ~10 Hz (run_policy.py line 107) and the dataset exporter counts
# "10 per s" goal changes (export_so101.sh line 5); one dataset row = one action step.
DEFAULT_RATE_HZ = 10.0
STILL_TICKS = 12
JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
GRIPPER_JOINT = 5                       # last joint; has its own delta limit (--gripper-max-delta-ticks)

QUEUE_ID = "inference/normvla"          # run_policy.py line 58
ST3215_TARGET_POS_REGISTER = 0x2A       # run_policy.py line 59


class Fatal(Exception):
    def __init__(self, code: int, msg: str):
        super().__init__(msg)
        self.code = code


def log(msg: str = "") -> None:
    print(msg, flush=True)


# =============================================================================
# CORE (pure Python, no torch): goal mapping, guard, stillness, chunk scheduler
# =============================================================================

def goals_to_raw(goal_rows: list[list[float]], ranges: list[tuple[int, int]]) -> list[list[int]]:
    """Mirror of run_policy.py lines 195-200 for every row: clamp goal_norm to [0, 1],
    raw = round(rmin + g * (rmax - rmin)), clipped to the calibrated range."""
    out = []
    for row in goal_rows:
        raws = []
        for g, (rmin, rmax) in zip(row, ranges):
            g = min(1.0, max(0.0, float(g)))
            raw = int(round(rmin + g * (rmax - rmin)))
            lo, hi = min(rmin, rmax), max(rmin, rmax)       # identical to the reference when rmin <= rmax
            raws.append(max(lo, min(hi, raw)))
        out.append(raws)
    return out


def max_abs_delta(a: list[int], b: list[int]) -> tuple[int, int]:
    """(max |a-b|, joint index)."""
    best, j = 0, 0
    for i, (x, y) in enumerate(zip(a, b)):
        d = abs(int(x) - int(y))
        if d > best:
            best, j = d, i
    return best, j


def joint_label(j: int, motor_ids: list[int]) -> str:
    name = JOINT_NAMES[j] if j < len(JOINT_NAMES) else f"joint{j}"
    mid = motor_ids[j] if j < len(motor_ids) else "?"
    return f"joint {j} (motor {mid}, {name})"


def delta_over_limit(a: list[int], b: list[int], max_delta: int,
                     gripper_max_delta: int) -> tuple[int, int, int] | None:
    """Largest |a-b| that exceeds its joint's limit, as (delta, joint, limit), or None.
    Arm joints use max_delta, the gripper (GRIPPER_JOINT) uses gripper_max_delta; a limit of 0 = off."""
    worst = None
    for i, (x, y) in enumerate(zip(a, b)):
        d = abs(int(x) - int(y))
        lim = gripper_max_delta if i == GRIPPER_JOINT else max_delta
        if lim and d > lim and (worst is None or d > worst[0]):
            worst = (d, i, lim)
    return worst


def limit_label(j: int) -> str:
    return "--gripper-max-delta-ticks" if j == GRIPPER_JOINT else "--max-delta-ticks"


@dataclass
class GuardResult:
    ok: bool
    first_delta: int = 0       # first goal vs measured position
    first_joint: int = 0
    step_delta: int = 0        # largest change between consecutive goals (incl. last sent -> first)
    step_joint: int = 0
    reason: str = ""


def check_chunk(raws: list[list[int]], measured: list[int], last_sent: list[int] | None,
                max_delta: int, motor_ids: list[int], gripper_max_delta: int = 1200) -> GuardResult:
    """Check every goal of the slice about to be sent, before anything is sent.
    Arm joints are limited by max_delta, the gripper joint by gripper_max_delta."""
    r = GuardResult(ok=True)
    r.first_delta, r.first_joint = max_abs_delta(raws[0], measured)
    bad = delta_over_limit(raws[0], measured, max_delta, gripper_max_delta)
    if bad:
        d, j, lim = bad
        r.ok = False
        r.reason = (f"first goal is {d} ticks from the measured position on "
                    f"{joint_label(j, motor_ids)} (limit {lim}, {limit_label(j)})")
        return r
    prev = last_sent
    for k, raw in enumerate(raws):
        if prev is not None:
            d, j = max_abs_delta(raw, prev)
            if d > r.step_delta:
                r.step_delta, r.step_joint = d, j
            bad = delta_over_limit(raw, prev, max_delta, gripper_max_delta)
            if bad:
                d, j, lim = bad
                r.ok = False
                what = "last sent goal -> action 0" if k == 0 else f"action {k - 1} -> action {k}"
                r.reason = (f"{what} jumps {d} ticks on {joint_label(j, motor_ids)} "
                            f"(limit {lim}, {limit_label(j)})")
                return r
        prev = raw
    return r


class StillDetector:
    """Still = the arm has moved at least once, and over the last window_s seconds of measured
    positions no joint moved by threshold ticks or more."""

    def __init__(self, window_s: float, threshold: int = STILL_TICKS):
        self.window_s = window_s
        self.threshold = threshold
        self.samples: deque[tuple[float, list[int]]] = deque()
        self.first: list[int] | None = None
        self.moved = False

    def add(self, t: float, pos: list[int]) -> None:
        if self.first is None:
            self.first = list(pos)
        elif max_abs_delta(pos, self.first)[0] >= self.threshold:
            self.moved = True
        self.samples.append((t, list(pos)))
        # keep one sample at or before t - window (so the window is fully covered)
        while len(self.samples) >= 2 and self.samples[1][0] <= t - self.window_s:
            self.samples.popleft()

    def is_still(self, now: float) -> bool:
        if self.window_s <= 0 or not self.moved or len(self.samples) < 2:
            return False
        if self.samples[0][0] > now - self.window_s:
            return False                      # not enough history yet
        n = len(self.samples[0][1])
        for j in range(n):
            vals = [p[j] for _, p in self.samples]
            if max(vals) - min(vals) >= self.threshold:
                return False
        return True


@dataclass
class RunConfig:
    execute: int = 20
    rate: float = DEFAULT_RATE_HZ
    max_delta: int = 300
    gripper_max_delta: int = 1200
    dry_run: bool = False
    max_seconds: float = 40.0
    max_cycles: int = 0
    still_s: float = 3.0
    motor_ids: list[int] = field(default_factory=lambda: [1, 2, 3, 4, 5, 6])
    print_actions: bool = False
    # start ramp (cycle 1 only): interpolate measured -> first goal in steps of at most ramp_step ticks.
    # 0 = off here (the CLI turns it on); ramp_max = largest arm-joint distance it will ramp over.
    ramp_step: int = 0
    ramp_max: int = 600
    # gripper hold (0 = off): see apply_grip_hold
    grip_hold: int = 0
    grip_thresh: int = 80
    grip_release_steps: int = 5
    grip_squeeze_max: int = 0
    log_gripper: bool = False


def ramp_to(measured: list[int], target: list[int], step: int) -> list[list[int]]:
    """Goals from measured towards target (target itself excluded), every joint moving at most
    `step` ticks per goal. Empty when target is already within one step."""
    d = max_abs_delta(target, measured)[0]
    if step <= 0 or d <= step:
        return []
    n = -(-d // step)                       # ceil
    return [[int(round(m + (t - m) * i / n)) for m, t in zip(measured, target)] for i in range(1, n)]


@dataclass
class GripInfo:
    gripping: bool = False     # at chunk start the last command was >= thresh past the measured gripper
    sign: int = 0              # +1 / -1: direction of "more closed" in ticks
    changed: int = 0           # gripper goals of this chunk that were changed
    released_at: int = -1      # action index from which the policy's "open" was let through


def apply_grip_hold(raws: list[list[int]], measured_g: int, last_cmd_g: int | None, hold: int, thresh: int,
                    release_steps: int, squeeze_max: int, grange: tuple[int, int] = (0, 4095)
                    ) -> tuple[list[list[int]], GripInfo]:
    """Keep squeezing an object the gripper is holding.

    Gripping = the last sent gripper goal is at least `thresh` ticks past the measured position (the
    object stops the jaws). The squeeze of a goal is how far it is past the measured position in that
    direction. While gripping, each goal gets squeeze >= hold (and <= squeeze_max if set), so a new
    chunk that starts at the measured position cannot relax the grip. The policy may still open:
    once `release_steps` goals in a row are more than `thresh` ticks on the open side, those goals
    and the rest of the chunk pass unchanged."""
    out = [list(r) for r in raws]
    info = GripInfo()
    if last_cmd_g is None or (hold <= 0 and squeeze_max <= 0):
        return out, info
    e = last_cmd_g - measured_g
    if abs(e) < thresh:
        return out, info
    s = 1 if e > 0 else -1
    info.gripping, info.sign = True, s
    lo, hi = min(grange), max(grange)
    opens = 0
    for k, r in enumerate(out):
        sq = s * (r[GRIPPER_JOINT] - measured_g)
        if sq < -thresh:
            opens += 1
            if opens >= release_steps:
                info.released_at = k
                for kk in range(k - release_steps + 1, k + 1):     # let the whole open run through
                    if out[kk][GRIPPER_JOINT] != raws[kk][GRIPPER_JOINT]:
                        info.changed -= 1
                    out[kk][GRIPPER_JOINT] = raws[kk][GRIPPER_JOINT]
                break
        else:
            opens = 0
        new = max(sq, hold) if hold > 0 else sq
        if squeeze_max > 0:
            new = min(new, squeeze_max)
        g = max(lo, min(hi, measured_g + s * new))
        if g != r[GRIPPER_JOINT]:
            info.changed += 1
            r[GRIPPER_JOINT] = g
    return out, info


async def control_loop(cfg: RunConfig, fetch, predict, send, clock=time.monotonic, sleep=asyncio.sleep,
                       observe=None) -> int:
    """fetch() -> (frame, measured_ticks, ranges)   [async]
    predict(frame) -> list of goal_norm rows (the full chunk, unnormalized)
    send(raw_goals: list[int])                       [async]
    observe() -> latest measured gripper ticks or None (optional, for --log-gripper)
    Returns the exit code."""
    period = 1.0 / cfg.rate
    still = StillDetector(0.0 if cfg.dry_run else cfg.still_s)
    t_start = clock()
    last_sent: list[int] | None = None
    cycle = 0
    total_sent = 0
    while True:
        if cfg.max_cycles and cycle >= cfg.max_cycles:
            log(f"stop: reached --max-cycles {cfg.max_cycles} ({total_sent} actions sent)")
            return 0
        if clock() - t_start >= cfg.max_seconds:
            log(f"stop: reached --max-seconds {cfg.max_seconds:g} ({total_sent} actions sent)")
            return 0
        cycle += 1
        t0 = clock()
        frame, measured, ranges = await fetch()
        t1 = clock()
        still.add(t1, measured)
        if still.is_still(t1):
            log(f"arm is still: assuming the task is finished (no joint moved {STILL_TICKS} ticks or more "
                f"in the last {cfg.still_s:g} s; {total_sent} actions sent)")
            return 0
        chunk = predict(frame)
        t2 = clock()
        n = min(cfg.execute, len(chunk))
        raws = goals_to_raw(chunk[:n], ranges)
        policy_g = [r[GRIPPER_JOINT] for r in raws]
        # In a dry run nothing was executed, so the previous plan is not where the arm is.
        prev = None if cfg.dry_run else last_sent
        raws, gi = apply_grip_hold(raws, measured[GRIPPER_JOINT], prev[GRIPPER_JOINT] if prev else None,
                                   cfg.grip_hold, cfg.grip_thresh, cfg.grip_release_steps, cfg.grip_squeeze_max,
                                   ranges[GRIPPER_JOINT])
        fetch_ms, predict_ms = (t1 - t0) * 1000, (t2 - t1) * 1000
        pf_delta, pf_joint = max_abs_delta(raws[0], measured)
        ramp: list[list[int]] = []
        if cycle == 1 and cfg.ramp_step > 0:
            bad = delta_over_limit(raws[0], measured, cfg.ramp_max, cfg.gripper_max_delta)
            if bad:
                d, j, lim = bad
                log(f"cycle {cycle}: fetch {fetch_ms:.0f} ms, predict {predict_ms:.0f} ms, sent 0/{n}")
                log(f"SAFETY STOP: first goal is {d} ticks from the measured position on "
                    f"{joint_label(j, cfg.motor_ids)} (limit {lim}, "
                    f"{'--gripper-max-delta-ticks' if j == GRIPPER_JOINT else '--ramp-max-ticks'}). "
                    f"Nothing was sent.")
                log(f"  measured {measured}")
                log(f"  planned  {raws[0]} ... {raws[-1]}")
                return 3
            ramp = ramp_to(measured, raws[0], cfg.ramp_step)
        plan = ramp + raws
        g = check_chunk(plan, measured, prev, cfg.max_delta, cfg.motor_ids, cfg.gripper_max_delta)
        if not g.ok:
            log(f"cycle {cycle}: fetch {fetch_ms:.0f} ms, predict {predict_ms:.0f} ms, sent 0/{len(plan)}")
            log(f"SAFETY STOP: {g.reason}. Nothing of this chunk was sent.")
            log(f"  measured {measured}")
            log(f"  planned  {plan[0]} ... {plan[-1]}")
            return 3
        if ramp:
            log(f"cycle {cycle}: start ramp of {len(ramp)} steps (<= {cfg.ramp_step} ticks each) to the first goal, "
                f"{pf_delta} ticks away on {joint_label(pf_joint, cfg.motor_ids)}")
        if cfg.log_gripper:
            hold = "off"
            if cfg.grip_hold > 0 or cfg.grip_squeeze_max > 0:
                hold = (f"gripping (dir {gi.sign:+d}), changed {gi.changed}/{n}"
                        + (f", policy opens from a{gi.released_at:02d}" if gi.released_at >= 0 else "")
                        if gi.gripping else "not gripping")
            log(f"  grip cycle {cycle}: measured {measured[GRIPPER_JOINT]}, last sent "
                f"{prev[GRIPPER_JOINT] if prev else '-'}, policy {policy_g[0]}..{policy_g[-1]} "
                f"(min {min(policy_g)}, max {max(policy_g)}), hold: {hold}")
        t_play = clock()
        sent = 0
        for k, raw in enumerate(plan):
            delay = t_play + k * period - clock()
            if delay > 0:
                await sleep(delay)
            if clock() - t_start >= cfg.max_seconds:
                break
            await send(raw)
            sent += 1
            total_sent += 1
            last_sent = raw
            if cfg.print_actions:
                log(f"    a{k:02d} {raw}")
            if cfg.log_gripper:
                pk = k - len(ramp)
                m = observe() if observe else None
                log(f"    grip t={clock() - t_start:5.1f} {'ramp' if pk < 0 else f'a{pk:02d}'} "
                    f"sent {raw[GRIPPER_JOINT]} policy {policy_g[pk] if pk >= 0 else '-'} "
                    f"measured {m if m is not None else '?'}")
        # let the last goal play for one period before looking again
        delay = t_play + sent * period - clock()
        if sent and delay > 0:
            await sleep(delay)
        log(f"cycle {cycle}: fetch {fetch_ms:.0f} ms, predict {predict_ms:.0f} ms, "
            f"{'would send' if cfg.dry_run else 'sent'} {sent}/{len(plan)}, "
            f"first-vs-measured {pf_delta} ticks ({joint_label(pf_joint, cfg.motor_ids)}), "
            f"max step {g.step_delta} ticks, t={clock() - t_start:.1f} s")


# =============================================================================
# SELF-TEST (no torch, no Station, no weights)
# =============================================================================

def self_test() -> int:
    results = []

    def check(name, cond, detail=""):
        results.append(bool(cond))
        log(f"{'PASS' if cond else 'FAIL'}  {name}{('  ' + detail) if detail else ''}")

    RANGES = [(0, 4095)] * 6

    class Sim:
        """Virtual clock + an arm that reaches each sent goal at once."""

        def __init__(self, start_norm=0.5, predict_s=1.5, chunk_fn=None):
            self.t = 0.0
            self.pos = [int(round(start_norm * 4095))] * 6
            self.sent: list[tuple[float, list[int]]] = []
            self.predict_s = predict_s
            self.chunk_fn = chunk_fn
            self.cycle = 0
            self.frozen = False

        def clock(self):
            return self.t

        async def sleep(self, s):
            self.t += s
            await asyncio.sleep(0)

        async def fetch(self):
            self.t += 0.03
            return None, list(self.pos), RANGES

        def predict(self, frame):
            self.t += self.predict_s
            self.cycle += 1
            return self.chunk_fn(self.cycle, self.pos)

        async def send(self, raw):
            self.sent.append((self.t, list(raw)))
            if not self.frozen:
                self.pos = list(raw)

    def ramp(step_norm=0.002, n=50):
        def fn(cyc, pos):
            base = [p / 4095 for p in pos]
            return [[b + step_norm * (k + 1) for b in base] for k in range(n)]
        return fn

    def run(cfg, sim):
        return asyncio.run(control_loop(cfg, sim.fetch, sim.predict, sim.send, sim.clock, sim.sleep))

    # 1. chunk scheduling: 3 cycles x 20 actions, spaced 1/rate apart within a chunk
    sim = Sim(chunk_fn=ramp())
    code = run(RunConfig(execute=20, rate=10, max_cycles=3, still_s=0), sim)
    gaps = [round(b[0] - a[0], 6) for a, b in zip(sim.sent[:19], sim.sent[1:20])]
    check("scheduling: 3 cycles x 20 actions, exit 0", code == 0 and len(sim.sent) == 60, f"sent={len(sim.sent)}")
    check("scheduling: 0.1 s between actions of a chunk", all(abs(g - 0.1) < 1e-6 for g in gaps), f"{set(gaps)}")
    check("scheduling: predict runs once per chunk", sim.cycle == 3)

    # 2. --execute 1 reproduces the reference (one action per inference)
    sim = Sim(chunk_fn=ramp())
    code = run(RunConfig(execute=1, max_cycles=4, still_s=0), sim)
    check("execute 1: one action per cycle", code == 0 and len(sim.sent) == 4 and sim.cycle == 4)

    # 3. execute larger than the chunk is clamped
    sim = Sim(chunk_fn=ramp(n=50))
    run(RunConfig(execute=80, max_cycles=1, still_s=0), sim)
    check("execute 80 with chunk 50: 50 sent", len(sim.sent) == 50)

    # 4. delta guard inside a chunk: jump at action 7 -> exit 3, nothing of the chunk sent
    def jump(cyc, pos):
        rows = ramp()(cyc, pos)
        for k in range(7, 50):
            rows[k][2] += 400 / 4095
        return rows
    sim = Sim(chunk_fn=jump)
    code = run(RunConfig(execute=20, max_delta=300, max_cycles=3, still_s=0), sim)
    check("guard: step jump -> exit 3, 0 sent", code == 3 and len(sim.sent) == 0)

    # 5. first goal far from the measured position -> exit 3
    sim = Sim(chunk_fn=lambda c, p: [[0.9] * 6 for _ in range(50)])
    code = run(RunConfig(execute=20, max_delta=300, max_cycles=3, still_s=0), sim)
    check("guard: first goal vs measured -> exit 3, 0 sent", code == 3 and len(sim.sent) == 0)

    # 6. jump between last sent goal of chunk 1 and first goal of chunk 2 (arm lags, measured stays)
    def flip(cyc, pos):
        v = 0.5 + (0.06 if cyc == 1 else 0.0)
        return [[v] * 6 for _ in range(50)]
    sim = Sim(chunk_fn=flip)
    sim.frozen = True                    # measured stays at 0.5 (2048); chunk 1 goal 0.56 (2293) = 245 ticks
    code = run(RunConfig(execute=5, max_delta=200, max_cycles=3, still_s=0), sim)
    check("guard: limit 200 with 245-tick first goal -> exit 3", code == 3 and len(sim.sent) == 0)
    sim = Sim(chunk_fn=lambda c, p: [[0.5 + (0.06 if c == 1 else -0.02)] * 6 for _ in range(50)])
    sim.frozen = True                    # chunk 1 at 2293, chunk 2 at 1966: 327-tick jump between sent goals
    code = run(RunConfig(execute=5, max_delta=300, max_cycles=3, still_s=0), sim)
    check("guard: last sent -> next chunk jump 327 > 300 -> exit 3 after chunk 1",
          code == 3 and len(sim.sent) == 5)

    # 7. dry run never calls the real sender and skips the cross-chunk check
    sim = Sim(chunk_fn=flip)
    sim.frozen = True
    calls = []

    async def dry_send(raw):
        calls.append(raw)
    cfg = RunConfig(execute=5, max_delta=300, max_cycles=3, dry_run=True)
    code = asyncio.run(control_loop(cfg, sim.fetch, sim.predict, dry_send, sim.clock, sim.sleep))
    check("dry run: exit 0, stillness off, goals go to the dry sender only",
          code == 0 and len(calls) == 15 and len(sim.sent) == 0)

    # 8. clipping and rounding (reference formula)
    r = goals_to_raw([[1.3, -0.2, 0.5, 0.25, 0.0, 1.0]],
                     [(100, 3000), (100, 3000), (1000, 2000), (0, 4095), (500, 600), (2047, 3100)])[0]
    check("clip: goals clipped to calibrated range, rounding as reference",
          r == [3000, 100, 1500, 1024, 500, 3100], f"{r}")

    # 9. stillness: moves for 2 cycles, then the policy holds still -> exit 0 before max-seconds
    def move_then_hold(cyc, pos):
        if cyc <= 2:
            return ramp(0.003)(cyc, pos)
        return [[p / 4095 for p in pos] for _ in range(50)]
    sim = Sim(chunk_fn=move_then_hold)
    code = run(RunConfig(execute=20, max_cycles=0, max_seconds=40, still_s=3.0), sim)
    check("still: exit 0 after the arm stops", code == 0 and sim.t < 20 and sim.cycle >= 3, f"t={sim.t:.1f}s")

    # 10. stillness never fires before the arm has moved
    sim = Sim(chunk_fn=lambda c, p: [[q / 4095 for q in p] for _ in range(50)])
    code = run(RunConfig(execute=20, max_seconds=12, still_s=3.0), sim)
    check("still: not triggered before first motion (stops on max-seconds)", code == 0 and sim.t >= 12)

    # 11. max-seconds is honoured mid-chunk
    sim = Sim(chunk_fn=ramp(0.0005), predict_s=0.5)
    code = run(RunConfig(execute=50, max_seconds=3.0, still_s=0), sim)
    check("max-seconds: stop mid-chunk, no send after the limit",
          code == 0 and all(t < 3.0 for t, _ in sim.sent), f"last send t={sim.sent[-1][0]:.2f}s")

    # 12. gripper has its own limit (closed on a box: its last goal is past the measured position)
    #     end to end: chunk 1 sends the gripper 320 ticks past measured (which does not follow), chunk 2
    #     starts at the measured value -> last sent -> action 0 jumps 320 on the gripper: must pass
    def grasp(j):
        def fn(cyc, pos):
            row = [2048] * 6
            if cyc == 1:
                row[j] += 320
            return [[v / 4095 for v in row] for _ in range(50)]
        return fn
    sim = Sim(chunk_fn=grasp(5))
    sim.pos, sim.frozen = [2048] * 6, True
    code = run(RunConfig(execute=5, max_delta=300, gripper_max_delta=1200, max_cycles=2, still_s=0), sim)
    check("gripper: run with 320-tick grasp jump completes, exit 0", code == 0 and len(sim.sent) == 10)
    sim = Sim(chunk_fn=grasp(2))
    sim.pos, sim.frozen = [2048] * 6, True
    code = run(RunConfig(execute=5, max_delta=300, gripper_max_delta=1200, max_cycles=2, still_s=0), sim)
    check("arm: same 320-tick jump on joint 2 -> exit 3 at first goal", code == 3 and len(sim.sent) == 0)

    # exact cases on check_chunk (measured 2048 on every joint)
    M = [2048] * 6
    MID = [1, 2, 3, 4, 5, 6]

    def goal(j, d):
        g = list(M)
        g[j] += d
        return g
    g = check_chunk([M] * 5, M, goal(5, 320), 300, MID, 1200)
    check("gripper: last sent -> action 0 jump 320 passes (gripper limit 1200)", g.ok, g.reason)
    g = check_chunk([M] * 5, M, goal(5, 1250), 300, MID, 1200)
    check("gripper: last sent -> action 0 jump 1250 > 1200 stops",
          not g.ok and "limit 1200" in g.reason and "gripper" in g.reason, g.reason)
    g = check_chunk([M] * 5, M, goal(1, 320), 300, MID, 1200)
    check("arm: last sent -> action 0 jump 320 on joint 1 still stops (limit 300)",
          not g.ok and "limit 300" in g.reason and "joint 1" in g.reason, g.reason)
    g = check_chunk([goal(5, 320)] * 5, M, None, 300, MID, 1200)
    check("gripper: first goal 320 from measured passes", g.ok, g.reason)
    g = check_chunk([goal(5, 1250)] * 5, M, None, 300, MID, 1200)
    check("gripper: first goal 1250 from measured stops", not g.ok and "limit 1200" in g.reason, g.reason)
    g = check_chunk([goal(4, 320)] * 5, M, None, 300, MID, 1200)
    check("arm: first goal 320 from measured on joint 4 stops", not g.ok and "limit 300" in g.reason, g.reason)
    g = check_chunk([M, M, goal(5, 320), goal(5, 320)], M, None, 300, MID, 1200)
    check("gripper: step 320 within a chunk passes", g.ok, g.reason)
    g = check_chunk([M, M, goal(5, 1250)], M, None, 300, MID, 1200)
    check("gripper: step 1250 within a chunk stops", not g.ok and "action 1 -> action 2" in g.reason, g.reason)
    g = check_chunk([M, M, goal(0, 320)], M, None, 300, MID, 1200)
    check("arm: step 320 within a chunk on joint 0 stops", not g.ok and "limit 300" in g.reason, g.reason)

    # 13. start ramp: first goal 301 ticks away on joint 2 (the unload start) -> ramped, no stop
    def const(row):
        return lambda c, p: [[v / 4095 for v in row] for _ in range(50)]
    sim = Sim(chunk_fn=const(goal(2, 301)))
    sim.pos = list(M)
    code = run(RunConfig(execute=5, max_delta=300, max_cycles=1, still_s=0, ramp_step=100), sim)
    steps = [max_abs_delta(b[1], a[1])[0] for a, b in zip(sim.sent, sim.sent[1:])]
    first = max_abs_delta(sim.sent[0][1], M)[0] if sim.sent else -1
    check("ramp: 301-tick start on elbow -> 3 ramp goals + 5, exit 0",
          code == 0 and len(sim.sent) == 8 and first <= 100 and max(steps) <= 100, f"sent={len(sim.sent)} first={first}")
    check("ramp: last ramp goal then policy goals", sim.sent[3][1] == goal(2, 301) and sim.sent[2][1][2] < 2048 + 301)
    sim = Sim(chunk_fn=const(goal(2, 700)))
    sim.pos = list(M)
    code = run(RunConfig(execute=5, max_delta=300, max_cycles=1, still_s=0, ramp_step=100, ramp_max=600), sim)
    check("ramp: 700 > ramp_max 600 -> exit 3, 0 sent", code == 3 and len(sim.sent) == 0)
    sim = Sim(chunk_fn=const(goal(2, 50)))
    sim.pos = list(M)
    code = run(RunConfig(execute=5, max_cycles=1, still_s=0, ramp_step=100), sim)
    check("ramp: within one step -> no ramp", code == 0 and len(sim.sent) == 5)

    # 14. grip hold (closed = lower ticks here: measured 1700 against the box, last sent 1400)
    def rows_g(gs):
        out = []
        for v in gs:
            r = list(M)
            r[GRIPPER_JOINT] = v
            out.append(r)
        return out
    o, gi = apply_grip_hold(rows_g([1700] * 6), 1700, 1400, 200, 80, 5, 0)
    check("hold: chunk starting at measured is held 200 past it",
          gi.gripping and gi.sign == -1 and all(r[5] == 1500 for r in o) and gi.changed == 6, f"{[r[5] for r in o]}")
    o, gi = apply_grip_hold(rows_g([1300] * 3), 1700, 1400, 200, 80, 5, 0)
    check("hold: goals squeezing harder pass unchanged", all(r[5] == 1300 for r in o) and gi.changed == 0)
    o, gi = apply_grip_hold(rows_g([1700] * 3), 1700, 1650, 200, 80, 5, 0)
    check("hold: not gripping (lag 50 < 80) -> unchanged", not gi.gripping and all(r[5] == 1700 for r in o))
    o, gi = apply_grip_hold(rows_g([1700] * 3 + [2000] * 10), 1700, 1400, 200, 80, 5, 0)
    gs = [r[5] for r in o]
    check("hold: 5 open goals in a row release the grip",
          gi.released_at == 7 and gs[:3] == [1500] * 3 and gs[3:] == [2000] * 10, f"{gs}")
    o, gi = apply_grip_hold(rows_g([1700, 2000, 2000, 1700, 2000]), 1700, 1400, 200, 80, 5, 0)
    check("hold: short open blips do not release", gi.released_at == -1 and all(r[5] == 1500 for r in o))
    o, gi = apply_grip_hold(rows_g([1300] * 3), 1700, 1400, 0, 80, 5, 100)
    check("squeeze max: 400 past measured capped to 100", all(r[5] == 1600 for r in o), f"{[r[5] for r in o]}")
    o, gi = apply_grip_hold(rows_g([2000] * 3), 2000, 2350, 200, 80, 5, 0)
    check("hold: other direction (closed = higher)", gi.sign == 1 and all(r[5] == 2200 for r in o))
    o, gi = apply_grip_hold(rows_g([1700] * 3), 1700, None, 200, 80, 5, 0)
    check("hold: first chunk (nothing sent yet) unchanged", not gi.gripping and all(r[5] == 1700 for r in o))

    # 15. end to end: chunk 1 closes 300 past a box (gripper does not follow), chunk 2 relaxes to measured
    def relax(cyc, pos):
        row = list(M)
        row[GRIPPER_JOINT] = 2048 - 300 if cyc == 1 else 2048
        return [[v / 4095 for v in row] for _ in range(50)]
    sim = Sim(chunk_fn=relax)
    sim.pos, sim.frozen = list(M), True
    run(RunConfig(execute=5, max_cycles=2, still_s=0, grip_hold=200), sim)
    check("hold end to end: chunk 2 gripper held at 1848, not relaxed to 2048",
          len(sim.sent) == 10 and all(r[5] == 1848 for _, r in sim.sent[5:]), f"{[r[5] for _, r in sim.sent]}")
    sim = Sim(chunk_fn=relax)
    sim.pos, sim.frozen = list(M), True
    run(RunConfig(execute=5, max_cycles=2, still_s=0, grip_hold=0), sim)
    check("hold off: chunk 2 relaxes to 2048 (the suspected bug)", all(r[5] == 2048 for _, r in sim.sent[5:]))

    ok = all(results)
    log(f"\nself-test: {'ALL PASS' if ok else 'FAILURES'} ({sum(results)}/{len(results)})")
    return 0 if ok else 1


# =============================================================================
# REAL RUN (torch, smolvla, Station)
# =============================================================================

def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--self-test", action="store_true", help="test scheduling/guard/stop logic, no torch needed")
    p.add_argument("--checkpoint", type=Path, help="dir with config.json, model.safetensors, stats.safetensors")
    p.add_argument("--task", help="sentence the policy was trained with")
    p.add_argument("--server", default="localhost", help="Station host")
    p.add_argument("--bus-serial", help="ST3215 bus serial of the SO-101 follower")
    p.add_argument("--motor-ids", default="1,2,3,4,5,6", help="same order as the joints in the Frame")
    p.add_argument("--cameras", default="0,1", help="Station image indices for cam0,cam1 (same as training)")
    p.add_argument("--norma-core", type=Path, default=None,
                   help="norma-core repo root (default: three levels above the current directory)")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    p.add_argument("--attn", default=None, choices=["sdpa", "eager"],
                   help="attention implementation (sets SMOLVLA_ATTN_IMPL; default sdpa)")
    p.add_argument("--execute", type=int, default=20, help="actions of each chunk to play before predicting again")
    p.add_argument("--rate", type=float, default=None, help=f"Hz at which actions are sent (default {DEFAULT_RATE_HZ:g})")
    p.add_argument("--max-delta-ticks", type=int, default=300, help="safety limit for arm joints 0-4, 0 = off")
    p.add_argument("--gripper-max-delta-ticks", type=int, default=1200,
                   help="safety limit for the gripper (joint 5), 0 = off")
    p.add_argument("--dry-run", action="store_true", help="never send; print goals and timings")
    p.add_argument("--max-seconds", type=float, default=40.0, help="stop after this long (from the first fetch)")
    p.add_argument("--max-cycles", type=int, default=0, help="stop after N inferences, 0 = unlimited")
    p.add_argument("--stop-when-still", type=float, default=0.0,
                   help="seconds without motion = done, 0 = off (default off)")
    p.add_argument("--ramp-step-ticks", type=int, default=100,
                   help="first cycle: ramp from the measured pose to the first goal in steps of at most this many "
                        "ticks (0 = off, the first goal is then sent directly)")
    p.add_argument("--ramp-max-ticks", type=int, default=600,
                   help="largest arm-joint distance to the first goal that the start ramp accepts")
    p.add_argument("--grip-hold-ticks", type=int, default=0,
                   help="while the gripper holds an object, keep every gripper goal at least this many ticks "
                        "past the measured position (0 = off; try 200-250)")
    p.add_argument("--grip-detect-ticks", type=int, default=80,
                   help="gripping = last gripper goal at least this far past the measured position")
    p.add_argument("--grip-release-steps", type=int, default=5,
                   help="consecutive 'open' goals from the policy before the hold lets go")
    p.add_argument("--grip-squeeze-max-ticks", type=int, default=0,
                   help="while gripping, cap how far a gripper goal may go past the measured position (0 = off)")
    p.add_argument("--log-gripper", action="store_true",
                   help="print gripper goal (sent and policy) and measured position for every action")
    p.add_argument("--timeout", type=float, default=5.0, help="frame fetch timeout, s")
    p.add_argument("--print-actions", action="store_true", help="print every goal (always on in --dry-run)")
    a = p.parse_args(argv)
    if not a.self_test:
        missing = [n for n in ("checkpoint", "task", "bus_serial") if getattr(a, n) is None]
        if missing:
            p.error("missing: " + ", ".join("--" + m.replace("_", "-") for m in missing))
        if a.execute < 1:
            p.error("--execute must be >= 1")
    return a


def install_stop_handlers() -> None:
    """Ctrl+C / SIGTERM: leave at once. os._exit runs no further Python code, so nothing more is sent.
    Torque is never touched; the arm holds the last goal it received."""
    def handler(signum, _frame):
        os.write(2, f"\nstopped by signal {signum}: nothing further sent\n".encode())
        os._exit(130 if signum == signal.SIGINT else 143)
    signal.signal(signal.SIGINT, handler)
    signal.signal(signal.SIGTERM, handler)


def setup_paths(args) -> Path:
    cwd = Path.cwd().resolve()
    repo = (args.norma_core.expanduser().resolve() if args.norma_core else cwd.parents[2])
    shared = repo / "software" / "station" / "shared"
    if not (shared / "station_py").exists():
        raise Fatal(2, f"{shared}/station_py not found. Run from norma-core/software/ai/smolvla_py "
                       f"or pass --norma-core <repo root>.")
    if not (cwd / "smolvla").is_dir():
        raise Fatal(2, f"no smolvla package in {cwd}; cd to norma-core/software/ai/smolvla_py first")
    # Same order as run_policy.py lines 45-46 (repo ends up first), plus cwd for `smolvla`
    # (uv run puts this script's folder on sys.path, not the cwd).
    sys.path.insert(0, str(shared))
    sys.path.insert(0, str(repo))
    sys.path.insert(0, str(cwd))
    return repo


def collapse(name: str) -> str:
    return re.sub(r"\.\d+(?=\.|$)", ".*", name)


def dtype_report(policy) -> str:
    pairs: dict[tuple[str, str], int] = {}
    for name, t in list(policy.named_parameters()) + list(policy.named_buffers()):
        mod = collapse(name.rsplit(".", 1)[0])
        key = (mod, str(t.dtype).replace("torch.", ""))
        pairs[key] = pairs.get(key, 0) + 1
    lines = [f"  {d:9s} x{n:<4d} {m}" for (m, d), n in sorted(pairs.items())]
    return "\n".join(lines)


def main_real(args) -> int:
    install_stop_handlers()
    if args.attn:
        os.environ["SMOLVLA_ATTN_IMPL"] = args.attn
    rate = args.rate or DEFAULT_RATE_HZ
    log("=" * 60)
    log("DRY RUN: nothing will be sent to the arm" if args.dry_run else
        "REAL MOTION: this script will move the SO-101 (Ctrl+C stops at once)")
    log("=" * 60)
    log(f"rate:      {rate:g} Hz ({'from --rate' if args.rate else 'default: normvla queue and dataset run at ~10 Hz'})")
    setup_paths(args)

    import torch
    from safetensors.torch import load_file
    import json, dataclasses, io                                          # noqa: E401
    import numpy as np
    from PIL import Image
    from station_py import new_station_client, send_commands               # run_policy.py line 48
    from target.gen_python.protobuf.drivers.inferences import normvla      # line 49
    from target.gen_python.protobuf.drivers.st3215 import st3215           # line 50
    from target.gen_python.protobuf.station import commands, drivers       # line 51
    from smolvla import SmolVLAPolicy
    from smolvla.config import SmolVLAConfig
    from smolvla.normalize import normalize_state, unnormalize_action
    from smolvla.stats import load_stats

    # ---- device ----
    mps_ok = getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()
    if args.device == "auto":
        dev = "cuda" if torch.cuda.is_available() else ("mps" if mps_ok else "cpu")
    else:
        dev = args.device
        if dev == "cuda" and not torch.cuda.is_available():
            raise Fatal(2, "--device cuda but CUDA is not available")
        if dev == "mps" and not mps_ok:
            raise Fatal(2, "--device mps but MPS is not available; use --device cpu")
    device = torch.device(dev)
    log(f"device:    {device} (float32)")

    motor_ids = [int(x) for x in args.motor_ids.split(",")]
    cams = tuple(int(x) for x in args.cameras.split(","))                  # run_policy.py lines 310-311
    image_keys = tuple(f"observation.images.cam{i}" for i in range(len(cams)))
    log(f"cameras:   {image_keys} <- station images {cams}")

    ckpt = args.checkpoint.expanduser()
    stats_path = ckpt / "stats.safetensors"                                # lines 316-319
    if not stats_path.exists():
        raise Fatal(2, f"No stats.safetensors in {ckpt}.")
    stats = {k: v.float().to(device) for k, v in load_stats(stats_path).items()}

    # ---- load: SmolVLAPolicy.from_pretrained (policy.py lines 195-208), with float32 BEFORE loading ----
    # Why: transformers 5.3 builds the SmolVLM backbone and the lm_expert from config with the
    # config's bfloat16 dtype (train.py: "smolvlm backbone loads as bf16"), while the cross-attention
    # k_proj/v_proj that smolvlm_with_expert.py lines 115-124 put into the expert are plain nn.Linear,
    # i.e. float32. In the cross layers, line 351 gives a bf16 query (expert q_proj) and lines 353-365
    # give float32 key/value (cast to k_proj/v_proj dtype), so SDPA at line 550 fails without autocast.
    # load_state_dict keeps each parameter's own dtype, so the cast must be done on the module.
    t_load = time.perf_counter()
    log(f"loading:   {ckpt} ...")
    with open(ckpt / "config.json") as f:
        config = SmolVLAConfig.from_dict(json.load(f))
    config = dataclasses.replace(config, load_vlm_weights=False)          # as run_policy.py line 324
    policy = SmolVLAPolicy(config)
    policy.float()
    sd = load_file(str(ckpt / "model.safetensors"), device="cpu")
    ck_dtypes: dict[str, int] = {}
    for v in sd.values():
        ck_dtypes[str(v.dtype)] = ck_dtypes.get(str(v.dtype), 0) + 1
    missing, unexpected = policy.load_state_dict(sd, strict=False)        # strict=False as reference
    log(f"checkpoint tensors by dtype: {ck_dtypes}; missing keys {len(missing)}, unexpected {len(unexpected)}")
    for k in list(missing)[:5]:
        log(f"  missing:    {k}")
    for k in list(unexpected)[:5]:
        log(f"  unexpected: {k}")
    del sd
    policy.to(device)
    policy.eval()
    torch.set_default_dtype(torch.float32)
    # Belt and braces: every floating parameter/buffer must be float32 now.
    fixed = 0
    for _, t in list(policy.named_parameters()) + list(policy.named_buffers()):
        if t.is_floating_point() and t.dtype != torch.float32:
            t.data = t.data.float()
            fixed += 1
    bad = [n for n, t in list(policy.named_parameters()) + list(policy.named_buffers())
           if t.is_floating_point() and t.dtype != torch.float32]
    if bad:
        raise Fatal(5, f"still not float32 after conversion: {bad[:10]}")
    log(f"dtypes:    all floating parameters/buffers float32 (converted late: {fixed})")
    log(f"loaded in {time.perf_counter() - t_load:.1f} s")

    trained_keys = tuple(getattr(policy.config, "image_keys", ()) or ())   # lines 328-331
    if trained_keys and trained_keys != image_keys:
        raise Fatal(2, f"Checkpoint was trained with {len(trained_keys)} cameras {trained_keys}, "
                       f"but --cameras gives {image_keys}. Use the same --cameras as in training.")
    chunk_size = int(config.chunk_size)
    execute = min(args.execute, chunk_size)
    if execute < args.execute:
        log(f"--execute {args.execute} > chunk_size {chunk_size}: using {execute}")

    tokens, mask = policy.tokenize_task(args.task, device=device)          # same call as line 189, done once

    # ---- frame / batch / command helpers (copied from run_policy.py) ----
    async def fetch_frame(client, timeout: float):                         # lines 94-99
        qr = client.read_from_tail(QUEUE_ID, offset=b"\x00", limit=1, step=1, buf_size=1)
        entry = await asyncio.wait_for(qr.data.get(), timeout=timeout)
        if entry is None:
            raise RuntimeError(f"{QUEUE_ID} closed without delivering a frame ({qr.err})")
        return normvla.FrameReader(memoryview(bytes(entry.Data)))

    async def fetch_fresh_frame(client, last_id: bytes, timeout: float, retry_ms: int = 20):  # lines 102-115
        while True:
            frame = await fetch_frame(client, timeout)
            fid = bytes(frame.get_global_frame_id())
            if fid != last_id:
                return frame, fid
            await asyncio.sleep(retry_ms / 1000.0)

    def frame_to_batch(frame):                                             # lines 121-149
        joints = frame.get_joints() or []
        images = frame.get_images() or []
        if len(images) <= max(cams):
            raise RuntimeError(f"Frame has {len(images)} images but policy expects {len(image_keys)}")
        state = torch.tensor(
            [j.get_position_norm() for j in joints], dtype=torch.float32, device=device
        ).unsqueeze(0)
        batch = {"observation.state": normalize_state(state, stats)}
        for i, key in enumerate(image_keys):
            jpeg = bytes(images[cams[i]].get_jpeg())
            with Image.open(io.BytesIO(jpeg)) as im:
                arr = np.asarray(im.convert("RGB"), dtype=np.uint8)
            batch[key] = (
                torch.from_numpy(arr.copy()).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.0
            )
        return batch

    def build_sync_write_command(raw_goals: list[int]):                    # lines 152-171
        motors = [
            st3215.ST3215SyncWriteCommand_MotorWrite(motor_id=mid, value=int(raw).to_bytes(2, byteorder="little"))
            for mid, raw in zip(motor_ids, raw_goals)
        ]
        sync = st3215.ST3215SyncWriteCommand(address=ST3215_TARGET_POS_REGISTER, motors=motors)
        cmd = st3215.Command(target_bus_serial=args.bus_serial, sync_write=sync)
        return commands.DriverCommand(type=drivers.StationCommandType.STC_ST3215_COMMAND, body=cmd.encode())

    state_box = {"last_id": b"", "first_pred": True}

    def predict(frame):                                                    # lines 188-195, full chunk
        try:
            with torch.no_grad():
                batch = frame_to_batch(frame)
                batch["observation.language.tokens"] = tokens
                batch["observation.language.attention_mask"] = mask
                pred_norm = policy.predict_action_chunk(batch)[0]          # (chunk_size, 6)
                pred_goal = unnormalize_action(pred_norm, stats)
                rows = pred_goal.float().cpu().clamp(0.0, 1.0).tolist()
        except (RuntimeError, NotImplementedError, TypeError) as e:
            msg = str(e)
            if any(s in msg for s in ("dtype", "BFloat16", "Half", "scalar type", "same type")):
                log("DTYPE ERROR in prediction:\n  " + msg.splitlines()[0])
                log("distinct (dtype, count, module) after loading:\n" + dtype_report(policy))
                raise Fatal(5, "dtype mismatch; paste the lines above. Also try --attn eager.")
            if device.type == "mps":
                raise Fatal(4, f"operation failed on MPS: {msg.splitlines()[0]}\n"
                               f"Run again with --device cpu (or try --attn eager).")
            raise
        if state_box["first_pred"]:
            state_box["first_pred"] = False
            log(f"first prediction OK: chunk {len(rows)} x {len(rows[0])}")
        return rows

    async def run() -> int:
        client = await new_station_client(args.server, __import__("logging").getLogger("policy_run"))

        async def fetch():
            frame, fid = await fetch_fresh_frame(client, state_box["last_id"], args.timeout)
            state_box["last_id"] = fid
            joints = frame.get_joints() or []
            if len(joints) != len(motor_ids) or len(joints) != config.action_dim:
                raise Fatal(2, f"Frame has {len(joints)} joints, --motor-ids has {len(motor_ids)}, "
                               f"policy action_dim {config.action_dim}")
            measured = [int(j.get_position()) for j in joints]
            ranges = [(int(j.get_range_min()), int(j.get_range_max())) for j in joints]  # line 148
            return frame, measured, ranges

        async def send(raw):
            cmd = build_sync_write_command(raw)                            # built in dry run too
            if args.dry_run:
                return
            await send_commands(client, [cmd])                             # line 252

        latest = {"g": None}

        async def poll_gripper():
            """Background reader of the measured gripper position, only for --log-gripper.
            Runs while the loop sleeps between actions (prediction blocks it, which is fine)."""
            while True:
                try:
                    fr = await fetch_frame(client, args.timeout)
                    js = fr.get_joints() or []
                    if len(js) > GRIPPER_JOINT:
                        latest["g"] = int(js[GRIPPER_JOINT].get_position())
                except asyncio.CancelledError:
                    raise
                except Exception as e:                                      # logging must never stop a run
                    log(f"gripper log: measured polling stopped ({type(e).__name__}: {e})")
                    latest["g"] = None
                    return
                await asyncio.sleep(0.05)

        if args.log_gripper:
            asyncio.get_running_loop().create_task(poll_gripper())

        cfg = RunConfig(execute=execute, rate=rate, max_delta=args.max_delta_ticks,
                        gripper_max_delta=args.gripper_max_delta_ticks, dry_run=args.dry_run,
                        max_seconds=args.max_seconds, max_cycles=args.max_cycles,
                        still_s=args.stop_when_still, motor_ids=motor_ids,
                        print_actions=args.print_actions or args.dry_run,
                        ramp_step=args.ramp_step_ticks, ramp_max=args.ramp_max_ticks,
                        grip_hold=args.grip_hold_ticks, grip_thresh=args.grip_detect_ticks,
                        grip_release_steps=args.grip_release_steps,
                        grip_squeeze_max=args.grip_squeeze_max_ticks, log_gripper=args.log_gripper)
        log(f"task:      {args.task!r}")
        log(f"bus:       {args.bus_serial} on {args.server}, motors {motor_ids}")
        log(f"plan:      execute {execute} of {chunk_size} actions per inference at {rate:g} Hz "
            f"({execute / rate:.2f} s of motion per cycle)")
        log(f"safety:    max_delta_ticks={args.max_delta_ticks} (arm), "
            f"gripper_max_delta_ticks={args.gripper_max_delta_ticks} (gripper), max_seconds={args.max_seconds:g}, "
            f"max_cycles={args.max_cycles}, stop_when_still={0 if args.dry_run else args.stop_when_still:g} s")
        log(f"start:     ramp_step_ticks={args.ramp_step_ticks}, ramp_max_ticks={args.ramp_max_ticks}")
        log(f"gripper:   hold_ticks={args.grip_hold_ticks}, detect_ticks={args.grip_detect_ticks}, "
            f"release_steps={args.grip_release_steps}, squeeze_max_ticks={args.grip_squeeze_max_ticks}, "
            f"log={'on' if args.log_gripper else 'off'}")
        return await control_loop(cfg, fetch, predict, send, observe=lambda: latest["g"])

    return asyncio.run(run())


def main() -> None:
    try:
        sys.stdout.reconfigure(line_buffering=True)   # loader reads stdout through a pipe
    except Exception:
        pass
    args = parse_args()
    if args.self_test:
        sys.exit(self_test())
    try:
        code = main_real(args)
    except Fatal as e:
        log(f"ERROR: {e}")
        code = e.code
    sys.exit(code)


if __name__ == "__main__":
    main()
