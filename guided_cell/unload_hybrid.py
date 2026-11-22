#!/usr/bin/env python3
"""Hybrid unload for the SO-101 (work area -> drop-off), like the sponge project's hover + policy:
a taught approach (hover over the box, then pre-grasp with the jaws around it) and then the SmolVLA
unload policy does the grasp, lift, carry to the drop-off and release.

Why taught poses and not IK: the box sits in the work-area fixture, so its position is fixed and a taught
pose is exact; the guided cell has no SO-101 kinematics or camera-to-SO-101 calibration (those are ElRobot's).
Small offsets are left to the policy (wrist camera).

Run everything from norma-core/software/ai/smolvla_py with `uv run python`.

Teach by hand (no teleop needed): SUPPORT THE ARM, then
    uv run python .../unload_hybrid.py limp             # SO-101 torque off: move it by hand
    uv run python .../unload_hybrid.py hold             # goal := where it is now, then torque on (no jump)
Teach (arm placed by hand with `limp`, or with Station teleop, which must be OFF before `run`):
    uv run python .../unload_hybrid.py teach hover      # about 8-10 cm straight above the box, gripper open
    uv run python .../unload_hybrid.py teach pregrasp   # jaws around the box, 1-2 cm above grasp height, open
    uv run python .../unload_hybrid.py show
Run:
    uv run python .../unload_hybrid.py run [--dry-run] [--execute 50] [any other policy_run.py option]
    uv run python .../unload_hybrid.py selftest         # no robot, no torch

policy_run.py is used UNCHANGED: this script imports it, lets it load the model, and moves the arm to the
taught poses just before policy_run's control loop starts (the model is already loaded, so there is no pause at
the box). Defaults: unload checkpoint/final, unload task, bus 5B7B015823, cameras 0,3, --execute 50,
--max-seconds 90; any policy_run.py option given on the command line overrides them.
Exit codes as policy_run.py; also 2 = poses not taught / outside the calibrated range, 3 = pose not reached.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import sys
import time
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import policy_run as pr  # noqa: E402  (unchanged runner; only its control_loop name is wrapped at run time)

POSES_FILE = HERE / "state" / "unload_hybrid_poses.json"
SEQUENCE = ("hover", "pregrasp")
DEFAULTS = [("--checkpoint", str(HERE / "policies" / "unload" / "final")),
            ("--task", "Move the red box from the work area to the drop-off area"),
            ("--bus-serial", "5B7B015823"), ("--cameras", "0,3"),
            ("--execute", "50"), ("--max-seconds", "90")]
ARM = range(5)                                  # joints 0-4; 5 is the gripper
log = pr.log


# ----------------------------------------------------------------------------- pure helpers
def eased_path(start: list[int], goal: list[int], max_step: float) -> list[list[int]]:
    """Joint-space path start -> goal (goal included) with smooth start/stop (cosine easing);
    no joint moves more than max_step ticks between two goals."""
    d = pr.max_abs_delta(goal, start)[0]
    if d == 0:
        return [list(goal)]
    n = max(1, math.ceil(d * math.pi / 2 / max_step))      # peak speed of the easing is pi/2 x average
    out = []
    for i in range(1, n + 1):
        f = (1 - math.cos(math.pi * i / n)) / 2
        out.append([int(round(s + (g - s) * f)) for s, g in zip(start, goal)])
    out[-1] = list(goal)
    return out


def arm_error(measured: list[int], goal: list[int]) -> tuple[int, int]:
    best, j = 0, 0
    for i in ARM:
        e = abs(measured[i] - goal[i])
        if e > best:
            best, j = e, i
    return best, j


def load_poses() -> dict:
    return json.loads(POSES_FILE.read_text()) if POSES_FILE.exists() else {}


def check_poses(poses: dict, ranges: list[tuple[int, int]], seq=SEQUENCE) -> None:
    missing = [n for n in seq if n not in poses]
    if missing:
        raise pr.Fatal(2, f"pose(s) {missing} not taught; run: unload_hybrid.py teach <name> (names: {seq})")
    for n in seq:
        t = poses[n]["ticks"]
        if len(t) != len(ranges):
            raise pr.Fatal(2, f"pose {n} has {len(t)} joints, the frame has {len(ranges)}")
        for j, (v, (a, b)) in enumerate(zip(t, ranges)):
            if not min(a, b) <= v <= max(a, b):
                raise pr.Fatal(2, f"pose {n}: {pr.JOINT_NAMES[j]} = {v} outside the calibrated range {a}..{b}")


async def approach(fetch, send, poses: dict, speeds: dict, rate: float, dry_run: bool, tol: int,
                   settle_s: float, fail_ticks: int, clock=time.monotonic, sleep=asyncio.sleep,
                   stop_at: str = "pregrasp") -> list[int]:
    """Move measured -> hover [-> pregrasp]. stop_at = last pose before the policy takes over.
    Returns the last commanded goal."""
    seq = SEQUENCE[:SEQUENCE.index(stop_at) + 1]
    _, measured, ranges = await fetch()
    check_poses(poses, ranges, seq)
    pos = list(measured)
    for name in seq:
        goal = list(poses[name]["ticks"])
        path = eased_path(pos, goal, speeds[name] / rate)
        d, j = pr.max_abs_delta(goal, pos)
        log(f"approach: -> {name}: {len(path)} goals, {len(path) / rate:.1f} s, largest move {d} ticks "
            f"({pr.JOINT_NAMES[j]}){'  [dry run: nothing sent]' if dry_run else ''}")
        t0 = clock()
        for k, raw in enumerate(path):
            delay = t0 + k / rate - clock()
            if delay > 0:
                await sleep(delay)
            await send(raw)
        pos = goal
        if dry_run:
            continue
        t_end = clock() + settle_s
        while True:
            _, measured, _ = await fetch()
            err, ej = arm_error(measured, goal)
            if err <= tol or clock() >= t_end:
                break
        if err > fail_ticks:
            raise pr.Fatal(3, f"{name} not reached: {pr.JOINT_NAMES[ej]} is {err} ticks off after {settle_s:g} s "
                              f"(blocked? measured {measured}, goal {goal})")
        log(f"approach: at {name}: largest arm error {err} ticks ({pr.JOINT_NAMES[ej]})"
            + ("" if err <= tol else f"  [above {tol}, continuing]")
            + f", gripper {measured[5]} (goal {goal[5]})")
    return pos


def install_hook(opts, poses_fn=load_poses) -> None:
    """Wrap policy_run.control_loop so the approach runs with policy_run's own fetch/send, after the model
    is loaded and the Ctrl+C handlers are installed, right before the policy starts."""
    orig = pr.control_loop

    async def hybrid_loop(cfg, fetch, predict, send, clock=time.monotonic, sleep=asyncio.sleep, observe=None):
        if opts.skip_approach:
            log("approach: skipped (--skip-approach)")
        else:
            await approach(fetch, send, poses_fn(),
                           {"hover": opts.hover_speed, "pregrasp": opts.pregrasp_speed},
                           opts.approach_rate, cfg.dry_run, opts.tol, opts.settle_s, opts.fail_ticks,
                           clock=clock, sleep=sleep, stop_at=opts.stop_at)
            if opts.pause_s > 0 and not cfg.dry_run:
                await sleep(opts.pause_s)
        log("approach done: policy takes over")
        return await orig(cfg, fetch, predict, send, clock, sleep, observe=observe)

    pr.control_loop = hybrid_loop


# ----------------------------------------------------------------------------- station (teach / show)
async def read_ticks(host: str) -> tuple[list[int], list[tuple[int, int]]]:
    pr.setup_paths(SimpleNamespace(norma_core=None))
    import logging
    from station_py import new_station_client
    from target.gen_python.protobuf.drivers.inferences import normvla
    client = await new_station_client(host, logging.getLogger("unload_hybrid"))
    qr = client.read_from_tail(pr.QUEUE_ID, offset=b"\x00", limit=1, step=1, buf_size=1)
    entry = await asyncio.wait_for(qr.data.get(), timeout=5.0)
    if entry is None:
        raise pr.Fatal(2, f"{pr.QUEUE_ID} delivered no frame (is Station running?)")
    joints = normvla.FrameReader(memoryview(bytes(entry.Data))).get_joints() or []
    return [int(j.get_position()) for j in joints], [(int(j.get_range_min()), int(j.get_range_max())) for j in joints]


RAM_TORQUE_ENABLE = 0x28                      # same as cellbot station_backend


async def set_torque(host: str, bus_serial: str, motor_ids: list[int], on: bool) -> None:
    """off: torque off. on: first goal := present position (so the arm cannot jump to an old goal), then torque on."""
    pr.setup_paths(SimpleNamespace(norma_core=None))
    import logging
    from station_py import new_station_client, send_commands
    from target.gen_python.protobuf.drivers.inferences import normvla
    from target.gen_python.protobuf.drivers.st3215 import st3215
    from target.gen_python.protobuf.station import commands, drivers
    client = await new_station_client(host, logging.getLogger("unload_hybrid"))

    def cmd(address: int, values: list[bytes]):
        sync = st3215.ST3215SyncWriteCommand(address=address, motors=[
            st3215.ST3215SyncWriteCommand_MotorWrite(motor_id=m, value=v) for m, v in zip(motor_ids, values)])
        body = st3215.Command(target_bus_serial=bus_serial, sync_write=sync).encode()
        return commands.DriverCommand(type=drivers.StationCommandType.STC_ST3215_COMMAND, body=body)

    if not on:
        await send_commands(client, [cmd(RAM_TORQUE_ENABLE, [b"\x00"] * len(motor_ids))])
        log("SO-101 torque OFF: move it by hand. Run `hold` when done (support the arm until then).")
        return
    qr = client.read_from_tail(pr.QUEUE_ID, offset=b"\x00", limit=1, step=1, buf_size=1)
    entry = await asyncio.wait_for(qr.data.get(), timeout=5.0)
    joints = normvla.FrameReader(memoryview(bytes(entry.Data))).get_joints() or []
    ticks = [int(j.get_position()) for j in joints]
    if len(ticks) != len(motor_ids):
        raise pr.Fatal(2, f"frame has {len(ticks)} joints, expected {len(motor_ids)}")
    await send_commands(client, [cmd(pr.ST3215_TARGET_POS_REGISTER, [t.to_bytes(2, "little") for t in ticks])])
    await asyncio.sleep(0.2)
    await send_commands(client, [cmd(RAM_TORQUE_ENABLE, [b"\x01"] * len(motor_ids))])
    log(f"SO-101 torque ON, holding {ticks}")


def cmd_teach(name: str, host: str) -> int:
    if name not in SEQUENCE:
        log(f"pose name must be one of {SEQUENCE}")
        return 2
    ticks, _ = asyncio.run(read_ticks(host))
    poses = load_poses()
    poses[name] = {"ticks": ticks, "saved": time.strftime("%Y-%m-%d %H:%M:%S")}
    POSES_FILE.parent.mkdir(parents=True, exist_ok=True)
    POSES_FILE.write_text(json.dumps(poses, indent=2))
    log(f"saved {name}: {dict(zip(pr.JOINT_NAMES, ticks))} -> {POSES_FILE}")
    return 0


def cmd_show(host: str) -> int:
    poses = load_poses()
    try:
        now, _ = asyncio.run(read_ticks(host))
    except Exception as e:                                   # show works without Station
        now = None
        log(f"(no Station reading: {type(e).__name__}: {e})")
    if now:
        log(f"now       {now}")
    for n in SEQUENCE:
        if n in poses:
            t = poses[n]["ticks"]
            extra = f"  (largest diff to now {pr.max_abs_delta(t, now)[0]} ticks)" if now else ""
            log(f"{n:9s} {t}  saved {poses[n]['saved']}{extra}")
        else:
            log(f"{n:9s} not taught")
    return 0


# ----------------------------------------------------------------------------- self-test
def self_test() -> int:
    res = []

    def check(name, cond, detail=""):
        res.append(bool(cond))
        log(f"{'PASS' if cond else 'FAIL'}  {name}{('  ' + detail) if detail else ''}")

    a, b = [2000, 900, 3100, 2900, 1000, 1300], [1500, 1400, 2500, 2600, 1100, 2000]
    p = eased_path(a, b, 20)
    steps = [pr.max_abs_delta(y, x)[0] for x, y in zip([a] + p, p)]
    check("eased path ends exactly at the goal", p[-1] == b)
    check("eased path: no step over 20 ticks", max(steps) <= 20, f"max {max(steps)}")
    check("eased path: slow start and end", steps[0] < 5 and steps[-1] < 5, f"{steps[0]}, {steps[-1]}")
    check("eased path to the same pose = one goal", eased_path(a, a, 20) == [a])

    RANGES = [(0, 4095)] * 6
    poses = {"hover": {"ticks": b, "saved": ""}, "pregrasp": {"ticks": [1500, 1500, 2400, 2650, 1100, 2000], "saved": ""}}

    class Sim:
        def __init__(self, follow=True):
            self.t, self.pos, self.sent, self.follow = 0.0, list(a), [], follow

        def clock(self):
            return self.t

        async def sleep(self, s):
            self.t += s

        async def fetch(self):
            self.t += 0.05
            return None, list(self.pos), RANGES

        async def send(self, raw):
            self.sent.append(list(raw))
            if self.follow:
                self.pos = list(raw)

    sim = Sim()
    last = asyncio.run(approach(sim.fetch, sim.send, poses, {"hover": 400, "pregrasp": 150}, 20, False, 40, 2.0, 150,
                                sim.clock, sim.sleep))
    st = [pr.max_abs_delta(y, x)[0] for x, y in zip([a] + sim.sent, sim.sent)]
    check("approach ends at pregrasp", last == poses["pregrasp"]["ticks"] and sim.pos == last)
    check("approach passes through hover", b in sim.sent)
    check("approach: steps <= 20 ticks (400 ticks/s at 20 Hz)", max(st) <= 20, f"max {max(st)}")
    sim = Sim(follow=False)
    try:
        asyncio.run(approach(sim.fetch, sim.send, poses, {"hover": 400, "pregrasp": 150}, 20, False, 40, 1.0, 150,
                             sim.clock, sim.sleep))
        code = 0
    except pr.Fatal as e:
        code = e.code
    check("approach: arm does not follow -> exit 3 after hover", code == 3 and b in sim.sent and sim.sent[-1] == b)
    try:
        asyncio.run(approach(Sim().fetch, Sim().send, {"hover": poses["hover"]}, {"hover": 400, "pregrasp": 150},
                             20, False, 40, 1.0, 150))
        code = 0
    except pr.Fatal as e:
        code = e.code
    check("approach: pregrasp not taught -> exit 2, nothing sent", code == 2)
    bad = {"hover": poses["hover"], "pregrasp": {"ticks": [1500, 1500, 2400, 2650, 1100, 5000], "saved": ""}}
    try:
        asyncio.run(approach(Sim().fetch, Sim().send, bad, {"hover": 400, "pregrasp": 150}, 20, False, 40, 1.0, 150))
        code = 0
    except pr.Fatal as e:
        code = e.code
    check("approach: pose outside range -> exit 2", code == 2)

    # hook: approach runs first, then the unchanged control loop starts from the pregrasp pose
    sim = Sim()
    seen = []

    def predict(frame):
        seen.append(list(sim.pos))
        return [[v / 4095 for v in sim.pos] for _ in range(50)]
    orig = pr.control_loop
    sim2 = Sim()
    last = asyncio.run(approach(sim2.fetch, sim2.send, {"hover": poses["hover"]}, {"hover": 400, "pregrasp": 150},
                                20, False, 40, 2.0, 150, sim2.clock, sim2.sleep, stop_at="hover"))
    check("stop-at hover: ends at hover, pregrasp not needed", last == b and sim2.pos == b)
    opts = SimpleNamespace(stop_at="pregrasp", skip_approach=False, hover_speed=400, pregrasp_speed=150, approach_rate=20, tol=40,
                           settle_s=2.0, fail_ticks=150, pause_s=0.3)
    install_hook(opts, poses_fn=lambda: poses)
    code = asyncio.run(pr.control_loop(pr.RunConfig(execute=5, max_cycles=1, still_s=0, ramp_step=100),
                                       sim.fetch, predict, sim.send, sim.clock, sim.sleep))
    pr.control_loop = orig
    check("hook: policy's first frame is at pregrasp, exit 0", code == 0 and seen == [poses["pregrasp"]["ticks"]])
    check("policy_run.py self-test still passes", pr.self_test() == 0)
    for line in ("run --log-gripper --grip-hold-ticks 250",
                 "run --stop-at hover --log-gripper --grip-hold-ticks 250"):
        c, n, argv = split_argv(line.split())
        o, rest = own_parser().parse_known_args(argv)
        a = pr.parse_args(rest + ["--checkpoint", "x", "--task", "t", "--bus-serial", "b"])
        check(f"cli: '{line}'", c == "run" and a.grip_hold_ticks == 250 and a.log_gripper
              and o.stop_at == ("hover" if "hover" in line else "pregrasp"))
    c, n, argv = split_argv(["teach", "hover"])
    check("cli: teach hover", c == "teach" and n == "hover" and argv == [])
    ok = all(res)
    log(f"\nunload_hybrid self-test: {'ALL PASS' if ok else 'FAILURES'} ({sum(res)}/{len(res)})")
    return 0 if ok else 1


# ----------------------------------------------------------------------------- main
COMMANDS = ("run", "teach", "show", "limp", "hold", "selftest")


def split_argv(argv: list[str]) -> tuple[str, str, list[str]]:
    """command, pose name (teach only), remaining options. Positionals are taken here, not by argparse,
    so a value like `250` after a policy_run option is never mistaken for a pose name."""
    if not argv or argv[0] not in COMMANDS:
        raise SystemExit(f"usage: unload_hybrid.py {{{'|'.join(COMMANDS)}}} [options]  (see the file header)")
    cmd, rest, name = argv[0], argv[1:], ""
    if cmd == "teach":
        if not rest or rest[0].startswith("-"):
            raise SystemExit("usage: unload_hybrid.py teach hover|pregrasp")
        name, rest = rest[0], rest[1:]
    return cmd, name, rest


def own_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bus", default="5B7B015823", help="SO-101 bus serial (limp/hold)")
    ap.add_argument("--host", default="localhost", help="Station host (teach/show)")
    ap.add_argument("--hover-speed", type=float, default=400, help="ticks/s (largest joint) to hover, ~35 deg/s")
    ap.add_argument("--pregrasp-speed", type=float, default=150, help="ticks/s from hover down to pregrasp")
    ap.add_argument("--approach-rate", type=float, default=20, help="goals per second during the approach")
    ap.add_argument("--tol", type=int, default=40, help="arm joints within this of a pose = reached")
    ap.add_argument("--settle-s", type=float, default=2.0, help="max wait at each pose")
    ap.add_argument("--fail-ticks", type=int, default=150, help="error left after settling that stops the run")
    ap.add_argument("--pause-s", type=float, default=0.3, help="still time at pregrasp before the policy starts")
    ap.add_argument("--skip-approach", action="store_true", help="policy only, from wherever the arm is")
    ap.add_argument("--stop-at", choices=SEQUENCE, default="pregrasp",
                    help="last taught pose before the policy takes over: hover = the policy also does the descent "
                         "(follows a box that is not in its usual spot; pregrasp need not be taught)")
    return ap


def main() -> None:
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass
    command, name, argv = split_argv(sys.argv[1:])
    opts, rest = own_parser().parse_known_args(argv)
    opts.command, opts.name = command, name
    if opts.command == "selftest":
        sys.exit(self_test())
    if opts.command == "teach":
        sys.exit(cmd_teach(opts.name or "", opts.host))
    if opts.command == "show":
        sys.exit(cmd_show(opts.host))
    if opts.command in ("limp", "hold"):
        try:
            asyncio.run(set_torque(opts.host, opts.bus, [1, 2, 3, 4, 5, 6], opts.command == "hold"))
            sys.exit(0)
        except pr.Fatal as e:
            log(f"ERROR: {e}")
            sys.exit(e.code)
    for flag, val in DEFAULTS:
        if flag not in rest:
            rest += [flag, val]
    args = pr.parse_args(rest)
    install_hook(opts)
    try:
        code = pr.main_real(args)
    except pr.Fatal as e:
        log(f"ERROR: {e}")
        code = e.code
    sys.exit(code)


if __name__ == "__main__":
    main()
