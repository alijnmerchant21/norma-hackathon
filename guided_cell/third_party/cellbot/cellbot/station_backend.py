"""Real NormaCore Station backend over its TCP API (port 8888).

Everything here is ported from code in the norma-core repo, not guessed:
  - client + send_commands:   software/station/shared/station_py
  - st3215/inference parsing:  software/station/examples/st3215-remote-teleop-py/state.py
  - sync_write commands:       software/station/examples/st3215-remote-teleop-py/commands.py
  - camera frames:             software/ai/smolvla_py/scripts/run_policy.py (inference/normvla)

The interface is in NORMALIZED joint units (0..1 of the calibrated arc), the
same as SimBackend, so the guard layer and MCP server do not care which is used.
"""

from __future__ import annotations

import asyncio
import logging
import struct
import sys
import time
from pathlib import Path

from .config import ArmConfig, CellConfig, ROOT
from .units import norm_to_ticks, ticks_to_norm

log = logging.getLogger("cellbot.station")

# ST3215 RAM register addresses (state.py in the teleop example).
RAM_TORQUE_ENABLE = 0x28
RAM_ACC = 0x29
RAM_GOAL_POSITION = 0x2A
RAM_GOAL_SPEED = 0x2E
RAM_PRESENT_POSITION = 0x38
RAM_STATUS = 0x40
RAM_PRESENT_CURRENT = 0x45
MAX_ANGLE_STEP = 4095
SIGN_BIT_MASK = 0x8000


def _u16(b: bytes, addr: int) -> int:
    return struct.unpack_from("<H", b, addr)[0] if len(b) >= addr + 2 else 0


def _u8(b: bytes, addr: int) -> int:
    return b[addr] if len(b) > addr else 0


def _normal_position(raw: int) -> int:
    if raw & SIGN_BIT_MASK:
        return (MAX_ANGLE_STEP + 1 - (raw & MAX_ANGLE_STEP)) & MAX_ANGLE_STEP
    return raw & MAX_ANGLE_STEP


class StationBackend:
    name = "station"

    def __init__(self, cfg: CellConfig):
        self.cfg = cfg
        st = cfg.station
        repo = Path(st["norma_core_dir"])
        if not repo.is_absolute():
            repo = (ROOT / repo).resolve()
        if not (repo / "target" / "gen_python").exists():
            raise RuntimeError(
                f"norma-core clone not found at {repo} (need target/gen_python). "
                "Set [station].norma_core_dir in cell.toml."
            )
        sys.path.insert(0, str(repo))
        from software.station.shared.station_py import new_station_client, send_commands
        from target.gen_python.protobuf.drivers.st3215 import st3215
        from target.gen_python.protobuf.drivers.inferences import normvla
        from target.gen_python.protobuf.station import commands, drivers

        self._new_client = new_station_client
        self._send = send_commands
        self._st3215 = st3215
        self._normvla = normvla
        self._commands = commands
        self._drivers = drivers
        self.client = None
        self.latest = None
        self.latest_t = 0.0
        self.bus_of: dict[str, str] = {}
        self._reader_task = None

    # ------------------------------------------------------------------ setup
    async def connect(self):
        if self.client is not None:
            return
        st = self.cfg.station
        self.client = await self._new_client(f"{st['host']}:{st['port']}", log)
        q: asyncio.Queue = asyncio.Queue()
        err_q = self.client.follow("st3215/inference", q)

        async def reader():
            while True:
                if not err_q.empty():
                    log.error("st3215/inference stream error: %s", err_q.get_nowait())
                entry = await q.get()
                if entry is None:
                    log.error("st3215/inference stream closed")
                    return
                try:
                    self.latest = self._st3215.InferenceStateReader(memoryview(bytes(entry.Data)))
                    self.latest_t = time.monotonic()
                except Exception:
                    log.exception("bad inference frame")

        self._reader_task = asyncio.create_task(reader())
        deadline = time.monotonic() + 10
        while self.latest is None:
            if time.monotonic() > deadline:
                raise RuntimeError("No st3215/inference frames from Station after 10 s")
            await asyncio.sleep(0.05)
        self._resolve_buses()

    def _buses(self):
        out = {}
        for b in self.latest.get_buses() or []:
            info = b.get_bus()
            if info:
                out[info.get_serial_number()] = b
        return out

    def _resolve_buses(self):
        buses = self._buses()
        log.info("Station buses: %s", {s: len(b.get_motors() or []) for s, b in buses.items()})
        for arm in self.cfg.arms.values():
            if arm.bus_serial != "auto":
                if arm.bus_serial not in buses:
                    raise RuntimeError(f"{arm.name}: bus {arm.bus_serial} not on Station ({list(buses)})")
                self.bus_of[arm.name] = arm.bus_serial
                continue
            # "auto": pick the only bus whose motor count matches this arm.
            # (GUESS that is safe here: SO-101 has 6 servos, ElRobot has 8.)
            want = len(arm.motor_ids)
            match = [s for s, b in buses.items() if len(b.get_motors() or []) == want]
            if len(match) != 1:
                raise RuntimeError(
                    f"{arm.name}: can't auto-pick a bus with {want} motors from "
                    f"{ {s: len(b.get_motors() or []) for s, b in buses.items()} }. Set bus_serial."
                )
            self.bus_of[arm.name] = match[0]
        log.info("Bus mapping: %s", self.bus_of)

    # ------------------------------------------------------------------ state
    def _motors(self, arm: ArmConfig) -> dict[int, dict]:
        if time.monotonic() - self.latest_t > 1.0:
            raise RuntimeError("Station joint data is stale (>1 s old)")
        bus = self._buses().get(self.bus_of[arm.name])
        if bus is None:
            raise RuntimeError(f"{arm.name}: bus missing from latest frame")
        out = {}
        for m in bus.get_motors() or []:
            sb = bytes(m.get_state())
            out[m.get_id()] = {
                "present": _normal_position(_u16(sb, RAM_PRESENT_POSITION)),
                "goal": _normal_position(_u16(sb, RAM_GOAL_POSITION)),
                "torque": _u8(sb, RAM_TORQUE_ENABLE) != 0,
                "status": _u8(sb, RAM_STATUS),
                "current": _u16(sb, RAM_PRESENT_CURRENT),
                "rmin": m.get_range_min(),
                "rmax": m.get_range_max(),
            }
        return out

    async def read_state(self) -> dict:
        state = {}
        for arm in self.cfg.arms.values():
            ms = self._motors(arm)
            joints = {}
            for j, mid in zip(arm.joint_names, arm.motor_ids):
                m = ms.get(mid)
                if m is None:
                    raise RuntimeError(f"{arm.name}: motor {mid} ({j}) missing")
                if m["rmin"] == 0 and m["rmax"] == 0:
                    raise RuntimeError(f"{arm.name}: motor {mid} ({j}) not calibrated in Station")
                joints[j] = {
                    "pos": ticks_to_norm(m["present"], m["rmin"], m["rmax"]),
                    "goal": ticks_to_norm(m["goal"], m["rmin"], m["rmax"]) if m["torque"] else None,
                    "current": m["current"],
                    "torque": m["torque"],
                    "error": m["status"],
                }
            state[arm.name] = joints
        return state

    # --------------------------------------------------------------- commands
    def _sync_write(self, bus: str, address: int, motors: list[tuple[int, bytes]]):
        st = self._st3215
        cmd = st.Command(
            target_bus_serial=bus,
            sync_write=st.ST3215SyncWriteCommand(
                address=address,
                motors=[st.ST3215SyncWriteCommand_MotorWrite(motor_id=i, value=v) for i, v in motors],
            ),
        )
        return self._commands.DriverCommand(
            type=self._drivers.StationCommandType.STC_ST3215_COMMAND, body=cmd.encode()
        )

    async def write_goals(self, arm_name: str, goals: dict[str, float], speed: int, accel: int):
        arm = self.cfg.arms[arm_name]
        bus = self.bus_of[arm_name]
        ms = self._motors(arm)
        ids, ticks = [], []
        for j, v in goals.items():
            mid = arm.motor_id(j)
            m = ms[mid]
            ids.append(mid)
            ticks.append(norm_to_ticks(v, m["rmin"], m["rmax"]))
        off = [i for i in ids if not ms[i]["torque"]]
        pack = []
        if off:
            pack.append(self._sync_write(bus, RAM_TORQUE_ENABLE, [(i, b"\x01") for i in off]))
        # Order matters: speed/accel before goal (commands.py in the teleop example).
        pack.append(self._sync_write(bus, RAM_GOAL_SPEED, [(i, speed.to_bytes(2, "little")) for i in ids]))
        pack.append(self._sync_write(bus, RAM_ACC, [(i, bytes([accel])) for i in ids]))
        pack.append(self._sync_write(bus, RAM_GOAL_POSITION, [(i, t.to_bytes(2, "little")) for i, t in zip(ids, ticks)]))
        await self._send(self.client, pack)

    async def freeze(self, arm_name: str):
        """Hold every motor where it is right now (goal := present). Torque stays on."""
        arm = self.cfg.arms[arm_name]
        ms = self._motors(arm)
        writes = [(mid, ms[mid]["present"].to_bytes(2, "little")) for mid in arm.motor_ids if mid in ms]
        await self._send(self.client, [self._sync_write(self.bus_of[arm_name], RAM_GOAL_POSITION, writes)])

    async def set_torque(self, arm_name: str, on: bool):
        arm = self.cfg.arms[arm_name]
        v = b"\x01" if on else b"\x00"
        await self._send(
            self.client,
            [self._sync_write(self.bus_of[arm_name], RAM_TORQUE_ENABLE, [(m, v) for m in arm.motor_ids])],
        )

    # ----------------------------------------------------------------- camera
    async def images(self) -> dict[str, bytes]:
        """Latest camera JPEGs from the normvla inference feed.

        Note: normvla images are resized by Station (usb-video resize_target,
        224 px by default). Raise resize_target in station.yaml for sharper marks.
        """
        qid = self.cfg.station["camera_queue"]
        qr = self.client.read_from_tail(qid, offset=b"\x00", limit=1, step=1, buf_size=1)
        entry = await asyncio.wait_for(qr.data.get(), timeout=3.0)
        if entry is None:
            raise RuntimeError(f"{qid} returned no frame ({qr.err})")
        frame = self._normvla.FrameReader(memoryview(bytes(entry.Data)))
        names = self.cfg.station["camera_names"]
        out = {}
        for i, img in enumerate(frame.get_images() or []):
            out[names[i] if i < len(names) else f"cam{i}"] = bytes(img.get_jpeg())
        return out

    def describe(self) -> str:
        return f"Station at {self.cfg.station['host']}:{self.cfg.station['port']}, buses {self.bus_of}"
