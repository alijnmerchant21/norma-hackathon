"""Load cell.toml and the taught-pose file."""

from __future__ import annotations

import json
import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


@dataclass
class ArmConfig:
    name: str
    role: str
    bus_serial: str
    motor_ids: list[int]
    joint_names: list[str]
    gripper: str
    gripper_open: float
    gripper_closed: float
    home_pose: str
    envelope: dict[str, tuple[float, float]] = field(default_factory=dict)

    @property
    def arm_joints(self) -> list[str]:
        """Joint names without the gripper."""
        return [j for j in self.joint_names if j != self.gripper]

    def motor_id(self, joint: str) -> int:
        return self.motor_ids[self.joint_names.index(joint)]


@dataclass
class CellConfig:
    raw: dict
    path: Path
    arms: dict[str, ArmConfig]

    @property
    def station(self) -> dict:
        return self.raw["station"]

    @property
    def guard(self) -> dict:
        return self.raw["guard"]

    @property
    def marking(self) -> dict:
        return self.raw["marking"]

    @property
    def sim(self) -> dict:
        return self.raw.get("sim", {})

    @property
    def state_dir(self) -> Path:
        d = Path(os.environ.get("CELLBOT_STATE", ROOT / "state"))
        d.mkdir(parents=True, exist_ok=True)
        return d

    def spot_ids(self) -> list[str]:
        return sorted(self.marking["spot_targets_px"].keys())


def load_config(path: str | os.PathLike | None = None) -> CellConfig:
    p = Path(path or os.environ.get("CELLBOT_CONFIG", ROOT / "cell.toml"))
    with open(p, "rb") as f:
        raw = tomllib.load(f)
    # Environment overrides, handy at the venue.
    if os.environ.get("CELLBOT_MODE"):
        raw["station"]["mode"] = os.environ["CELLBOT_MODE"]
    if os.environ.get("STATION_HOST"):
        raw["station"]["host"] = os.environ["STATION_HOST"]
    arms = {}
    for name, a in raw["arms"].items():
        arms[name] = ArmConfig(
            name=name,
            role=a["role"],
            bus_serial=a.get("bus_serial", "auto"),
            motor_ids=list(a["motor_ids"]),
            joint_names=list(a["joint_names"]),
            gripper=a["gripper"],
            gripper_open=float(a["gripper_open"]),
            gripper_closed=float(a["gripper_closed"]),
            home_pose=a.get("home_pose", "home"),
            envelope={k: (float(v[0]), float(v[1])) for k, v in a.get("envelope", {}).items()},
        )
    return CellConfig(raw=raw, path=p, arms=arms)


class PoseStore:
    """Named poses, taught by a human (see cellbot.teach).

    Format: {arm: {pose_name: {"joints": {joint: value}, "zone": "work"|"clear"}}}
    Claude can use poses but cannot create or change them.
    """

    def __init__(self, cfg: CellConfig):
        mode = cfg.station["mode"]
        # Separate files so sim poses can never be sent to the real arms.
        self.path = cfg.state_dir / f"poses.{mode}.json"
        if not self.path.exists():
            if mode == "sim":
                self.path.write_text((ROOT / "poses.sim.json").read_text())
            else:
                self.path.write_text("{}")
        self.reload()

    def reload(self):
        self.poses: dict[str, dict[str, dict]] = json.loads(self.path.read_text())

    def get(self, arm: str, name: str) -> dict:
        try:
            return self.poses[arm][name]
        except KeyError:
            raise KeyError(f"No pose '{name}' for arm '{arm}'. Known: {sorted(self.poses.get(arm, {}))}")

    def names(self, arm: str) -> list[str]:
        return sorted(self.poses.get(arm, {}))

    def save(self, arm: str, name: str, joints: dict[str, float], zone: str):
        self.poses.setdefault(arm, {})[name] = {
            "joints": {k: round(v, 4) for k, v in joints.items()},
            "zone": zone,
        }
        self.path.write_text(json.dumps(self.poses, indent=2))
