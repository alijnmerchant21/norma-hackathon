"""Teach named poses on the real arms (human only; Claude cannot create poses).

    python -m cellbot.teach show                       # print current joints of both arms
    python -m cellbot.teach limp so101                 # torque off: move the arm by hand
    python -m cellbot.teach save so101 pick clear      # store current joints as pose "pick", zone clear
    python -m cellbot.teach hold so101                 # torque on again

Poses Claude's tools expect:
  so101:   home, pick_hover, pick, place_hover, place, drop_hover, drop
  elrobot: home, mark_hover_1..4, mark_touch_1..4
Zone is "work" for any pose over the shared work area, else "clear".
Tip: with the leader arm and Station teleop you can steer the follower into place
instead of going limp.
"""

import asyncio
import json
import sys

from .config import PoseStore, load_config
from .station_backend import StationBackend


async def amain(argv):
    cfg = load_config()
    if cfg.station["mode"] != "station":
        print("teach needs mode='station' in cell.toml (or CELLBOT_MODE=station)")
        return
    be = StationBackend(cfg)
    await be.connect()
    cmd = argv[0] if argv else "show"
    if cmd == "show":
        st = await be.read_state()
        print(json.dumps({a: {j: round(v["pos"], 3) for j, v in js.items()} for a, js in st.items()}, indent=2))
    elif cmd == "limp":
        await be.set_torque(argv[1], False)
        print(f"{argv[1]} torque OFF. Support the arm!")
    elif cmd == "hold":
        await be.set_torque(argv[1], True)
        print(f"{argv[1]} torque ON")
    elif cmd == "save":
        arm, name, zone = argv[1], argv[2], argv[3]
        assert zone in ("work", "clear"), "zone must be work or clear"
        st = await be.read_state()
        acfg = cfg.arms[arm]
        joints = {j: st[arm][j]["pos"] for j in acfg.arm_joints}
        PoseStore(cfg).save(arm, name, joints, zone)
        print(f"saved {arm}.{name} ({zone}): {json.dumps({k: round(v, 3) for k, v in joints.items()})}")
    else:
        print(__doc__)


if __name__ == "__main__":
    asyncio.run(amain(sys.argv[1:]))
