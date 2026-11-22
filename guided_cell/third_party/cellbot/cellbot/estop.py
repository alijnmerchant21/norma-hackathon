"""Human e-stop control.

    python -m cellbot.estop stop [reason]   # latch: the MCP server freezes both arms at the next check
    python -m cellbot.estop release         # human only: allow motion again
    python -m cellbot.estop status

Claude can latch the e-stop (the `stop` tool) but has no tool to release it.
Keep the Station web UI open too: it is the hardware-level fallback.
"""

import json
import sys
import time

from .config import load_config


def main():
    cfg = load_config()
    path = cfg.state_dir / "estop.json"
    cmd = sys.argv[1] if len(sys.argv) > 1 else "status"
    if cmd == "stop":
        reason = " ".join(sys.argv[2:]) or "human e-stop"
        path.write_text(json.dumps({"latched": True, "reason": reason, "t": time.time()}))
        print("E-STOP LATCHED:", reason)
    elif cmd == "release":
        path.write_text(json.dumps({"latched": False, "released_t": time.time()}))
        print("E-stop released. Arms may move again.")
    else:
        print(path.read_text() if path.exists() else '{"latched": false}')


if __name__ == "__main__":
    main()
