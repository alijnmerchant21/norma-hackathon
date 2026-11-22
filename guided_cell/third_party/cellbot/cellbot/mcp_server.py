"""MCP server: the only way Claude touches the robots.

Run (stdio):   python -m cellbot.mcp_server
Add to Claude Code:  claude mcp add cellbot -- python -m cellbot.mcp_server
Set CELLBOT_MODE=sim|station and STATION_HOST=<mac ip> to override cell.toml.

Every tool returns JSON text; refused commands come back as errors that say why,
so Claude can correct itself instead of the arm doing something surprising.
"""

from __future__ import annotations

import json
import logging
import sys

try:  # mcp >= 2
    from mcp.server.mcpserver import MCPServer as _Server
    from mcp.server.mcpserver.exceptions import ToolError
    from mcp.server.mcpserver.utilities.types import Image
except ImportError:  # mcp 1.x
    from mcp.server.fastmcp import FastMCP as _Server
    from mcp.server.fastmcp.exceptions import ToolError
    from mcp.server.fastmcp.utilities.types import Image

from .cell import Cell, GuardError

logging.basicConfig(stream=sys.stderr, level=logging.INFO)  # stdout is the MCP channel

cell = Cell()
mcp = _Server(
    "cellbot",
    instructions=(
        "Guarded control of a 2-arm marking cell (SO-101 handler + ElRobot marker) on a NormaCore "
        "Station. Joint values are normalized 0..1 of each joint's calibrated range. Big moves use "
        "taught poses; move_joints is for small adjustments only. Call get_status first."
    ),
)


def _j(obj) -> str:
    return json.dumps(obj, indent=1, default=str)


async def _guarded(coro):
    try:
        return _j(await coro)
    except GuardError as e:
        cell.log_event("refused", str(e))
        raise ToolError(f"REFUSED by guardrail: {e}")
    except Exception as e:  # hardware/link problems: tell Claude, don't hide them
        cell.log_event("error", repr(e))
        raise ToolError(f"ERROR: {e!r}")


@mcp.tool()
async def get_status() -> str:
    """Joint positions of both arms (normalized 0..1), zones (work/clear/unknown), last taught pose,
    gripper, e-stop state, the part in progress, and the taught poses available per arm."""
    return await _guarded(cell.status())


@mcp.tool()
async def look() -> list:
    """Camera images of the cell (top camera and any others) plus a short status line.
    Use it to check the part is at pickup, seated in the work area, or removed."""
    imgs = await cell.look()
    st = await cell.status()
    brief = {a: {"zone": v["zone"], "pose": v["last_pose"], "gripper": v["gripper"]} for a, v in st["arms"].items()}
    out: list = [f"cameras: {list(imgs)}; arms: {json.dumps(brief)}; estop: {st['estop'].get('latched')}"]
    for name, jpeg in imgs.items():
        out.append(f"camera '{name}':")
        out.append(Image(data=jpeg, format="jpeg"))
    return out


@mcp.tool()
async def go_to_pose(arm: str, pose: str, speed: str = "normal") -> str:
    """Move an arm ('so101' or 'elrobot') to a taught pose (see get_status for names).
    speed: 'normal' or 'slow'. A 'work' pose is refused while the other arm is in the work area."""
    return await _guarded(cell.go_to_pose(arm, pose, speed))


@mcp.tool()
async def go_home(arm: str) -> str:
    """Move an arm slowly to its home pose (a clear pose, outside the work area)."""
    return await _guarded(cell.go_home(arm))


@mcp.tool()
async def move_joints(arm: str, joints: dict[str, float], relative: bool = True) -> str:
    """Small adjustment of named joints, e.g. {"wrist_flex": -0.03}. relative=True adds to the current
    position. Limited per call and to a small distance from the last taught pose. Not for the gripper."""
    return await _guarded(cell.move_joints(arm, joints, relative))


@mcp.tool()
async def gripper(arm: str, action: str) -> str:
    """'open' or 'close' the gripper. After 'close' the result says whether an object is held
    (gripper stopped early) or nothing was grasped (closed fully)."""
    return await _guarded(cell.gripper(arm, action))


@mcp.tool()
async def new_part_ready() -> str:
    """Call at the start of each part. In the simulator this puts a new part at pickup."""
    return await _guarded(cell.new_part_ready())


@mcp.tool()
async def start_marking(part_id: str = "") -> str:
    """Begin marking a part that is now in the work area. Both arms must be at clear poses: this
    takes the camera reference image used to find the new marks."""
    return await _guarded(cell.start_marking(part_id or None))


@mcp.tool()
async def mark_spot(spot: str) -> str:
    """ElRobot marks one spot ('1'..'4'): hover, touch, dwell, lift. The learned correction for
    that spot is applied automatically and reported. Leaves the ElRobot above the part."""
    return await _guarded(cell.mark_spot(spot))


@mcp.tool()
async def inspect_marks() -> list:
    """Measure every mark against its target with the top camera (both arms must be clear).
    Returns per-spot miss in mm, pass/fail against tolerance, and an annotated image.
    Measured misses are fed to the learning loop automatically."""
    try:
        res, img = await cell.inspect_marks()
    except GuardError as e:
        raise ToolError(f"REFUSED by guardrail: {e}")
    except Exception as e:
        raise ToolError(f"ERROR: {e!r}")
    return [_j(res), Image(data=img, format="jpeg")]


@mcp.tool()
async def learning_status() -> str:
    """Current learned corrections per spot (mm), the refined arm Jacobian, and recent part results."""
    return _j(cell.learner.status())


@mcp.tool()
async def log_result(kind: str, message: str) -> str:
    """Log an event for the report, e.g. kind='exception' message='grasp failed, retried'."""
    return _j(cell.log_event(kind, message))


@mcp.tool()
async def finish_part(outcome: str, notes: str = "") -> str:
    """Close the current part. outcome: 'pass', 'fail' or 'aborted'. Notes go in the log."""
    return await _guarded(cell.finish_part(outcome, notes))


@mcp.tool()
async def stop(reason: str) -> str:
    """EMERGENCY STOP: freeze both arms and latch. Only a human can release it. Use when anything
    looks unsafe or unexpected (dropped part, collision, arm not where it should be)."""
    return _j(await cell.stop(reason))


@mcp.prompt()
def run_parts(parts: str = "3") -> str:
    """Run the full cell procedure (pick, mark 4 spots, inspect, remove, report) for N parts."""
    from .operator_prompt import run_parts_text
    try:
        n = max(1, min(int(parts), 20))
    except ValueError:
        n = 3
    return run_parts_text(n)


@mcp.resource("cellbot://procedure")
def procedure() -> str:
    """Operator procedure, safety rules and learning loop for the marking cell."""
    from .operator_prompt import SYSTEM
    return SYSTEM


def main():
    mcp.run()


if __name__ == "__main__":
    main()
