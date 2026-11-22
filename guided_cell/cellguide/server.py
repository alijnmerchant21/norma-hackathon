"""MCP server for the guided cell: all of cellbot's tools plus camera-guided work on a part.

It reuses cellbot's server unchanged: cellbot.mcp_server is imported after cellbot.cell.Cell is pointed at
GuidedCell, so its module-level `cell` is a GuidedCell and every existing cellbot tool (get_status, look,
go_home, stop, finish_part, ...) keeps working with the same guards.

Run (stdio) from guided_cell/.venv (made by setup_mac.sh, which also prints the Claude desktop config block):
    PYTHONPATH=<guided_cell> CELLBOT_CONFIG=<guided_cell>/cell.guided.toml CELLBOT_STATE=<guided_cell>/state \\
        CELLBOT_MODE=station STATION_HOST=127.0.0.1 .venv/bin/python -m cellguide.server
Optional env: CELLGUIDE_CONFIG (another guided.toml).
Use this server instead of cellbot's own one, not next to it (two servers would drive the same arms).
"""
from __future__ import annotations

from . import guided_cell as gc  # loads guided.toml and puts cellbot on sys.path

import cellbot.cell as _cellmod  # noqa: E402

_cellmod.Cell = gc.GuidedCell     # so cellbot.mcp_server builds a GuidedCell

from cellbot import mcp_server as base  # noqa: E402  (all existing cellbot tools stay available)

from .operator_prompt import SYSTEM, run_guided_text  # noqa: E402
from .vision_source import VisionError, install as _install_camera  # noqa: E402

_install_camera(gc)                # camera:auto + self-healing camera:N (guided_cell.py unchanged)

cell = base.cell
mcp = base.mcp
ToolError, Image, GuardError = base.ToolError, base.Image, gc.GuardError
assert isinstance(cell, gc.GuidedCell), "cellbot.mcp_server did not build a GuidedCell"


async def _run(coro):
    """Like cellbot's _guarded, plus a readable message for vision problems. Returns the raw result."""
    try:
        return await coro
    except GuardError as e:
        cell.log_event("refused", str(e))
        raise ToolError(f"REFUSED by guardrail: {e}")
    except VisionError as e:
        cell.log_event("vision", str(e))
        raise ToolError(f"VISION problem: {e}. Check that the vision server runs and sees the part "
                        "(arm out of the view), then try again.")
    except ToolError:
        raise
    except Exception as e:  # hardware/link problems: tell Claude, don't hide them
        cell.log_event("error", repr(e))
        raise ToolError(f"ERROR: {e!r}")


@mcp.tool()
async def new_part(part_id: str = "") -> str:
    """Start work on a new part that is now in the work area. Clears the old hole layout and waits until the
    camera has locked onto the new part. The elrobot must be out of the camera's view (retreat or home).
    Returns the part id, the hole ids and the box angle."""
    return base._j(await _run(cell.new_part(part_id or None)))


@mcp.tool()
async def see_part() -> list:
    """What the top camera sees now: vision status, box pose, and per hole its id, position on the lid
    (fractions 0..1), robot coordinates in mm, whether the arm can reach it (and why not), and whether it is
    done on this part. Plus the annotated camera image. Use it before writing a program and after any stop."""
    res, img = await _run(cell.scene())
    return [base._j(res), Image(data=img, format="jpeg")]


@mcp.tool()
async def run_program(steps: list[dict], retreat_after: bool = True, look_each_step: bool = False) -> str:
    """Run a program you wrote for this part, in the part's own frame. steps is an ordered list, each step one
    of {"hole": <id>} or {"at": [fx, fy]} (fractions of the lid), optional "action": "dab" (default, touch
    with the pen) or "hover", optional "dwell_s". Example: [{"hole": 0}, {"hole": 2}, {"at": [0.5, 0.5],
    "action": "hover"}]. With the top camera only, the arm takes ONE clear look at the start (retreating first
    if needed) and every step aims from it; the part is not watched while the arm is over it. If the part may
    be moved between steps, set look_each_step (or "look": true on a step) to retreat and look again first.
    Stops at the first step that fails and returns the reason and the remaining steps. With retreat_after the
    arm leaves the camera's view at the end. Per step: sighting_age_s and aimed_from (age of the clear look),
    touch_error_mm (measured minus commanded touch height); re_aims/followed_mm stay 0 without a camera that
    sees under the arm."""
    return base._j(await _run(cell.run_program(steps, retreat_after, look_each_step)))


@mcp.tool()
async def plan_program(steps: list[dict]) -> str:
    """Check a program WITHOUT moving (same steps as run_program). For the current camera view and arm pose,
    per step: target, pixel, robot mm, learned correction, reachable or the refusal reason, tool tilt, path
    (direct or via the high point) and the biggest joint change in degrees per leg; each step is planned from
    the previous step's hover pose. "ok" is true only if every step is reachable. Use it before run_program
    for a new kind of part and after any refusal."""
    return base._j(await _run(cell.plan_program(steps)))


@mcp.tool()
async def preflight() -> str:
    """Read-only readiness checklist (never moves the arms): Python and packages, file paths and calibration,
    cellbot config and guards, e-stop, Station link and joint ranges, taught poses, tool tip, part geometry,
    vision and image size, hole reachability, learning and logs. Each check has status ok/warn/fail/skip,
    detail and fix. The verdict says READY (no fail) or NOT READY with the number of blocking items."""
    from .preflight import report
    return base._j(await _run(report(cell)))


@mcp.tool()
async def inspect_part() -> list:
    """Measure every mark made on this part against its target with the top camera (the arm must be out of
    the view). Returns per target the miss in mm and pass/fail against the tolerance, stray marks, the
    learning status, and an annotated image. Measured misses train the aim correction automatically."""
    res, img = await _run(cell.inspect())
    return [base._j(res), Image(data=img, format="jpeg")]


@mcp.tool()
async def retreat() -> str:
    """Lift the elrobot straight up, then move it to the 'look' pose (out of the camera's view), or home if no
    look pose is taught. Use it before see_part or inspect_part when the arm hides the part."""
    return base._j(await _run(cell.retreat()))


@mcp.tool()
async def guided_status() -> str:
    """Guided-cell set-up and state: vision source, pen tip source, calibration file, part height, where the
    tip is now, droop correction, targets done on this part, the learning trend (mean miss per part, and per
    known target position: marks, mean error and the aim correction applied there now), and
    the active guard limits (guards)."""
    res = await _run(cell.guided_status())
    try:                                   # loader episode counts (no change to guided_cell.py needed)
        st = cell.loader_stats(last=0)
        res["loader_episodes"] = {"total": st["total"], "by_phase_variant": st["by_phase_variant"]}
    except Exception as e:  # noqa: BLE001  never let the summary break the status
        res["loader_episodes"] = {"error": f"{type(e).__name__}: {e}"}
    return base._j(res)


@mcp.tool()
async def load_part(variant: str = "primary") -> str:
    """so101 brings a part from the pickup area to the work area. Refused unless the elrobot is at a clear pose.
    Mode from guided.toml [loader]: "taught" (taught poses, all guards, grasp retried up to 2 times; returns the
    steps and the failed step), "policy" (a learned policy outside the guards; returns exit_code and
    output_tail; if it did not exit cleanly so101 stays at an unknown position and go_home so101 is needed
    before the elrobot may move), or "human" (no motion: ask the human to place the part). load_mode /
    unload_mode override mode per phase; taught_<phase>_seq (taught mode) replaces the fixed poses with a taught
    sequence (e.g. a sweep), ending at home; a failed step stops it and is named in failed_step.
    variant (policy mode only): "primary" (default: policy_load_cmd with its pre-pose/pre-seq) or "fallback"
    (policy_load_fallback_cmd, or policy_load_cmd if only policy_load_fallback_pre_seq is set; the primary's
    pre-pose/pre-seq never run, policy_load_fallback_pre_seq taught steps run first if set; refused if neither).
    variant "teleop" (any mode): a HANDOVER, no motion: a human moves the so101 with the leader arm through
    Station's teleoperation (pickup -> work area); returns instructions_for_human. While it is open every so101
    motion is refused; close it with finish_teleop.
    Every attempt is logged with an attempt_id; label it afterwards with label_loader_attempt.
    Then call new_part: it verifies that a part is really there."""
    return base._j(await _run(cell.load_part(variant)))


@mcp.tool()
async def unload_part(variant: str = "primary") -> str:
    """so101 takes the part from the work area to the drop-off area. Refused unless the elrobot is at a clear
    pose. Same modes and results as load_part. In taught mode, if the gripper closes on nothing the so101
    opens, goes home and the call fails with "nothing grasped". variant (policy mode only): "primary" (default:
    policy_unload_cmd, e.g. the raw policy) or "fallback" (policy_unload_fallback_pre_seq taught steps first if
    set, then policy_unload_fallback_cmd, or policy_unload_cmd if that is empty; refused if neither).
    variant "teleop" (any mode): a HANDOVER, no motion: a human moves the so101 with the leader arm through
    Station's teleoperation (work area -> drop-off); returns instructions_for_human and an open attempt_id. While
    it is open every so101 motion is refused; when the human says done and teleoperation is OFF: finish_teleop.
    A teleop attempt labelled success is a human demonstration (training data). Every attempt is logged with an
    attempt_id: look with see_part from a clear pose, then label it with label_loader_attempt."""
    return base._j(await _run(cell.unload_part(variant)))


@mcp.tool()
async def finish_teleop(attempt_id: str = "") -> str:
    """Close the open teleop handover (opened by load_part / unload_part with variant "teleop") once the human
    said they are done and teleoperation is OFF. Never moves, never labels: it records the end time. Then:
    confirm teleoperation is OFF with the human, go_home so101, look with the camera, label_loader_attempt.
    attempt_id "" = the open one. Refused if no handover is open."""
    async def go():
        return cell.finish_teleop(attempt_id)
    return base._j(await _run(go()))


@mcp.tool()
async def label_loader_attempt(success: bool, note: str = "", attempt_id: str = "") -> str:
    """Set the outcome of a load/unload attempt after checking with the camera (see_part / new_part from a clear
    pose): load success = the part is in the work area and seen; unload success = the work area is empty.
    attempt_id "" = the latest attempt. Label EVERY attempt: successful ones become training data."""
    async def go():
        return cell.label_loader_attempt(bool(success), note, attempt_id)
    return base._j(await _run(go()))


@mcp.tool()
async def loader_stats() -> str:
    """Loader attempts per phase and variant (attempts, successes, fails, unlabelled, success rate, mean
    seconds, open teleop handovers), the last 10 attempts, and dataset_ranges: the time ranges of successful
    attempts per phase/variant, each with its source ("policy" or "teleop demonstration"), used by a human to cut training episodes from Station's recording. Never moves."""
    async def go():
        return cell.loader_stats()
    return base._j(await _run(go()))


@mcp.tool()
async def alerts() -> str:
    """What happened since the last call, as an operator feed: alerts (level info/warn/critical, source,
    message, needs_human) from cellbot events, vision events and accuracy metrics; metrics (parts done, marks,
    mean miss per part, re-aims, followed mm and seconds per touch); and a recommendation: "continue",
    "check and continue" or "pause and ask a human". Call it after run_program and whenever a step fails,
    and follow the recommendation."""
    return base._j(await _run(cell.alerts()))


@mcp.prompt()
def run_guided(parts: str = "3") -> str:
    """Run the guided cell (see part, write a program, run, inspect, finish, report) for N parts."""
    try:
        n = max(1, min(int(parts), 20))
    except ValueError:
        n = 3
    return run_guided_text(n)


@mcp.resource("cellguide://procedure")
def procedure() -> str:
    """Operator procedure and safety rules for the guided cell."""
    return SYSTEM


def main():
    mcp.run()


if __name__ == "__main__":
    main()
