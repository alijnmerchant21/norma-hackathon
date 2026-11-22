"""Fixed settings for the offline tests, applied in memory to every GuidedCell a test builds.

guided.toml holds the user's field-test values (touch_mm, dwell_s, all_slow, ...). The tests must not depend on
them: after constructing a cell, call apply(cell). guided.toml is never changed.
"""
from __future__ import annotations

MOTION = dict(arm="elrobot", hover_mm=40.0, high_mm=110.0, clearance_mm=25.0, touch_mm=0.0, dwell_s=0.05,
              ik_tol_mm=3.0, workspace_margin_mm=30.0, reaim_mm=3.0, max_reaim_mm=80.0, max_reaims=6,
              aim_frames=3, follow_s=8.0, commit_age_s=8.0, look_pose="look", settle_rounds=4,
              settle_max_deg=4.0, all_slow=False, sight_max_age_s=60.0, touch_sag_abort_mm=6.0,
              touch_tol_mm=2.0, touch_correct_rounds=4, touch_correct_max_deg=3.0, touch_cmd_below_mm=3.0,
              pretouch_mm=8.0, settle_tol_mm=1.5, settle_still_s=0.1, pretouch_tol_mm=1.0)
VISION = dict(max_state_age_s=0.7, live_wait_s=4.0, sees_under_arm=False)
MARKS = dict(method="color", hsv_lo=[95, 90, 50], hsv_hi=[130, 255, 255], min_area_px=12, search_radius_mm=15.0,
             tolerance_mm=3.0, gain=0.8, max_correction_mm=12.0, window=40)
PART = dict(surface_height_mm=40.0, camera_height_mm=0.0, camera_nadir_px=[])
TOOL = dict(tip_local_m=[], close_gripper=True)


def apply(cell):
    """Override the values the tests rely on, in this process only (call before ensure_guided)."""
    g = cell.gcfg
    cell.m.update(MOTION)
    g["vision"].update(VISION)
    g["marks"].update(MARKS)
    g["part"].update(PART)
    g["tool"].update(TOOL)
    cell.model.gain, cell.model.max_c, cell.model.window = MARKS["gain"], MARKS["max_correction_mm"], MARKS["window"]
    return cell
