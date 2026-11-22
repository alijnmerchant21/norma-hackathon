"""Offline end-to-end check: cellbot's simulator + Manasa's real test frames + the real ElRobot model.

No hardware, no camera, no network:
    python tests/test_guided_sim.py

It checks the joins between the three code drops, not accuracy on the real rig:
  1. the hole layout locks and every hole gets a reachable robot target;
  2. a program in the part's frame runs through cellbot's guards and the modelled tip lands on each target;
  3. the part moves while the arm hovers: sees_under_arm = true -> re-aims; false (top camera) -> not
     watched, aims from the clear look; run_program look_each_step looks again between steps;
  4. the part disappears -> the step is refused with a reason;
  5. marks are measured and the next target is shifted against the measured miss;
  6. guards; 7. the so101 loader (taught, policy, human); 8. the alerts feed;
  9. vision while the arm is over the part is flagged (scene) and ignored (alerts);
  10. gravity droop (modelled in the simulator): the offset learned at hover carries into touch and lift;
  11. joint-2 tracking error near the part (short / over): the closed loop runs at the pre-touch height (never in
      contact), the final straight descent lands within tolerance, no sideways tip motion near the lid; an arm
      that does not respond to the correction, or is too high only at the touch, is lifted and the step fails;
      in a program the pre-touch correction seeds the next hole; the settle loop does not wind the carried
      offset up on a slow servo or on a joint pinned at its limit;
  12. the hover is closed on the measured TIP (FK of joint feedback): with drooping / creeping servos, joint 6 pinned
      short and a wound-up carried offset, every hole arrives at the pre-touch height within 3 mm (0-1 rounds);
  13. the pre-touch loop corrects the full 3D tip (height first, then xy): a 3-5 mm sideways tracking error
      (joint 7, or joint 1 only near the part) ends within ~1 mm in plane at pre-touch and at the touch; an
      error beyond the cap proceeds and reports the residual.
"""
import asyncio
import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ["CELLBOT_MODE"] = "sim"
os.environ["CELLBOT_STATE"] = tempfile.mkdtemp(prefix="cellguide_test_")

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from cellguide import config as gconfig  # noqa: E402
from cellguide.guided_cell import GuidedCell  # noqa: E402
from cellguide.vision_source import FrameVision  # noqa: E402
import sim_settings  # noqa: E402  (tests/: fixed motion values, independent of guided.toml)

G = gconfig.load()
V = gconfig.load_module("part_vision", G["_paths"]["vision_file"])
IMG = G["_paths"]["vision_file"].parent / "images"


class Feed:
    """A scripted camera. `frame` is what the camera shows now; `ink` adds dabs relative to the detected holes."""

    def __init__(self):
        self.base = {i: cv2.imread(str(IMG / f"test_{i}.jpeg")) for i in (1, 2, 3)}
        self.show(1)
        self.ink: list[tuple[int, float, float]] = []     # (hole id, dx px, dy px)
        self.holes_px = {}

    def show(self, i, gone=False):
        self.cur, self.gone = self.base[i], gone

    async def __call__(self):
        f = self.cur.copy()
        if self.gone:
            scene = V.detect_scene(f)
            m = np.zeros(f.shape[:2], np.uint8)
            cv2.fillPoly(m, [np.int32(scene["_face_px"])], 255)
            m = cv2.dilate(m, np.ones((25, 25), np.uint8))
            f[m > 0] = np.median(f[m == 0], axis=0)
        for hid, dx, dy in self.ink:
            x, y = self.holes_px[hid]
            cv2.circle(f, (int(round(x + dx)), int(round(y + dy))), 3, (200, 60, 20), -1)   # blue ink (BGR)
        return f


def check(name, cond, detail=""):
    print(f"  [{'ok' if cond else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    if not cond:
        check.failed += 1


check.failed = 0


async def main():
    feed = Feed()
    cell = sim_settings.apply(GuidedCell())
    cell.vision = FrameVision(V, feed, "test frames")
    await cell.ensure_guided()
    k = cell.kin
    print(f"backend: {cell.backend.describe()};  tool tip: {k.tool_source}")

    print("1. lock the hole layout")
    part = await cell.new_part("t1")
    scene, jpeg = await cell.scene()
    check("3 holes locked", part["holes"] == [0, 1, 2], str(part["holes"]))
    check("all holes reachable", all(h["reachable"] for h in scene["holes"]),
          "; ".join(h.get("why_not", "") for h in scene["holes"]))
    check("annotated image returned", len(jpeg) > 10000)
    st = await cell.vision.state()
    feed.holes_px = {h["id"]: h["px"] for h in st["holes"]}

    print("1b. plan a program without moving")
    import copy
    j0, z0 = copy.deepcopy(cell.backend.joints), dict(cell.zone)
    pl = await cell.plan_program([{"hole": 0}, {"hole": 2}, {"at": [0.5, 0.2], "action": "hover"}])
    check("3-step plan: all reachable", pl["ok"] and len(pl["steps"]) == 3 and all(s["reachable"] for s in pl["steps"]),
          "; ".join(f'{s["target"]}: {s.get("path")} tilt {s.get("tool_tilt_deg")} travel {s.get("joint_travel_max_deg")}'
                    for s in pl["steps"]))
    check("planning did not move the arm or change zones", cell.backend.joints == j0 and cell.zone == z0)
    pl = await cell.plan_program([{"hole": 0}, {"hole": 9}])
    bad = pl["steps"][1]
    check("unknown hole reported unreachable with a reason, ok False", not pl["ok"] and pl["steps"][0]["reachable"]
          and not bad["reachable"] and "no hole 9" in bad.get("why_not", ""), bad.get("why_not", ""))

    print("2. run a program in the part's frame")
    res = await cell.run_program([{"hole": 0}, {"hole": 2}, {"at": [0.5, 0.2], "action": "hover"}], retreat_after=False)
    check("program completed", res["ok"], str([r.get("reason") for r in res["results"]]))
    gr = cell.cfg.arms["elrobot"]
    check("gripper closed before the first dab, not again later",
          res["results"][0].get("gripper_closed_first") is True
          and not any(r.get("gripper_closed_first") for r in res["results"][1:])
          and cell.backend.joints["elrobot"][gr.gripper] <= gr.gripper_closed + 0.08,
          str([r.get("gripper_closed_first") for r in res["results"]]))
    for r in res["results"]:
        check(f"{r['target']}: {r['action']}", r["ok"] and r["hover_error_model_mm"] < 1.0,
              f"model error {r['hover_error_model_mm']} mm, tilt {r['tool_tilt_deg']} deg, {r['seconds']} s")
    hv_ok = all(abs(r.get("tip_above_part_mm", -99) - (cell.m["hover_mm"] + cell.m["touch_mm"])) < 1.5
                for r in res["results"])
    dab_ok = all(abs(r.get("lowest_tip_above_part_mm", -99) - cell.m["touch_mm"]) < 1.5
                 for r in res["results"] if r.get("action") == "dab")
    check("results report tip_above_part_mm (and lowest_tip_above_part_mm for dabs)", hv_ok and dab_ok,
          str([(r.get("tip_above_part_mm"), r.get("lowest_tip_above_part_mm")) for r in res["results"]]))
    q = await cell._q_now()
    a = cell._aim(st, {"at": [0.5, 0.2]})
    err = np.linalg.norm(k.tip(q) - (a["P"] + k.n * cell.m["hover_mm"] / 1000)) * 1000
    check("tip ends above the last target", err < 1.0, f"{err:.2f} mm")
    check("zone bookkeeping: arm is in the work area", cell.zone["elrobot"] == "work")
    r = await cell.retreat()
    check("retreat to a taught clear pose", r["ok"] and cell.zone["elrobot"] == "clear", str(r.get("pose")))

    print("2b. neighbouring holes use neighbouring joint solutions")
    for order in ([0, 1, 2], [0, 2, 1]):
        prog = [{"hole": h} for h in order]
        pl = await cell.plan_program(prog)
        tr = [s.get("joint_travel_max_deg", {}).get("to_hover", 999) for s in pl["steps"]]
        check(f"holes {order}: hover-to-hover joint travel < 20 deg for steps 1, 2",
              pl["ok"] and tr[1] < 20 and tr[2] < 20, f"to_hover {tr}, tilt {[s.get('tool_tilt_deg') for s in pl['steps']]}")
        res = await cell.run_program(prog, retreat_after=True)
        check(f"holes {order}: program runs, modelled tip on each target < 1 mm",
              res["ok"] and all(r["ok"] and r["hover_error_model_mm"] < 1.0 for r in res["results"]),
              str([r.get("hover_error_model_mm", r.get("reason")) for r in res["results"]]))

    print("3. the part moves while the arm hovers (sees_under_arm = true: a camera that sees under the arm)")
    cell.gcfg["vision"]["sees_under_arm"] = True
    feed.show(1)
    for _ in range(4):
        await cell.vision.state()

    # Deterministic: the simulator moves instantly, so the part "moves" on the first frame requested after
    # the arm has reached hover height above it (not on a wall-clock timer).
    real_move = cell._guided_move
    flipped = []

    async def move_then_flip(q, *a, **kw):
        res = await real_move(q, *a, **kw)
        hover = cell.m["hover_mm"] / 1000
        if not flipped and res["ok"] and abs(cell._above_part(q) - hover) < 0.002:
            flipped.append(True)
            feed.show(2)
        return res
    cell._guided_move = move_then_flip
    try:
        r = await cell.work_on({"hole": 1})
    finally:
        cell._guided_move = real_move
    check("part was moved while hovering", bool(flipped))
    check("followed the part", r["ok"] and r["re_aims"] >= 1, f"re-aims {r.get('re_aims')}, followed {r.get('followed_mm')} mm"
          if r["ok"] else str(r.get("reason")))
    await cell.retreat()

    print("3b. the part jumps more than max_reaim_mm while the arm hovers")
    feed.show(1)
    for _ in range(4):
        await cell.vision.state()
    flipped.clear()
    m0 = cell.m.get("max_reaim_mm", 80.0)
    cell.m["max_reaim_mm"] = 5.0
    cell._guided_move = move_then_flip
    try:
        r = await cell.work_on({"hole": 1})
    finally:
        cell._guided_move = real_move
        cell.m["max_reaim_mm"] = m0
    q = await cell._q_now()
    check("jump is refused, not chased", bool(flipped) and not r["ok"] and "jumped" in r.get("reason", "")
          and "max_reaim_mm" in r["reason"], r.get("reason", ""))
    check("arm lifted to travel height after the jump",
          abs(cell._above_part(q) - cell.m["high_mm"] / 1000) < 0.005, f"{cell._above_part(q) * 1000:.0f} mm")
    gs = await cell.guided_status()
    check("guided_status reports the guards", {"max_reaim_mm", "aim_frames", "commit_age_s", "sees_under_arm",
                                               "sight_max_age_s"} <= set(gs["guards"]["cellguide"])
          and "current_limit" in gs["guards"]["cellbot"], str(gs["guards"]))
    await cell.retreat()
    cell.gcfg["vision"]["sees_under_arm"] = False

    print("3c. top camera only (sees_under_arm = false): one clear look, no following")
    feed.show(1)
    for _ in range(4):
        await cell.vision.state()
    flipped.clear()
    seen = []
    real_vstate = cell._vstate

    async def spy_vstate():
        seen.append(cell.zone["elrobot"])
        return await real_vstate()
    cell._vstate = spy_vstate
    cell._guided_move = move_then_flip
    try:
        r = await cell.work_on({"hole": 1})
    finally:
        cell._guided_move = real_move
        cell._vstate = real_vstate
    check("part moved while hovering, not chased: re_aims 0, aimed from the clear view", bool(flipped) and r["ok"]
          and r["re_aims"] == 0 and r["followed_mm"] == 0 and r["aimed_from"].startswith("clear view"),
          f'{r.get("aimed_from")}, re-aims {r.get("re_aims")}')
    check("vision was read only while the arm was clear", seen and all(z == "clear" for z in seen), str(set(seen)))
    check("the note says the part is not watched while the arm is over it", "NOT watched" in r.get("note", ""))
    sc, _ = await cell.scene()
    check("scene with the arm over the part: top-level warning, holes and pose marked not reliable",
          "warning" in sc and "not reliable" in sc["pose"]
          and all(h.get("reliability") == "not reliable: arm in view" for h in sc.get("holes", [])), sc.get("pose"))
    t_old = cell._sight["t"]
    cell._sight["t"] -= 1000                              # stale: the next step must retreat and look again
    feed.show(1)
    r = await cell.work_on({"hole": 0, }, "hover")
    check("stale sighting with the arm in view: retreat, fresh look, then the step", r["ok"]
          and r["sighting_age_s"] < 5 and cell._sight["t"] > t_old, f'{r.get("aimed_from")}')
    await cell.retreat()

    async def prog(look):
        feed.show(1)
        for _ in range(4):
            await cell.vision.state()
        flipped.clear()
        cell._guided_move = move_then_flip
        try:
            return await cell.run_program([{"hole": 1, "action": "hover"}, {"hole": 1, "action": "hover"}],
                                          retreat_after=True, look_each_step=look)
        finally:
            cell._guided_move = real_move
    r0 = await prog(False)
    a0 = [x.get("aimed_robot_mm") for x in r0["results"]]
    check("run_program: one look, the part moved after step 1 -> step 2 aims from the old look", r0["ok"]
          and a0[0] == a0[1], str(a0))
    r1 = await prog(True)
    a1 = [x.get("aimed_robot_mm") for x in r1["results"]]
    d = float(np.linalg.norm(np.subtract(a1[1], a1[0]))) if r1["ok"] else -1
    check("run_program look_each_step: retreat and look before step 2 -> aims at the moved part", r1["ok"] and d > 3,
          f"{a1}, {d:.0f} mm")
    feed.show(1)
    for _ in range(4):
        await cell.vision.state()

    print("4. the part disappears")
    feed.show(2, gone=True)
    t0 = cell.gcfg["vision"]["live_wait_s"]
    cell.gcfg["vision"]["live_wait_s"] = 0.6
    r = await cell.run_program([{"hole": 0}], retreat_after=False)
    cell.gcfg["vision"]["live_wait_s"] = t0
    check("refused with a reason, arm did not move", not r["ok"] and cell.zone["elrobot"] == "clear",
          r["results"][0].get("reason", ""))
    feed.show(2)
    for _ in range(4):
        await cell.vision.state()
    r = await cell.run_program([{"hole": 9}], retreat_after=False)
    reason = r["results"][0].get("reason", "")
    check("unknown hole id refused with a reason", not r["ok"] and "no hole 9" in reason
          and cell.zone["elrobot"] == "clear", reason)

    print("5. measure marks, learn, shift the next target")
    feed.show(1)
    await cell.new_part("t2")
    st = await cell.vision.state()
    feed.holes_px = {h["id"]: h["px"] for h in st["holes"]}
    r = await cell.run_program([{"hole": 0}, {"hole": 1}, {"hole": 2}])
    check("3 dabs + retreat", r["ok"] and r["retreat"]["ok"])
    feed.ink = [(0, 6, 3), (1, 6, 3), (2, 6, 3)]            # the "arm" lands 6 px right, 3 px down of each hole
    before = cell._aim(st, {"hole": 1})
    out, jpeg = await cell.inspect()
    found = [m for m in out["marks"].values() if m["found"]]
    check("3 marks measured", len(found) == 3, str(out["marks"]))
    mm_px = np.linalg.norm(k.surface_point(np.array(feed.holes_px[1]) + [20, 0], st["image_size"]) - before["P0"]) / 20 * 1000
    expect = float(np.hypot(6, 3) * mm_px)
    got = float(np.mean([m["miss_dist_mm"] for m in found])) if found else -1
    check("miss size is right", abs(got - expect) < 0.8, f"measured {got:.1f} mm, drawn {expect:.1f} mm")
    after = cell._aim(st, {"hole": 1})
    shift = (after["P"] - before["P"]) * 1000
    ink_dir = k.surface_point(np.array(feed.holes_px[1]) + [6, 3], st["image_size"]) - before["P0"]
    cosang = float(shift @ ink_dir / (np.linalg.norm(shift) * np.linalg.norm(ink_dir) + 1e-12))
    check("next target shifted against the miss", cosang < -0.95 and 0.6 * expect < np.linalg.norm(shift) < expect,
          f"shift {np.linalg.norm(shift):.1f} mm, direction cos {cosang:.2f}")
    fin = await cell.finish_part("pass", "sim test")
    check("cellbot finish_part still works", fin["outcome"] == "pass" and len(fin["misses_mm"]) == 3)

    print("6. guards")
    try:
        await cell.work_on({"at": [3.0, 0.5]})
        check("off-part target refused", False)
    except Exception as e:
        check("off-part target refused", "fractions" in str(e), str(e)[:70])
    (cell.cfg.state_dir / "estop.json").write_text('{"latched": true, "reason": "test"}')
    await cell.new_part("t3")
    r = await cell.run_program([{"hole": 0}], retreat_after=False)
    check("e-stop blocks guided motion", not r["ok"] and "E-STOP" in r["results"][0]["reason"], r["results"][0]["reason"][:60])
    (cell.cfg.state_dir / "estop.json").write_text('{"latched": false}')
    cell.zone["so101"] = "unknown"
    r = await cell.run_program([{"hole": 0}], retreat_after=False)
    check("other arm not clear blocks guided motion", not r["ok"] and "work area is not free" in r["results"][0]["reason"])

    await cell.go_home("so101")

    print("7. loader (so101)")
    from cellguide.guided_cell import GuardError
    lc = cell.gcfg.setdefault("loader", {})
    lc.update(mode="taught", arm="so101")
    retries = 0
    unload_fail_ok, unload_fails = True, 0
    cycles_ok = 0
    for i in range(4):
        cell.backend.new_part()
        r = await cell.load_part()
        retries += sum(1 for s_ in r.get("steps", []) if s_.get("holding") is False)
        if not (r["ok"] and cell.backend.part == "work" and cell.zone["so101"] == "clear"):
            check(f"taught load {i}", False, str(r))
            continue
        for _ in range(3):
            u = await cell.unload_part()
            if u["ok"] or "nothing grasped" not in u.get("reason", ""):
                break
            unload_fails += 1
            unload_fail_ok &= cell.zone["so101"] == "clear" and cell.last_pose["so101"] == "home"
        if u["ok"] and cell.backend.part == "removed" and cell.last_pose["so101"] == "home":
            cycles_ok += 1
        else:
            check(f"taught unload {i}", False, str(u))
    check("4 taught load/unload cycles in the simulator", cycles_ok == 4, f"{retries} grasp retries on load")
    check("a failed unload grasp went home and stayed clear", unload_fail_ok, f"{unload_fails} failed unload grasps")
    real_names = cell.poses.names
    cell.poses.names = lambda arm: [n for n in real_names(arm) if n not in ("drop", "drop_hover")]
    try:
        await cell.unload_part()
        check("missing taught poses refused", False)
    except GuardError as e:
        check("missing taught poses refused", "drop_hover" in str(e) and "drop" in str(e), str(e)[:90])
    finally:
        del cell.poses.names
    cell.zone["elrobot"] = "work"
    try:
        await cell.load_part()
        check("loader refused while the worker is not clear", False)
    except GuardError as e:
        check("loader refused while the worker is not clear", "not at a clear pose" in str(e))
    cell.zone["elrobot"] = "clear"

    lc.update(mode="policy", policy_load_cmd="python3 -c \"print('ok')\"", policy_unload_cmd="", policy_cwd="",
              policy_timeout_s=5)
    r = await cell.load_part()
    check("policy load: clean exit, loader homed, zone clear", r["ok"] and r["exit_code"] == 0
          and r["output_tail"] == ["ok"] and cell.zone["so101"] == "clear" and r["go_home"]["ok"], str(r)[:120])
    lc["policy_load_cmd"] = "exit 3"
    r = await cell.load_part()
    check("policy load fails: zone stays unknown, no auto home", not r["ok"] and r["exit_code"] == 3
          and cell.zone["so101"] == "unknown" and "go_home" not in r and "required" in r["note"])
    rp = await cell.run_program([{"hole": 0}], retreat_after=False)
    check("worker refused after a failed policy run", not rp["ok"] and "unknown position" in rp["results"][0]["reason"],
          rp["results"][0]["reason"][:80])
    await cell.go_home("so101")
    try:
        await cell.unload_part()
        check("empty policy command refused", False)
    except GuardError as e:
        check("empty policy command refused", "empty" in str(e))
    lc.update(policy_load_cmd="sleep 10", policy_timeout_s=0.5)
    r = await cell.load_part()
    check("policy killed on timeout", not r["ok"] and "timeout" in r.get("killed", "") and r["seconds"] < 3
          and cell.zone["so101"] == "unknown", str(r.get("killed")))
    await cell.go_home("so101")
    lc.update(mode="human")
    r = await cell.load_part()
    check("human mode: no motion, a note", r["ok"] and "human" in r["note"])
    lc.update(mode="taught")

    print("8. alerts feed")
    a0 = await cell.alerts()
    check("first alerts call does not replay old events", not any(a["source"] in ("cellbot", "guided")
                                                                 for a in a0["alerts"]), str(a0["alerts"])[:150])
    check("metrics on the first call", a0["metrics"]["marks_logged"] == 3 and a0["metrics"]["touches"] >= 5,
          str(a0["metrics"]))
    cell.zone["so101"] = "unknown"
    await cell.run_program([{"hole": 0}], retreat_after=False)
    await cell.go_home("so101")
    a1 = await cell.alerts()
    check("refused step shows up as an alert", any(a["source"] in ("cellbot", "guided") and "refused" in a["message"]
                                                    for a in a1["alerts"]), str(a1["alerts"])[:150])
    await cell.stop("test alert")
    a2 = await cell.alerts()
    check("after an e-stop: pause and ask a human", a2["recommendation"] == "pause and ask a human"
          and any(a["level"] == "critical" and a["needs_human"] for a in a2["alerts"]), a2["recommendation"])
    (cell.cfg.state_dir / "estop.json").write_text('{"latched": false}')

    print("9. vision while the arm is over the part")
    feed.show(1)
    for _ in range(4):
        await cell.vision.state()
    await cell.alerts()                                    # cursor to now
    r = await cell.work_on({"hole": 0}, "hover")
    feed.show(1, gone=True)                                # the "arm" hides the part: tracker raises events
    for _ in range(6):
        await cell.vision.state()
    feed.show(1)
    for _ in range(3):
        await cell.vision.state()
    a3 = await cell.alerts()
    ign = [a for a in a3["alerts"] if "not reliable while the arm is over the part" in a["message"]]
    check("vision events raised with the arm over the part are ignored, counted in one info line",
          r["ok"] and len(ign) == 1 and ign[0]["level"] == "info"
          and not any(a["source"] == "vision" and "NO_BOX" in a["message"] for a in a3["alerts"]),
          str([a["message"] for a in a3["alerts"] if a["source"] == "vision"]))
    await cell.retreat()
    await asyncio.sleep(1.1)                               # past the grace time after the arm left
    feed.show(1, gone=True)
    for _ in range(6):
        await cell.vision.state()
    feed.show(1)
    for _ in range(3):
        await cell.vision.state()
    a4 = await cell.alerts()
    check("the same events with the arm clear are reported", any(a["source"] == "vision" and "not reliable"
                                                                 not in a["message"] for a in a4["alerts"]),
          str([a["message"] for a in a4["alerts"] if a["source"] == "vision"]))
    await cell.new_part("t4")

    print("10. gravity droop (simulated: joints 2 and 4 stop short by A deg * cos(joint 2))")
    be, J = cell.backend, cell.joints
    real_wg = be.write_goals
    droop = {"A": 0.0, "s": 1.0}

    def dq(q):
        d = np.zeros(7)
        d[1] = d[3] = droop["s"] * np.deg2rad(droop["A"]) * np.cos(q[1])
        return d

    async def drooped(arm, goals, speed, accel):
        if arm == cell.arm and droop["A"] and all(j in goals for j in J):
            q = k.norm_to_rad([goals[j] for j in J]) - dq(k.norm_to_rad([goals[j] for j in J]))
            goals = {**goals, **dict(zip(J, (float(v) for v in k.rad_to_norm(q))))}
        await real_wg(arm, goals, speed, accel)
    be.write_goals = drooped
    tol0, rounds0 = cell.g["reach_tolerance"], cell.m["settle_rounds"]
    cell.g["reach_tolerance"] = 0.1                        # the real cell copy also widened it for the droop
    cell.m["touch_mm"], cell.m["pretouch_mm"] = 10.0, 18.0        # the live geometry: pre-touch 8 mm above the touch
    st = await cell.vision.state()
    qp = cell._plan(cell._aim(st, {"hole": 0})["P"], await cell._q_now())["q_touch"]
    droop["A"] = 3.0
    lows = {s_: cell._above_part(qp - (lambda d: d * s_ / droop["s"])(dq(qp))) for s_ in (1.0, -1.0)}
    droop["s"] = min(lows, key=lows.get)                   # the sign that lowers the tip = gravity
    try:
        cell._offset[:] = 0
        cell.m["settle_rounds"], cell.m["touch_sag_abort_mm"] = 0, 1000.0
        ra = await cell.work_on({"hole": 0})
        await cell.retreat()
        check("(a) model sanity: with no offset carried, the arm is low by several mm at the pre-touch height; the "
              "closed loop there brings it within 2 mm and the final descent carries it to the touch",
              ra["ok"] and ra["pretouch_error_before_mm"] < -3.0 and abs(ra["pretouch_error_after_mm"]) <= 2.0
              and ra["pretouch_correction_rounds"] >= 1 and abs(ra["touch_error_mm"]) <= 2.0
              and ra["touch_correction_rounds"] == 0,
              f'pre-touch before {ra.get("pretouch_error_before_mm")} after {ra.get("pretouch_error_after_mm")} mm, '
              f'{ra.get("pretouch_correction_rounds")} round(s); touch {ra.get("touch_error_mm")} mm')
        cell._offset[:] = 0
        cell.m["settle_rounds"], cell.m["touch_sag_abort_mm"] = rounds0, 6.0
        rb = await cell.work_on({"hole": 0})
        want = cell.m["hover_mm"] + cell.m["touch_mm"]
        check("(b) offset learned at hover carries into the touch: within 2 mm of the commanded height",
              rb["ok"] and abs(rb["touch_error_mm"]) < 2.0 and abs(rb["tip_above_part_mm"] - want) < 2.0,
              f'touch_error_mm {rb.get("touch_error_mm")}, after the lift {rb.get("tip_above_part_mm")} mm '
              f'(commanded {want}), offset {[round(float(v), 2) for v in np.degrees(cell._offset)]} deg')
        rb2 = await cell.work_on({"hole": 2})
        # the settle stops refining once every joint is within 0.25 deg (servo resolution ~0.09 deg), and the
        # droop changes over the 40 mm descent: a carried offset leaves up to ~3 mm here (no pressing in)
        check("(b) next hole with the carried offset: within 3.5 mm", rb2["ok"] and abs(rb2["touch_error_mm"]) < 3.5,
              f'touch_error_mm {rb2.get("touch_error_mm")}')
        await cell.retreat()
        cell._offset[:] = 0
        cell.m["settle_rounds"] = 0
        droop["A"] = 6.0
        rc = await cell.work_on({"hole": 0})
        q = await cell._q_now()
        check("(c) large droop with the offset forced to zero: sag abort at the pre-touch height, before any "
              "contact, lifted to hover, step failed",
              not rc["ok"] and "sagged" in rc["reason"] and "before touching" in rc["reason"]
              and rc["pretouch_error_after_mm"] < -6.0 and rc["pretouch_correction_rounds"] == 0
              and "touch_error_mm" not in rc
              and cell._above_part(q) * 1000 > rc["pretouch_mm"] + rc["pretouch_error_after_mm"] + 20.0, rc.get("reason", ""))
        print(f"     droop numbers: no carry {ra.get('touch_error_mm')} mm, with carry {rb.get('touch_error_mm')} / "
              f"{rb2.get('touch_error_mm')} mm, abort case (pre-touch) {rc.get('pretouch_error_after_mm')} mm")
        await cell.retreat()
        dcl = cell.droop_clear()
        check("droop_clear zeroes the offset", dcl["ok"] and not cell._offset.any())
    finally:
        be.write_goals = real_wg
        cell.g["reach_tolerance"], cell.m["settle_rounds"] = tol0, rounds0
        cell.m.update(touch_mm=sim_settings.MOTION["touch_mm"], pretouch_mm=sim_settings.MOTION["pretouch_mm"],
                      touch_sag_abort_mm=6.0)
        droop["A"] = 0.0

    print("11. joint-2 tracking error near the part (stiction / pose-dependent droop the hover offset misses): "
          "corrected at the pre-touch height, final straight descent, straight lift")
    await cell.new_part("t5")
    st = await cell.vision.state()
    feed.holes_px = {h["id"]: h["px"] for h in st["holes"]}
    inj = {"bias_deg": 0.0, "frozen": False, "lowest_cmd": 1e9, "touch_bias_deg": 0.0}
    cell.m["touch_mm"], cell.m["pretouch_mm"] = 10.0, 18.0  # the live geometry: pre-touch 8 mm above the touch
    region = (cell.m["touch_mm"] + 15.0) / 1000            # goals below this height above the part: near the part
    near_touch = (cell.m["touch_mm"] + 3.0) / 1000         # touch_bias_deg applies only below this (not at pre-touch)
    real_tc = cell._touch_correct
    # rec: every goal the simulated arm actually receives (after the injected error) -> (height, tip), and, during
    # the pre-touch correction rounds, the heights of the received and of the commanded goals
    rec = {"phase": None, "path": [], "round_h": [], "round_cmd_h": [], "arrive_h": None, "pre_h": None}

    async def tracked(arm, goals, speed, accel):
        if arm == cell.arm and all(j in goals for j in J):
            qg = k.norm_to_rad([goals[j] for j in J])
            hg = cell._above_part(qg)
            if hg < region:
                inj["lowest_cmd"] = min(inj["lowest_cmd"], hg)
            if inj["frozen"]:                              # the arm does not respond: position joints stay put
                qg[:4] = k.norm_to_rad([be.joints[arm][j] for j in J])[:4]
            elif hg < region:
                qg[1] += np.deg2rad(inj["bias_deg"] + (inj["touch_bias_deg"] if hg < near_touch else 0.0))
            goals = {**goals, **dict(zip(J, (float(v) for v in k.rad_to_norm(qg))))}
            rec["path"].append((cell._above_part(qg), qg.copy()))
            if rec["phase"] == "pre":
                rec["round_h"].append(cell._above_part(qg))
                rec["round_cmd_h"].append(hg)
        await real_wg(arm, goals, speed, accel)

    async def rec_tc(q_goal, cmd0, q_meas, **kw):
        rec.update(phase="pre", arrive_h=cell._above_part(q_meas), pre_h=cell._above_part(q_goal))
        try:
            return await real_tc(q_goal, cmd0, q_meas, **kw)
        finally:
            rec["phase"] = None

    async def frozen_tc(*a, **kw):
        inj["frozen"] = True
        try:
            return await rec_tc(*a, **kw)
        finally:
            inj["frozen"] = False

    def reset_rec():
        rec.update(phase=None, path=[], round_h=[], round_cmd_h=[], arrive_h=None, pre_h=None)

    nn = np.asarray(k.n, float) / np.linalg.norm(k.n)

    def sideways_near_touch(h_max_m):
        """Max sideways (perpendicular to the table normal) spread of the tip while it is below h_max_m above the
        part (= from arriving at the touch height until the lift has left it), mm. The simulated joints jump from
        goal to goal; a real arm moves between them, so each step is sampled along the joint-space line."""
        pts = []
        for (_, qa), (_, qb) in zip(rec["path"], rec["path"][1:]):
            for t in np.linspace(0.0, 1.0, 11):
                qx = qa + t * (qb - qa)
                if cell._above_part(qx) < h_max_m:
                    pts.append(k.tip(qx))
        if not pts:
            return None, 0
        d = [(p - pts[0]) - float(np.dot(p - pts[0], nn)) * nn for p in pts]
        return max(float(np.linalg.norm(v)) for v in d) * 1000, len(pts)

    be.write_goals = tracked
    cell._touch_correct = rec_tc
    tol0 = cell.g["reach_tolerance"]
    cell.g["reach_tolerance"] = 0.04                       # the real cell copy's value
    want_hover = cell.m["hover_mm"] + cell.m["touch_mm"]
    tol_m = cell.m["touch_tol_mm"] / 1000
    below_m = cell.m["touch_cmd_below_mm"] / 1000
    touch_m = cell.m["touch_mm"] / 1000
    try:
        # which sign of joint 2 raises the tip at this touch pose ("short" = stops above the target)
        qp = cell._plan(cell._aim(st, {"hole": 0})["P"], await cell._q_now())["q_touch"]
        up = 1.0 if cell._above_part(qp + np.eye(7)[1] * 0.01) > cell._above_part(qp) else -1.0
        lim_mm = cell.m["touch_mm"] - cell.m["touch_cmd_below_mm"] - 0.05
        rad = np.deg2rad
        cases = (  # name, joint-2 tracking error near the part (deg, + = short = tip high), carried j2 offset, check
            ("short 0.6 deg (stiction at the touch)", 0.6, 0.0, lambda b: b > 2.0),
            ("morning's case: carried offset wound up (2.65 deg too far up, settle off), tracks it faithfully",
             0.0, 2.65, lambda b: b > 10.0),
            ("over 1.0 deg (tip low, within the sag abort)", -1.0, 0.0, lambda b: -6.0 < b < -2.0))
        for name, bias, off, sign_ok in cases:
            cell._offset[:] = 0
            cell._offset[1] = rad(off * up)
            off0 = cell._offset.copy()
            cell.m["settle_rounds"] = 0 if off else rounds0
            inj.update(bias_deg=bias * up, lowest_cmd=1e9)
            reset_rec()
            r = await cell.work_on({"hole": 0})
            lift_ok = off or abs(r.get("tip_above_part_mm", -99) - want_hover) < 2.0    # (a wrong offset lifts wrong)
            check(f"(a) {name}: corrected at the pre-touch height to within 2 mm, final descent within 2 mm, "
                  "carried offset untouched by the correction",
                  r["ok"] and sign_ok(r["pretouch_error_before_mm"]) and abs(r["pretouch_error_after_mm"]) <= 2.0
                  and 1 <= r["pretouch_correction_rounds"] <= 4 and abs(r["touch_error_mm"]) <= 2.0
                  and r["touch_error_mm"] == r["touch_error_after_mm"] and r["touch_correction_rounds"] == 0
                  and np.allclose(cell._offset, off0) and lift_ok,
                  f'pre-touch before {r.get("pretouch_error_before_mm")} after {r.get("pretouch_error_after_mm")} mm, '
                  f'{r.get("pretouch_correction_rounds")} round(s), correction {r.get("pretouch_correction_deg")} deg; '
                  f'touch {r.get("touch_error_mm")} mm; after the lift {r.get("tip_above_part_mm")} mm')
            lo_rx = min(rec["round_h"], default=1e9)
            floor_rx = min(rec["pre_h"] - tol_m, rec["arrive_h"]) - 0.00005
            check("    corrections ran at the pre-touch height: no goal received during the rounds below "
                  "min(pre-touch - tol, where it arrived), none near the lid, none commanded below pre-touch - "
                  "touch_cmd_below_mm",
                  rec["round_h"] and lo_rx >= floor_rx and lo_rx > touch_m + 0.003
                  and min(rec["round_cmd_h"]) >= rec["pre_h"] - below_m - 0.00005,
                  f'lowest received {lo_rx * 1000:.1f} mm (pre-touch {rec["pre_h"] * 1000:.1f}, arrived '
                  f'{rec["arrive_h"] * 1000:.1f}), lowest commanded {min(rec["round_cmd_h"], default=0) * 1000:.1f} mm, '
                  f'{len(rec["round_h"])} goal(s)')
            side, npts = sideways_near_touch(touch_m + 0.005)
            check("    sideways tip motion within 5 mm of the touch height (final descent, dwell, lift) < 1 mm",
                  side is not None and side < 1.0 and npts >= 2, f'{side if side is None else round(side, 3)} mm '
                  f'over {npts} goal(s)')
            check("    the commanded tip never went below touch - touch_cmd_below_mm", inj["lowest_cmd"] * 1000 >= lim_mm,
                  f'lowest commanded {inj["lowest_cmd"] * 1000:.1f} mm')
        cell.m["settle_rounds"] = rounds0
        # joint 2 stuck 2.65 deg short: reaching the pre-touch height would need a goal ~12 mm below it -> refused
        cell._offset[:] = 0
        inj.update(bias_deg=2.65 * up, lowest_cmd=1e9)
        reset_rec()
        r = await cell.work_on({"hole": 0})
        q = await cell._q_now()
        check("(d) joint 2 stuck 2.65 deg short: not pushed below the pre-touch command floor; fails without "
              "touching, lifted to hover",
              not r["ok"] and "not reached" in r["reason"] and r["pretouch_error_after_mm"] > 2.0
              and "touch_error_mm" not in r and min(h for h, _ in rec["path"]) > touch_m + 0.003
              and min(rec["round_cmd_h"], default=1e9) >= rec["pre_h"] - below_m - 0.00005
              and inj["lowest_cmd"] * 1000 >= lim_mm and r["lift"]["ok"]
              and abs(cell._above_part(q) * 1000 - want_hover) < 2.0,
              f'{r.get("reason")}; pre-touch before {r.get("pretouch_error_before_mm")} after '
              f'{r.get("pretouch_error_after_mm")} mm, {r.get("pretouch_correction_rounds")} round(s), lowest '
              f'commanded {inj["lowest_cmd"] * 1000:.1f} mm')
        await cell.retreat()
        # fine at the pre-touch height, but 2 deg short only in the last 3 mm (e.g. the lid is higher than modelled):
        # the touch is measured, not corrected; too high -> straight lift, step fails
        cell._offset[:] = 0
        inj.update(bias_deg=0.0, touch_bias_deg=2.0 * up, lowest_cmd=1e9)
        reset_rec()
        r = await cell.work_on({"hole": 0})
        q = await cell._q_now()
        check("(d) too high only at the touch: no correction in contact, step fails, lifted to hover",
              not r["ok"] and "not reached" in r["reason"] and "after the final descent" in r["reason"]
              and r["touch_error_mm"] > 2.0 and r["touch_correction_rounds"] == 0
              and abs(r["pretouch_error_after_mm"]) <= 2.0 and r["lift"]["ok"]
              and abs(cell._above_part(q) * 1000 - want_hover) < 2.0,
              f'{r.get("reason")}; touch {r.get("touch_error_mm")} mm, now {cell._above_part(q) * 1000:.1f} mm')
        inj["touch_bias_deg"] = 0.0
        await cell.retreat()
        # the arm stops 1 deg over (tip ~4 mm low) and then does not respond to the correction at all
        cell._offset[:] = 0
        inj.update(bias_deg=-1.0 * up, lowest_cmd=1e9)
        cell._touch_correct = frozen_tc
        reset_rec()
        n_alerts = len((await cell.alerts())["alerts"])
        res = await cell.run_program([{"hole": 0}, {"hole": 2}], retreat_after=False)
        r = res["results"][0]
        q = await cell._q_now()
        check("(d) arm does not respond to the correction: step fails with a reason before touching, program stops, "
              "arm lifted to hover",
              not res["ok"] and res["failed_step"] == 0 and not r["ok"] and "not reached" in r["reason"]
              and r["pretouch_correction_rounds"] == cell.m["touch_correct_rounds"]
              and r["pretouch_error_after_mm"] < -2.0 and "touch_error_mm" not in r and r["lift"]["ok"]
              and min(h for h, _ in rec["path"]) > touch_m + 0.003
              and abs(cell._above_part(q) * 1000 - want_hover) < 2.0 and cell._seed is None,
              f'{r.get("reason")}; pre-touch before {r.get("pretouch_error_before_mm")} after '
              f'{r.get("pretouch_error_after_mm")} mm, {r.get("pretouch_correction_rounds")} round(s), now '
              f'{cell._above_part(q) * 1000:.1f} mm')
        check("    commanded tip never below touch - touch_cmd_below_mm",
              inj["lowest_cmd"] * 1000 >= cell.m["touch_mm"] - cell.m["touch_cmd_below_mm"] - 0.05,
              f'lowest commanded {inj["lowest_cmd"] * 1000:.1f} mm')
        al = (await cell.alerts())["alerts"]
        check("    raised as a guided alert", any(a["source"] == "guided" and "not reached" in a["message"]
                                                 for a in al), str([a["message"] for a in al[-3:]]))
        cell._touch_correct = rec_tc
        await cell.retreat()

        # seeding: in one program the pre-touch correction of a hole is the starting correction of the next.
        # (A carried offset no longer survives the program's first look, so the error to seed is one the hover
        # cannot see: joint 2 stopping 0.8 deg short only near the part.)
        cell._offset[:] = 0
        inj.update(bias_deg=0.8 * up, lowest_cmd=1e9)
        reset_rec()
        res = await cell.run_program([{"hole": 0}, {"hole": 1}, {"hole": 2}], retreat_after=False)
        rs = res["results"]
        cap = cell.m["touch_correct_max_deg"] + 0.01
        check("(e) seeded holes: hole 0 corrects, the later holes start from its correction and need 0-1 rounds; "
              "every hole touches within 2 mm; seed within the correction cap; seed cleared after the run",
              res["ok"] and rs[0]["pretouch_correction_rounds"] >= 1 and not any(rs[0]["pretouch_seeded_deg"])
              and all(x["pretouch_correction_rounds"] <= 1 and any(x["pretouch_seeded_deg"]) for x in rs[1:])
              and all(abs(x["touch_error_mm"]) <= 2.0 for x in rs)
              and all(max(abs(v) for v in x["pretouch_correction_deg"]) <= cap for x in rs)
              and inj["lowest_cmd"] * 1000 >= lim_mm and cell._seed is None,
              "; ".join(f'{x.get("target")}: before {x.get("pretouch_error_before_mm")} mm, '
                        f'{x.get("pretouch_correction_rounds")} round(s), touch {x.get("touch_error_mm")} mm'
                        for x in rs))
        inj["bias_deg"] = 0.0
        cell._touch_correct = real_tc
        await cell.retreat()

        # settle anti-windup: a servo that is still creeping when read (half way at the first read; the rest only
        # once the joints have stopped) must not make the settle wind the carried offset up (it did, to the cap)
        lag, pending = {"on": False}, {}

        async def lagging(arm, goals, speed, accel):
            if lag["on"] and arm == cell.arm and all(j in goals for j in J):
                qg = k.norm_to_rad([goals[j] for j in J])
                qc = k.norm_to_rad([be.joints[arm][j] for j in J])
                pending["g"] = dict(goals)
                half = qc + 0.5 * (qg - dq(qg) - qc)       # half way to where the drooping servo will end up
                goals = {**goals, **dict(zip(J, (float(v) for v in k.rad_to_norm(half))))}
                be.goals[arm].update(goals)
                for j in J:
                    be.joints[arm][j] = goals[j]
                return
            await drooped(arm, goals, speed, accel)

        real_arrive, real_settle = cell._arrive, cell._settle

        async def arrive_completes(timeout: float = 4.0):  # the joints really stop: the lag finishes
            if "g" in pending:
                await drooped(cell.arm, pending.pop("g"), 0, 0)

        async def lag_settle(q_goal):
            lag["on"] = True
            try:
                await real_settle(q_goal)
            finally:
                lag["on"] = False
                await arrive_completes()
        be.write_goals, cell._arrive, cell._settle = lagging, arrive_completes, lag_settle
        cell.g["reach_tolerance"] = 0.1
        cell._offset[:] = 0
        droop["A"] = 3.0
        q_h = cell._plan(cell._aim(st, {"hole": 0})["P"], await cell._q_now())["q_hover"]
        await cell._guided_move(q_h)
        off_deg, true_deg = np.degrees(cell._offset), np.degrees(dq(q_h))
        q = await cell._q_now()
        check("(c) settle with a lagging servo learns the droop, not a wound-up multiple of it; hover lands",
              abs(abs(off_deg[1]) - abs(true_deg[1])) < 0.5 and np.abs(off_deg).max() < float(cell.m["settle_max_deg"])
              - 0.5 and abs(cell._above_part(q) - cell._above_part(q_h)) * 1000 <= cell.m["settle_tol_mm"],
              f'offset {[round(float(v), 2) for v in off_deg]} deg, true droop j2 {abs(true_deg[1]):.2f} deg, '
              f'hover j2 error {np.degrees(q - q_h)[1]:+.2f} deg')
        cell._arrive, cell._settle = real_arrive, real_settle
        be.write_goals = real_wg
        droop["A"] = 0.0
        await cell.retreat()
        # joint 6 commanded at its limit but stopping 0.88 deg inside it (the real arm at every touch/hover pose)
        writes = {"n": 0}
        pin = k.hi[5] - np.deg2rad(0.88)

        async def pinned(arm, goals, speed, accel):
            if arm == cell.arm and all(j in goals for j in J):
                writes["n"] += 1
                qg = k.norm_to_rad([goals[j] for j in J])
                qg[5] = min(qg[5], pin)
                goals = {**goals, **dict(zip(J, (float(v) for v in k.rad_to_norm(qg))))}
            await real_wg(arm, goals, speed, accel)
        q_pin = cell._plan(cell._aim(st, {"hole": 0})["P"], await cell._q_now())["q_hover"].copy()
        q_pin[5] = k.hi[5]
        be.write_goals = pinned
        cell._offset[:] = 0
        await cell._guided_move(q_pin, settle=False)
        writes["n"] = 0
        await cell._settle(q_pin)
        check("(d) a joint pinned at its limit neither keeps the settle loop running nor gets an offset",
              writes["n"] == 0 and cell._offset[5] == 0.0,
              f'{writes["n"]} settle write(s), offset {[round(float(v), 2) for v in np.degrees(cell._offset)]} deg')
        be.write_goals = real_wg
        await cell.retreat()
    finally:
        be.write_goals = real_wg
        cell._touch_correct = real_tc
        cell.__dict__.pop("_arrive", None)
        cell.__dict__.pop("_settle", None)
        cell.g["reach_tolerance"] = tol0
        cell.m.update(touch_mm=sim_settings.MOTION["touch_mm"], pretouch_mm=sim_settings.MOTION["pretouch_mm"])
        droop["A"] = 0.0
        cell._offset[:] = 0

    print("12. hover closed on the TIP (real-arm servo model: pose-dependent droop on joints 2-4, joint 3 creeping "
          "after each goal, joint 6 stopping 0.88 deg short of its limit; a wound-up carried offset)")
    await cell.retreat()
    await cell.new_part("t6")
    st = await cell.vision.state()
    rs0, J = be.read_state, cell.joints
    pin6 = k.hi[5] - np.deg2rad(0.88)
    A0 = np.deg2rad(np.array([0.0, 1.45, 1.0, 1.95, 0.0, 0.0, 0.0]))   # ~1.1 / 0.75 / 1.5 deg at the touch, as logged
    srv = {"s": 0.0, "scale": 1.0, "creep": None, "extra": np.zeros(7)}
    CJ = 2                                                  # joint 3 creeps: 40 % at the write, half the rest per read

    def track(qg):
        """Where the servos settle for the goal qg (rad): pose-dependent tracking error, joint 6 pinned short."""
        qe = np.array(qg, float) - srv["s"] * srv["scale"] * A0 * np.cos(qg[1]) + srv["extra"]
        qe[5] = min(qe[5], pin6)
        return qe

    def q_meas():
        return k.norm_to_rad([be.joints[cell.arm][j] for j in J])

    def set_meas(q):
        for j, v in zip(J, k.rad_to_norm(q)):
            be.joints[cell.arm][j] = float(v)

    async def servo_wg(arm, goals, speed, accel):
        if arm == cell.arm and all(j in goals for j in J):
            qg = k.norm_to_rad([goals[j] for j in J])
            qe, qc = track(qg), q_meas()
            srv["creep"] = qe[CJ]
            qe[CJ] = qc[CJ] + 0.4 * (qe[CJ] - qc[CJ])
            goals = {**goals, **dict(zip(J, (float(v) for v in k.rad_to_norm(qe))))}
        await real_wg(arm, goals, speed, accel)

    async def servo_rs():
        if srv["creep"] is not None:
            q = q_meas()
            q[CJ] += 0.5 * (srv["creep"] - q[CJ])
            set_meas(q)
        return await rs0()

    qp = cell._plan(cell._aim(st, {"hole": 0})["P"], await cell._q_now())["q_pre"]
    srv.update(s=1.0, scale=1.0)
    low = 1.0 if cell._above_part(track(qp)) < cell._above_part(qp) else -1.0      # the sign that lowers the tip
    tol0 = cell.g["reach_tolerance"]
    cell.g["reach_tolerance"] = 0.04                       # the real cell copy's value
    cap_deg = float(cell.m["settle_max_deg"])

    def hole_ok(x, first):
        return (x["ok"] and x.get("hover_status") == "ok" and abs(x["hover_error_after_mm"]) <= cell.m["settle_tol_mm"]
                and abs(x["pretouch_error_before_mm"]) <= 3.0 and x["pretouch_correction_rounds"] <= 1
                and abs(x["touch_error_mm"]) <= cell.m["touch_tol_mm"]
                and x["hover_rounds"] <= (3 if first else 2)
                and max(abs(v) for v in x["hover_offset_after_deg"]) < cap_deg - 0.5
                and x["hover_offset_after_deg"][5] == 0.0)

    def show(x):
        return (f'{x.get("target")}: hover {x.get("hover_error_before_mm")} -> {x.get("hover_error_after_mm")} mm in '
                f'{x.get("hover_rounds")} round(s) ({x.get("hover_settle_s")} s), pre-touch arrival '
                f'{x.get("pretouch_error_before_mm")} mm, {x.get("pretouch_correction_rounds")} round(s), touch '
                f'{x.get("touch_error_mm")} mm, offset {x.get("hover_offset_after_deg")}' + ("" if x.get("ok") else
                                                                                            f' FAILED: {x.get("reason")}'))
    be.write_goals, be.read_state = servo_wg, servo_rs
    try:
        for name, sgn, scale in (("too low (gravity droop, ~12 mm at the touch with no offset)", low, 1.0),
                                 ("too high (servos stop above, within touch_cmd_below_mm)", -low, 0.2)):
            srv.update(s=sgn, scale=scale, creep=None)
            res = await cell.run_program([{"hole": 0}, {"hole": 1}, {"hole": 2}], retreat_after=False)
            rs = res["results"]
            true_off = np.degrees(track(qp) - qp)[:4]
            check(f"(a) {name}: every hole's hover lands within settle_tol_mm, pre-touch arrival within 3 mm, "
                  "0-1 pre-touch rounds, touch within tolerance; joint 6 gets no offset",
                  res["ok"] and all(hole_ok(x, i == 0) for i, x in enumerate(rs)), "; ".join(show(x) for x in rs))
            off = np.degrees(cell._offset)[:4]
            check("    carried offset = the servos' real tracking error (creep waited out, nothing wound up)",
                  np.abs(off + true_off).max() < 0.35,
                  f'offset j1-4 {np.round(off, 2).tolist()} deg, tracking error at pre-touch {np.round(-true_off, 2).tolist()}')
        # the logged case: an offset wound up to [-0.46, -3.21, -3.25, -2.63] deg (it lifted the tip ~30 mm) carried
        # into the next hole of the same program (no look in between)
        srv.update(s=low, scale=1.0, creep=None)
        await cell.run_program([{"hole": 0}], retreat_after=False)
        wound = np.deg2rad([-0.46, -3.21, -3.25, -2.63, 0.0, 0.0, 0.0])
        q_h = cell._plan(cell._aim(cell._sight["state"], {"hole": 2})["P"], await cell._q_now())["q_hover"]
        lift_mm = (cell._above_part(q_h + wound) - cell._above_part(q_h)) * 1000
        rounds0 = cell.m["settle_rounds"]
        cell._offset = wound.copy()
        cell.m["settle_rounds"] = 0                       # model sanity: without the hover loop
        r0 = await cell.work_on({"hole": 2}, _program=True)
        cell.m["settle_rounds"] = rounds0
        await cell.work_on({"hole": 0}, _program=True)
        cell._offset = wound.copy()
        r1 = await cell.work_on({"hole": 2}, _program=True)
        check("(b) model sanity: the wound-up offset alone arrives at pre-touch far too high (the logged +10..+16 mm)",
              lift_mm > 20 and r0.get("pretouch_error_before_mm", 0) > 8.0,
              f'offset lifts the commanded tip {lift_mm:.0f} mm; without the hover loop: {show(r0)}')
        check("(b) with the hover tip loop: the wound-up offset is measured at hover (too high) and replaced; "
              "pre-touch arrival within 3 mm, 0-1 rounds",
              r1.get("hover_error_before_mm", 0) > 8.0 and hole_ok(r1, True), show(r1))
        print("     " + show(r1))

        print("13. pre-touch loop on the full 3D tip: droop + creep on joints 2-4, joint 6 pinned, and a joint-7 / "
              "joint-1 tracking error the hover offset does not carry (pure sideways error of 3-5 mm)")
        P0 = cell._aim(cell._sight["state"], {"hole": 0})["P"]
        qp0 = cell._plan(P0, await cell._q_now())["q_pre"]

        def sideways(j, deg):
            d = np.zeros(7)
            d[j] = np.deg2rad(deg)
            v = k.tip(qp0 + d) - k.tip(qp0)
            return float(np.linalg.norm(k.in_plane(v))) * 1000, abs(float(v @ k.n)) * 1000
        cmds = []

        async def spy_wg(arm, goals, speed, accel):
            if arm == cell.arm and all(j in goals for j in J):
                cmds.append(k.norm_to_rad([goals[j] for j in J]))
            await servo_wg(arm, goals, speed, accel)
        be.write_goals = spy_wg
        tmm, below = float(cell.m["touch_mm"]), float(cell.m["touch_cmd_below_mm"])

        def xy_ok(x):
            return (x["ok"] and x.get("pretouch_xy_error_after_mm", 99) <= 1.0 and x.get("touch_xy_error_mm", 99) <= 1.2
                    and abs(x["pretouch_error_after_mm"]) <= cell.m["touch_tol_mm"]
                    and abs(x["touch_error_mm"]) <= cell.m["touch_tol_mm"]
                    and x["pretouch_correction_rounds"] <= cell.m["touch_correct_rounds"]
                    and len(x.get("touch_xy_error_vec_mm", [])) == 2)

        def xshow(x):
            return (f'{x.get("target")}: xy {x.get("pretouch_xy_error_before_mm")} -> {x.get("pretouch_xy_error_after_mm")}'
                    f' mm ({x.get("pretouch_xy_status")}) in {x.get("pretouch_correction_rounds")} round(s), touch xy '
                    f'{x.get("touch_xy_error_mm")} {x.get("touch_xy_error_vec_mm")} mm, height pre '
                    f'{x.get("pretouch_error_after_mm")} touch {x.get("touch_error_mm")} mm'
                    + ("" if x.get("ok") else f' FAILED: {x.get("reason")}'))
        import time as _t
        # a joint-1 error that only appears close to the part (stiction / pose-dependent; below hover_mm / 2): the
        # hover offset cannot have learned it, so it reaches pre-touch as a pure sideways error
        d1 = 4.0 / sideways(0, 1.0)[0]
        h_lim = 0.75 * float(cell.m["hover_mm"]) / 1000          # commanded goal: pre-touch / touch, not hover
        near = {"deg": 0.0, "j": 0}

        async def near_wg(arm, goals, speed, accel):
            if arm == cell.arm and all(j in goals for j in J):
                qg = k.norm_to_rad([goals[j] for j in J])
                e = np.zeros(7)
                e[near["j"]] = np.deg2rad(near["deg"]) if cell._above_part(qg) < h_lim else 0.0
                srv["extra"] = e
            await spy_wg(arm, goals, speed, accel)
        be.write_goals = near_wg
        srv.update(s=low, scale=1.0, creep=None, extra=np.zeros(7))
        t0 = _t.monotonic()
        base_res = await cell.run_program([{"hole": 0}, {"hole": 1}, {"hole": 2}], retreat_after=False)
        t_base = (_t.monotonic() - t0) / 3
        check("    reference run without the sideways error: ok, xy within ~1 mm", base_res["ok"]
              and all(xy_ok(x) for x in base_res["results"]), "; ".join(xshow(x) for x in base_res["results"]))
        for sg in (1.0, -1.0):
            near["deg"] = sg * d1
            cmds.clear()
            t0 = _t.monotonic()
            res = await cell.run_program([{"hole": 0}, {"hole": 1}, {"hole": 2}], retreat_after=False)
            t_xy = (_t.monotonic() - t0) / 3
            rs = res["results"]
            low_cmd = min(cell._above_part(c) for c in cmds) * 1000
            check(f"(a) joint 1 {near['deg']:+.2f} deg near the part (~4 mm sideways) + droop/creep, joint 6 pinned: "
                  "pre-touch and touch xy <= ~1 mm, heights within tolerance, rounds within budget",
                  res["ok"] and all(xy_ok(x) for x in rs) and rs[0]["pretouch_xy_error_before_mm"] >= 2.5,
                  "; ".join(xshow(x) for x in rs))
            check("    no commanded goal below the floor (touch - touch_cmd_below_mm)", low_cmd >= tmm - below - 0.05,
                  f"lowest commanded tip {low_cmd:.2f} mm above the part")
            check("    the dab is at most ~1 s slower on average than without the sideways error",
                  t_xy - t_base <= 1.0, f"{t_xy:.2f} s vs {t_base:.2f} s per dab")
        near["deg"] = 0.0
        be.write_goals = spy_wg
        # uncorrectable: the per-joint cap is too small for the sideways error -> proceeds, residual reported
        near["deg"] = d1
        srv.update(creep=None)
        cap0 = cell.m["touch_correct_max_deg"]
        cell.m["touch_correct_max_deg"] = 0.3
        try:
            res = await cell.run_program([{"hole": 0}, {"hole": 1}], retreat_after=False)
        finally:
            cell.m["touch_correct_max_deg"] = cap0
        rs = res["results"]
        check("(c) sideways error beyond the cap: the dab proceeds (height ok), residual xy reported, status cap/rounds",
              res["ok"] and all(x["ok"] and x["pretouch_xy_error_after_mm"] > 1.0
                                and x["pretouch_xy_status"] in ("cap", "rounds")
                                and abs(x["touch_error_mm"]) <= cell.m["touch_tol_mm"]
                                and x["touch_xy_error_mm"] > 1.0 for x in rs), "; ".join(xshow(x) for x in rs))
        lines = [json.loads(ln) for ln in open(cell.glog)][-2:]
        check("    guided_log.jsonl has the xy fields", all(
            all(kk in ln for kk in ("pretouch_xy_error_before_mm", "pretouch_xy_error_after_mm", "touch_xy_error_mm",
                                    "touch_xy_error_vec_mm")) for ln in lines), str([sorted(ln)[-6:] for ln in lines]))
        near["deg"] = 0.0
        srv.update(extra=np.zeros(7))
    finally:
        be.write_goals, be.read_state = real_wg, rs0
        cell.g["reach_tolerance"] = tol0
        cell._offset[:] = 0
    await cell.retreat()
    check("retreat forgets the carried offset", not cell._offset.any())

    print(f"\n{'ALL CHECKS PASSED' if not check.failed else str(check.failed) + ' CHECK(S) FAILED'}")
    return check.failed


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
