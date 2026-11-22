"""Offline check of the terminal CLI (cellguide.cli): sim backend + recorded frames, no hardware, no network.
    python tests/test_cli.py

  1. the step short-form parser;
  2. a whole worker flow in `shell`, fed line by line: status, new, see, plan all, hover 0, run 0 2, retreat,
     inspect (no ink -> note), alerts, finish pass; images saved under <state>/cli/;
  3. with a fake "station" backend name every mover asks "Move the real arm? [y/N]" and "n" cancels without
     motion; --yes is refused; stop never asks;
  4. jog and teach the lid aim offset, aim / aim clear;  5. set <key> <value>;  6. station banner;
  7. quit with pending background tasks exits quietly (subprocess, stderr captured).
"""
import asyncio
import copy
import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ["CELLBOT_MODE"] = "sim"
STATE = Path(tempfile.mkdtemp(prefix="cellguide_cli_test_"))
os.environ["CELLBOT_STATE"] = str(STATE)

from cellguide import config as gconfig  # noqa: E402
from cellguide import cli  # noqa: E402
import sim_settings  # noqa: E402  (tests/: fixed motion values, independent of guided.toml)

_make_cell = cli.make_cell


def _fixed_make_cell(*a, **kw):
    """Every cell the CLI builds in this test gets the fixed test settings (guided.toml is not changed)."""
    cell, frames = _make_cell(*a, **kw)
    sim_settings.apply(cell)
    return cell, frames


cli.make_cell = _fixed_make_cell

G = gconfig.load()
IMG = G["_paths"]["vision_file"].parent / "images"


def check(name, cond, detail=""):
    print(f"  [{'ok' if cond else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    if not cond:
        check.failed += 1


check.failed = 0


def last_json(lines):
    """The JSON object printed by a command (the lines from the first '{' on)."""
    txt = "\n".join(lines)
    i = txt.find("{")
    if i < 0:
        return None
    try:
        return json.JSONDecoder().raw_decode(txt[i:])[0]
    except ValueError:
        return None


def parser_checks():
    print("1. short-form parser")
    P = cli.parse_steps
    check("0 2 1 = dab holes", P("0 2 1") == [{"hole": 0}, {"hole": 2}, {"hole": 1}])
    check("h0 = hover hole 0", P("h0") == [{"hole": 0, "action": "hover"}])
    check("0.5,0.2 = point on the lid", P("0.5,0.2") == [{"at": [0.5, 0.2]}])
    check("h0.5,0.2 = hover a point", P("h0.5,0.2") == [{"at": [0.5, 0.2], "action": "hover"}])
    check("mixed", P("0 h1 0.5,0.5") == [{"hole": 0}, {"hole": 1, "action": "hover"}, {"at": [0.5, 0.5]}])
    check("all expands to the given ids", P("all", [0, 1, 2]) == [{"hole": 0}, {"hole": 1}, {"hole": 2}])
    js = '[{"hole":0},{"at":[0.5,0.2],"action":"hover"}]'
    check("JSON list passes through", P(js) == json.loads(js))
    check("JSON object -> one step", P('{"hole": 2}') == [{"hole": 2}])
    for bad, why in (("", "empty"), ("all", "all without a scene"), ("1,2,3", "3 coordinates"),
                     ("a,b", "not numbers")):
        try:
            P(bad)
            check(f"bad input refused: {why}", False)
        except ValueError:
            check(f"bad input refused: {why}", True)


def shell_flow():
    print("2. worker flow in the shell (sim, recorded frames)")
    script = ["status", "new t1", "see", "plan all", "hover 0", "run 0 2", "run --look 0 1", "droop", "droop clear", "retreat", "inspect", "alerts",
              "finish pass", "look", "nonsense", "quit"]
    feed, out = list(script), {}
    cur = ["(start)"]

    def inp(prompt=""):
        if not feed:
            raise EOFError
        cur[0] = feed.pop(0)
        out[cur[0]] = []
        return cur[0]

    def pr(s=""):
        out.setdefault(cur[0], []).append(str(s))

    rc = cli.main(["--frames", str(IMG), "shell"], input_fn=inp, out=pr)
    start = "\n".join(out.get("(start)", []))
    check("shell exits 0 on quit", rc == 0)
    check("backend line SIMULATION shown at start", "BACKEND: SIMULATION" in start, start.splitlines()[0])
    st = last_json(out["status"])
    check("status: sim backend, both arms", st and "SIMULATION" in st["backend"] and set(st["arms"]) >= {"so101", "elrobot"})
    nw = last_json(out["new t1"])
    check("new: part t1 with 3 holes", nw and nw["ok"] and nw["part"] == "t1" and nw["holes"] == [0, 1, 2], str(nw))
    sc = last_json(out["see"])
    imgs = [ln for ln in out["see"] if ln.startswith("image: ")]
    check("see: holes listed with robot_mm", sc and len(sc.get("holes", [])) == 3
          and all("robot_mm" in h for h in sc["holes"]))
    check("see: image saved under state/cli", imgs and Path(imgs[0][7:]).is_file()
          and Path(imgs[0][7:]).parent == STATE / "cli" and imgs[0].endswith("_see.jpg"), imgs[0] if imgs else "")
    pl = last_json(out["plan all"])
    check("plan all: 3 reachable steps", pl and pl["ok"] and [s["target"] for s in pl["steps"]] ==
          ["hole 0", "hole 1", "hole 2"], str([s.get("target") for s in (pl or {}).get("steps", [])]))
    hv = last_json(out["hover 0"])
    check("hover 0: one hover step, said what it does first", hv and hv["ok"] and hv["results"][0]["action"] == "hover"
          and "retreat" not in hv and out["hover 0"][0].startswith("WILL DO:"), out["hover 0"][0])
    rn = last_json(out["run 0 2"])
    check("run 0 2: two dabs then retreat", rn and rn["ok"] and [r["target"] for r in rn["results"]] ==
          ["hole 0", "hole 2"] and rn["retreat"]["ok"])
    check("gripper closed once, on the first guided step (hover 0), not again in run 0 2",
          hv and rn and hv["results"][0].get("gripper_closed_first") is True
          and not any(r.get("gripper_closed_first") for r in rn["results"]))
    rl = last_json(out["run --look 0 1"])
    check("run --look 0 1: two dabs, each aimed from a fresh clear view", rl and rl["ok"]
          and [r["target"] for r in rl["results"]] == ["hole 0", "hole 1"]
          and all(r.get("aimed_from", "").startswith("clear view") and r["re_aims"] == 0 for r in rl["results"]),
          str([r.get("aimed_from") for r in (rl or {}).get("results", [])]))
    dr, dc = last_json(out["droop"]), last_json(out["droop clear"])
    check("droop shows the offset, droop clear zeroes it", dr and len(dr["droop_correction_deg"]) == 7
          and dc and dc["ok"] and dc["droop_correction_deg"] == [0.0] * 7, str(dc))
    rt = last_json(out["retreat"])
    check("retreat ok", rt and rt["ok"])
    ins = last_json(out["inspect"])
    check("inspect: no ink -> note, image saved", ins and "no marks found" in ins.get("note", "")
          and any(ln.endswith("_inspect.jpg") for ln in out["inspect"]))
    al = last_json(out["alerts"])
    check("alerts: feed with a recommendation", al and "recommendation" in al and "metrics" in al,
          al.get("recommendation") if al else str(out["alerts"]))
    fn = last_json(out["finish pass"])
    check("finish pass: part closed", fn and fn["part"] == "t1" and fn["outcome"] == "pass")
    check("look: camera image saved", any("_look_" in ln for ln in out["look"]), str(out["look"]))
    check("unknown command: REFUSED, shell keeps running", out["nonsense"] and out["nonsense"][0].startswith("REFUSED:")
          and "quit" in out)


async def station_confirm():
    print("3. fake station: movers ask first, 'n' cancels")
    cell, frames = cli.make_cell(str(IMG))
    asked, lines = [], []

    def inp(prompt=""):
        asked.append(prompt)
        return "n"

    c = cli.CLI(cell, input_fn=inp, out=lambda s="": lines.append(str(s)), frames=frames)
    await c.execute("new")
    cell.backend.name = "station"                    # pretend: no real Station backend is built
    j0, z0 = copy.deepcopy(cell.backend.joints), dict(cell.zone)
    movers = ["run 0 2", "hover 0", "touch 0", "retreat", "home so101", "gripper elrobot close", "load", "unload"]
    for cmd in movers:
        n0, l0 = len(asked), len(lines)
        await c.execute(cmd)
        said = lines[l0:]
        check(f"{cmd}: printed the plan, asked, cancelled", len(asked) == n0 + 1
              and asked[-1] == "Move the real arm? [y/N] " and any(s.startswith("WILL DO:") for s in said)
              and said[-1] == "cancelled, nothing moved", " | ".join(said)[:160])
        if cmd in ("run 0 2", "touch 0"):
            check(f"{cmd}: shows touch_mm and hover_mm", any("touch_mm =" in s and "hover_mm =" in s for s in said))
    check("nothing moved, zones unchanged", cell.backend.joints == j0 and cell.zone == z0)
    l0 = len(lines)
    await c.execute("hover 0 --yes")
    check("--yes refused on the station", lines[l0].startswith("REFUSED: --yes") and len(asked) == len(movers))
    l0, n0 = len(lines), len(asked)
    await c.execute("stop test of the cli")
    st = last_json(lines[l0:])
    check("stop never asks and latches the e-stop", len(asked) == n0 and st and st["estop"].get("latched"), str(st))
    # 'y' would proceed: refused now by the latched e-stop, through the same guard
    l0 = len(lines)
    c.input = lambda p="": "y"
    await c.execute("gripper elrobot open")
    check("after stop, a confirmed move is REFUSED by the e-stop", any(s.startswith("REFUSED:") and "E-STOP" in s
                                                                       for s in lines[l0:]), lines[-1][:120])


class Script:
    """Scripted input: pops lines, records prompts; EOF when empty."""

    def __init__(self):
        self.lines, self.prompts = [], []

    def __call__(self, prompt=""):
        self.prompts.append(prompt)
        if not self.lines:
            raise EOFError
        return self.lines.pop(0)


def unit(v):
    import numpy as np
    return v / np.linalg.norm(v)


async def jog_checks():
    import numpy as np
    print("4. jog and teach the lid aim offset (sim)")
    (STATE / "estop.json").write_text('{"latched": false}')       # section 3 latched it
    cell, frames = cli.make_cell(str(IMG))
    inp, lines = Script(), []
    c = cli.CLI(cell, input_fn=inp, out=lambda s="": lines.append(str(s)), frames=frames)
    await c.execute("new")
    k = cell.kin
    l0 = len(lines)
    c.in_shell = False
    await c.execute("jog")
    check("jog outside the shell is refused", any("shell only" in x for x in lines[l0:]), " | ".join(lines[l0:]))
    c.in_shell = True
    l0 = len(lines)
    await c.execute("jog")
    check("jog before any hover/touch is refused", any(x.startswith("REFUSED:") and "hover or touch" in x
                                                         for x in lines[l0:]), " | ".join(lines[l0:]))
    st = await cell.vision.state()
    aim_before = cell._aim(st, {"hole": 0})["P"].copy()
    l0 = len(lines)
    await c.execute("touch 0")
    tr = last_json(lines[l0:])
    r0 = (tr or {}).get("results", [{}])[0]
    if not (tr and tr.get("ok")):
        print("\n".join(lines[l0:]))
    check("touch result: tip_above_part_mm and lowest_tip_above_part_mm",
          tr and tr["ok"] and abs(r0.get("tip_above_part_mm", -99) - cell.m["hover_mm"]) < 1.5
          and abs(r0.get("lowest_tip_above_part_mm", -99) - cell.m["touch_mm"]) < 1.5,
          f"{r0.get('tip_above_part_mm')} / {r0.get('lowest_tip_above_part_mm')}")
    la = cell._last_aim
    size = la["image_size"]
    shape = (size[1], size[0])
    px = np.asarray(la["px"], float)
    p0 = k.table.pixel_to_base(px, shape)
    e_r = unit(k.table.pixel_to_base(px + [10, 0], shape) - p0)       # image right, from the calibration
    e_u = unit(k.table.pixel_to_base(px + [0, -10], shape) - p0)      # image top
    tip0 = k.tip(await cell._q_now())
    inp.lines = ["dd", "w", "p", "k", "y"]
    n_prompts = len(inp.prompts)
    l0 = len(lines)
    await c.execute("jog")
    said = lines[l0:]
    jog_lines = said
    tip1 = k.tip(await cell._q_now())
    moved = (tip1 - tip0) * 1000
    expect = 4.0 * e_r + 2.0 * e_u                                     # dd = 2 x 2 mm right, w = 2 mm up (mm)
    check("dd w: modelled tip moved 4 mm image-right + 2 mm image-up (robot frame)",
          np.linalg.norm(moved - expect) < 0.5, f"moved {np.round(moved, 2).tolist()} expected {np.round(expect, 2).tolist()}")
    check("sim: no motion question, only jog> prompts and the parallax question",
          all(p in ("jog> ", "Save this offset anyway? [y/N] ") for p in inp.prompts[n_prompts:]), str(inp.prompts[n_prompts:]))
    pj = last_json(said[next((i for i, x in enumerate(said) if x.startswith("{")), 0):])
    check("p prints the offset in image directions and robot frame", pj and abs(pj["offset_image_mm"]["right"] - 4.0) < 0.3
          and abs(pj["offset_image_mm"]["up"] - 2.0) < 0.3 and len(pj["offset_robot_mm"]) == 3, str(pj))
    check("k at 40 mm above the part: parallax warning and a confirmation", any("WARNING" in x and "parallax" in x
                                                                              for x in said))
    f = cell.cfg.state_dir / "aim_lid.json"
    data = json.loads(f.read_text()) if f.exists() else {}
    smp = data.get("samples", [])
    check("aim_lid.json written with one sample (offset_m, at_m, height_above_part_mm, t)", len(smp) == 1
          and {"offset_m", "at_m", "height_above_part_mm", "t"} <= set(smp[0])
          and np.linalg.norm(np.array(smp[0]["offset_m"]) * 1000 - k.in_plane(expect)) < 0.5, str(smp)[:160])
    aim_after = cell._aim(st, {"hole": 0})["P"]
    shift = (aim_after - aim_before) * 1000
    check("next target for hole 0 shifted by the saved offset", np.linalg.norm(shift - k.in_plane(expect)) < 0.5,
          f"shift {np.round(shift, 2).tolist()}")
    l0 = len(lines)
    await c.execute("hover 0")
    hv = last_json(lines[l0:])
    am = np.array(hv["results"][0]["aimed_robot_mm"], float) if hv and hv["ok"] else np.zeros(3)
    check("hover 0 aims at the shifted target", hv and hv["ok"]
          and np.abs(am - aim_after * 1000).max() <= 0.6, str(am))
    l0 = len(lines)
    await c.execute("aim")
    a = last_json(lines[l0:])
    check("aim prints aim_file part, lid part, samples", a and "aim_file" in a and "lid_jog" in a
          and len(a["samples"]) == 1, str(a)[:120])
    gs = await cell.guided_status()
    check("guided_status shows both parts", {"aim_file", "lid_jog", "total"} <= set(gs.get("aim_offset_mm", {})))
    inp.lines = ["n"]
    await c.execute("aim clear")
    check("aim clear + n: nothing changed", len(k.aim_lid_samples()) == 1)
    inp.lines = ["y"]
    await c.execute("aim clear")
    data = json.loads(f.read_text())
    check("aim clear + y: empty samples list, file kept, target back", data == {"samples": []}
          and np.linalg.norm(cell._aim(st, {"hole": 0})["P"] - aim_before) < 1e-6)

    print("4b. jog limits")
    tip0 = k.tip(await cell._q_now())
    inp.lines = ["]]]", "dddd", "q"]                     # 8 mm steps: the 4th would make 32 mm > 30 mm
    l0 = len(lines)
    await c.execute("jog")
    said = lines[l0:]
    tip1 = k.tip(await cell._q_now())
    side = float(np.linalg.norm(k.in_plane(tip1 - tip0)) * 1000)
    check("a step beyond 30 mm sideways is refused, 24 mm done, nothing saved",
          any("REFUSED" in x and "max 30 mm" in x for x in said) and abs(side - 24.0) < 0.7
          and json.loads(f.read_text()) == {"samples": []} and any("nothing saved" in x for x in said),
          f"{side:.1f} mm; " + " | ".join(x for x in said if "REFUSED" in x))
    inp.lines = ["]]]", "ffffff", "q"]
    l0 = len(lines)
    await c.execute("jog")
    h = cell._above_part(await cell._q_now()) * 1000
    check("f never goes below 3 mm above the part top", abs(h - 3.0) < 0.7
          and any("lower jog limit" in x for x in lines[l0:]), f"{h:.1f} mm")
    l0 = len(lines)
    await c.execute("retreat")
    await c.execute("jog")
    check("after retreat, jog is refused", any("REFUSED" in x and "hover or touch" in x for x in lines[l0:]))

    print("4c. fake station: jog asks once on entering, n cancels")
    await c.execute("hover 0")
    cell.backend.name = "station"
    cell.backend.latest_t = float("inf")              # fake: joint data always fresh
    try:
        j0 = cell.backend.joints["elrobot"].copy()
        inp.lines = ["n"]
        n0 = len(inp.prompts)
        l0 = len(lines)
        await c.execute("jog")
        check("station: one question, n cancels, nothing moved", inp.prompts[n0:] == ["Move the real arm? [y/N] "]
              and lines[-1] == "cancelled, nothing moved" and cell.backend.joints["elrobot"] == j0
              and any("Jog the real arm in steps of 2 mm, max 30 mm total" in x for x in lines[l0:]),
              str(inp.prompts[n0:]))
        inp.lines = ["y", "d", "a", "q"]
        n0 = len(inp.prompts)
        await c.execute("jog")
        check("station: y then two moves, still asked only once", inp.prompts[n0:].count("Move the real arm? [y/N] ") == 1
              and inp.prompts[n0:].count("jog> ") == 3, str(inp.prompts[n0:]))
    finally:
        cell.backend.name = "sim"
    return jog_lines


async def set_checks():
    import numpy as np
    print("5. set <key> <value>")
    P = cli.parse_setting
    m = {"hover_mm": 40.0, "high_mm": 110.0}
    check("parse float", P(m, "touch_mm", "5") == 5.0 and P(m, "touch_mm", "-3") == -3.0)
    check("parse bool", P(m, "all_slow", "true") is True and P(m, "all_slow", "false") is False)
    check("parse pretouch_mm", P(m, "pretouch_mm", "12") == 12.0 and P(m, "pretouch_mm", "5") == 5.0
          and P({**m, "touch_mm": 4.0}, "pretouch_mm", "40") == 40.0)
    for key, val, why in (("touch_mm", "41", "out of range"), ("touch_mm", "-3.5", "out of range"),
                          ("hover_mm", "10", "out of range"), ("high_mm", "39", "out of range"),
                          ("high_mm", "40", "must stay above hover_mm"), ("hover_mm", "115", "must stay below"),
                          ("dwell_s", "31", "out of range"), ("reaim_mm", "0.5", "out of range"),
                          ("commit_age_s", "x", "not a number"), ("all_slow", "maybe", "true or false"),
                          ("max_speed", "1", "cannot be changed"), ("pretouch_mm", "4.5", "out of range"),
                          ("pretouch_mm", "41", "out of range"), ("pretouch_mm", "40", "must stay below the hover")):
        try:
            P(m, key, val)
            check(f"set {key} {val} refused", False)
        except cli.GuardError as e:
            check(f"set {key} {val} refused ({why})", why in str(e) and ("allowed" in str(e) or "true or false" in str(e)),
                  str(e)[:90])
    cell, frames = cli.make_cell(str(IMG))
    inp, lines = Script(), []
    c = cli.CLI(cell, input_fn=inp, out=lambda s="": lines.append(str(s)), frames=frames)
    await c.execute("new")
    saved = dict(cell.m)
    try:
        l0 = len(lines)
        await c.execute("set")
        ls = last_json(lines[l0:])
        check("set alone lists the whitelist", ls and set(ls) == set(cli.SETTABLE), str(ls))
        st = await cell.vision.state()
        p_before = cell._aim(st, {"hole": 1})["P"]
        l0 = len(lines)
        await c.execute("set touch_mm 5")
        check("set touch_mm 5 prints old -> new", any("touch_mm: 0.0 -> 5.0" in x for x in lines[l0:]), lines[-1])
        d = (cell._aim(st, {"hole": 1})["P"] - p_before) * 1000
        check("effect: the touch target rises 5 mm along the table normal", abs(d @ cell.kin.n - 5.0) < 1e-6
              and np.linalg.norm(cell.kin.in_plane(d)) < 1e-6)
        await c.execute("set touch_mm 99")
        check("out of range refused in the shell, value kept", lines[-1].startswith("REFUSED:") and cell.m["touch_mm"] == 5.0)
        await c.execute("set all_slow true")
        check("set all_slow true", cell.m["all_slow"] is True)
        cell.backend.name = "station"
        try:
            inp.lines = ["n"]
            l0 = len(lines)
            await c.execute("set touch_mm 1")
            check("station: touch_mm < 3 explains and asks, n keeps the value", cell.m["touch_mm"] == 5.0
                  and inp.prompts[-1] == "Set touch_mm = 1.0? [y/N] " and any("nearly touching" in x for x in lines[l0:]))
            inp.lines = ["y"]
            await c.execute("set touch_mm -1")
            check("station: y applies a pressing touch_mm", cell.m["touch_mm"] == -1.0)
            inp.lines = ["n"]
            await c.execute("set all_slow false")
            check("station: all_slow false asks, n keeps it", cell.m["all_slow"] is True and "normal" in lines[-2])
            n0 = len(inp.prompts)
            await c.execute("set hover_mm 30")
            check("station: harmless change does not ask", len(inp.prompts) == n0 and cell.m["hover_mm"] == 30.0)
        finally:
            cell.backend.name = "sim"
    finally:
        cell.m.clear()
        cell.m.update(saved)
    check("guided.toml unchanged by set", gconfig.load()["motion"]["touch_mm"] == G["motion"]["touch_mm"])


async def banner_checks():
    print("6. station banner")
    cell, frames = cli.make_cell(str(IMG))
    lines = []
    c = cli.CLI(cell, input_fn=Script(), out=lambda s="": lines.append(str(s)), frames=frames)
    cell.backend.name = "station"
    cell.backend.bus_of = {"5B7B015823": "so101", "5A7C": "elrobot"}
    try:
        cell._connected = False
        b = c.banner()
        st = cell.cfg.station
        check("before connecting: no 'buses {}', says when it connects",
              f"Station at {st.get('host')}:{st.get('port')} (connects on the first command)" in b and "buses" not in b, b)
        await c.execute("help")
        check("a command that does not connect prints no 'connected:' line", not any(x.startswith("connected:") for x in lines))
        await c.execute("status")
        con = [x for x in lines if x.startswith("connected:")]
        await c.execute("status")
        check("first connecting command prints 'connected: buses {...}' once",
              len([x for x in lines if x.startswith("connected:")]) == 1 and "5B7B015823" in con[0], str(con))
    finally:
        cell.backend.name = "sim"


SHUTDOWN_SCRIPT = r"""
import asyncio, os, sys
sys.path.insert(0, sys.argv[2]); sys.path.insert(0, os.path.join(sys.argv[2], "tests"))
from cellguide import cli
orig = cli.CLI._dispatch

async def disp(self, cmd, args):
    if cmd != "spawn":
        return await orig(self, cmd, args)
    async def forever_queue():            # like station_py Client.read_loop
        await asyncio.Queue().get()
    async def forever_sleep():            # like keep_alive_loop
        while True:
            await asyncio.sleep(0.05)
    async def slow_cleanup():
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            await asyncio.sleep(0.2)
            raise
    async def agen():
        while True:
            yield 1
            await asyncio.sleep(0.05)
    async def use_agen():
        async for _ in agen():
            pass
    loop = asyncio.get_running_loop()
    self._dummy = [loop.create_task(f()) for f in (forever_queue, forever_sleep, slow_cleanup, use_agen)]
    self.out("spawned")

cli.CLI._dispatch = disp
if sys.argv[1] == "shell":
    feed = ["spawn", "status", "quit"]
    rc = cli.main(["shell"], input_fn=lambda p="": feed.pop(0))
else:
    rc = cli.main(["spawn"])
print("RC", rc)
"""


def shutdown_checks():
    import subprocess
    print("7. quit with pending background tasks: quiet, clean exit")
    for mode in ("shell", "single"):
        env = dict(os.environ, CELLBOT_STATE=tempfile.mkdtemp(prefix="cellguide_quit_"))
        t0 = __import__("time").monotonic()
        p = subprocess.run([sys.executable, "-c", SHUTDOWN_SCRIPT, mode, str(ROOT)], capture_output=True, text=True,
                           env=env, timeout=60)
        dt = __import__("time").monotonic() - t0
        noise = [w for w in ("Task was destroyed", "Event loop is closed", "Exception ignored", "Traceback",
                             "never awaited") if w in p.stderr]
        check(f"{mode}: exit 0, 'spawned', no shutdown noise on stderr", p.returncode == 0 and "spawned" in p.stdout
              and "RC 0" in p.stdout and not noise, f"{dt:.1f} s; rc {p.returncode}; noise {noise}; "
              f"stderr tail {p.stderr[-300:]!r}")


def main():
    parser_checks()
    shell_flow()
    asyncio.run(station_confirm())
    jog_lines = asyncio.run(jog_checks())
    asyncio.run(set_checks())
    asyncio.run(banner_checks())
    shutdown_checks()
    if os.environ.get("SHOW_JOG"):
        print("\n".join(jog_lines))
    rc = cli.main(["--yes", "status"], out=lambda s="": None)
    check("--yes accepted in sim for a one-shot command", rc == 0)
    if check.failed:
        print(f"{check.failed} CHECK(S) FAILED")
        sys.exit(1)
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
