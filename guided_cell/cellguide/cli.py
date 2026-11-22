"""Run any guided-cell step from a terminal, without the Claude app. Thin wrappers over GuidedCell.

    python -m cellguide.cli <command> [args]            one command, one process
    python -m cellguide.cli shell                       prompt loop on ONE long-lived GuidedCell (normal use)
    python -m cellguide.cli --frames <folder> shell     sim only: recorded images instead of the camera
    python -m cellguide.cli --source camera:2 shell     override guided.toml [vision] source for this run

Use `shell` on hardware: zones, the current part and the hole layout live in memory, so a fresh process per
command starts with zones "unknown" and no part. Same env as the MCP server (PYTHONPATH, CELLBOT_CONFIG,
CELLBOT_STATE, CELLBOT_MODE, STATION_HOST, optional CELLGUIDE_CONFIG). Every command that can move an arm prints
what it will do first and, on the real Station, asks "Move the real arm? [y/N]". `stop` never asks.
Results print as JSON; images go to <state>/cli/<timestamp>_<command>.jpg.
"""
from __future__ import annotations

import asyncio
import json
import shlex
import sys
import time
from pathlib import Path

from . import guided_cell as gc  # loads guided.toml and puts cellbot on sys.path (before any cellbot import)

from cellbot.cell import GuardError  # noqa: E402

from .vision_source import FrameVision, VisionError, install as _install_camera  # noqa: E402

_install_camera(gc)                 # camera:auto + self-healing camera:N (guided_cell.py unchanged)

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp"}
MOVERS = {"run", "hover", "touch", "retreat", "home", "gripper", "load", "unload", "jog"}

# `set <key> <value>`: [motion] keys that may change in a running session (guided.toml is not changed).
SETTABLE = {"touch_mm": (-3.0, 40.0), "hover_mm": (15.0, 120.0), "high_mm": (40.0, 200.0), "dwell_s": (0.0, 30.0),
            "all_slow": None, "reaim_mm": (1.0, 20.0), "commit_age_s": (1.0, 30.0), "touch_tol_mm": (1.0, 5.0),
            "pretouch_mm": (5.0, 40.0)}
JOG_KEYS = "wasdrf[]pkq"
JOG_HELP = ("jog keys (several per line, applied in order, e.g. ddw): w/s = towards the TOP/BOTTOM of the camera "
            "image, a/d = LEFT/RIGHT, r/f = up/down along the table normal, [ ] = halve/double the step "
            "(0.5..8 mm), p = print the offset, k = save and leave, q = leave without saving")

HELP = """commands (results print as JSON, images are saved under <state>/cli/):
  status                     cellbot status: zones, joints, poses, e-stop
  preflight                  read-only readiness checklist
  guided                     guided_status: vision, tool tip, calibration, tip position, guards
  see                        what the camera sees: holes, robot mm, reachability (+ image)
  new [part_id]              start a new part: lock the hole layout
  plan <steps>               check a program without moving
  run <steps>                one clear look, run a program from it, then retreat (moves)
  run --look <steps>         retreat and look again before EVERY step (part moved between steps)  (moves)
  hover <hole_id>            hover over one hole and stay there                 (moves)
  touch <hole_id>            dab one hole, stay hovering above it              (moves)
  inspect                    measure the marks on this part (+ image)
  retreat                    lift, then go to the look pose (or home)          (moves)
  home <arm>                 move an arm slowly to its home pose               (moves)
  gripper <arm> open|close   open or close a gripper                           (moves)
  load / unload [fallback]   loader arm brings / removes the part; fallback: see [loader] policy_<phase>_fallback_*  (moves)
  load / unload teleop       HANDOVER: no motion; a human moves the so101 with the leader arm (Station teleoperation)
  teleop done                close the open teleop handover (no motion); then go_home so101, see, label
  label ok|fail [note]       outcome of the latest load/unload attempt (check with `see` first)
  loader                     loader attempts per phase/variant, success rate, dataset time ranges
  jog                        shell only, after hover/touch: nudge the tip, k saves the lid aim offset  (moves)
  aim                        aim offset: aim_file part + lid part (jog) and the lid samples
  aim clear                  forget the lid aim samples (asks y/N)
  droop                      learned gravity-droop correction per joint (deg)
  droop clear                zero it (relearned at the next hover)
  set [<key> <value>]        change a [motion] value for this session only (no key: list them)
  alerts                     operator feed since the last call
  finish pass|fail|aborted [notes]   close the current part
  stop [reason]              e-stop: freeze both arms, latch (never asks)
  look                       save every camera image from cellbot
  help / quit                (Ctrl+C while a command runs = emergency stop)
<steps>: JSON ('[{"hole":0},{"at":[0.5,0.2],"action":"hover"}]') or short form:
  0 2 1 = dab holes 0, 2, 1;  h0 = hover over hole 0;  0.5,0.2 = dab that point of the lid (h0.5,0.2 hovers);
  all = every reachable hole of the current scene, in id order."""


# ------------------------------------------------------------------ steps
def _hole_id(tok: str):
    try:
        return int(tok)
    except ValueError:
        return tok


def parse_steps(text, all_ids=None) -> list[dict]:
    """JSON or short form -> list of run_program steps. `all` needs all_ids (reachable hole ids)."""
    if isinstance(text, (list, tuple)):
        text = " ".join(str(t) for t in text)
    text = text.strip()
    if not text:
        raise ValueError("no steps given")
    if text[0] in "[{":
        v = json.loads(text)
        v = [v] if isinstance(v, dict) else v
        if not isinstance(v, list) or not all(isinstance(s, dict) for s in v):
            raise ValueError("JSON steps must be a list of objects")
        return v
    steps = []
    for tok in text.replace(";", " ").split():
        if tok.lower() == "all":
            if all_ids is None:
                raise ValueError("'all' needs the current scene")
            steps += [{"hole": h} for h in all_ids]
            continue
        hover = tok[0] in "hH" and len(tok) > 1
        body = tok[1:] if hover else tok
        if "," in body:
            parts = body.split(",")
            if len(parts) != 2:
                raise ValueError(f"{tok!r}: a point on the lid is fx,fy (fractions 0..1)")
            try:
                s = {"at": [float(parts[0]), float(parts[1])]}
            except ValueError:
                raise ValueError(f"{tok!r}: a point on the lid is fx,fy (fractions 0..1)") from None
        else:
            s = {"hole": _hole_id(body)}
        if hover:
            s["action"] = "hover"
        steps.append(s)
    if not steps:
        raise ValueError("no steps given")
    return steps


def parse_setting(m: dict, key: str, text: str):
    """Validate `set <key> <value>` against SETTABLE and the current values. Returns the new value."""
    if key not in SETTABLE:
        raise GuardError(f"set: {key!r} cannot be changed here; allowed: " + ", ".join(
            f"{k} ({'true/false' if r is None else f'{r[0]:g} .. {r[1]:g}'})" for k, r in SETTABLE.items()))
    rng = SETTABLE[key]
    if rng is None:
        t = text.strip().lower()
        if t not in ("true", "false", "1", "0", "on", "off", "yes", "no"):
            raise GuardError(f"set {key}: use true or false")
        return t in ("true", "1", "on", "yes")
    try:
        v = float(text)
    except ValueError:
        raise GuardError(f"set {key}: {text!r} is not a number (allowed {rng[0]:g} .. {rng[1]:g})") from None
    if not (rng[0] <= v <= rng[1]) or v != v:
        raise GuardError(f"set {key}: {v:g} is out of range (allowed {rng[0]:g} .. {rng[1]:g})")
    if key == "high_mm" and v <= float(m["hover_mm"]):
        raise GuardError(f"set high_mm: must stay above hover_mm = {m['hover_mm']} (allowed {rng[0]:g} .. {rng[1]:g})")
    if key == "hover_mm" and v >= float(m["high_mm"]):
        raise GuardError(f"set hover_mm: must stay below high_mm = {m['high_mm']} (allowed {rng[0]:g} .. {rng[1]:g})")
    if key == "pretouch_mm" and v >= float(m.get("touch_mm", 0.0)) + float(m["hover_mm"]):
        raise GuardError(f"set pretouch_mm: must stay below the hover height (touch_mm + hover_mm = "
                         f"{float(m.get('touch_mm', 0.0)) + float(m['hover_mm']):g}; allowed {rng[0]:g} .. {rng[1]:g}); "
                         "at or below touch_mm the pre-touch stage is skipped")
    return v


def describe_steps(steps) -> str:
    out = []
    for s in steps:
        what = f"hole {s['hole']}" if "hole" in s else f"lid point {s.get('at')}"
        out.append(f"{s.get('action', 'dab')} {what}")
    return ", ".join(out)


# ------------------------------------------------------------------ recorded frames (sim/testing)
class FolderFrames:
    """Images of a folder in name order. The same image is shown until advance() (called by `new` after the
    first part), so the tracker sees a still part during a program and each new part is the next image."""

    def __init__(self, folder):
        import cv2
        self.folder = Path(folder).expanduser()
        files = sorted(p for p in self.folder.iterdir() if p.suffix.lower() in IMAGE_EXT) if self.folder.is_dir() else []
        self.frames = [(p.name, cv2.imread(str(p))) for p in files]
        self.frames = [(n, f) for n, f in self.frames if f is not None]
        if not self.frames:
            raise SystemExit(f"--frames {folder}: no readable images in that folder")
        self.i = 0
        self.used = False

    def advance(self):
        if self.used:
            self.i = (self.i + 1) % len(self.frames)
        self.used = True

    @property
    def current(self) -> str:
        return self.frames[self.i][0]

    async def __call__(self):
        return self.frames[self.i][1].copy()


# ------------------------------------------------------------------ the CLI
def _jsonable(o):
    try:
        import numpy as np
        if isinstance(o, np.generic):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
    except ImportError:
        pass
    if isinstance(o, (bytes, bytearray)):
        return f"<{len(o)} bytes>"
    if isinstance(o, Path):
        return str(o)
    return repr(o)


class CLI:
    def __init__(self, cell, yes: bool = False, input_fn=input, out=print, frames: FolderFrames | None = None):
        self.cell, self.yes, self.input, self.out, self.frames = cell, yes, input_fn, out, frames
        self.in_shell = False
        self._announced = False          # "connected: buses ..." printed once after the first connecting command
        self._runner = None

    # -------------------------------------------------------------- helpers
    @property
    def station(self) -> bool:
        return self.cell.backend.name != "sim"

    def banner(self) -> str:
        if self.station:
            if getattr(self.cell, "_connected", False):
                where = self.cell.backend.describe()
            else:
                st = self.cell.cfg.station
                where = f"Station at {st.get('host')}:{st.get('port')} (connects on the first command)"
            line = f"BACKEND: {where}  -- REAL ARMS, every motion asks first"
        else:
            line = "BACKEND: SIMULATION (no hardware)"
        if self.frames:
            line += f"\nVISION: recorded frames from {self.frames.folder} ({len(self.frames.frames)} images)"
        else:
            v = self.cell.gcfg["vision"]
            if v["source"].startswith("camera:"):
                port = int(v.get("view_port", 8765))
                view = f"viewer http://127.0.0.1:{port}/frame.jpg" if port else "viewer off"
                line += f"\nVISION: camera {v['source'].split(':', 1)[1].strip()} in-process ({view})"
            else:
                line += f"\nVISION: {v['source']}"
        return line

    def show(self, obj):
        self.out(json.dumps(obj, indent=2, default=_jsonable))

    def save_image(self, data: bytes, command: str) -> Path:
        d = self.cell.cfg.state_dir / "cli"
        d.mkdir(parents=True, exist_ok=True)
        ext = ".png" if data[:4] == b"\x89PNG" else ".jpg"
        ts = time.strftime("%Y%m%d_%H%M%S") + f"_{int(time.time() * 1000) % 1000:03d}"
        p = d / f"{ts}_{command}{ext}"
        p.write_bytes(data)
        self.out(f"image: {p}")
        return p

    async def confirm(self, what: str, cmd: str) -> bool:
        """Print what will move; ask on the real Station. False = cancelled.
        The question is asked in a worker thread: the event loop (Station link) must keep running meanwhile."""
        m = self.cell.m
        self.out(f"WILL DO: {what}")
        if not self.station:
            self.out("(simulation: no question asked)")
            return True
        if cmd in ("run", "touch"):
            self.out(f"  touch_mm = {m['touch_mm']} (tip height above the lid when touching), "
                     f"hover_mm = {m['hover_mm']}")
        try:
            ans = await asyncio.get_running_loop().run_in_executor(None, self.input, "Move the real arm? [y/N] ")
        except EOFError:
            ans = ""
        if ans.strip().lower() != "y":
            self.out("cancelled, nothing moved")
            return False
        return True

    async def ask(self, prompt: str) -> str:
        """One line of input, read in a worker thread (the event loop keeps running). EOF = empty."""
        try:
            return await asyncio.get_running_loop().run_in_executor(None, self.input, prompt)
        except EOFError:
            return ""

    async def yes_no(self, prompt: str) -> bool:
        return (await self.ask(prompt + " [y/N] ")).strip().lower() == "y"

    async def _reachable_ids(self) -> list:
        scene, _ = await self.cell.scene()
        if not scene.get("holes"):
            raise GuardError(f"'all': no holes in the current view ({scene.get('status')})")
        holes = sorted(scene["holes"], key=lambda h: (str(type(h["id"])), h["id"]))
        return [h["id"] for h in holes if h.get("reachable")]

    async def _steps(self, args) -> list[dict]:
        text = " ".join(args)
        ids = await self._reachable_ids() if any(t.lower() == "all" for t in text.split()) else None
        try:
            return parse_steps(text, ids)
        except (ValueError, json.JSONDecodeError) as e:
            raise GuardError(f"bad steps: {e}") from None

    def _arm(self, args, i=0) -> str:
        if len(args) <= i or args[i] not in self.cell.cfg.arms:
            raise GuardError(f"name an arm: {list(self.cell.cfg.arms)}")
        return args[i]

    # -------------------------------------------------------------- one line
    async def execute(self, line: str) -> bool:
        """Run one command line. Returns False on quit. Never raises for command errors."""
        try:
            args = shlex.split(line)
        except ValueError as e:
            self.out(f"ERROR: {e}")
            return True
        if not args:
            return True
        yes_here = False
        while args and args[-1] in ("--yes", "-y"):
            yes_here = True
            args.pop()
        cmd, args = args[0].lower(), args[1:]
        if cmd in ("quit", "exit", "q"):
            return False
        if yes_here and self.station:
            self.out("REFUSED: --yes is not allowed with the real Station; answer the question instead")
            return True
        cell = self.cell
        was_connected = getattr(cell, "_connected", False)
        try:
            await self._dispatch(cmd, args)
            if self.station and not self._announced and not was_connected and getattr(cell, "_connected", False):
                self._announced = True
                self.out(f"connected: buses {getattr(cell.backend, 'bus_of', {})}")
        except GuardError as e:
            cell.log_event("refused", str(e))
            self.out(f"REFUSED: {e}")
        except VisionError as e:
            cell.log_event("vision", str(e))
            self.out(f"VISION: {e}")
        except Exception as e:  # noqa: BLE001  hardware/link problems: show them, keep the shell alive
            try:
                cell.log_event("error", repr(e))
            except Exception:  # noqa: BLE001
                pass
            self.out(f"ERROR: {e!r}")
        return True

    async def _dispatch(self, cmd: str, args: list[str]):
        cell, m = self.cell, self.cell.m
        if cmd == "help":
            self.out(HELP)
        elif cmd == "status":
            self.show(await cell.status())
        elif cmd == "preflight":
            from .preflight import format_text, report
            self.out(format_text(await report(cell)))
        elif cmd == "guided":
            self.show(await cell.guided_status())
        elif cmd == "see":
            res, img = await cell.scene()
            self.show(res)
            self.save_image(img, "see")
        elif cmd == "new":
            if self.frames:
                self.frames.advance()
                self.out(f"frame: {self.frames.current}")
            self.show(await cell.new_part(args[0] if args else None))
        elif cmd == "plan":
            self.show(await cell.plan_program(await self._steps(args)))
        elif cmd == "run":
            look = "--look" in args
            steps = await self._steps([a for a in args if a != "--look"])
            how = ("retreat and look before every step" if look else "one clear look first (retreat if needed)")
            if await self.confirm(f"{cell.arm}: {how}; {len(steps)} step(s): {describe_steps(steps)}; then retreat "
                            "out of the camera's view", cmd):
                self.show(await cell.run_program(steps, retreat_after=True, look_each_step=look))
        elif cmd in ("hover", "touch"):
            if len(args) != 1:
                raise GuardError(f"usage: {cmd} <hole_id>")
            h = _hole_id(args[0])
            if cmd == "hover":
                what = (f"{cell.arm}: go {m['hover_mm']} mm above hole {h} and stay there"
                        + (" (follows the part if it moves)" if cell._sees_under_arm() else
                           " (aimed from a clear look; the part is not watched while the arm is over it)"))
                step = {"hole": h, "action": "hover"}
            else:
                what = (f"{cell.arm}: go above hole {h}, lower to touch_mm = {m['touch_mm']} mm above the lid, "
                        f"lift back to {m['hover_mm']} mm and stay there")
                step = {"hole": h}
            if await self.confirm(what, cmd):
                self.show(await cell.run_program([step], retreat_after=False))
        elif cmd == "inspect":
            res, img = await cell.inspect()
            self.show(res)
            self.save_image(img, "inspect")
        elif cmd == "retreat":
            pose = m["look_pose"]
            if await self.confirm(f"{cell.arm}: lift straight up to {m['high_mm']} mm above the lid if low, then move "
                            f"slowly to pose '{pose}' (or home if '{pose}' is not taught)", cmd):
                self.show(await cell.retreat())
        elif cmd == "home":
            arm = self._arm(args)
            if await self.confirm(f"{arm}: move slowly to its home pose '{cell.cfg.arms[arm].home_pose}'", cmd):
                self.show(await cell.go_home(arm))
        elif cmd == "gripper":
            arm = self._arm(args)
            if len(args) < 2 or args[1] not in ("open", "close"):
                raise GuardError("usage: gripper <arm> open|close")
            if await self.confirm(f"{arm}: {args[1]} the gripper", cmd):
                self.show(await cell.gripper(arm, args[1]))
        elif cmd in ("load", "unload") and args and args[0].lower() == "teleop":
            if len(args) > 1:
                raise GuardError(f"usage: {cmd} teleop")
            r = await (cell.load_part("teleop") if cmd == "load" else cell.unload_part("teleop"))
            self.show(r)
            self.out(f"HUMAN: {r['instructions_for_human']}")
            self.out("when the human is done and teleoperation is OFF: teleop done")
        elif cmd == "teleop":
            if not args or args[0].lower() != "done" or len(args) > 2:
                raise GuardError("usage: teleop done [attempt_id]")
            self.show(cell.finish_teleop(args[1] if len(args) > 1 else ""))
        elif cmd in ("load", "unload"):
            if len(args) > 1 or (args and args[0].lower() not in ("primary", "fallback")):
                raise GuardError(f"usage: {cmd} [fallback|teleop]")
            lc = cell._loader_cfg()
            mode = cell._phase_mode(cmd, lc)
            variant = cell._check_variant(cmd, args[0] if args else "primary", mode, lc)   # refuses before asking
            fb = variant == "fallback"
            run = (lambda: cell.load_part(variant)) if cmd == "load" else (lambda: cell.unload_part(variant))
            if mode == "human":
                self.out(f"loader mode human: no motion, {cmd} only prompts")
                self.show(await run())
            else:
                where = "pickup -> work area" if cmd == "load" else "work area -> drop-off"
                pre = str(lc.get(f"policy_{cmd}_pre_pose") or "").strip() if mode == "policy" and not fb else ""
                seq = cell._phase_seq(cmd, lc) if mode == "taught" else []
                if seq:
                    plan = [t for _, _, t in cell._parse_seq(lc["arm"], cmd, seq)]   # refuses a bad seq here
                pre_seq = ((cell._fallback_pre_seq(cmd, lc) if fb else cell._phase_pre_seq(cmd, lc))
                           if mode == "policy" else [])                       # refuses pre_pose+pre_seq
                if pre_seq:
                    seq_key = f"policy_{cmd}_fallback_pre_seq" if fb else f"policy_{cmd}_pre_seq"
                    pre_plan = [t for _, _, t in cell._parse_seq(lc["arm"], cmd, pre_seq,
                                                                 key=seq_key, auto_home=False)]
                if await self.confirm(f"{lc['arm']}: {cmd} the part ({where}), loader mode {mode}"
                                + (f"; FALLBACK command ({cell._fallback_cmd_key(cmd, lc)}): "
                                   f"{lc[cell._fallback_cmd_key(cmd, lc)]}" if fb else "")
                                + (f"; FIRST taught pre-pose {pre!r} (cellbot guards, slow), THEN the policy"
                                   if pre else "")
                                + (f"; FIRST taught steps: {' -> '.join(pre_plan)} (cellbot guards, slow), "
                                   "THEN the policy" if pre_seq else "")
                                + (" (policy runs OUTSIDE cellbot's guards)" if mode == "policy" else "")
                                + (f"; taught sequence (cellbot guards, slow): {' -> '.join(plan)}"
                                   if seq else ""),
                                cmd):
                    self.show(await run())
        elif cmd == "label":
            if not args or args[0].lower() not in ("ok", "fail"):
                raise GuardError("usage: label ok|fail [note]   (outcome of the latest load/unload attempt)")
            self.show(cell.label_loader_attempt(args[0].lower() == "ok", " ".join(args[1:])))
        elif cmd == "loader":
            self.show(cell.loader_stats())
        elif cmd == "jog":
            await self._jog(args)
        elif cmd == "aim":
            await self._aim_cmd(args)
        elif cmd == "set":
            await self._set(args)
        elif cmd == "droop":
            await cell.ensure_guided()
            if args and args[0].lower() == "clear":
                self.show(cell.droop_clear())
            elif args:
                raise GuardError("usage: droop  |  droop clear")
            else:
                self.show({"droop_correction_deg": [round(float(v), 2) for v in (cell._offset * 57.29577951308232)],
                           "settle_max_deg": m.get("settle_max_deg", 4.0)})
        elif cmd == "alerts":
            self.show(await cell.alerts())
        elif cmd == "finish":
            if not args or args[0] not in ("pass", "fail", "aborted"):
                raise GuardError("usage: finish pass|fail|aborted [notes]")
            self.show(await cell.finish_part(args[0], " ".join(args[1:])))
        elif cmd == "stop":
            self.show(await cell.stop(" ".join(args) or "stop from cellguide.cli"))
        elif cmd == "look":
            imgs = await cell.look()
            if not imgs:
                self.out("no camera images")
            for cam, data in imgs.items():
                self.save_image(data, f"look_{cam}")
        else:
            raise GuardError(f"unknown command {cmd!r}; type help")

    # -------------------------------------------------------------- jog / aim / set
    async def _jog(self, args):
        cell = self.cell
        if not self.in_shell:
            raise GuardError("jog works in the shell only: python -m cellguide.cli shell")
        step = 2.0
        info = await cell.jog_begin(step)              # preconditions; no motion
        what = (f"{cell.arm}: Jog the {'real ' if self.station else ''}arm in steps of {step:g} mm, max {cell.JOG_MAX_MM:.0f} mm total, around "
                f"{info['target']} at the current height ({info['target_above_part_mm']} mm above the part)")
        if not await self.confirm(what, "jog"):
            cell.jog_end()
            return
        self.out(JOG_HELP)
        try:
            while True:
                try:
                    cell._check_estop()
                except GuardError as e:
                    self.out(f"REFUSED: {e}")
                    return
                line = (await self.ask("jog> ")).strip().lower()
                if not line:
                    continue
                try:
                    cell._check_estop()                # e.g. Ctrl+C was pressed while waiting for this line
                except GuardError as e:
                    self.out(f"REFUSED: {e}")
                    return
                for ch in line:
                    if ch.isspace():
                        continue
                    if ch not in JOG_KEYS:
                        self.out(f"unknown jog key {ch!r}; rest of the line ignored. {JOG_HELP}")
                        break
                    if ch == "q":
                        self.out("left jog, nothing saved")
                        return
                    if ch == "p":
                        self.show(cell.jog_info())
                        continue
                    if ch == "k":
                        if await self._jog_save():
                            return
                        continue
                    try:
                        r = await cell.jog_key(ch)
                    except GuardError as e:
                        self.out(f"REFUSED: {e}; rest of the line ignored")
                        if "E-STOP" in str(e):
                            return
                        break
                    if not r["ok"]:
                        self.out("move failed, jog ended:")
                        self.show(r)
                        return
                    if "how" in r:
                        o = r["offset_image_mm"]
                        self.out(f"{ch}: offset up {o['up']} right {o['right']} mm, robot {r['offset_robot_mm']} mm, "
                                 f"{r['target_above_part_mm']} mm above the part (step {r['step_mm']} mm)")
                    else:
                        self.out(f"step {r['step_mm']} mm")
        finally:
            cell.jog_end()

    async def _jog_save(self) -> bool:
        """k: True = saved (leave jog), False = not saved (stay in jog)."""
        cell = self.cell
        info = cell.jog_info()
        h = info["target_above_part_mm"]
        if h > cell.JOG_WARN_ABOVE_MM:
            self.out(f"WARNING: the tip is {h:.0f} mm above the part top (more than {cell.JOG_WARN_ABOVE_MM:.0f} mm): "
                     "parallax makes judging 'above the hole' unreliable. Lower with f first.")
            if not await self.yes_no("Save this offset anyway?"):
                self.out("not saved; still jogging")
                return False
        res = await cell.jog_save()
        self.show(res)
        self.out(f"saved to {cell.kin.aim_lid_path}; applied now (next targets shift by it)")
        return True

    async def _aim_cmd(self, args):
        cell = self.cell
        await cell.ensure_guided()
        if args and args[0].lower() == "clear":
            n = len(cell.kin.aim_lid_samples())
            if not await self.yes_no(f"Remove the {n} lid aim sample(s) in {cell.kin.aim_lid_path.name}?"):
                self.out("cancelled, nothing changed")
                return
            cell.kin.clear_aim_lid()
            self.out(f"cleared {n} sample(s)")
            self.show(cell.aim_parts())
            return
        if args:
            raise GuardError("usage: aim  |  aim clear")
        self.show({**cell.aim_parts(), "samples": cell.kin.aim_lid_samples()})

    async def _set(self, args):
        m = self.cell.m
        if not args:
            self.show({k: m.get(k) for k in SETTABLE})
            return
        if len(args) != 2:
            raise GuardError("usage: set <key> <value>   (set alone lists the values)")
        key = args[0]
        v = parse_setting(m, key, args[1])
        old = m.get(key)
        if self.station:
            why = None
            if key == "touch_mm" and v < 3.0:
                why = (f"touch_mm {v:g}: the tip goes to {v:g} mm above the lid top"
                       + (" and presses into it" if v < 0 else ", nearly touching it") + " on every dab/touch")
            elif key == "all_slow" and v is False:
                why = "all_slow false: guided moves other than descents and lifts use cellbot's normal (faster) speed"
            if why:
                self.out(why)
                if not await self.yes_no(f"Set {key} = {v}?"):
                    self.out("cancelled, nothing changed")
                    return
        m[key] = v
        self.out(f"{key}: {old} -> {v}  (this session only; guided.toml unchanged)")

    # -------------------------------------------------------------- shell
    def shell(self, loop: asyncio.AbstractEventLoop):
        try:
            import readline  # noqa: F401  (line editing and history when available)
        except ImportError:
            pass
        self.in_shell = True
        self.out("type help for the commands, quit to leave")
        prompt = "cellguide[STATION]> " if self.station else "cellguide[sim]> "
        import threading
        runner = threading.Thread(target=loop.run_forever, daemon=True)
        self._runner = runner
        runner.start()                      # keeps the Station reader alive while we wait for typing
        try:
            while True:
                try:
                    line = self.input(prompt)
                except (EOFError, KeyboardInterrupt):
                    self.out("")
                    break
                fut = asyncio.run_coroutine_threadsafe(self.execute(line), loop)
                try:
                    if not fut.result():
                        break
                except KeyboardInterrupt:   # Ctrl+C while a command runs = emergency stop
                    self.out("\nCtrl+C: stopping")
                    if self.station:
                        try:
                            asyncio.run_coroutine_threadsafe(self.cell.stop("Ctrl+C in the CLI"), loop).result(5)
                            self.out("E-STOP latched, both arms frozen. A human releases it: python -m cellbot.estop release")
                        except Exception as e:  # noqa: BLE001
                            self.out(f"ERROR: stop failed: {e!r}  -> cut the 12 V power")
                    try:
                        fut.result(10)
                    except BaseException:  # noqa: BLE001
                        pass
        finally:
            loop.call_soon_threadsafe(loop.stop)
            runner.join(3)


def shutdown_loop(loop: asyncio.AbstractEventLoop, cell=None, runner=None) -> None:
    """Quiet, clean exit: stop the loop thread, close the Station client if it can, cancel every pending task
    (Station client read/write/keep-alive loops, cellbot's reader), wait at most 2 s, then shut down async
    generators. Sends nothing to the arms (torque state is left as it is). Never raises."""
    try:
        loop.set_exception_handler(lambda _loop, _ctx: None)      # last resort: no leftover noise
    except Exception:  # noqa: BLE001
        pass
    try:
        if runner is not None and runner.is_alive():
            loop.call_soon_threadsafe(loop.stop)
            runner.join(3)
        if loop.is_running() or loop.is_closed():
            return
        client = getattr(getattr(cell, "backend", None), "client", None)
        for name in ("aclose", "close"):
            fn = getattr(client, name, None) if client is not None else None
            if callable(fn):
                try:
                    r = fn()
                    if asyncio.iscoroutine(r):
                        loop.run_until_complete(asyncio.wait_for(r, 2))
                except Exception:  # noqa: BLE001
                    pass
                break
        tasks = [t for t in asyncio.all_tasks(loop) if not t.done()]
        for t in tasks:
            t.cancel()
        if tasks:
            gathered = asyncio.gather(*tasks, return_exceptions=True)
            loop.run_until_complete(asyncio.wait([gathered], timeout=2.0))
        loop.run_until_complete(loop.shutdown_asyncgens())
    except Exception:  # noqa: BLE001
        pass


def make_cell(frames_dir: str | None = None, source: str | None = None):
    cell = gc.GuidedCell()
    if source:                                   # --source: this run only, guided.toml is not changed
        cell.gcfg = {**cell.gcfg, "vision": {**cell.gcfg["vision"], "source": source}}
    frames = None
    if frames_dir:
        if cell.backend.name != "sim":
            raise SystemExit("--frames is for simulation/testing only (CELLBOT_MODE=sim); refused with the real "
                             "Station")
        frames = FolderFrames(frames_dir)
        V = gc.gconfig.load_module("part_vision", cell.gcfg["_paths"]["vision_file"])
        cell.vision = FrameVision(V, frames, f"frames in {frames.folder}")
    return cell, frames


def main(argv=None, input_fn=input, out=print) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    frames_dir, yes, source = None, False, None
    while argv and argv[0].startswith("--"):
        opt = argv.pop(0)
        if opt == "--frames" and argv:
            frames_dir = argv.pop(0)
        elif opt == "--source" and argv:
            source = argv.pop(0)
        elif opt == "--yes":
            yes = True
        elif opt in ("--help", "-h"):
            argv = ["help"]
            break
        else:
            out(f"unknown option {opt}")
            return 2
    if not argv:
        argv = ["help"]
    from cellbot.config import load_config
    if yes and load_config().station["mode"] != "sim":
        out("REFUSED: --yes is not allowed with the real Station: every motion must be confirmed by a person.")
        return 2
    try:
        cell, frames = make_cell(frames_dir, source)
    except Exception as e:  # noqa: BLE001  e.g. norma-core or Station config missing
        out(f"ERROR: could not set up the cell: {e!r}")
        return 1
    cli = CLI(cell, yes, input_fn, out, frames)
    if yes and cli.station:
        out("REFUSED: --yes is not allowed with the real Station: every motion must be confirmed by a person.")
        return 2
    out(cli.banner())
    loop = asyncio.new_event_loop()
    try:
        if argv[0].lower() == "shell":
            cli.shell(loop)
            return 0
        line = " ".join(shlex.quote(a) for a in argv)
        printed = []

        def capture(s=""):
            printed.append(str(s))
            out(s)
        cli.out = capture
        loop.run_until_complete(cli.execute(line))
        failed = any(p.startswith(("REFUSED:", "VISION:", "ERROR:")) for p in printed)
        return 1 if failed else 0
    finally:
        try:
            cell.close_vision()                  # camera:N: release the camera
        except Exception:  # noqa: BLE001
            pass
        shutdown_loop(loop, cell, cli._runner)
        try:
            loop.close()
        except Exception:  # noqa: BLE001
            pass


if __name__ == "__main__":
    sys.exit(main())
