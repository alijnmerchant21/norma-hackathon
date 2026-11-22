#!/usr/bin/env python3
"""Build the guided-cell evidence pack (markdown + CSV + pictures) from the cell's real logs.

    python3 demo/make_pack.py --state <state dir> --transcript <claude run .txt> --out demo/pack

--state holds guided_log.jsonl, events.jsonl, cli/*.jpg and, once the automatic inspection has run, marks/*
and guided_marks.jsonl. Every input is optional: a missing file gives an empty-state note, never a crash.
--code is the guided_cell folder (default: the parent of this script's folder); it is only READ, for config
values and short code excerpts. Standard library + Pillow (for the picture crops; skipped if missing).
"""
from __future__ import annotations

import argparse
import base64
import datetime as dt
import html
import io
import json
import re
import sys
from pathlib import Path

try:
    import tomllib
except ImportError:  # Python 3.10
    try:
        import tomli as tomllib
    except ImportError:
        tomllib = None

try:
    from PIL import Image
except ImportError:
    Image = None

try:
    from zoneinfo import ZoneInfo
    TZ = ZoneInfo("Europe/Berlin")      # events.jsonl is written in Berlin local time
except Exception:
    TZ = dt.timezone(dt.timedelta(hours=1))



# ------------------------------------------------------------------ inputs
def read_jsonl(p: Path) -> list[dict]:
    out = []
    if not p or not p.exists():
        return out
    for ln in p.read_text(errors="replace").splitlines():
        ln = ln.strip()
        if ln:
            try:
                out.append(json.loads(ln))
            except json.JSONDecodeError:
                pass
    return out


def read_toml(p: Path) -> dict:
    if tomllib is None or not p.exists():
        return {}
    try:
        return tomllib.loads(p.read_text())
    except Exception:
        return {}


def local(t: float) -> dt.datetime:
    return dt.datetime.fromtimestamp(t, TZ)


def ev_time(e) -> dt.datetime | None:
    try:
        return dt.datetime.fromisoformat(e["t"])
    except Exception:
        return None


def hms(e) -> str:
    t = ev_time(e)
    return t.strftime("%H:%M:%S") if t else "?"


def excerpt(path: Path, start: str, n: int, keep: list[str] | None = None) -> tuple[str, int]:
    """n lines from the first line containing `start`; with keep, only lines containing one of those strings
    (plus the first line). Returns (text, first line number) or ("", 0)."""
    if not path.exists():
        return "", 0
    lines = path.read_text(errors="replace").splitlines()
    for i, ln in enumerate(lines):
        if start in ln:
            block = lines[i:i + n]
            if keep:
                block = [block[0]] + [b for b in block[1:] if any(k in b for k in keep)]
            ind = min((len(b) - len(b.lstrip()) for b in block if b.strip()), default=0)
            return "\n".join(b[ind:] for b in block), i + 1
    return "", 0


def crop_jpeg(p: Path, box=(560, 170, 890, 420), width=440, q=80) -> str | None:
    """Crop (in 1280x720 pixel units, scaled to the real size) and downscale, as a data URI."""
    if Image is None or not p.exists():
        return None
    try:
        im = Image.open(p).convert("RGB")
        sx, sy = im.width / 1280, im.height / 720
        if box:
            im = im.crop((int(box[0] * sx), int(box[1] * sy), int(box[2] * sx), int(box[3] * sy)))
        h = round(im.height * width / im.width)
        im = im.resize((width, h), Image.LANCZOS)
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=q, optimize=True)
        return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
    except Exception:
        return None


# --------------------------------------------------------------- transcript
def parse_transcript(p: Path) -> dict:
    """Claude Code CLI transcript -> {prompt, steps:[{calls, failed, text, human}], report_lines}."""
    out = {"prompt": "", "steps": [], "report": [], "header": ""}
    if not p or not p.exists():
        return out
    lines = p.read_text(errors="replace").splitlines()
    for ln in lines[:4]:
        if "Opus" in ln or "Claude Code" in ln:
            out["header"] += ln.strip() + " "
    i = 0
    while i < len(lines) and not lines[i].startswith("❯"):
        i += 1
    if i < len(lines):
        out["prompt"] = lines[i].lstrip("❯ ").strip()
    blocks, cur = [], None
    for ln in lines[i + 1:]:
        m = re.match(r"\s*Called (\w+)(?: (\d+) times)?(.*)", ln)
        if m:
            cur = {"server": m.group(1), "calls": int(m.group(2) or 1),
                   "failed": int((re.search(r"(\d+) failed", m.group(3)) or [0, 0])[1]), "body": []}
            blocks.append(cur)
            continue
        if ln.startswith("Answered Claude's questions"):
            cur = {"human": True, "body": []}
            blocks.append(cur)
            continue
        if cur is not None:
            cur["body"].append(ln)
    for b in blocks:
        paras, buf = [], []
        for ln in b["body"] + [""]:
            if ln.strip():
                buf.append(ln)
            elif buf:
                paras.append(buf)
                buf = []
        b["paras"] = paras
    for k, b in enumerate(blocks):
        last = k == len(blocks) - 1
        if b.get("human"):
            txt = " ".join(x.strip() for x in (b["paras"][0] if b["paras"] else []))
            txt = txt.lstrip("· ").strip()
            q, _, a = txt.partition("→")
            out["steps"].append({"human": True, "question": q.strip(), "answer": a.strip()})
            continue
        if last and len(b["paras"]) > 1:
            out["steps"].append({"calls": b["calls"], "failed": b["failed"], "text": ""})
            out["report"] = [ln for p_ in b["paras"] for ln in p_ + [""]]
            continue
        text = " ".join(x.strip() for x in (b["paras"][0] if b["paras"] else []))
        out["steps"].append({"calls": b["calls"], "failed": b["failed"],
                             "text": re.sub(r"\s*·\s*summary$", "", text)})
    return out


# Tool names per step of the 22 Nov run. The CLI prints only "Called cellguide N times", so these are read from
# Claude's own narration around each call and checked against events.jsonl; they are applied only when the
# transcript has exactly this many steps (another transcript shows its steps without names).
KNOWN_RUN = [
    (["preflight", "get_status", "guided_status"], "read", []),
    (["look"], "read", []),
    (["go_home so101", "go_home elrobot"], "motion", []),
    (["load_part", "+1 call"], "policy", ["11:37:00", "11:38:49"]),
    (["look"], "read", []),
    (["new_part"], "read", ["11:39:06"]),
    (["alerts", "log_result"], "read", ["11:39:13"]),
    None,                                   # human answer
    (["load_part"], "policy", ["11:39:28", "11:40:49"]),
    (["look"], "read", []),
    (["new_part", "see_part"], "read", ["11:40:54"]),
    (["plan_program"], "read", []),
    (["run_program"], "motion", ["11:41:36"]),
    (["alerts"], "read", []),
    (["retreat", "see_part"], "motion", []),
    (["run_program"], "motion", ["11:42:06"]),
    (["alerts", "retreat", "inspect_part"], "motion", []),
    (["unload_part"], "policy", ["11:42:23", "11:44:10"]),
    (["look"], "read", []),
    (["log_result", "finish_part"], "read", ["11:44:21"]),
]


# ------------------------------------------------------------------ data
def is_sim(r) -> bool:
    """A dab whose reached joints equal the commanded ones to 1e-9 came from the simulator."""
    qc, qr = r.get("q_cmd"), r.get("q_reached")
    return bool(qc and qr) and max(abs(a - b) for a, b in zip(qc, qr)) < 1e-9


def height_rows(glog: list[dict], since: str | None) -> tuple[list[dict], str]:
    """One row per real dab of the chosen day. Closed-loop dabs carry their own measured error fields; older
    lines only carry aimed_m and tip_reached_m, so their error is the robot-z difference (computed here)."""
    real = [r for r in glog if not is_sim(r) and "t" in r]
    if not real:
        return [], ""
    day = since or local(real[-1]["t"]).strftime("%Y-%m-%d")
    rows = []
    for r in real:
        if local(r["t"]).strftime("%Y-%m-%d") < day:
            continue
        row = {"time": local(r["t"]).strftime("%H:%M"), "part": r.get("part", "?"), "target": r.get("target", "?")}
        if "pretouch_error_before_mm" in r:
            row.update(kind="pretouch", before=r["pretouch_error_before_mm"], after=r["pretouch_error_after_mm"],
                       rounds=r.get("pretouch_correction_rounds"), touch=r.get("touch_error_mm"))
        elif "touch_error_before_mm" in r:
            row.update(kind="touch", before=r["touch_error_before_mm"], after=r["touch_error_after_mm"],
                       rounds=r.get("touch_correction_rounds"))
        elif r.get("aimed_m") and r.get("tip_reached_m"):
            row.update(kind="open", before=None, after=round((r["tip_reached_m"][2] - r["aimed_m"][2]) * 1000, 1))
        else:
            continue
        rows.append(row)
    return rows, day




def save_crop(src: Path, dst: Path, box=(560, 170, 890, 420), scale=2.0) -> bool:
    """Crop the box/hole area (1280x720 pixel units) and enlarge it, as PNG."""
    if Image is None or not src.exists():
        return False
    im = Image.open(src).convert("RGB")
    sx, sy = im.width / 1280, im.height / 720
    im = im.crop((int(box[0] * sx), int(box[1] * sy), int(box[2] * sx), int(box[3] * sy)))
    im = im.resize((int(im.width * scale), int(im.height * scale)), Image.LANCZOS)
    dst.parent.mkdir(parents=True, exist_ok=True)
    im.save(dst)
    return True


def md_table(head, rows) -> str:
    esc = lambda x: str(x).replace("|", "\\|").replace("\n", " ")
    return "\n".join(["| " + " | ".join(head) + " |", "|" + "---|" * len(head)] +
                     ["| " + " | ".join(esc(c) for c in r) + " |" for r in rows]) + "\n"


def write_csv(p: Path, head, rows):
    import csv
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(head)
        w.writerows(rows)


def box_table(lines) -> tuple[list, list]:
    rows = [[c.strip() for c in ln.strip().strip("│").split("│")] for ln in lines if ln.strip().startswith("│")]
    return (rows[0], rows[1:]) if rows else ([], [])


# ------------------------------------------------------------ fixed texts
PROJECT = ("**Guided cell.** Claude operates a two-arm robot cell (HackLab x BRISA Berlin Robotics & AI Hackathon, "
           "NormaCore track \"AI-Powered Robot Control\"). An SO-101 \"loader\" arm runs SmolVLA policies trained on "
           "the team's own NormaCore datasets to bring a red box to the work area and take it away. A 7-joint ElRobot "
           "\"worker\" arm with a marker dabs the holes on the box lid: a top camera sees the part, Claude writes a "
           "program in the part's own frame (which holes, in which order), and camera calibration + inverse kinematics "
           "turn each hole into joint targets. Claude never talks to the robot directly: Claude -> MCP tools "
           "(cellguide, {ntools} tools) -> cellbot guard layer -> NormaCore Station API (TCP 8888) -> servos, and a "
           "human approves every motion call. After each run the camera picture is measured and the next aim corrected. "
           "Team: Max (kinematics/IK, integration), Ali (cellbot: guarded Station I/O + MCP server), Manasa (vision: "
           "red box, holes, tracker).")

WORKS = """**What works**
- Claude runs a whole part through the MCP tools: preflight, writes and dry-runs a program (plan_program), runs it, reads alerts, asks the human before a retry, and reports failures as failures.
- Guards stop bad steps on real hardware today: 8.8 mm sag abort, capped height correction, \"no hole 2 on this part\", other arm in the work area, stale Station data, no vision lock (see 02_program_and_guardrails/guardrails.md).
- Vision -> part frame -> IK: hole 0 miss 24.3 mm -> about 5.6 mm -> about 2-3 mm (last value a visual estimate) after two corrections.
- Closed-loop height control: errors before correction up to +15.6 mm, after correction under 2 mm in every logged dab (joint-feedback measurement).
- Load policy: box into the work area on the second try in the full Claude run.

**What does not work yet**
- Pen height at the lid: in the full run the log said 1.2 mm touch error, but the pen stayed 5.2 mm above the lid (no ink); another hole sagged 8.8 mm. Tool length / droop / part height need recalibration.
- Unload policy timed out with the box still in the work area; grasp releases after lifting ~5 cm.
- Hybrid loader (taught pose, then policy) is built, not tried on the arm.
- Automatic \"diff\" inspection -> MissModel: validated on one real picture (dots within 0.3 mm), not yet a full run on the cell. Today's two aim corrections were applied outside the MissModel (corr_mm = 0.0 in every guided_log line).
"""

FLYWHEEL = """# Data flywheel

## What is recorded after each run
- `guided_log.jsonl` (one line per completed dab): part id, target, target pixel, box angle, aimed point (m), learned correction applied (`corr_mm`), commanded / sent / reached joints, droop offset, height error before and after the closed-loop correction (touch or pre-touch), correction rounds, tip position reached, seconds. See `height_error.csv`.
- `events.jsonl`: every guard refusal, failed step, policy start/end with exit code, operator exceptions, and the `finish_part` outcome with Claude's notes.
- Top-camera pictures (`state/cli/*.jpg`; with the automatic inspection also `state/marks/<time>_<part>_{{before,after,diff,result}}`).
- With inspection: `guided_marks.jsonl` + `guided_learning.json` (MissModel): per mark the miss in mm, the correction that was applied, and the arm's own error = miss - applied.

## How the next run uses it
- **Aim.** MissModel (cellguide/learn.py) estimates the arm's error as a function of position on the table (mean offset with fewer than 8 marks, plane fit from 8), and shifts the next target by -gain x error (`[marks] gain` = {gain}, capped at {maxc} mm). The IK applies the shift; no joint Jacobian needed.
- **Height.** The droop offset learned while settling is carried into the next move; the pre-touch correction of one hole seeds the next hole of the same program.
- **Today, by hand.** The aim at hole 0 was corrected twice from the camera pictures (`aim_miss_hole0.csv`): 24.3 mm -> about 5.6 mm -> about 2-3 mm (visual estimate). These corrections were applied outside the MissModel: `corr_mm` is [0.0, 0.0] in all {nlog} guided_log lines.

## The automatic part: "diff" inspection (built today)
1. Before the dabs, a clear look at the part is kept as the reference picture.
2. After the dabs, `inspect_part` registers the lid between the two pictures (hole centres, then ECC refinement) and finds NEW dark dots (cellguide/marks_dark.py), so a black marker on black rings still shows.
3. Each dot is assigned to its target; the miss in mm is logged per hole.
4. MissModel turns those misses into the next aim correction.

**Status (honest):** built today and switched on (`[marks] method = "{method}"` in guided.toml). Validated on one real picture (`pictures/original/20261122_105310_840_inspect.jpg`, dots inpainted for the "before" image, tests/test_marks_dark.py part B): dots found within 0.3 mm. It has **not yet completed a full run on the cell**: in Claude's full run no ink reached the lid, so inspection found 0 marks and the model got no data. MissModel marks in the logs now: {nmarks}.

## Further runs
`make_pack.py` picks up `guided_marks.jsonl` and `state/marks/*` automatically; when present they appear below and in `inspections.csv`.

{further}
"""

LOADER_STAGES = [
    ("1", "Reference NormaCore runner crashed on a dtype mismatch -> own runner (policy_run.py). Dry run OK, ~0.3 s per prediction on the Mac.", "fixed", "team run notes"),
    ("2", "20 actions per prediction: reached the box, then hesitated and stalled.", "replaced", "team run notes"),
    ("3", "50 actions per prediction: approach and grasp good.", "kept (--execute 50)", "team run notes; guided.toml policy_load_cmd"),
    ("4", "Stopped by the jump guard at the grasp (gripper command 320 ticks from last goal, limit 300) -> separate gripper limit added.", "fixed (--gripper-max-delta-ticks, default 1200)", "team run notes; policy_run.py"),
    ("5", "Grasp, then released after lifting ~5 cm.", "OPEN problem", "team run notes"),
    ("6", "Full Claude-operated run: load failed on try 1 (time limit, box not moved), succeeded on try 2; unload timed out with the box still in the work area.", "partly working", "events.jsonl, transcript"),
]


def guard_rows(events, gcfg, ccfg, refused_hole):
    g, m, ld = ccfg.get("guard", {}), gcfg.get("motion", {}), gcfg.get("loader", {})
    sel = lambda f: [e for e in events if f(e)]
    q = lambda lst: (f'events.jsonl {lst[0]["t"]}: "{lst[0]["message"]}"' + (f" ({len(lst)} times in events.jsonl)" if len(lst) > 1 else "")) if lst else ""
    zone = sel(lambda e: "work area is not free" in e.get("message", ""))
    stale = sel(lambda e: "stale" in e.get("message", ""))
    nolock = sel(lambda e: "did not lock onto a part" in e.get("message", ""))
    sag = sel(lambda e: "sagged" in e.get("message", ""))
    capd = sel(lambda e: "correction round" in e.get("message", ""))
    tmo = sel(lambda e: e.get("kind") == "exception" and "time limit" in e.get("message", ""))
    nothing = sel(lambda e: "nothing was marked" in e.get("message", ""))
    env = ccfg.get("arms", {}).get("so101", {}).get("envelope", {}).get("shoulder_lift", "?")
    R = [
        ("Human approves every motion call", "Claude Code permission prompt + operator rule", "cellguide/operator_prompt.py (Rules)",
         "unreviewed motion; retries nobody agreed to",
         'transcript: Claude asked "How should I continue?" after the failed load; human answered "Retry policy once". (The individual approval prompts are not in the transcript text.)'),
        ("One arm in the work area at a time", "other arm must be in zone 'clear'", "third_party/cellbot/cellbot/cell.py _check_zone",
         "the two arms colliding over the part", q(zone)),
        ("Learned-policy turn-taking", "SO-101 zone 'unknown' from policy start until go_home succeeds", "cellguide/loader.py (zone set before start)",
         "worker entering while an unguarded policy drives the SO-101", "same refusals as the row above (\"so101 is at an unknown position\")" if zone else ""),
        ("Station data must be fresh", "joint data older than 1 s refused", "third_party/cellbot/cellbot/station_backend.py _motors",
         "moving from a stale idea of where the joints are", q(stale)),
        ("Vision must lock onto the part", f"live state <= {gcfg.get('vision', {}).get('max_state_age_s', '?')} s, reference line + all holes", "cellguide/guided_cell.py new_part; guided.toml [vision] max_state_age_s",
         "aiming at a part that is absent or not understood", q(nolock)),
        ("Target must exist on this part", "step hole id checked against holes vision sees", "cellguide/guided_cell.py _target_px",
         "dabbing where Claude assumed a hole", q([refused_hole]) if refused_hole else ""),
        ("Sag abort", f"guided.toml touch_sag_abort_mm = {m.get('touch_sag_abort_mm', '?')} now (6 at the 11:41 run per transcript: \"abort limit is 6 mm\")", "guided.toml [motion] touch_sag_abort_mm; guided_cell.py _touch_correct / work_on",
         "pen digging into the lid / pushing the part", q(sag)),
        ("Height-correction caps", f"touch_correct_rounds = {m.get('touch_correct_rounds', '?')}, touch_correct_max_deg = {m.get('touch_correct_max_deg', '?')}, touch_tol_mm = {m.get('touch_tol_mm', '?')} (current file; log lines show values at the time)", "guided.toml [motion]",
         "a correction loop winding a joint far from its plan", q(capd)),
        ("Policy time limit", f"--max-seconds 90 per policy run; killed after policy_timeout_s = {ld.get('policy_timeout_s', '?')} s", "guided.toml [loader]; cellguide/loader.py",
         "a learned policy running on indefinitely", q(tmo)),
        ("Policy jump guard", "goal jump <= 300 ticks on arm joints (--max-delta-ticks); gripper own limit, default 1200", "policy_run.py delta_over_limit",
         "a policy chunk snapping the arm", "team run notes (policy test today, not in these log files): stopped at the grasp, gripper command 320 ticks from last goal > 300; gripper limit then added"),
        ("Inspect needs marks", "inspect_part refuses before any dab", "cellguide/guided_cell.py inspect", "reporting nothing as a measurement", q(nothing)),
        ("Commanded tip floor", f"commanded tip never below touch height - {m.get('touch_cmd_below_mm', '?')} mm", "guided.toml [motion] touch_cmd_below_mm",
         "a correction commanding the tip into the lid", ""),
        ("Joint limits", f"limit_margin = {g.get('limit_margin', '?')} (normalised) from each end; so101 shoulder_lift envelope {env}", "cell.guided.toml [guard], [arms.so101.envelope]; cell.py _check_limits",
         "end-stop hits", ""),
        ("Step size", f"max_delta_per_call = {g.get('max_delta_per_call', '?')}; waypoints interp_step = {g.get('interp_step', '?')}", "cell.guided.toml [guard]",
         "large unchecked jumps", ""),
        ("Over-current freeze + e-stop", f"current_limit = {g.get('current_limit', '?')} (raw ST3215)", "cell.guided.toml [guard]; cell.py _wait_reached",
         "pushing through a collision", ""),
        ("Reach / timeout", f"reach_tolerance = {g.get('reach_tolerance', '?')}, move_timeout_s = {g.get('move_timeout_s', '?')}", "cell.guided.toml [guard]",
         "a stuck joint reported as success", ""),
        ("Reachability / clearance", f"ik_tol_mm = {m.get('ik_tol_mm', '?')}, workspace_margin_mm = {m.get('workspace_margin_mm', '?')}, clearance_mm = {m.get('clearance_mm', '?')}", "guided.toml [motion]; guided_cell.py _plan",
         "unreachable targets, low paths over the part", ""),
        ("E-stop latch", "no motion while latched; only a human releases it", "third_party/cellbot/cellbot/cell.py _check_estop", "motion after an emergency", ""),
    ]
    return [(a, b, c, d, "fired" if e else "config only", e or "config only") for a, b, c, d, e in R]


def build_pack(args):
    out, state, code = args.out, args.state, args.code
    glog = read_jsonl(state / "guided_log.jsonl") if state else []
    events = read_jsonl(state / "events.jsonl") if state else []
    marks_log = read_jsonl(state / "guided_marks.jsonl") if state else []
    tr = parse_transcript(args.transcript)
    gcfg, ccfg = read_toml(code / "guided.toml"), read_toml(code / "cell.guided.toml")
    mk, ld = gcfg.get("marks", {}), gcfg.get("loader", {})
    ntools = sum(p.read_text().count("@mcp.tool") for p in (code / "cellguide/server.py", code / "third_party/cellbot/cellbot/mcp_server.py") if p.exists())
    ev_by = {}
    for e in events:
        ev_by.setdefault(hms(e), []).append(e)
    out.mkdir(parents=True, exist_ok=True)
    import shutil
    d1, d2, d3, dr = out / "01_control_flow", out / "02_program_and_guardrails", out / "03_data_flywheel", out / "raw_logs"
    for d in (d1, d2, d3, dr):
        d.mkdir(exist_ok=True)

    # ---------------- raw logs
    for n in ("guided_log.jsonl", "events.jsonl", "guided_marks.jsonl"):
        if state and (state / n).exists():
            shutil.copy2(state / n, dr / n)

    # ---------------- 01 control flow
    if args.transcript and args.transcript.exists():
        shutil.copy2(args.transcript, d1 / "transcript.txt")
    steps = tr["steps"]
    known = len(steps) == len(KNOWN_RUN)
    done = next((e for e in reversed(events) if e.get("kind") == "part_done"), None)
    rows, k = [], 0
    for i, s in enumerate(steps):
        if s.get("human"):
            rows.append(["(human)", f'Claude asked: "{s["question"]}"', "question to human", f'Human: "{s["answer"]}"'])
            continue
        k += 1
        hint = KNOWN_RUN[i] if known else None
        evs = [e for t in (hint[2] if hint else []) for e in ev_by.get(t, [])]
        tools = (", ".join(hint[0]) + f" [{hint[1]}]") if hint else f'{s["calls"]} call(s)'
        if s["failed"]:
            tools += f' ({s["failed"]} failed)'
        res = "; ".join(f'{hms(e)} {e.get("kind")}: {e.get("message", "")}' + (f' (exit {e["exit_code"]})' if "exit_code" in e else "") for e in evs)
        rows.append([str(k), s["text"] or ("(no narration printed; the final report follows)" if i == len(steps) - 1 else "(no narration printed; next block is Claude's question to the human)"), tools, res or "see Claude's next narration"])
    head, body = box_table(tr["report"])
    rep_tbl = md_table(head, body) if head else "(no table found in the transcript)\n"
    ncalls = sum(s.get("calls", 0) for s in steps if not s.get("human"))
    cf = f"""# Control flow of the real Claude-operated run

Source: `transcript.txt` (verbatim; {tr['header'].strip() or 'Claude Code CLI'}), cross-checked with `../raw_logs/events.jsonl`.
Part `{done.get('part') if done else '?'}`, outcome logged by Claude: **{done.get('message') if done else '?'}** ({done.get('t') if done else '?'}). {ncalls} tool calls in {k} steps + 1 question to the human.

Request given to Claude:
> {tr['prompt']}

**How to read the tool column:** the CLI prints only "Called cellguide N times", so tool names were read from Claude's narration around each call and checked against events.jsonl. [read] = read only, [motion] = arm motion through the guards (human-approved), [policy] = learned SO-101 policy.
"Result" quotes the matching events.jsonl lines (clock time, Berlin); where none exists, the result is in Claude's next narration.

{md_table(['#', 'What Claude said / decided (verbatim narration)', 'Tool(s) called', 'Result (events.jsonl)'], rows)}
## Final report table (verbatim from the transcript)

{rep_tbl}
Rest of the report (verbatim):

```
{chr(10).join(ln for ln in tr['report'] if not ln.strip().startswith(('┌', '│', '├', '└')))}
```

Logged by Claude with finish_part: {done.get('notes') if done else '?'}

## What this shows
- Claude ran a whole part end to end through MCP tools only: checks first (preflight, status), then load, look, plan, run, inspect, unload, log.
- It verified instead of trusting exit codes: the load policy "ended cleanly" twice, but Claude looked and found the box had not moved the first time.
- It asked the human before retrying the policy, and did not retry the unload because only the load retry was approved.
- Two dabs were refused by guards (sag 8.8 mm > 6 mm; height not reached within the correction cap); Claude re-looked (part had rotated) and re-planned the order instead of repeating the call.
- The report is honest: outcome "fail", no ink on the one dab that ran, the part left in the work area for a human, and a plausible root cause (pen-height model, not aim).
"""
    (d1 / "control_flow.md").write_text(cf)

    # ---------------- 02 program / call chain / guards / architecture
    refused_hole = next((e for e in events if "no hole" in e.get("message", "")), None)
    doc, _ = excerpt(code / "cellguide/server.py", "async def run_program", 15)
    prog = f"""# The program Claude writes

Claude never sends coordinates or joint values for the worker arm. It writes an ordered list of steps **in the part's own frame**: a hole id that vision assigned, or a point on the lid as fractions of its length and width. The tool converts each step to pixel -> table point -> IK joint targets when it runs.

## Tool signature and docstring (cellguide/server.py)
```python
{doc}
```

Each step: `{{"hole": <id>}}` or `{{"at": [fx, fy]}}`, optional `"action": "dab" | "hover"`, `"dwell_s"`, `"look": true`. `plan_program(steps)` checks the same list without moving (reachability, tool tilt, path, joint travel).

## In the real run (part {done.get('part') if done else '?'})
Reconstructed from Claude's narration (steps 12-16 in control_flow.md); **the exact step lists were not logged**:
```python
plan_program(steps)                       # "all 4 steps are reachable on direct paths, tool tilt is 11-14 deg"
run_program([{{"hole": 0}}, {{"hole": 1}}, ...], look_each_step=True)   # hole 0 dabbed, hole 1 refused (sag 8.8 mm)
# alerts -> retreat -> see_part: box had rotated; Claude re-ordered
run_program([{{"hole": 2}}, {{"hole": 3}}, {{"hole": 1}}], look_each_step=True)   # "starting with hole 2, leaving hole 1 for last"; hole 2 refused
```
Known for certain: the first run started with hole 0 then hole 1; order of the remaining steps not logged.

## A real step refused before any motion
{('events.jsonl ' + refused_hole['t'] + ': `' + refused_hole['message'] + '`') if refused_hole else '(none in the logs)'}
"""
    (d2 / "program.md").write_text(prog)

    ex = [
        ("1. MCP tool (what Claude calls)", "cellguide/server.py", "async def run_program", 16, ["async def run_program", "cell.run_program("]),
        ("2. Guard wrapper: refusals go back to Claude as text", "cellguide/server.py", "async def _run(coro)", 9, ["except GuardError", 'log_event("refused"', "REFUSED"]),
        ("3. Guided move: every worker motion passes cellbot's guards", "cellguide/guided_cell.py", "async def _guided_move", 22, ["_check_limits", "_station_fresh", "_check_estop", "_check_zone", "self._move("]),
        ("4. Zone guard (one arm in the work area)", "third_party/cellbot/cellbot/cell.py", "def _check_zone", 9, None),
        ("5. Waypointed, checked move", "third_party/cellbot/cellbot/cell.py", "async def _move", 22, ["_check_estop", "for i in range", "write_goals", "_wait_reached"]),
        ("6. NormaCore Station API: goals written as ST3215 sync_write commands", "third_party/cellbot/cellbot/station_backend.py", "async def write_goals", 18, ["async def", "RAM_GOAL_SPEED", "RAM_ACC", "RAM_GOAL_POSITION", "self._send("]),
        ("7. NormaCore Station API: client, joint-state stream", "third_party/cellbot/cellbot/station_backend.py", "from software.station.shared.station_py", 4, None),
        ("8. Station connection (TCP 8888) and state queue", "third_party/cellbot/cellbot/station_backend.py", "self.client = await self._new_client", 3, None),
        ("9. Stale-data guard", "third_party/cellbot/cellbot/station_backend.py", "def _motors", 4, None),
    ]
    cc = ["# Call chain: MCP tool -> guard layer -> NormaCore Station API\n",
          "Real code, shortened to the lines that matter (\"selected lines\" = lines in between omitted). Line numbers are where the excerpt starts in the file as of this build.\n",
          "Claude -> `run_program` (MCP) -> `GuidedCell.work_on` -> `_guided_move` (limits, Station freshness, e-stop, zone) -> cellbot `Cell._move` (waypoints, over-current, timeout) -> `StationBackend.write_goals` -> NormaCore `station_py.send_commands` -> ST3215 servos.\n"]
    for title, rel, start, n, keep in ex:
        txt, ln = excerpt(code / rel, start, n, keep)
        cc.append(f"## {title}\n`{rel}`, line {ln}{', selected lines' if keep else ''}\n```python\n{txt or '(not found)'}\n```\n")
    port = (code / "cellguide.toml") if False else None
    cc.append(f"Station host/port from `cell.guided.toml [station]`: host = {ccfg.get('station', {}).get('host')}, port = {ccfg.get('station', {}).get('port')} (web UI 8889), mode = {ccfg.get('station', {}).get('mode')}.\n")
    (d2 / "call_chain.md").write_text("\n".join(cc))

    G = guard_rows(events, gcfg, ccfg, refused_hole)
    head = ["guard", "limit", "source (file / key)", "prevents", "fired today?", "real instance (quoted) or config only"]
    write_csv(d2 / "guardrails.csv", head, G)
    nf = sum(1 for r in G if r[4] == "fired")
    (d2 / "guardrails.md").write_text(f"# Guardrails\n\n{nf} of {len(G)} guards fired for real in today's logs/transcript; the rest are shown from config only. "
                                      "Limit values are read from cell.guided.toml / guided.toml when the pack is built (they may have been changed after a run: the quoted log line shows the value at the time).\n\n"
                                      + md_table(head, G))
    (d2 / "architecture.md").write_text(f"""# Architecture

```mermaid
flowchart LR
  C[Claude<br/>Opus in Claude Code CLI] -->|MCP stdio| T[cellguide MCP server<br/>{ntools} tools]
  H[Human<br/>approves each motion call] -.-> C
  T --> G[Guard layer<br/>cellbot Cell + GuidedCell<br/>limits, zones, sag, caps, e-stop]
  G -->|TCP 8888| S[NormaCore Station API]
  S --> A1[SO-101 loader]
  S --> A2[ElRobot worker, 7 joints + marker]
  P[SmolVLA policy<br/>policy_run.py, own jump guards] -->|direct, outside guards;<br/>worker blocked by zone 'unknown'| S
  T -. load_part / unload_part .-> P
  CAM[Top camera 1280x720] --> V[Vision: red box, holes, tracker] --> F[Part frame<br/>hole id / lid fractions] --> IK[Calibration + IK] --> G
  CAM --> I[inspect: misses] --> L[MissModel] --> F
```

Plain text:
```
Claude --MCP--> cellguide tools ({ntools}) --> guard layer (cellbot) --TCP 8888--> NormaCore Station --> SO-101 + ElRobot
   ^ human approves every motion call
top camera -> vision (box, holes) -> part frame -> calibration + IK -> joint targets -> guard layer
after a run: camera -> measured misses -> MissModel -> next aim
SO-101 policy (SmolVLA) -> Station directly (outside the guards; the worker may not move until SO-101 is home)
```
Tool count: @mcp.tool in cellguide/server.py + third_party/cellbot/cellbot/mcp_server.py = {ntools}.
""")

    # ---------------- 03 flywheel
    pics = [("20261122_105310_840_inspect.jpg", "1", 24.3, "", "no", "raw picture; two first marker dots ~24 mm from hole 0"),
            ("20261122_105832_802_see.jpg", "2", 5.6, "", "approx (about 5.6, dot 5-6 mm)", "after the first correction"),
            ("20261122_110703_175_see.jpg", "3", 2.5, "2-3", "YES, visual estimate", "after the second correction; dots inside the ring of hole 0; hole 1 ~5-6 mm")]
    write_csv(d3 / "aim_miss_hole0.csv", ["round", "time", "miss_mm", "range_mm", "estimate", "picture", "note"],
              [[r, f[9:11] + ":" + f[11:13], v if not rg else "", rg, est, "pictures/original/" + f, note] for f, r, v, rg, est, note in pics])
    for f, *_ in pics:
        src = state / "cli" / f if state else None
        if src and src.exists():
            (d3 / "pictures/original").mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, d3 / "pictures/original" / f)
            save_crop(src, d3 / "pictures/crops" / (Path(f).stem + "_hole_area_2x.png"))

    hr = []
    for r in glog:
        base = [local(r["t"]).strftime("%Y-%m-%d %H:%M:%S") if "t" in r else "", r.get("part"), r.get("target"), "sim" if is_sim(r) else "real"]
        if "pretouch_error_before_mm" in r:
            hr.append(base + ["closed loop at pre-touch", r["pretouch_error_before_mm"], r["pretouch_error_after_mm"], r.get("pretouch_correction_rounds"), r.get("touch_error_mm"), "logged"])
        elif "touch_error_before_mm" in r:
            hr.append(base + ["closed loop at touch", r["touch_error_before_mm"], r["touch_error_after_mm"], r.get("touch_correction_rounds"), r["touch_error_after_mm"], "logged"])
        elif r.get("aimed_m") and r.get("tip_reached_m"):
            z = round((r["tip_reached_m"][2] - r["aimed_m"][2]) * 1000, 1)
            hr.append(base + ["open loop (no correction)", "", z, 0, z, "computed: robot z of tip_reached_m - aimed_m (commanded touch offset at the time not logged)"])
    write_csv(d3 / "height_error.csv", ["time_berlin", "part", "target", "sim_or_real", "method", "error_before_mm", "error_after_mm", "correction_rounds", "touch_error_mm", "source"], hr)

    runs, cur = [], None
    for e in events:
        msg = e.get("message", "")
        if e.get("kind") == "loader" and msg.endswith("started"):
            cur = {"phase": "unload" if "unload" in msg else "load", "start": ev_time(e), "end": None, "exit": "", "outcome": ""}
            runs.append(cur)
        elif e.get("kind") == "loader" and msg.endswith("ended") and cur:
            cur["end"], cur["exit"] = ev_time(e), e.get("exit_code")
        elif e.get("kind") == "exception" and cur and "_part" in msg:
            cur["outcome"] = msg
    run_rows = [[r["phase"], r["start"].strftime("%H:%M:%S") if r["start"] else "", int((r["end"] - r["start"]).total_seconds()) if r["end"] and r["start"] else "", r["exit"],
                 r["outcome"] or ("box reached the work area (Claude: look + new_part locked)" if r["phase"] == "load" else "")] for r in runs]
    write_csv(d3 / "loader_progress.csv", ["stage", "what happened", "status", "source"], LOADER_STAGES)
    write_csv(d3 / "loader_runs_in_cell.csv", ["phase", "start_berlin", "seconds_incl_model_load", "exit_code", "outcome"], run_rows)
    (d3 / "loader_progress.md").write_text("# Loader (SO-101 SmolVLA policy) progress today\n\n" + md_table(["stage", "what happened", "status", "source"], LOADER_STAGES)
        + "\nCurrent status: load works on some tries (1 of 2 in the full run), unload not yet, grasp-then-release open. "
          f"Hybrid mode (taught pre-pose under the guards, then the policy) is built (guided.toml policy_load_pre_pose = \"{ld.get('policy_load_pre_pose', '')}\", i.e. off) and not yet tried on the arm.\n\n"
          "## Policy runs inside the cell (events.jsonl)\n\n" + md_table(["phase", "start (Berlin)", "seconds incl. model load", "exit", "outcome logged"], run_rows)
        + "\nExit 0 means the runner stopped cleanly (time limit or arm still), not that the task was done. Policy command: --execute 50 --max-seconds 90 --stop-when-still 6.\n")

    further = "No automatic inspection results in the logs yet."
    shots = sorted((state / "marks").glob("*")) if state and (state / "marks").exists() else []
    if marks_log:
        mr = [[local(s["t"]).strftime("%Y-%m-%d %H:%M:%S") if "t" in s else "", s.get("part"), s.get("target"), round(sum(v * v for v in s.get("miss_mm", [0, 0])) ** .5, 2),
               s.get("miss_mm"), s.get("applied_mm"), s.get("error_mm"), s.get("method", "color")] for s in marks_log]
        head = ["time", "part", "target", "miss_mm", "miss_xy_mm", "applied_mm", "arm_error_mm", "method"]
        write_csv(d3 / "inspections.csv", head, mr)
        further = md_table(head, mr)
    if shots:
        (d3 / "pictures/marks").mkdir(parents=True, exist_ok=True)
        for p in shots:
            shutil.copy2(p, d3 / "pictures/marks" / p.name)
        further += f"\n{len(shots)} inspection pictures copied to pictures/marks/.\n"
    (d3 / "flywheel.md").write_text(FLYWHEEL.format(gain=mk.get("gain", "?"), maxc=mk.get("max_correction_mm", "?"), nlog=len(glog),
                                                   method=mk.get("method", "?"), nmarks=len(marks_log), further=further))

    # ---------------- README
    files = sorted(p.relative_to(out) for p in out.rglob("*") if p.is_file() and p.name != "README.md")
    desc = {
        "01_control_flow/transcript.txt": "verbatim Claude Code transcript of the real run",
        "01_control_flow/control_flow.md": "the run as a numbered timeline + final report table + what it shows",
        "02_program_and_guardrails/program.md": "shape of the program Claude writes (run_program), the real run's steps",
        "02_program_and_guardrails/call_chain.md": "MCP tool -> guards -> NormaCore Station API, real code excerpts with file/line",
        "02_program_and_guardrails/guardrails.md": "guard table: limit, source, what it prevents, real fired instance or config only",
        "02_program_and_guardrails/guardrails.csv": "same table as CSV",
        "02_program_and_guardrails/architecture.md": "mermaid + plain-text architecture diagram",
        "03_data_flywheel/aim_miss_hole0.csv": "hole 0 aim miss per correction round, picture, estimate flag",
        "03_data_flywheel/height_error.csv": "every guided_log dab: height error before/after closed loop, rounds",
        "03_data_flywheel/loader_progress.md": "the six loader stages + policy runs in the cell",
        "03_data_flywheel/loader_progress.csv": "the six loader stages",
        "03_data_flywheel/loader_runs_in_cell.csv": "policy runs from events.jsonl (duration, exit, outcome)",
        "03_data_flywheel/flywheel.md": "what is recorded, how the next run uses it, diff inspection status",
        "03_data_flywheel/inspections.csv": "automatic inspection results (MissModel log)",
        "raw_logs/guided_log.jsonl": "raw: one line per dab",
        "raw_logs/events.jsonl": "raw: cell events, refusals, loader, part outcomes",
        "raw_logs/guided_marks.jsonl": "raw: MissModel marks",
    }
    lines = []
    for f in files:
        s = str(f)
        d = desc.get(s) or ("cropped, 2x enlarged hole area" if "/crops/" in s else "original top-camera picture (1280x720)" if "/original/" in s else "inspection picture" if "/marks/" in s else "")
        lines.append(f"- `{s}`" + (f": {d}" if d else ""))
    (out / "README.md").write_text(f"""# Guided cell: evidence pack

{PROJECT.format(ntools=ntools)}

All numbers come from the cell's logs (`raw_logs/`), its config files, the run transcript, or the team's run notes (marked as such); estimates are labelled.
Built {dt.datetime.now(TZ).strftime('%Y-%m-%d %H:%M')} (Berlin) by `demo/make_pack.py` from {len(glog)} guided_log lines, {len(events)} events, {len(marks_log)} MissModel marks.

## Files
{chr(10).join(lines)}

## What works / what does not yet
{WORKS}
## Regenerate after a new run
```
python3 demo/make_pack.py --state <guided_cell>/state --transcript <run transcript .txt> --out demo/pack
```
""")
    return out


def main(argv=None):
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description="Build the guided-cell evidence pack from the logs")
    ap.add_argument("--state", type=Path, default=None, help="dir with guided_log.jsonl, events.jsonl, cli/, marks/")
    ap.add_argument("--transcript", type=Path, default=None, help="Claude Code CLI transcript of a run (.txt)")
    ap.add_argument("--out", type=Path, default=here / "pack")
    ap.add_argument("--code", type=Path, default=here.parent, help="guided_cell folder (read only)")
    args = ap.parse_args(argv)
    if args.state and not args.state.exists():
        print(f"warning: --state {args.state} not found; building with empty logs", file=sys.stderr)
        args.state = None
    o = build_pack(args)
    print(f"wrote {o}")


if __name__ == "__main__":
    main()
