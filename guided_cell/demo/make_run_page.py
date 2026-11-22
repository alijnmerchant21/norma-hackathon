#!/usr/bin/env python3
"""Guided cell: one-page, human-friendly run report (self-contained HTML).

    python demo/make_run_page.py --state <state dir> --transcript <txt> [--transcript <txt> ...] \
        [--part <id>] --out <html>

Sections 1-7 come only from the cell's logs (events.jsonl, guided_log.jsonl, guided_marks.jsonl,
guided_learning.json, loader_episodes.jsonl, marks/) and the operator transcript(s). Section 8 is a
SIMULATION that runs the real cellguide.learn.MissModel; it is tagged as such on the page.
Standard library + numpy + Pillow.
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
import tempfile
from pathlib import Path

import numpy as np

try:
    from PIL import Image, ImageDraw
except Exception:  # pragma: no cover
    Image = None

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from cellguide.learn import MissModel  # noqa: E402

E = html.escape

# Known code changes on the cell today (team notes; times local).
CHANGES = [("10:15", "closed-loop height"), ("11:25", "gentle touch"), ("12:20", "arrival-height fix"),
           ("12:30", "per-hole learning"), ("13:00", "camera auto-detect"), ("14:22", "sideways correction")]


# ------------------------------------------------------------------ reading
def jl(p: Path) -> list[dict]:
    if not p.exists():
        return []
    out = []
    for ln in p.read_text(errors="replace").splitlines():
        try:
            out.append(json.loads(ln))
        except Exception:
            pass
    return out


def ts(x) -> dt.datetime:
    if isinstance(x, (int, float)):
        return dt.datetime.fromtimestamp(x)
    return dt.datetime.fromisoformat(str(x)[:19])


def hm(x) -> str:
    return ts(x).strftime("%H:%M:%S")


def is_sim(r) -> bool:
    qc, qr = r.get("q_cmd"), r.get("q_reached")
    return bool(qc and qr) and max(abs(a - b) for a, b in zip(qc, qr)) < 1e-9


def touch_err(r):
    if "touch_error_mm" in r:
        return r["touch_error_mm"]
    if r.get("aimed_m") and r.get("tip_reached_m"):
        return round((r["tip_reached_m"][2] - r["aimed_m"][2]) * 1000, 1)
    return None


def load_state(sd: Path) -> dict:
    learn = json.loads((sd / "guided_learning.json").read_text())["samples"] if (sd / "guided_learning.json").exists() else []
    return dict(events=jl(sd / "events.jsonl"), glog=jl(sd / "guided_log.jsonl"), marks=jl(sd / "guided_marks.jsonl"),
                learn=learn, eps=jl(sd / "loader_episodes.jsonl"), sd=sd)


def parse_transcripts(paths: list[Path]) -> dict:
    text = "\n".join(p.read_text(errors="replace") for p in paths if p and p.exists())
    prog = {}
    for m in re.finditer(r"[Pp]rogram:?\s*(\[\{.*?\}\])", text):
        try:
            prog[json.dumps(json.loads(m.group(1)))] = json.loads(m.group(1))
        except Exception:
            pass
    return dict(text=text, programs=list(prog.values()),
                loader_lines=[l.strip() for l in text.splitlines() if l.strip().startswith(("LOADER |", "RECORDING |"))])


# ------------------------------------------------------------------ the run
def run_window(st, part):
    ev = st["events"]
    done = [e for e in ev if e["kind"] == "part_done" and e.get("part") == part]
    start = [e for e in ev if e["kind"] == "part" and e.get("part") == part and "started" in e["message"]]
    if not start:
        raise SystemExit(f"part {part} not found in events.jsonl")
    t1 = ts(done[-1]["t"]) if done else ts(ev[-1]["t"])
    prev = [e for e in ev if e["kind"] == "part_done" and ts(e["t"]) < ts(start[0]["t"])]
    t_prev = ts(prev[-1]["t"]) if prev else ts(start[0]["t"]) - dt.timedelta(minutes=10)
    # the run begins with the first load attempt after the previous part was closed
    loads = [e for e in ev if ts(e["t"]) > t_prev and e["kind"] == "loader" and "policy load started" in e["message"]]
    t0 = ts(loads[0]["t"]) if loads else ts(start[0]["t"])
    return t0, t1


def pick_part(st) -> tuple[str, str]:
    """The most complete real run: score each closed part on what it exercised."""
    best, why = None, ""
    for e in st["events"]:
        if e["kind"] != "part_done":
            continue
        p = e["part"]
        try:
            t0, t1 = run_window(st, p)
        except SystemExit:
            continue
        dabs = [r for r in st["glog"] if r.get("part") == p and not is_sim(r)]
        marks = [m for m in st["marks"] if m["part"] == p]
        eps = [x for x in st["eps"] if t0 <= ts(x["t_start"]) <= t1]
        labelled = [x for x in eps if x.get("outcome")]
        evs = [x for x in st["events"] if t0 <= ts(x["t"]) <= t1]
        stages = dict(load=any(x["phase"] == "load" and x.get("outcome") for x in eps), dabs=bool(dabs),
                      guard=any(x["kind"] in ("refused", "guided_fail", "estop") for x in evs), marks=bool(marks),
                      unload=any(x["phase"] == "unload" and x.get("outcome") for x in eps),
                      closed_loop=any("hover_status" in r for r in dabs))
        score = (sum(stages.values()), len(marks), t1.timestamp())
        if best is None or score >= best[0]:
            best = (score, p, "every stage is in the logs: " + ", ".join(k.replace("_", " ") for k, v in stages.items() if v)
                    + f"; {len(dabs)} dab attempts, {len(marks)} measured mark(s), {len(labelled)} labelled loader attempts")
    return best[1], best[2]


def build_steps(st, part, t0, t1, tr):
    S = []
    add = lambda t, actor, what, result, status, why="": S.append(dict(t=ts(t), actor=actor, what=what, why=why,
                                                                       result=result, status=status))
    eps = {x["id"]: x for x in st["eps"]}
    ev = [e for e in st["events"] if t0 - dt.timedelta(seconds=1) <= ts(e["t"]) <= t1]
    dabs = {r["t"]: r for r in st["glog"] if r.get("part") == part}
    for r in dabs.values():
        if not r.get("ok", True):
            continue
        add(r["t"], "Worker arm ElRobot", f"Dab {r['target']}",
            f"touch height error {touch_err(r):+.1f} mm" + (f", learned correction {np.hypot(*r['corr_mm']):.1f} mm applied" if r.get("corr_mm") else ""),
            "ok", "Program step: camera fix, IK to joint targets, descend, touch, lift.")
    for e in ev:
        k, m = e["kind"], e["message"]
        if k == "loader":
            mm = re.match(r"attempt (\w+) labelled (\w+)", m)
            if mm:
                x = eps.get(mm.group(1), {})
                ok = mm.group(2) == "success"
                add(e["t"], "Camera", f"Label {x.get('phase','')} attempt {mm.group(1)} ({x.get('variant','')})",
                    f"{mm.group(2)}: {e.get('note','')}", "ok" if ok else "warning",
                    "Every loader attempt gets a camera-verified label so successful ones become training data.")
            elif m.endswith("started") and "pre-sequence" not in m:
                ph = "load" if "load started" in m and "unload" not in m else "unload"
                x = next((x for x in st["eps"] if x["phase"] == ph and ts(x["t_start"]) - dt.timedelta(seconds=3) <= ts(e["t"]) <= ts(x["t_end"])), {})
                add(e["t"], "Holder arm SO-101", f"{ph.capitalize()} by SmolVLA policy ({e.get('variant') or x.get('variant','primary')})",
                    f"ran {x.get('seconds','?')} s, exit code {x.get('exit_code')}", "ok" if x.get("exit_code") == 0 else "warning",
                    "Move the red box " + ("from pickup into the work area." if ph == "load" else "out of the work area."))
            elif "pre-sequence started" in m:
                add(e["t"], "Holder arm SO-101", "Fallback unload: taught moves, then policy",
                    "gripper open, sweep, grasp, close", "ok", "The primary unload failed; the fallback starts from a taught grasp.")
        elif k == "estop":
            add(e["t"], "Claude", "E-stop", m.split(":", 1)[-1].strip(), "refused", "A human hand was seen on the box.")
        elif k == "refused":
            add(e["t"], "Guard", "Command refused", m, "refused")
        elif k == "guided_fail":
            if "loader" in m:
                continue
            add(e["t"], "Guard", "Dab refused by height guard", m, "refused")
        elif k == "exception":
            add(e["t"], "Claude", "Logged exception for the human", m, "warning")
        elif k == "part" and "marking" in m:
            add(e["t"], "Claude", "Program started (each motion approved by a human)", "fresh camera look before each hole", "human",
                "Claude wrote the hole order; the cell turns each hole into joint targets.")
        elif k == "part" and "started" in m:
            add(e["t"], "Camera", f"Part {part} recognised", f"{e.get('holes')} holes mapped in the part's frame", "ok",
                "The program is written in the part's own frame, so the box angle does not matter.")
        elif k == "part_done":
            add(e["t"], "Claude", "Closed the part", f"logged as {m}", "warning" if m != "ok" else "ok")
    # human moments that only the transcript shows
    if "You released it" in tr["text"] and any(s["what"] == "E-stop" for s in S):
        te = next(s["t"] for s in S if s["what"] == "E-stop")
        t = min([s["t"] for s in S if s["t"] - te > dt.timedelta(seconds=60)] or [te]) - dt.timedelta(seconds=1)
        S.append(dict(t=t, actor="Human", what="Checked the cell, released the e-stop",
                      why="Only a human may release the latch.", result="preflight READY afterwards (time not logged; from the transcript)",
                      status="human", tlabel="later"))
    if "teleop" in tr["text"] and part in tr["text"] and "unload_part refused: variant 'teleop'" in json.dumps(ev):
        t = t1 - dt.timedelta(seconds=1)
        S.append(dict(t=t, actor="Human", what="Unloaded the part by teleoperation",
                      why="Unload by policy does not work yet; the teleop is itself a demonstration.",
                      result="work area empty; not logged (older cell server), so not yet a training episode",
                      status="human", tlabel="not logged"))
    S.sort(key=lambda s: s["t"])
    return S


# ------------------------------------------------------------------ pictures
def crop_part(st, part, px_list):
    if Image is None or not px_list:
        return None, None
    files = sorted((st["sd"] / "marks").glob(f"*_{part}_*"))
    pick = {k: next((f for f in files if f.name.endswith(k)), None) for k in ("before.jpg", "result.jpg")}
    xs, ys = [p[0] for p, _ in px_list], [p[1] for p, _ in px_list]
    pad = 140
    box = (max(0, min(xs) - pad), max(0, min(ys) - pad), min(1280, max(xs) + pad), min(720, max(ys) + pad))
    out = {}
    for k, f in pick.items():
        if not f:
            continue
        im = Image.open(f).convert("RGB")
        sx, sy = im.width / 1280, im.height / 720
        c = im.crop((int(box[0] * sx), int(box[1] * sy), int(box[2] * sx), int(box[3] * sy)))
        scale = 520 / c.width
        c = c.resize((520, round(c.height * scale)), Image.LANCZOS)
        if k == "before.jpg":
            d = ImageDraw.Draw(c)
            for (x, y), lab in px_list:
                cx, cy = (x - box[0]) * scale, (y - box[1]) * scale
                d.ellipse((cx - 13, cy - 13, cx + 13, cy + 13), outline=(232, 37, 26), width=3)
                d.rectangle((cx + 14, cy - 34, cx + 40, cy - 8), fill=(255, 255, 255))
                d.text((cx + 20, cy - 32), lab, fill=(20, 20, 20), font_size=20) if hasattr(d, "text") else None
        buf = io.BytesIO()
        c.save(buf, "JPEG", quality=78, optimize=True)
        out[k] = ("data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode(), f.name)
    return out, box


# ------------------------------------------------------------------ charts (inline SVG)
def svg_time(points, t_lo, t_hi, y_max, unit, changes=True, h=230, refline=None, ylab=""):
    W, H, L, R, T, B = 1100, h, 60, 16, 30, 34
    X = lambda t: L + (t - t_lo) / max(1e-9, t_hi - t_lo) * (W - L - R)
    Y = lambda v: T + (1 - v / y_max) * (H - T - B)
    o = [f'<svg viewBox="0 0 {W} {H}" role="img" preserveAspectRatio="xMidYMid meet">']
    step = 2 if y_max <= 10 else 5 if y_max <= 25 else 10
    for v in np.arange(0, y_max + 1e-9, step):
        o.append(f'<line x1="{L}" x2="{W-R}" y1="{Y(v):.1f}" y2="{Y(v):.1f}" class="grid"/><text x="{L-8}" y="{Y(v)+5:.1f}" class="ax" text-anchor="end">{v:g}</text>')
    o.append(f'<text x="4" y="{T-12}" class="ax">{ylab or unit}</text>')
    for hh in range(24):
        t = DAY0 + hh * 3600
        if t_lo <= t <= t_hi:
            o.append(f'<text x="{X(t):.1f}" y="{H-10}" class="ax" text-anchor="middle">{hh:02d}:00</text>')
    if changes:
        for i, (c, lab) in enumerate(CHANGES):
            hcc, mcc = map(int, c.split(":"))
            t = DAY0 + hcc * 3600 + mcc * 60
            if t_lo <= t <= t_hi:
                o.append(f'<line x1="{X(t):.1f}" x2="{X(t):.1f}" y1="{T}" y2="{H-B}" class="chg"/>')
    if refline is not None:
        o.append(f'<line x1="{L}" x2="{W-R}" y1="{Y(refline[0]):.1f}" y2="{Y(refline[0]):.1f}" class="ref"/><text x="{W-R}" y="{Y(refline[0])-6:.1f}" class="ax" text-anchor="end">{refline[1]}</text>')
    for t, v, cls, lab in points:
        vv = min(v, y_max)
        o.append(f'<circle cx="{X(t):.1f}" cy="{Y(vv):.1f}" r="6" class="{cls}"><title>{E(lab)}</title></circle>')
    o.append("</svg>")
    return "".join(o)


def change_legend(t_lo, t_hi):
    return "".join(f'<span class="chip"><b>{c}</b> {E(l)}</span>' for c, l in CHANGES)


DAY0 = 0.0


# ------------------------------------------------------------------ simulation (real MissModel)
def simulate(seed=7, runs=12):
    rng = np.random.default_rng(seed)
    holes = np.array([[-15.8, 3.0], [5.0, 18.0], [22.0, -6.0], [0.0, -18.0]])   # part frame, mm
    # aim error field of the size seen today: smooth in table position, 3-8 mm at the holes
    base = np.array([[4.5, -5.5], [-2.5, 2.0], [-3.0, 6.0], [5.0, 1.5]])
    def err(p_table, k):
        return base[k] + 0.04 * (p_table - holes[k])     # gentle gradient: placement jitter changes error a little
    d = Path(tempfile.mkdtemp(prefix="run_page_sim_"))
    m = MissModel(d)
    out = []
    for r in range(runs):
        jitter = rng.normal(0, 2.0, 2)                    # box placement jitter, mm
        misses = []
        for k in range(len(holes)):
            p = holes[k] + jitter                         # MissModel keys on position in table frame
            c = m.correction_mm(p)
            miss = c + err(p, k) + rng.normal(0, 0.5, 2)  # measurement noise 0.5 mm
            m.record(f"sim{r}", f"hole {k}", p, miss, c)
            misses.append(float(np.hypot(*miss)))
        out.append((r + 1, float(np.mean(misses)), float(np.max(misses))))
    return out, [float(np.hypot(*b)) for b in base]


def svg_sim(rows):
    W, H, L, R, T, B = 900, 260, 52, 20, 26, 40
    n, ym = len(rows), max(8.0, max(r[2] for r in rows) + 0.5)
    X = lambda i: L + (i - 1) / max(1, n - 1) * (W - L - R)
    Y = lambda v: T + (1 - v / ym) * (H - T - B)
    o = [f'<svg viewBox="0 0 {W} {H}" role="img">']
    for v in range(0, int(ym) + 1, 2):
        o.append(f'<line x1="{L}" x2="{W-R}" y1="{Y(v):.1f}" y2="{Y(v):.1f}" class="grid"/><text x="{L-8}" y="{Y(v)+5:.1f}" class="ax" text-anchor="end">{v}</text>')
    o.append(f'<text x="{L-8}" y="{T-10}" class="ax" text-anchor="end">mm</text>')
    band = " ".join(f"{X(i):.1f},{Y(mx):.1f}" for i, _, mx in rows) + " " + " ".join(f"{X(i):.1f},{Y(0):.1f}" for i, _, _ in reversed(rows))
    o.append(f'<polygon points="{band}" class="simband"/>')
    o.append('<polyline fill="none" class="simline" points="' + " ".join(f"{X(i):.1f},{Y(mn):.1f}" for i, mn, _ in rows) + '"/>')
    for i, mn, mx in rows:
        o.append(f'<circle cx="{X(i):.1f}" cy="{Y(mn):.1f}" r="5" class="simdot"><title>run {i}: mean {mn:.1f} mm, worst {mx:.1f} mm</title></circle>')
        o.append(f'<text x="{X(i):.1f}" y="{H-14}" class="ax" text-anchor="middle">{i}</text>')
    o.append(f'<text x="{X(rows[0][0])+10:.1f}" y="{Y(rows[0][1])-10:.1f}" class="lab">{rows[0][1]:.1f} mm</text>')
    o.append(f'<text x="{X(rows[-1][0]):.1f}" y="{Y(rows[-1][1])-12:.1f}" class="lab" text-anchor="end">{rows[-1][1]:.1f} mm</text>')
    o.append(f'<text x="{(L+W-R)/2:.0f}" y="{H-0}" class="ax" text-anchor="middle">simulated run</text>')
    o.append("</svg>")
    return "".join(o)


# ------------------------------------------------------------------ page
def table(head, rows, cls="stack"):
    h = "".join(f"<th>{E(x)}</th>" for x in head)
    b = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f'<div class="scroll"><table class="{cls}"><thead><tr>{h}</tr></thead><tbody>{b}</tbody></table></div>'


def loader_stats(eps):
    out = {}
    for x in eps:
        k = f"{x['phase']} / {x['mode'] if x['mode']!='policy' else x['variant']}"
        d = out.setdefault(k, dict(n=0, ok=0, fail=0, unl=0))
        d["n"] += 1
        o = x.get("outcome")
        d["ok" if o == "success" else "fail" if o == "fail" else "unl"] += 1
    return out


def build(args):
    global DAY0
    st = load_state(Path(args.state))
    tr = parse_transcripts([Path(p) for p in args.transcript or []])
    why_pick = ""
    part = args.part
    if not part:
        part, why_pick = pick_part(st)
    t0, t1 = run_window(st, part)
    day = t0.date()
    DAY0 = dt.datetime.combine(day, dt.time()).timestamp()
    steps = build_steps(st, part, t0, t1, tr)
    dabs = [r for r in st["glog"] if r.get("part") == part]
    dab_ok = [r for r in dabs if r.get("ok", True)]
    marks = [m for m in st["marks"] if m["part"] == part]
    learn_keys = {(s["part"], s["target"], round(s["t"], 3)) for s in st["learn"]}
    eps_run = [x for x in st["eps"] if t0 - dt.timedelta(seconds=2) <= ts(x["t_start"]) <= t1]
    done = next((e for e in st["events"] if e["kind"] == "part_done" and e.get("part") == part), None)
    exc = [e for e in st["events"] if t0 <= ts(e["t"]) <= t1 and e["kind"] in ("exception", "refused", "guided_fail", "estop", "error", "vision")]
    nholes = next((e.get("holes") for e in st["events"] if e["kind"] == "part" and e.get("part") == part and e.get("holes")), len(dabs))
    loads = [x for x in eps_run if x["phase"] == "load"]
    unloads = [x for x in eps_run if x["phase"] == "unload"]
    moved = any(re.search(r"part moved|box slid|lid moved", e["message"]) for e in exc + ([done] if done else []))
    allmsg = json.dumps(exc)
    not_measured = "NOT measured" in allmsg or "could not measure" in json.dumps(done or {})
    teleop = "teleop" in allmsg

    # verdict
    bits = [f"Part {part} was loaded ({'first try' if len(loads)==1 and loads[0]['outcome']=='success' else f'{sum(x['outcome']=='success' for x in loads)} of {len(loads)} load attempts succeeded'})",
            f"{len(dab_ok)} of {nholes} holes were dabbed"]
    tail = []
    if moved:
        tail.append("the box slid under the pen, so only " + (f"{len(marks)} mark could be measured" if marks else "no mark could be measured"))
    elif not_measured:
        tail.append("the camera refused to measure the marks, so this run proves nothing about aim")
    if unloads and not any(x.get("outcome") == "success" for x in unloads):
        tail.append(f"the unload policy failed {len(unloads)} of {len(unloads)} times")
    if teleop:
        tail.append("a human unloaded the box by hand")
    verdict = "; ".join(bits) + (", but " + " and ".join(tail) if tail else "") + f". The cell logged the part as {done['message'] if done else 'open'}."

    miss_txt = ", ".join(f"{np.hypot(*m['miss_mm']):.1f} mm" for m in marks) or "none"
    keynums = [(f"{len(dab_ok)} / {nholes}", "holes dabbed"),
               (f"{len(marks)} / {len(dab_ok)}", f"marks measured ({miss_txt})" if marks else "marks measured"),
               (f"{sum(x.get('outcome')=='success' for x in eps_run)} / {len(eps_run)}", "loader attempts labelled success")]

    # ---------- timeline
    def mark(s):
        return {"ok": "OK", "warning": "Warning", "refused": "Refused", "human": "Human"}[s]
    tl = "".join(f'''<li class="st {s['status']}"><div class="tm">{s.get('tlabel') or s['t'].strftime('%H:%M:%S')}</div>
<div class="dot"></div><div class="body"><div class="who">{E(s['actor'])} <span class="tag {s['status']}">{mark(s['status'])}</span></div>
<div class="what">{E(s['what'])}</div><div class="res">{E(s['result'])}</div>{f'<div class="why">{E(s["why"])}</div>' if s['why'] else ''}</div></li>''' for s in steps)

    # ---------- program
    prog = next((p for p in tr["programs"] if len(p) == nholes or all(f"hole {d['hole']}" in {r['target'] for r in dabs} for d in p)), None)
    targets_in_order = [r["target"] for r in sorted(dabs, key=lambda r: r["t"])]
    px = [(r["px"], r["target"].split()[-1]) for r in dabs if r.get("px")]
    imgs, box = crop_part(st, part, px)
    prog_rows = []
    order = [f"hole {d['hole']}" for d in prog] if prog else targets_in_order
    for i, tname in enumerate(order, 1):
        r = next((x for x in dabs if x["target"] == tname), None)
        m = next((x for x in marks if x["target"] == tname), None)
        res = "not run" if r is None else ("done" if r.get("ok", True) else "refused by guard")
        prog_rows.append([str(i), E(tname), "" if not r or not r.get("px") else f"{r['px'][0]:.0f}, {r['px'][1]:.0f}",
                          "" if not r or touch_err(r) is None else f"{touch_err(r):+.1f} mm", res,
                          f"{np.hypot(*m['miss_mm']):.1f} mm" if m else "not measured"])
    prog_src = "reconstructed from the operator's narration" if prog else "reconstructed from the dab log (no program line in the transcript)"
    prog_code = json.dumps(prog) if prog else json.dumps([{"hole": int(t.split()[-1])} for t in targets_in_order])
    imgs_html = ""
    if imgs:
        imgs_html = '<div class="pics">' + "".join(
            f'<figure><img src="{u}" alt="{k}"><figcaption>{"Before marking. Red rings: the pixel the camera reported for each hole at the moment of its dab (the box later slid)" if k=="before.jpg" else "Inspection result picture"}. <span class="src">{E(n)}</span></figcaption></figure>'
            for k, (u, n) in imgs.items()) + "</div>"

    # ---------- exceptions
    def guard_of(e):
        m = e["message"]
        if "safety stop" in m.lower() or "exit code 3" in m:
            return "Policy runner's joint-jump limit (300 ticks); then Claude's camera check of the work area"
        if e["kind"] == "exception" and ("Inspection" in m or "inspect" in m):
            return "Inspection check (before/after picture alignment) and the cell's alert monitor"
        if e["kind"] == "estop":
            return "Claude's own check of the camera (hand in the work area) and the e-stop latch"
        if "work area is not free" in m:
            return "Guard layer: work-area interlock between the two arms"
        if "touch height" in m:
            return "Height guard: touch must be within 3.5 mm of the target height"
        if "registration" in m:
            return "Inspection check: lid alignment between before/after pictures must be within 3.0 px"
        if "variant" in m:
            return "Tool argument check in the cell server"
        if "SAFETY STOP" in m or "safety stop" in m:
            return "Policy runner's joint-jump limit (300 ticks)"
        return "Cell event log"
    def action_of(e):
        m = e["message"]
        if e["kind"] == "exception":
            return "Claude logged it for the human and paused instead of guessing."
        if "exit code 3" in m:
            return "The SO-101 opened its gripper and went home; the attempt was labelled fail."
        if e["kind"] == "estop":
            return "Run paused; the arms stayed still until a human checked the cell and released the latch."
        if "work area is not free" in m:
            return "Nothing moved. Claude sent the SO-101 home first, then the ElRobot retreated."
        if "touch height" in m:
            return "Pen lifted, step not done, no blind retry. The run continued with the other holes."
        if "unload_part refused" in m:
            return "Nothing moved. The human unloaded by teleoperation outside the cell's logging; Claude reported the gap."
        if e["kind"] == "exception":
            return "Claude logged it for the human and paused instead of guessing."
        return "Logged."
    cards = "".join(f'''<article class="card {e['kind']}"><div class="ct"><b>{hm(e['t'])}</b> <span class="tag {'refused' if e['kind'] in ('refused','estop','guided_fail') else 'warning'}">{E(e['kind'].replace('_',' '))}</span></div>
<p class="noticed">{E(e['message'])}</p><dl><dt>Noticed by</dt><dd>{E(guard_of(e))}</dd><dt>What happened next</dt><dd>{E(action_of(e))}</dd></dl></article>''' for e in exc)

    # ---------- data points
    mark_rows = [[E(m["target"]), f"{m['pos_mm'][0]:.1f}, {m['pos_mm'][1]:.1f}", f"{m['miss_mm'][0]:+.1f}, {m['miss_mm'][1]:+.1f}",
                  f"<b>{np.hypot(*m['miss_mm']):.1f} mm</b>", f"{np.hypot(*m['applied_mm']):.1f} mm",
                  "yes" if (m["part"], m["target"], round(m["t"], 3)) in learn_keys else "no (not in the learning set)"] for m in marks]
    ep_rows = [[f"<code class='nw'>{x['id']}</code>", E(f"{x['phase']} / {x['mode'] if x['mode']!='policy' else x['variant']}"),
                f'<span class="nw">{x["iso_start"][11:19]} → {x["iso_end"][11:19]}</span>', f"{x['seconds']:.1f} s",
                f"<b>{E(x.get('outcome') or 'unlabelled')}</b>", E((x.get("outcome_note") or "")[:140])] for x in eps_run]
    ranges = [x for x in eps_run if x.get("outcome") == "success"]
    pics = sorted((st["sd"] / "marks").glob(f"*_{part}_*"))
    n_ev = sum(1 for e in st["events"] if t0 <= ts(e["t"]) <= t1)
    data_rows = [["Marks measured", str(len(marks))], ["Loader attempts logged", str(len(eps_run))],
                 ["New training ranges (labelled success)", str(len(ranges))], ["Pictures saved in marks/", str(len(pics))],
                 ["Dab log lines (guided_log.jsonl)", str(len(dabs))], ["Event log lines (events.jsonl)", str(n_ev)]]
    range_txt = "\n".join(f"{x['phase']}/{x['variant']}  {x['iso_start'][11:23]} -> {x['iso_end'][11:23]}  ({x['iso_start'][:10]}, attempt {x['id']})" for x in ranges)

    # ---------- improvement
    mm = MissModel(st["sd"])
    tg = mm.targets()
    learn_rows = [[E(g["target"]), E(", ".join(g["parts"])), f"{g['pos_mm'][0]:.1f}, {g['pos_mm'][1]:.1f}", str(g["used"]),
                   "" if not g["mean_error_mm"] else f"{np.hypot(*g['mean_error_mm']):.1f} mm",
                   f"{g['correction_now_mm'][0]:+.1f}, {g['correction_now_mm'][1]:+.1f}"] for g in tg]
    nxt_rows = []
    for r in dabs:
        if r.get("px") is None and not marks:
            continue
    for m in marks:
        c = mm.correction_mm(m["pos_mm"])
        nxt_rows.append([E(m["target"]), f"{np.hypot(*m['miss_mm']):.1f} mm", f"{c[0]:+.1f}, {c[1]:+.1f} mm"])
    before = loader_stats([x for x in st["eps"] if ts(x["t_start"]) < t0 - dt.timedelta(seconds=2)])
    after = loader_stats([x for x in st["eps"] if ts(x["t_start"]) <= t1])
    ls_rows = []
    for k in after:
        b, a = before.get(k, dict(n=0, ok=0, fail=0, unl=0)), after[k]
        lab_b, lab_a = b["ok"] + b["fail"], a["ok"] + a["fail"]
        rate = lambda d, l: "–" if l == 0 else f"{100*d['ok']/l:.0f}%"
        ls_rows.append([E(k), f"{b['ok']}/{lab_b} ({rate(b, lab_b)})", f"<b>{a['ok']}/{lab_a} ({rate(a, lab_a)})</b>", str(a["unl"])])
    excluded = [m for m in marks if (m["part"], m["target"], round(m["t"], 3)) not in learn_keys]
    if marks and excluded:
        meas_note = (f"This run measured {len(marks)} mark{'s' if len(marks)!=1 else ''}, but none is in the learning set now: guided_learning.json holds "
                     f"{len(st['learn'])} samples and no sample from {part}. The mark was taken while the box slid "
                     "(the operator flagged it and asked the human whether to discard it); it stays in guided_marks.jsonl as a record. "
                     "Who removed it and when is not logged. So the next run's aim at this part's holes uses the earlier samples only.")
    elif marks:
        meas_note = f"This run added {len(marks)} learning sample(s)."
    else:
        meas_note = "This run added no measurement, so the aim corrections are unchanged. Reason from the log: " + \
                    next((e["message"] for e in exc if "measure" in e["message"].lower()), "no inspection result.")
    diffs = []
    if any("touch height" in e["message"] for e in exc):
        diffs.append("Hole 2 was not retried blindly. The height guard stays at 3.5 mm; the droop that caused it is on the human's check list.")
    if ranges:
        diffs.append(f"{len(ranges)} new load episode(s) can be cut from the Station recording (below), so the load policy gets more training data.")
    if unloads and not any(x.get("outcome") == "success" for x in unloads):
        diffs.append(f"{len(unloads)} more failed unload attempts are labelled. They show where the unload policy breaks (pushes instead of picks, releases inside the work area).")
    if teleop:
        diffs.append("The teleoperated unload was not logged by this older server; the current server has a teleop variant that records its time range, so the next teleop becomes a training episode.")
    if moved:
        diffs.append("The box sliding under the pen is now a known failure: inspection detects the lid moved and the miss is not trusted.")

    # ---------- today
    today = lambda t: ts(t).date() == day
    mk = [(m["t"], float(np.hypot(*m["miss_mm"])), "dotA" if (m["part"], m["target"], round(m["t"], 3)) in learn_keys else "dotX",
           f"{hm(m['t'])} part {m['part']} {m['target']}: {np.hypot(*m['miss_mm']):.1f} mm") for m in st["marks"] if today(m["t"])]
    dz = [(r["t"], abs(touch_err(r)), "dotA" if r.get("ok", True) else "dotX", f"{hm(r['t'])} part {r['part']} {r['target']}: {touch_err(r):+.1f} mm")
          for r in st["glog"] if "t" in r and today(r["t"]) and not is_sim(r) and touch_err(r) is not None]
    tlo = DAY0 + 9 * 3600
    early = [p for p in dz if p[0] < tlo]
    mk = [p for p in mk if p[0] >= tlo]
    dz = [p for p in dz if p[0] >= tlo]
    thi = max([p[0] for p in mk + dz] + [t1.timestamp()]) + 900
    ch1 = svg_time(mk, tlo, thi, 20, "mm", ylab="miss, mm", refline=(3.0, "3 mm tolerance"))
    ch2 = svg_time(dz, tlo, thi, max(10, min(30, max([p[1] for p in dz] + [1]) + 2)), "mm", ylab="|height error|, mm", refline=(3.5, "3.5 mm guard"))
    st_all = loader_stats([x for x in st["eps"] if today(x["t_start"])])
    bars = "".join(f'''<div class="bar"><div class="bl">{E(k)}</div><div class="bt"><div class="bf" style="width:{(100*v['ok']/max(1,v['ok']+v['fail'])):.0f}%"></div></div>
<div class="bv">{v['ok']} of {v['ok']+v['fail']} labelled{f", {v['unl']} unlabelled" if v['unl'] else ''}</div></div>''' for k, v in st_all.items())
    day_counts = [["Dabs logged (real cell)", str(sum(1 for r in st["glog"] if "t" in r and today(r["t"]) and not is_sim(r)))],
                  ["Marks measured", str(sum(1 for m in st["marks"] if today(m["t"])))],
                  ["Marks kept as learning samples", str(sum(1 for s in st["learn"] if today(s["t"])))],
                  ["Loader attempts logged", str(sum(1 for x in st["eps"] if today(x["t_start"])))],
                  ["Load episodes ready to cut (labelled success)", str(sum(1 for x in st["eps"] if today(x["t_start"]) and x.get("outcome") == "success"))],
                  ["Inspection picture sets (marks/)", str(len({p.name[:22] for p in (st["sd"] / "marks").glob("2*") if p.name.startswith(day.strftime('%Y%m%d'))}))],
                  ["Events logged", str(sum(1 for e in st["events"] if ts(e["t"]).date() == day))]]

    # ---------- simulation
    sim, base_mag = simulate()
    sim_svg = svg_sim(sim)

    T = dict(part=part, date=t0.strftime("%d %B %Y"), t0=t0.strftime("%H:%M"), t1=t1.strftime("%H:%M"))
    css = CSS
    return f'''<!DOCTYPE html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Guided cell run report</title><link rel="preconnect" href="https://fonts.googleapis.com"><link href="https://fonts.googleapis.com/css2?family=Inter+Tight:wght@600;700&family=Inter:wght@400;500;600&display=swap" rel="stylesheet">
<style>{css}</style></head><body>
<header class="band hero"><div class="wrap">
<p class="crumb">HackLab x BRISA Berlin Robotics &amp; AI Hackathon, NormaCore track</p>
<h1>Guided cell — run report</h1>
<p class="meta">Part <b>{part}</b> · {T['date']} · {T['t0']}–{T['t1']}</p>
<p class="verdict">{E(verdict)}</p>
<div class="kn">{''.join(f'<div class="k"><div class="kv">{E(a)}</div><div class="kl">{E(b)}</div></div>' for a, b in keynums)}</div>
<p class="note">{('Why this run: the most complete real run of the day (' + E(why_pick) + ').') if why_pick else 'Run chosen by --part.'}</p>
</div></header>
<nav class="band nav"><div class="wrap scroll"><a href="#happened">What happened</a><a href="#program">Program</a><a href="#wrong">What went wrong</a><a href="#data">Data points</a><a href="#next">Next run</a><a href="#today">Today so far</a><a href="#sim">Capability (simulated)</a></div></nav>
<main>
<section class="band" id="happened"><div class="wrap">
<h2>What happened, step by step</h2>
<p class="lede">Claude calls the cell's tools; a guard layer checks every command; a human approves each motion. Times are the cell's clock.</p>
<ol class="tl">{tl}</ol></div></section>

<section class="band alt" id="program"><div class="wrap">
<h2>The program Claude wrote</h2>
<p class="lede">A program is a list of holes in the part's own frame. Calibration and inverse kinematics (IK: turning a point into joint angles) make each hole a set of joint targets. <span class="src">Program {prog_src}.</span></p>
<pre class="code"><code>{E(prog_code)}</code></pre>
{table(["Step", "Hole", "Camera pixel (x, y)", "Touch height error", "Result", "Miss"], prog_rows)}
<p>Why this order: the narration does not say. Claude asked for a fresh camera look before each hole, so a slid box would be seen.</p>
{imgs_html}
</div></section>

<section class="band" id="wrong"><div class="wrap">
<h2>What went wrong, and what the cell did about it</h2>
<p class="lede">Each card is one event from the cell's log: what was noticed, which check noticed it, and what happened next.</p>
<div class="cards">{cards or '<p>No refusals, failures or alerts in this run.</p>'}</div></div></section>

<section class="band alt" id="data"><div class="wrap">
<h2>Data points this run recorded</h2>
<div class="two"><div>{table(["What", "Count"], data_rows, "plain")}</div>
<div><h3>Training ranges to cut from the NormaCore Station recording</h3><p>Load and unload attempts labelled success by the camera. Each range becomes a new training episode.</p>
<pre class="code"><code>{E(range_txt) or 'none'}</code></pre></div></div>
<h3>Marks measured (miss = how far the ink dot is from the hole)</h3>
{table(["Hole", "Position on part (mm)", "Miss x, y (mm)", "Miss", "Correction applied", "Used for learning"], mark_rows) if mark_rows else '<p>No mark was measured in this run.</p>'}
<h3>Loader attempts</h3>
{table(["Attempt", "Phase / variant", "Time", "Length", "Label", "Camera note"], ep_rows)}
</div></section>

<section class="band" id="next"><div class="wrap">
<h2>How this run improves the next</h2>
<p class="lede">{E(meas_note)}</p>
<h3>Learned aim corrections stored per hole position (now, from cellguide.learn.MissModel)</h3>
{table(["Hole", "Part", "Position (mm)", "Samples", "Mean error", "Correction next time (mm)"], learn_rows)}
<div class="two"><div><h3>Loader success, before and after this run</h3>
{table(["Phase / variant", "Before", "After", "Unlabelled"], ls_rows)}</div>
<div><h3>What the cell will do differently</h3><ul class="dl">{''.join(f'<li>{E(d)}</li>' for d in diffs)}</ul></div></div>
</div></section>

<section class="band alt" id="today"><div class="wrap">
<h2>Today so far, all real runs</h2>
<p class="lede">Vertical lines mark code changes (team notes): {change_legend(tlo, thi)}</p>
<div class="charts">
<figure class="chart"><figcaption><b>Miss per measured mark</b>. Filled: kept for learning. Ringed: excluded (box slid, or outlier).</figcaption>{ch1}</figure>
<figure class="chart"><figcaption><b>Touch height error per dab</b> (size of the error, how far the pen tip was from the touch height). Ringed: dab refused by the height guard. From 09:00; {len(early)} earlier dabs ({hm(min(x[0] for x in early)) if early else ''}–{hm(max(x[0] for x in early)) if early else ''}, before height control) are left out.</figcaption>{ch2}</figure>
</div>
<div class="two"><div><h3>Loader success per variant</h3><div class="bars">{bars}</div></div>
<div><h3>Data points collected today</h3>{table(["What", "Count"], day_counts, "plain")}</div></div>
<p class="tn">Team notes, not in the logs: aim at hole 0 by hand measurement this morning went 24.3 mm, then about 5.6 mm, then about 2–3 mm (estimate). A calibration fit on today's dabs gives about 3.6 mm RMS when the two dabs where the box slid are excluded. The arm's own sideways tracking error at the touch averaged 2.1 mm (max 5.2 mm); a correction for that was installed at 14:22 and has not been measured on the cell yet. The top camera is 56.5 cm above the table.</p>
</div></section>

<section class="band sim" id="sim"><div class="wrap">
<h2>What the loop is capable of <span class="simtag">Simulated</span></h2>
<p class="lede"><b>Simulated, not a cell result.</b> The real learning code (cellguide.learn.MissModel, default settings) runs {len(sim)} times on a simulated 4-hole part.</p>
<figure class="chart wide"><figcaption><b>Mean miss per run</b> (line) and worst hole (shaded) falls from {sim[0][1]:.1f} mm to {sim[-1][1]:.1f} mm.</figcaption>{sim_svg}</figure>
<p class="assume">Assumptions: each hole has its own fixed aim error of {min(base_mag):.1f}–{max(base_mag):.1f} mm (the size seen today), measurement noise 0.5 mm, box placement jitter 2 mm per run, the box does not slide, and every mark is measured every run. Seed 7.</p>
</div></section>
</main>
<footer class="band foot"><div class="wrap">
<p>Generated by <code>python demo/make_run_page.py --state {E(args.state)} {' '.join('--transcript ' + E(p) for p in args.transcript or [])} --part {part} --out {E(args.out)}</code> at {dt.datetime.now():%H:%M}.</p>
<p>Sections 1–7 come from the cell's logs (events.jsonl, guided_log.jsonl, guided_marks.jsonl, guided_learning.json, loader_episodes.jsonl, marks/) and the operator transcript; lines marked team notes do not. Section 8 is a simulation.</p>
</div></footer></body></html>'''


CSS = """/* VOICE work: a calm engineering run report for judges on a projector
   IDEA the cell's own shift log: a printed run sheet with a time rail
   PICTURE the vertical timeline of real steps, plus real camera crops and log charts
   COLOUR white ground, near-black ink, steel panels, alarm red as the one accent for refusals and warnings
   TYPE Inter Tight 700 headings + Inter text
   LAYOUT title band with three key numbers; sections stacked full width; two columns above 900px */
:root{color-scheme:light dark;--bg:light-dark(#ffffff,#16191c);--ink:light-dark(#121416,#eceff1);--mut:light-dark(#4a535b,#a9b3bb);--panel:light-dark(#eef1f3,#1f2428);--steel:light-dark(#d5dbe0,#2c333a);--acc:light-dark(#d4210f,#ff5a4a);--line:light-dark(#c9d0d6,#39424a)}
:root[data-theme=light]{color-scheme:light}
:root[data-theme=dark]{color-scheme:dark}
*{box-sizing:border-box;min-width:0}
body{margin:0;background:var(--bg);color:var(--ink);font:17px/1.55 Inter,system-ui,-apple-system,Segoe UI,Roboto,sans-serif;overflow-x:clip}
h1,h2,h3{font-family:"Inter Tight",Inter,system-ui,sans-serif;letter-spacing:-.02em;line-height:1.08;text-wrap:balance}
h1{font-size:clamp(34px,5vw,52px);margin:.1em 0 .2em}
h2{font-size:clamp(26px,3.2vw,36px);margin:0 0 .4em}
h3{font-size:19px;margin:1.4em 0 .5em}
.wrap{max-width:1180px;margin:0 auto}
.band{padding:clamp(28px,4vw,56px) max(5vw,16px)}
.alt{background:var(--panel)}
.hero{background:var(--steel);padding-bottom:36px}
.crumb,.meta,.note,.src,.tn,.assume{color:var(--mut)}
.crumb{margin:0;font-size:15px}
.meta{font-size:18px;margin:0 0 1em}
.verdict{font-size:clamp(19px,2vw,23px);max-width:62em;margin:0 0 1.2em;font-weight:500}
.kn{display:grid;grid-template-columns:repeat(auto-fit,minmax(14em,1fr));gap:12px}
.k{background:var(--bg);padding:16px 20px;border-radius:6px}
.kv{font:700 clamp(34px,4vw,46px)/1 "Inter Tight",Inter,sans-serif;white-space:nowrap}
.kl{color:var(--mut);margin-top:6px}
.note{font-size:15px;margin:1.2em 0 0}
.nav{padding-block:10px;border-bottom:1px solid var(--line);position:sticky;top:0;background:var(--bg);z-index:2}
.nav .wrap{display:flex;gap:6px 22px;white-space:nowrap}
.nav a{flex:none;color:var(--ink);text-decoration:none;padding:10px 0;font-weight:500}
.nav a:hover{color:var(--acc)}
.lede{max-width:70ch;color:var(--mut)}
.tl{list-style:none;margin:24px 0 0;padding:0;position:relative}
.tl::before{content:"";position:absolute;left:104px;top:6px;bottom:6px;width:2px;background:var(--line)}
.st{display:grid;grid-template-columns:90px 30px minmax(0,1fr);gap:0 0;padding:0 0 18px}
.tm{font-variant-numeric:tabular-nums;color:var(--mut);text-align:right;padding-right:6px;white-space:nowrap;font-size:15px;padding-top:2px}
.dot{width:14px;height:14px;border-radius:50%;background:var(--ink);margin:5px 0 0 8px;position:relative;z-index:1;border:2px solid var(--bg)}
.st.warning .dot{background:var(--bg);border:3px solid var(--acc)}
.st.refused .dot{background:var(--acc)}
.st.human .dot{background:var(--bg);border:3px solid var(--ink)}
.who{font-weight:600}
.what{font-size:18px}
.res{overflow-wrap:anywhere}
.why{color:var(--mut);font-size:15px}
.tag{display:inline-block;font-size:13px;font-weight:600;padding:1px 8px;border-radius:3px;margin-left:6px;vertical-align:2px;border:1px solid var(--line)}
.tag.ok{color:var(--mut)}
.tag.warning{color:var(--acc);border-color:var(--acc)}
.tag.refused{background:var(--acc);color:#fff;border-color:var(--acc)}
.tag.human{background:var(--ink);color:var(--bg);border-color:var(--ink)}
.code{background:var(--bg);border:1px solid var(--line);padding:12px 16px;border-radius:4px;font-size:15px;overflow-x:auto}
pre,.scroll{overflow-x:auto}
code{overflow-wrap:anywhere;font-family:ui-monospace,Menlo,Consolas,monospace}
table{border-collapse:collapse;width:100%;font-size:15.5px;margin:.4em 0 1em}
th{text-align:left;font-weight:600;border-bottom:2px solid var(--ink);padding:8px 10px;vertical-align:bottom}
td{border-bottom:1px solid var(--line);padding:8px 10px;vertical-align:top}
.nw{white-space:nowrap;font-variant-numeric:tabular-nums}
.pics{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:16px;margin-top:16px}
figure{margin:0}
img{max-width:100%;display:block;border-radius:4px}
figcaption{font-size:15px;margin-top:6px}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,340px),1fr));gap:16px}
.card{border:1px solid var(--line);border-radius:6px;padding:16px 18px;background:var(--bg)}
.card .noticed{overflow-wrap:anywhere;margin:.5em 0}
.card dl{margin:0;font-size:15.5px}
.card dt{color:var(--mut);font-size:14px;margin-top:6px}
.card dd{margin:0}
.two{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,420px),1fr));gap:12px 40px;align-items:start}
.dl li{margin-bottom:.5em}
.chip{display:inline-block;margin:2px 10px 2px 0;white-space:nowrap}
.charts{display:grid;gap:16px}
.chart svg{width:100%;height:auto;display:block;margin-top:6px}
.chart{background:var(--bg);border:1px solid var(--line);border-radius:6px;padding:12px 14px}
svg text{font:17px Inter,system-ui,sans-serif;paint-order:stroke;stroke:var(--bg);stroke-width:4px}
.ax{fill:var(--mut)}
.lab{fill:var(--ink);font-weight:600}
.grid{stroke:var(--line);stroke-width:1}
.chg{stroke:var(--mut);stroke-width:1.2;stroke-dasharray:4 4}
.ref{stroke:var(--acc);stroke-width:1.5;stroke-dasharray:8 5}
.dotA{fill:var(--ink)}
.dotX{fill:var(--bg);stroke:var(--acc);stroke-width:2.5}
.bar{display:grid;grid-template-columns:9em minmax(0,1fr) 11em;gap:10px;align-items:center;margin:10px 0}
.bt{height:20px;background:var(--steel);border-radius:3px;overflow:hidden}
.bf{height:100%;background:var(--ink)}
.bv{font-size:15px;color:var(--mut)}
.tn{font-size:15px;border-left:3px solid var(--line);padding-left:12px;max-width:80ch}
.sim{background:repeating-linear-gradient(135deg,var(--panel) 0 14px,var(--bg) 14px 28px)}
.sim .wrap{background:var(--bg);padding:24px;border:2px dashed var(--acc);border-radius:8px}
.simtag{display:inline-block;background:var(--acc);color:#fff;font-size:16px;padding:3px 10px;border-radius:3px;vertical-align:6px;letter-spacing:0}
.simband{fill:color-mix(in srgb,var(--acc) 14%,transparent)}
.simline{stroke:var(--acc);stroke-width:3}
.simdot{fill:var(--acc)}
.assume{font-size:15px}
.foot{font-size:15px;color:var(--mut);border-top:1px solid var(--line)}
@media(max-width:600px){table{font-size:13px}th,td{padding:6px 4px}.stack thead{display:none}.stack tr{display:grid;padding-block:10px}.stack td{display:flex;justify-content:space-between;gap:12px;border:0;text-align:right}.stack td::before{content:attr(data-l);color:var(--mut);text-align:left}.tl::before{left:70px}.st{grid-template-columns:60px 26px minmax(0,1fr)}.tm{font-size:13px}.bar{grid-template-columns:7em minmax(0,1fr);}.bv{grid-column:1/-1}.nav{position:static}}
"""

LABELS_JS = """<script>{for(const t of document.querySelectorAll('table.stack')){const h=[...t.rows[0].cells].map(c=>c.textContent);for(const r of [...t.rows].slice(1))[...r.cells].forEach((c,i)=>c.dataset.l=h[i]||'')}}</script>"""


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required=True)
    ap.add_argument("--transcript", action="append")
    ap.add_argument("--part")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    page = build(a).replace("</body>", LABELS_JS + "</body>")
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page)
    print(f"wrote {out} ({out.stat().st_size/1024:.0f} KB)")


if __name__ == "__main__":
    main()
