"""Offline check of the preflight checklist (simulator, Manasa's test frames):  python tests/test_preflight.py"""
import asyncio
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ["CELLBOT_MODE"] = "sim"
os.environ["CELLBOT_STATE"] = tempfile.mkdtemp(prefix="cellguide_pf_")
os.environ["CELLBOT_CONFIG"] = str(ROOT / "cell.guided.toml")

import cv2  # noqa: E402

from cellguide import config as gconfig  # noqa: E402
from cellguide import preflight as P  # noqa: E402
from cellguide.guided_cell import GuidedCell  # noqa: E402
import sim_settings  # noqa: E402
from cellguide.vision_source import FrameVision, VisionError, image_size_check  # noqa: E402

G = gconfig.load()
V = gconfig.load_module("part_vision", G["_paths"]["vision_file"])
FRAME = cv2.imread(str(G["_paths"]["vision_file"].parent / "images" / "test_1.jpeg"))


def check(name, cond, detail=""):
    print(f"  [{'ok' if cond else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    if not cond:
        check.failed += 1


check.failed = 0


def by_name(checks, name):
    return next((c for c in checks if c["name"] == name), None)


async def main():
    async def frame():
        return FRAME.copy()

    async def small():
        return cv2.resize(FRAME, (224, 224))

    cell = sim_settings.apply(GuidedCell())
    cell.vision = FrameVision(V, frame, "test frames")
    await cell.ensure_guided()
    await cell.new_part("pf")
    j0 = {a: dict(v) for a, v in cell.backend.joints.items()}

    rep = await P.report(cell)
    cs = rep["checks"]
    print(P.format_text(rep))
    print()
    check("no check crashed", not any(c["detail"].startswith("internal error") for c in cs),
          str([c for c in cs if c["detail"].startswith("internal error")]))
    check("statuses from the allowed set", all(c["status"] in P.STATUSES for c in cs))
    check("every check has name, status, detail, fix", all(set(c) == {"name", "status", "detail", "fix"} for c in cs))
    check("calibration ok", (by_name(cs, "calibration") or {}).get("status") == "ok")
    check("guards ok", (by_name(cs, "guards") or {}).get("status") == "ok")
    check("Station skipped in sim", (by_name(cs, "Station link") or {}).get("status") == "skip")
    look = by_name(cs, "pose elrobot look") or {}
    check("missing look pose is a warn", look.get("status") == "warn", look.get("detail", ""))
    check("image size ok", (by_name(cs, "image size") or {}).get("status") == "ok")
    hp = by_name(cs, "hole plans") or {}
    check("hole plans made", hp.get("status") == "ok" and hp["detail"].count(": ok") == 3, hp.get("detail", ""))
    check("verdict READY without blocking items", rep["verdict"] == P.READY, rep["verdict"])
    check("preflight did not move the arms", cell.backend.joints == j0)

    (cell.cfg.state_dir / "estop.json").write_text('{"latched": true, "reason": "test"}')
    rep = await P.report(cell)
    es = by_name(rep["checks"], "e-stop")
    check("latched e-stop -> fail with the release command", es["status"] == "fail" and "estop release" in es["fix"])
    check("latched e-stop -> NOT READY", rep["verdict"].startswith("NOT READY: 1 "), rep["verdict"])
    (cell.cfg.state_dir / "estop.json").write_text('{"latched": false}')

    check("helper: 224x224 vs 1280x720 is a fail", image_size_check([224, 224], [1280, 720])[0] == "fail")
    check("helper: 640x360 vs 1280x720 is a warn", image_size_check([640, 360], [1280, 720])[0] == "warn")
    cell.vision = FrameVision(V, small, "224 px frames")
    rep = await P.report(cell)
    isz = by_name(rep["checks"], "image size") or {}
    check("224x224 frames -> image size fail, NOT READY", isz.get("status") == "fail"
          and "224 px square" in isz.get("detail", "") and rep["verdict"].startswith("NOT READY"), isz.get("detail", ""))
    try:
        await cell._vstate()
        check("run-time guard refuses 224x224 frames", False)
    except VisionError as e:
        check("run-time guard refuses 224x224 frames", "aspect" in str(e), str(e)[:80])

    print(f"\n{'ALL CHECKS PASSED' if not check.failed else str(check.failed) + ' CHECK(S) FAILED'}")
    return check.failed


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
