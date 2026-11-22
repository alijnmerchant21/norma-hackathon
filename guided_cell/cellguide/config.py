"""Load guided.toml and make the other code drops importable without copying them."""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:          # Python 3.10: tomli is the same parser; cellbot's own `import tomllib`
    import tomli as tomllib          # (cellbot/config.py) then finds it too, without editing cellbot
    sys.modules["tomllib"] = tomllib

ROOT = Path(__file__).resolve().parent.parent


def load(path: str | os.PathLike | None = None) -> dict:
    p = Path(path or os.environ.get("CELLGUIDE_CONFIG", ROOT / "guided.toml"))
    cfg = tomllib.loads(p.read_text())

    def resolve(v: str) -> Path | None:
        if not v:
            return None
        q = Path(os.path.expanduser(v))
        return q if q.is_absolute() else (p.parent / q).resolve()

    cfg["_paths"] = {k: resolve(v) for k, v in cfg["paths"].items()}
    cfg["_file"] = p
    for k in ("cellbot_dir", "elrobot_ik_dir", "vision_file", "calib_file"):
        q = cfg["_paths"].get(k)
        if q is None or not q.exists():
            raise SystemExit(f"guided.toml [paths] {k} = {cfg['paths'].get(k)!r}: not found ({q})")
    return cfg


def use_cellbot(cfg: dict) -> None:
    """Put the cellbot repo on sys.path (its package is imported as `cellbot`)."""
    d = str(cfg["_paths"]["cellbot_dir"])
    if d not in sys.path:
        sys.path.insert(0, d)


def load_module(name: str, file: Path):
    """Import one file under a private name, so `vision.py`/`table.py` of different projects don't clash."""
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, file)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod
