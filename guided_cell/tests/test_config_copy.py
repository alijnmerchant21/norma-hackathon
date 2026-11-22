"""cell.guided.toml must equal Ali's cell.toml except [station] host and norma_core_dir:  python tests/test_config_copy.py

Ali's file is found through guided.toml [paths] cellbot_dir (or CELLGUIDE_CONFIG). Skipped if it is missing.
"""
import copy
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from cellguide.config import tomllib  # noqa: E402  (tomli on Python 3.10)
DIFFERENT = ("host", "norma_core_dir")
GUARD_DIFFERENT = {"reach_tolerance": 0.04}      # the one guard value we changed on purpose (see the file header)


def check(name, cond, detail=""):
    print(f"  [{'ok' if cond else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    if not cond:
        check.failed += 1


check.failed = 0


def alis_file() -> Path:
    p = Path(os.environ.get("CELLGUIDE_CONFIG", ROOT / "guided.toml"))
    v = tomllib.loads(p.read_text())["paths"]["cellbot_dir"]
    q = Path(os.path.expanduser(v))
    return (q if q.is_absolute() else (p.parent / q).resolve()) / "cell.toml"


def main() -> int:
    ali = alis_file()
    if not ali.exists():
        print(f"SKIPPED: Ali's cell.toml not found at {ali}")
        return 0
    ours_text = (ROOT / "cell.guided.toml").read_text()
    a, b = tomllib.loads(ali.read_text()), tomllib.loads(ours_text)
    print(f"comparing {ROOT / 'cell.guided.toml'} with {ali}")
    check("host is 127.0.0.1", b["station"]["host"] == "127.0.0.1", b["station"]["host"])
    check("norma_core_dir is the absolute Mac path",
          b["station"]["norma_core_dir"] == "/Users/MXH37/Documents/dev/lerobot/robot-pizza-cup/elrobot/norma-core")
    a2, b2 = copy.deepcopy(a), copy.deepcopy(b)
    for k, v in GUARD_DIFFERENT.items():
        check(f"[guard] {k} is {v}", b["guard"].get(k) == v, str(b["guard"].get(k)))
        a2["guard"].pop(k, None)
        b2["guard"].pop(k, None)
    check("every other [guard] value is identical", a2["guard"] == b2["guard"])
    for k in DIFFERENT:
        a2["station"].pop(k, None)
        b2["station"].pop(k, None)
    diff = sorted(set(a2) ^ set(b2)) + [k for k in a2 if k in b2 and a2[k] != b2[k]]
    check("everything else is identical", a2 == b2, f"differs: {diff}" if diff else "")
    check("header comment", ours_text.splitlines()[:4] and all(
        line.startswith("#") for line in ours_text.splitlines()[:4]) and "CELLBOT_CONFIG" in ours_text[:900])
    print(f"\n{'ALL CHECKS PASSED' if not check.failed else str(check.failed) + ' CHECK(S) FAILED'}")
    return check.failed


if __name__ == "__main__":
    sys.exit(main())
