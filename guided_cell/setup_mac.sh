#!/usr/bin/env bash
# Set up guided_cell on the Mac. Safe to re-run. It never talks to the robot or Station and only writes inside
# this folder (.venv). Steps: find Python >= 3.10, make .venv, install cellbot's requirements + scipy
# (+ tomli on 3.10), run the offline tests, print the Claude desktop config block and the preflight command.
#   bash setup_mac.sh
#   PYTHON=$(which python) bash setup_mac.sh     # use a given interpreter (e.g. the active conda env)
#   SKIP_INSTALL=1 bash setup_mac.sh             # re-run without pip (no network)
set -u

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE" || exit 1
CONFIG="${CELLGUIDE_CONFIG:-$HERE/guided.toml}"
VENV="$HERE/.venv"

version_of() { "$1" -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>/dev/null; }
version_ok() { "$1" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' 2>/dev/null; }

how_to_get_python() {
    echo "No Python >= 3.10 found. Do one of these, then run this script again:"
    echo "  conda create -n cellguide python=3.12 && conda activate cellguide"
    echo "  PYTHON=\$(which python) bash setup_mac.sh"
    echo "or:"
    echo "  brew install python@3.12 && bash setup_mac.sh"
}

# ------------------------------------------------------------------ 1. Python
PY=""
if [ -n "${PYTHON:-}" ]; then
    if version_ok "$PYTHON"; then PY="$PYTHON"; else
        echo "PYTHON=$PYTHON is not Python >= 3.10 ($(version_of "$PYTHON" || echo 'not runnable'))."
        how_to_get_python; exit 1
    fi
else
    # 3.12 / 3.11 / 3.13 first (no tomli needed), then python3 (may be 3.10, e.g. the active conda env).
    for c in python3.12 python3.11 python3.13 python3; do
        p="$(command -v "$c" 2>/dev/null)" || continue
        if version_ok "$p"; then PY="$p"; break; fi
    done
fi
if [ -z "$PY" ]; then how_to_get_python; exit 1; fi
echo "Python: $PY ($(version_of "$PY"))"

# ------------------------------------------------------------------ 2. venv (separate: conda env stays untouched)
if [ -x "$VENV/bin/python" ] && ! version_ok "$VENV/bin/python"; then
    echo ".venv has Python $(version_of "$VENV/bin/python"), too old: making it again"
    rm -rf "$VENV"
fi
if [ ! -x "$VENV/bin/python" ]; then
    "$PY" -m venv "$VENV" || { echo "could not create $VENV"; exit 1; }
fi
VPY="$VENV/bin/python"
VVER="$(version_of "$VPY")"
echo "venv: $VENV (Python $VVER)"
OLD_PY=0
"$VPY" -c 'import sys; sys.exit(0 if sys.version_info < (3, 11) else 1)' && OLD_PY=1

# ------------------------------------------------------------------ 3. cellbot_dir from guided.toml
CELLBOT_DIR="$("$VPY" - "$CONFIG" <<'EOF'
import os, sys
from pathlib import Path
try:
    import tomllib
except ModuleNotFoundError:
    try:
        import tomli as tomllib
    except ModuleNotFoundError:      # first run on 3.10, before tomli is installed: read the one line by hand
        tomllib = None
p = Path(sys.argv[1])
if tomllib:
    v = tomllib.loads(p.read_text())["paths"]["cellbot_dir"]
else:
    import re
    v = re.search(r'^\s*cellbot_dir\s*=\s*"([^"]*)"', p.read_text(), re.M).group(1)
q = Path(os.path.expanduser(v))
print(q if q.is_absolute() else (p.parent / q).resolve())
EOF
)"
REQ="$CELLBOT_DIR/requirements.txt"
echo "guided.toml: $CONFIG"
echo "cellbot: $CELLBOT_DIR"
if [ ! -f "$REQ" ]; then
    echo "cellbot requirements not found at $REQ: fix [paths] cellbot_dir in $CONFIG"; exit 1
fi

# ------------------------------------------------------------------ 4. install
if [ "${SKIP_INSTALL:-0}" = "1" ]; then
    echo "SKIP_INSTALL=1: pip skipped"
else
    EXTRA="scipy"
    [ "$OLD_PY" = "1" ] && EXTRA="scipy tomli"
    "$VPY" -m pip install --upgrade pip >/dev/null
    # shellcheck disable=SC2086
    if ! "$VPY" -m pip install -r "$REQ" $EXTRA; then
        echo "pip install failed (network?). Nothing outside $VENV was changed; run the script again."; exit 1
    fi
fi
if [ "$OLD_PY" = "1" ]; then
    # Python 3.10: cellbot imports tomllib (3.11+). cellguide maps it to tomli itself; this shim inside .venv
    # makes cellbot's own commands (python -m cellbot.teach / cellbot.estop) work too. cellbot is not edited.
    SITE="$("$VPY" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
    printf '%s\n' '"""Python 3.10 shim made by guided_cell/setup_mac.sh: tomllib is tomli."""' \
        'from tomli import *  # noqa: F401,F403' 'from tomli import TOMLDecodeError, load, loads  # noqa: F401' \
        > "$SITE/tomllib.py"
    echo "Python 3.10: tomllib shim -> tomli in $SITE"
fi
MODS="numpy, cv2, scipy, mcp, PIL"
[ "$OLD_PY" = "1" ] && MODS="$MODS, tomli"
if ! "$VPY" -c "import $MODS"; then
    echo "packages missing in .venv (see above): run the script again without SKIP_INSTALL"; exit 1
fi

# ------------------------------------------------------------------ 5. offline tests
TMPSTATE="$(mktemp -d "${TMPDIR:-/tmp}/cellguide_setup.XXXXXX")"
trap 'rm -rf "$TMPSTATE"' EXIT
echo
echo "Offline tests (CELLBOT_MODE=sim, temporary state in $TMPSTATE):"
FAILED=0
for t in test_guided_sim.py test_teach_tool.py test_alerts.py test_config_copy.py test_preflight.py test_cli.py test_camera_vision.py; do
    log="$TMPSTATE/$t.log"
    mkdir -p "$TMPSTATE/${t%.py}"
    CELLBOT_MODE=sim CELLBOT_STATE="$TMPSTATE/${t%.py}" PYTHONDONTWRITEBYTECODE=1 \
        "$VPY" "tests/$t" >"$log" 2>&1
    rc=$?
    if [ $rc -eq 0 ] && grep -q "ALL CHECKS PASSED" "$log"; then
        echo "  PASS  $t"
    elif [ $rc -eq 0 ] && grep -q "^SKIPPED" "$log"; then
        echo "  SKIP  $t  ($(grep '^SKIPPED' "$log" | head -1))"
    else
        echo "  FAIL  $t  (exit $rc), last lines:"
        tail -n 15 "$log" | sed 's/^/        /'
        FAILED=$((FAILED + 1))
    fi
done

# ------------------------------------------------------------------ 6. what to do next (printed, not written)
echo
"$VPY" - "$HERE" "$VPY" "${CELLGUIDE_CONFIG:-}" <<'EOF'
import json, shlex, sys
here, py, gcfg = sys.argv[1], sys.argv[2], sys.argv[3]
env = {"PYTHONPATH": here, "CELLBOT_CONFIG": f"{here}/cell.guided.toml", "CELLBOT_STATE": f"{here}/state",
       "CELLBOT_MODE": "station", "STATION_HOST": "127.0.0.1"}
if gcfg:
    env["CELLGUIDE_CONFIG"] = gcfg
block = {"cellguide": {"command": py, "args": ["-m", "cellguide.server"], "env": env}}
print('Paste this under "mcpServers" in ~/Library/Application Support/Claude/claude_desktop_config.json')
print("(use it instead of cellbot's own server, not next to it), then restart Claude desktop:\n")
print(json.dumps(block, indent=2)[1:-1].strip("\n"))
print("\nNext, with Station running (--tcp), check readiness (read-only, never moves the arms):\n")
print("cd " + shlex.quote(here) + " && env " + " ".join(f"{k}={shlex.quote(v)}" for k, v in env.items())
      + " " + shlex.quote(py) + " -m cellguide.preflight")
print("\nTeach poses, the pen tip and release the e-stop with the same CELLBOT_STATE, or they are not seen.")
EOF

echo
if [ "$FAILED" -ne 0 ]; then echo "$FAILED test file(s) failed: fix them before using the robot."; exit 1; fi
echo "Setup done."
