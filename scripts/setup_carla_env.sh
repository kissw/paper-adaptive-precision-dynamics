#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# Setup CARLA PythonAPI for uv-managed project environment
# =============================================================================
# Usage:
#   bash scripts/setup_carla_env.sh
#
# This script:
#   1. Checks uv project environment
#   2. Installs CARLA Python wheel into .venv
#   3. Registers CARLA PythonAPI path for agents.navigation.BasicAgent
#   4. Installs missing BasicAgent dependency: shapely
#   5. Verifies carla and BasicAgent imports
# =============================================================================

REPO_DIR="${REPO_DIR:-$HOME/av/paper-adaptive-precision-dynamics}"
CARLA_ROOT="${CARLA_ROOT:-/home2/kdh/av/carla}"
CARLA_WHEEL="${CARLA_WHEEL:-$CARLA_ROOT/PythonAPI/carla/dist/carla-0.9.16-cp310-cp310-manylinux_2_31_x86_64.whl}"
CARLA_PYTHONAPI="${CARLA_PYTHONAPI:-$CARLA_ROOT/PythonAPI/carla}"

cd "$REPO_DIR"

echo "[1/7] Repository: $REPO_DIR"
echo "[2/7] CARLA root: $CARLA_ROOT"

if ! command -v uv >/dev/null 2>&1; then
    echo "ERROR: uv is not installed or not in PATH."
    exit 1
fi

if [ ! -f "$CARLA_WHEEL" ]; then
    echo "ERROR: CARLA wheel not found:"
    echo "  $CARLA_WHEEL"
    echo
    echo "Available CARLA wheel/egg files:"
    find "$CARLA_ROOT/PythonAPI/carla" -maxdepth 3 -type f \( -name "carla*.whl" -o -name "carla*.egg" \) -print || true
    exit 1
fi

if [ ! -d "$CARLA_PYTHONAPI/agents" ]; then
    echo "ERROR: CARLA agents directory not found:"
    echo "  $CARLA_PYTHONAPI/agents"
    exit 1
fi

echo "[3/7] Installing CARLA wheel into uv .venv..."
uv pip install "$CARLA_WHEEL"

echo "[4/7] Installing shapely..."
uv pip install shapely

echo "[5/7] Registering CARLA PythonAPI path for agents..."
SITE_PACKAGES="$(uv run python - <<'PY'
import site
print(site.getsitepackages()[0])
PY
)"

echo "$CARLA_PYTHONAPI" > "$SITE_PACKAGES/carla_pythonapi.pth"

echo "[6/7] Creating local directories..."
mkdir -p logs data

echo "[7/7] Verifying imports..."
uv run python - <<'PY'
import sys
print("python:", sys.executable)

import carla
print("carla OK:", carla.__file__)

from agents.navigation.basic_agent import BasicAgent
print("BasicAgent OK:", BasicAgent)

import shapely
print("shapely OK:", shapely.__version__)
PY

echo
echo "CARLA uv environment setup complete."