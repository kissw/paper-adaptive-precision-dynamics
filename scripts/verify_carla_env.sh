#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/av/paper-adaptive-precision-dynamics}"

cd "$REPO_DIR"

uv run python - <<'PY'
import sys
print("python:", sys.executable)

try:
    import torch
    print("torch:", torch.__version__)
    print("cuda available:", torch.cuda.is_available())
    print("cuda version:", torch.version.cuda)
    if torch.cuda.is_available():
        print("gpu:", torch.cuda.get_device_name(0))
except Exception as e:
    print("torch ERROR:", repr(e))

try:
    import carla
    print("carla OK:", carla.__file__)
except Exception as e:
    print("carla ERROR:", repr(e))
    raise

try:
    from agents.navigation.basic_agent import BasicAgent
    print("BasicAgent OK:", BasicAgent)
except Exception as e:
    print("BasicAgent ERROR:", repr(e))
    raise

try:
    import active_inference
    print("active_inference OK")
except Exception as e:
    print("active_inference ERROR:", repr(e))
    raise
PY