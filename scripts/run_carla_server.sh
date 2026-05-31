#!/usr/bin/env bash
set -euo pipefail

CARLA_ROOT="${CARLA_ROOT:-/home2/kdh/av/carla}"
CARLA_PORT="${CARLA_PORT:-2000}"

cd "$CARLA_ROOT"

echo "Starting CARLA server..."
echo "CARLA_ROOT=$CARLA_ROOT"
echo "CARLA_PORT=$CARLA_PORT"

./CarlaUE4.sh \
    -RenderOffScreen \
    -nosound \
    -carla-port="$CARLA_PORT"