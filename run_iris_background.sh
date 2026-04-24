#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="/home/badal/Documents/projects/Iris"
VENV_PYTHON="$PROJECT_DIR/venv/bin/python"
IRIS_SCRIPT="$PROJECT_DIR/iris.py"
LOG_DIR="$HOME/.local/state/iris"
LOG_FILE="$LOG_DIR/iris.log"

mkdir -p "$LOG_DIR"
cd "$PROJECT_DIR"

# Run with GUI enabled so Iris window appears on login.
export IRIS_GUI=1

exec "$VENV_PYTHON" -u "$IRIS_SCRIPT" >>"$LOG_FILE" 2>&1
