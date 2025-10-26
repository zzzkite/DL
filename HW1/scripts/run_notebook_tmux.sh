#!/usr/bin/env bash
# Execute a notebook headlessly in a tmux session using papermill with incremental saving
# Usage:
#   ./scripts/run_notebook_tmux.sh INPUT_NOTEBOOK [OUTPUT_NOTEBOOK] [SESSION_NAME]
# Defaults:
#   OUTPUT_NOTEBOOK=<INPUT_BASENAME>.executed.ipynb
#   SESSION_NAME=nb-run
# Notes:
# - Papermill writes outputs to the OUTPUT_NOTEBOOK after each cell finishes.
# - A live log is also written to logs/papermill_<timestamp>.log
# - Detach with Ctrl-b then d; reattach with: tmux attach -t nb-run

set -euo pipefail

if [ $# -lt 1 ]; then
  echo "Usage: $0 INPUT_NOTEBOOK [OUTPUT_NOTEBOOK] [SESSION_NAME]"
  exit 1
fi

INPUT_NB=$1
BASENAME="${INPUT_NB%.ipynb}"
OUTPUT_NB=${2:-"${BASENAME}.executed.ipynb"}
SESSION_NAME=${3:-nb-run}

# Optional: activate your env here (uncomment and adjust)
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate your-env
# or for venv
# source /path/to/venv/bin/activate

if ! command -v tmux >/dev/null 2>&1; then
  echo "[ERROR] tmux is not installed. Please install it: sudo apt-get install -y tmux"
  exit 1
fi
if ! command -v papermill >/dev/null 2>&1; then
  echo "[ERROR] papermill is not installed. Install: pip install papermill"
  exit 1
fi

mkdir -p logs
TS=$(date +"%Y%m%d_%H%M%S")
LOGFILE="logs/papermill_${TS}.log"

# Ensure output directory exists (e.g., results/...) before writing the notebook
OUT_DIR="$(dirname "${OUTPUT_NB}")"
mkdir -p "${OUT_DIR}"

CMD="papermill '${INPUT_NB}' '${OUTPUT_NB}' --log-output"

echo "[INFO] Starting papermill in tmux session '$SESSION_NAME'"

echo "[INFO] Command: $CMD"

echo "[INFO] Output notebook: $OUTPUT_NB"

echo "[INFO] Logs: $LOGFILE"

# Start or reuse tmux session
if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
  echo "[INFO] Session '$SESSION_NAME' exists. Sending command to a new window."
  tmux new-window -t "$SESSION_NAME" -n "pm-${TS}" "$CMD 2>&1 | tee -a '$LOGFILE'"
else
  tmux new-session -d -s "$SESSION_NAME" "$CMD 2>&1 | tee -a '$LOGFILE'"
fi

echo "[INFO] Attach with: tmux attach -t $SESSION_NAME"