#!/usr/bin/env bash
# One-command live demo on the Mac: narrator service (:8081) + API in live mode (:8000).
# Run from Terminal.app / iTerm (Claude Code's shell can't get camera permission):
#
#   bash scripts/spikes/run_demo.sh              # live camera + mic
#   MODE=replay bash scripts/spikes/run_demo.sh  # no camera: replay mode
#   NARRATOR=template bash scripts/spikes/run_demo.sh   # skip the LLM (template narrator)
#
# Ctrl-C stops both servers.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
PY="$ROOT/.venv/bin/python"
MODE="${MODE:-live}"
API_PORT="${VALENCE_API_PORT:-8000}"
NARRATOR_PORT=8081
LOGS="${LOGS:-$ROOT/var/demo-logs}"   # server logs only (no media); /var/ is gitignored
mkdir -p "$LOGS"

fail() { echo "PREFLIGHT FAIL: $*" >&2; exit 1; }
warn() { echo "warning: $*" >&2; }

# --- preflight ---------------------------------------------------------------------------------
[[ -x "$PY" ]] || fail "no .venv (uv venv --python 3.12 .venv && uv pip install -r requirements-mac.txt)"
[[ "$("$PY" -c 'import platform;print(platform.machine())')" == arm64 ]] || fail "not an arm64 Python"
for f in models/yolo26n-pose.pt models/gesture_recognizer.task models/face_landmarker.task models/yamnet.tflite \
         models/vosk-model-small-en-us-0.15 models/llm/Qwen3-1.7B-4bit; do
  [[ -e "$f" ]] || fail "missing $f (bash scripts/download_models.sh)"
done
if [[ "$MODE" == live && "${__CFBundleIdentifier:-}" == com.anthropic.* ]]; then
  fail "launched from Claude's shell, which macOS can't give the camera; run this in Terminal.app"
fi
for port in "$API_PORT" "$NARRATOR_PORT"; do
  if pid=$(lsof -nP -t -iTCP:"$port" -sTCP:LISTEN 2>/dev/null | head -1) && [[ -n "$pid" ]]; then
    fail "port $port is taken by: $(ps -o command= -p "$pid" | cut -c1-120) (stop it first: kill $pid)"
  fi
done
[[ "$(ioreg -r -k AppleClamshellState -d 4 2>/dev/null | grep -c '"AppleClamshellState" = Yes')" == 0 ]] || \
  warn "the lid is closed: the built-in camera is off (an iPhone Continuity Camera may be used instead)"
pmset -g batt 2>/dev/null | grep -q "AC Power" || warn "on battery: plug in (thermal/power throttling lowers FPS)"

# --- env: offline + one OpenMP runtime (torch's) for torch + LightGBM in one process ------------
TORCH_LIB="$("$PY" -c 'import os, torch; print(os.path.join(os.path.dirname(torch.__file__), "lib"))')"
export DYLD_LIBRARY_PATH="$TORCH_LIB${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}"
export YOLO_OFFLINE=1 YOLO_AUTOINSTALL=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export VALENCE_MODE="$MODE"

PIDS=()
cleanup() { echo; echo "stopping..."; for p in "${PIDS[@]}"; do kill "$p" 2>/dev/null; done; wait 2>/dev/null; }
trap cleanup EXIT INT TERM

http_code() { curl -s -m 1 -o /dev/null -w '%{http_code}' "$1" 2>/dev/null; }
wait_http() { # wait_http <url> <seconds>: until the server answers at all (the narrator says 503 = template mode)
  for _ in $(seq 1 "$2"); do [[ "$(http_code "$1")" =~ ^[2-5] ]] && return 0; sleep 1; done
  return 1
}

# --- narrator ------------------------------------------------------------------------------------
NARGS=()
[[ "${NARRATOR:-llm}" == template ]] && NARGS+=(--template-only)
"$PY" -m valence.narrator.service --port "$NARRATOR_PORT" "${NARGS[@]}" > "$LOGS/narrator.log" 2>&1 &
PIDS+=($!)
echo "narrator: starting on :$NARRATOR_PORT (model load + warm-up ~10 s) ..."
if wait_http "http://127.0.0.1:$NARRATOR_PORT/health" 90 && [[ "$(http_code "http://127.0.0.1:$NARRATOR_PORT/health")" == 200 ]]; then
  echo "narrator: LLM ready"
elif [[ "${NARRATOR:-llm}" == template ]]; then
  echo "narrator: template mode"
else
  warn "narrator LLM not ready (explanations fall back to templates); see $LOGS/narrator.log"
fi

# --- API -----------------------------------------------------------------------------------------
"$PY" -m uvicorn valence.api.app:app --host 127.0.0.1 --port "$API_PORT" > "$LOGS/api.log" 2>&1 &
PIDS+=($!)
echo "api: starting on :$API_PORT in $MODE mode (camera + models ~10 s) ..."
wait_http "http://127.0.0.1:$API_PORT/api/health" 60 || { tail -20 "$LOGS/api.log"; fail "API did not start"; }
health="$(curl -s "http://127.0.0.1:$API_PORT/api/health")"
echo "api: $health"
if [[ "$MODE" == live && "$health" == *'"edge_error":"'* ]]; then
  warn "edge failed (camera/mic). Switch to a recording: curl -X POST localhost:$API_PORT/api/replay_recording/capture_typing"
fi
echo
echo "  events:     ws://127.0.0.1:$API_PORT/ws/events"
echo "  governance: http://127.0.0.1:$API_PORT/api/governance/stats"
echo "  console:    $PY scripts/console.py"
echo "Ctrl-C to stop."
wait
