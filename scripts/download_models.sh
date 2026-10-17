#!/usr/bin/env bash
# One-time model downloads into models/ (gitignored via .git/info/exclude; never commit weights).
# After this, Valence runs fully offline.
#   bash scripts/download_models.sh            # everything
#   SKIP_LLM=1 bash scripts/download_models.sh # perception/audio only
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
M="$ROOT/models"
mkdir -p "$M"
cd "$M"

fetch() { # fetch <url> <dest>
  local url="$1" dest="$2"
  if [[ -s "$dest" ]]; then echo "ok    $dest"; return; fi
  echo "get   $dest"
  curl -fL --retry 3 --progress-bar -o "$dest.part" "$url"
  mv "$dest.part" "$dest"
}

MP=https://storage.googleapis.com/mediapipe-models

# Video
fetch https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n-pose.pt yolo26n-pose.pt
fetch "$MP/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task" gesture_recognizer.task
fetch "$MP/face_landmarker/face_landmarker/float16/1/face_landmarker.task" face_landmarker.task
fetch "$MP/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task" pose_landmarker_lite.task

# Audio
fetch "$MP/audio_classifier/yamnet/float32/1/yamnet.tflite" yamnet.tflite
if [[ ! -d vosk-model-small-en-us-0.15 ]]; then
  fetch https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip vosk-model-small-en-us-0.15.zip
  unzip -q vosk-model-small-en-us-0.15.zip && rm vosk-model-small-en-us-0.15.zip
fi
echo "ok    vosk-model-small-en-us-0.15/"

# Narrator LLM (MLX 4-bit). Primary, then mirror, then fallback.
if [[ -z "${SKIP_LLM:-}" ]]; then
  HF="${HF:-$ROOT/.venv/bin/hf}"
  for repo in mlx-community/Qwen3-1.7B-4bit lmstudio-community/Qwen3-1.7B-MLX-4bit; do
    dir="llm/$(basename "$repo")"
    if [[ -s "$dir/model.safetensors" ]] || "$HF" download "$repo" --local-dir "$dir"; then
      echo "ok    $dir"; break
    fi
  done
  dir=llm/Llama-3.2-3B-Instruct-4bit
  [[ -s "$dir/model.safetensors" ]] || "$HF" download mlx-community/Llama-3.2-3B-Instruct-4bit --local-dir "$dir"
  echo "ok    $dir (fallback)"
fi

echo "models in $M:"
du -sh "$M"/* | sed "s#$M/##"
