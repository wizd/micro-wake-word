#!/usr/bin/env bash
# microWakeWord service control script (AutoDL / native deploy)
set -euo pipefail

ROOT="${MWW_ROOT:-/root/autodl-tmp/micro-wake-word}"
WORKSPACE="${MWW_WORKSPACE:-/root/autodl-tmp/mww}"
ASSETS="${MWW_ASSETS_DIR:-/root/autodl-tmp/assets}"
CONDA_ENV="${MWW_CONDA_ENV:-/root/autodl-tmp/envs/mww}"
PORT="${MWW_PORT:-6006}"
PIDFILE="${WORKSPACE}/mww-service.pid"
LOGFILE="${WORKSPACE}/mww-service.log"

export MWW_WORKSPACE="$WORKSPACE"
export MWW_ASSETS_DIR="$ASSETS"
export MWW_PORT="$PORT"
export MWW_HOST="${MWW_HOST:-0.0.0.0}"
export MICROWAKEWORD_HF_ENDPOINT="${MICROWAKEWORD_HF_ENDPOINT:-https://hf-mirror.com}"
export MICROWAKEWORD_VOICE_MODEL="${MICROWAKEWORD_VOICE_MODEL:-$ASSETS/piper-voices/zh_CN-huayan-medium.onnx}"
export MICROWAKEWORD_VOICE_CONFIG="${MICROWAKEWORD_VOICE_CONFIG:-$ASSETS/piper-voices/zh_CN-huayan-medium.onnx.json}"
export MICROWAKEWORD_NEGATIVE_DATASETS_DIR="${MICROWAKEWORD_NEGATIVE_DATASETS_DIR:-$ASSETS/negative-datasets}"
export MICROWAKEWORD_TRAIN_BATCH="${MICROWAKEWORD_TRAIN_BATCH:-256}"
export MICROWAKEWORD_CUDNN9_DIR="${MICROWAKEWORD_CUDNN9_DIR:-$ASSETS/cudnn9}"
export PYTHONUNBUFFERED=1
# Prefer soundfile over torchcodec for datasets audio decoding
export HF_DATASETS_AUDIO_DECODER=soundfile
export DATASETS_AUDIO_BACKEND=soundfile

# NOTE: network_turbo breaks conda/pip mirrors — do not source it for this service.
# HuggingFace assets are fetched via MICROWAKEWORD_HF_ENDPOINT (hf-mirror).

activate_env() {
  if [[ -f /root/miniconda3/etc/profile.d/conda.sh ]]; then
    # shellcheck disable=SC1091
    source /root/miniconda3/etc/profile.d/conda.sh
  elif [[ -f /root/anaconda3/etc/profile.d/conda.sh ]]; then
    # shellcheck disable=SC1091
    source /root/anaconda3/etc/profile.d/conda.sh
  elif [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
    # shellcheck disable=SC1091
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
  fi
  conda activate "$CONDA_ENV"
  # Ensure pip-installed NVIDIA libs are visible to TF, and shadow host CuDNN 8.6.
  # Prepend isolated cuDNN9 (+ cublas) so Piper/onnxruntime-gpu can use CUDA EP
  # without replacing TF's nvidia-cudnn-cu12 8.9 packages (libcudnn.so.8).
  local nv_paths ort_paths
  nv_paths="$(python - <<'PY'
import site
from pathlib import Path
paths = []
for p in site.getsitepackages():
    n = Path(p) / "nvidia"
    if not n.is_dir():
        continue
    for name in (
        "cudnn", "cublas", "cuda_runtime", "cufft", "curand",
        "cusolver", "cusparse", "nccl", "nvtx", "cuda_nvrtc", "cuda_cupti",
    ):
        lib = n / name / "lib"
        if lib.is_dir():
            paths.append(str(lib))
print(":".join(paths))
PY
)"
  ort_paths=""
  if [[ -d "${MICROWAKEWORD_CUDNN9_DIR}/lib" ]]; then
    ort_paths="${MICROWAKEWORD_CUDNN9_DIR}/lib"
  fi
  if [[ -d "${MICROWAKEWORD_CUDNN9_DIR}/cublas" ]]; then
    ort_paths="${ort_paths:+$ort_paths:}${MICROWAKEWORD_CUDNN9_DIR}/cublas"
  fi
  # Drop host /usr/lib cudnn entries so TF does not load CuDNN 8.6
  filtered=""
  IFS=':' read -r -a old_paths <<< "${LD_LIBRARY_PATH:-}"
  for p in "${old_paths[@]}"; do
    [[ -z "$p" ]] && continue
    [[ "$p" == *"/usr/lib"* ]] && continue
    [[ "$p" == *"x86_64-linux-gnu"* ]] && continue
    filtered="${filtered:+$filtered:}$p"
  done
  export LD_LIBRARY_PATH="${ort_paths:+$ort_paths:}${nv_paths}${filtered:+:$filtered}"
  export MICROWAKEWORD_USE_CUDA="${MICROWAKEWORD_USE_CUDA:-auto}"
}

is_running() {
  if [[ -f "$PIDFILE" ]]; then
    local pid
    pid="$(cat "$PIDFILE")"
    if kill -0 "$pid" 2>/dev/null; then
      return 0
    fi
  fi
  return 1
}

cmd_start() {
  mkdir -p "$WORKSPACE" "$ASSETS" "$(dirname "$PIDFILE")"
  if is_running; then
    echo "already running pid=$(cat "$PIDFILE")"
    return 0
  fi
  activate_env
  cd "$ROOT"
  setsid nohup python -m microwakeword.service.main \
    >>"$LOGFILE" 2>&1 < /dev/null &
  echo $! >"$PIDFILE"
  sleep 2
  if is_running; then
    echo "started pid=$(cat "$PIDFILE") port=$PORT log=$LOGFILE"
  else
    echo "failed to start; see $LOGFILE" >&2
    tail -n 50 "$LOGFILE" >&2 || true
    exit 1
  fi
}

cmd_stop() {
  if ! is_running; then
    echo "not running"
    rm -f "$PIDFILE"
    return 0
  fi
  local pid
  pid="$(cat "$PIDFILE")"
  kill "$pid" 2>/dev/null || true
  for _ in $(seq 1 20); do
    if ! kill -0 "$pid" 2>/dev/null; then
      break
    fi
    sleep 0.5
  done
  if kill -0 "$pid" 2>/dev/null; then
    kill -9 "$pid" 2>/dev/null || true
  fi
  rm -f "$PIDFILE"
  echo "stopped"
}

cmd_restart() {
  cmd_stop || true
  cmd_start
}

cmd_status() {
  if is_running; then
    echo "running pid=$(cat "$PIDFILE")"
    curl -fsS "http://127.0.0.1:${PORT}/healthz" || true
    echo
  else
    echo "stopped"
    exit 1
  fi
}

case "${1:-}" in
  start) cmd_start ;;
  stop) cmd_stop ;;
  restart) cmd_restart ;;
  status) cmd_status ;;
  *)
    echo "usage: $0 {start|stop|restart|status}" >&2
    exit 2
    ;;
esac
