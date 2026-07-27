#!/usr/bin/env python3
"""Deploy microWakeWord service to an AutoDL 4090D instance via SSH/SFTP."""

from __future__ import annotations

import argparse
import json
import os
import posixpath
import sys
import time
from pathlib import Path

import paramiko

PROJECT_ROOT = Path(__file__).resolve().parents[1]

REMOTE_ROOT = "/root/autodl-tmp/micro-wake-word"
REMOTE_WORKSPACE = "/root/autodl-tmp/mww"
REMOTE_ASSETS = "/root/autodl-tmp/assets"
REMOTE_SCRIPTS = "/root/autodl-tmp/mww-deploy"
REMOTE_ENV = "/root/autodl-tmp/envs/mww"
CONDA_ENV = "mww"

EXCLUDE_DIRS = {
    ".git",
    "__pycache__",
    ".pytest_cache",
    "microwakeword.egg-info",
    "notebooks",
    "benchmarks",
    ".cursor",
}


def connect(host: str, port: int, user: str, password: str) -> paramiko.SSHClient:
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    print(f"[ssh] connecting {user}@{host}:{port} ...")
    client.connect(
        hostname=host,
        port=port,
        username=user,
        password=password,
        timeout=30,
        allow_agent=False,
        look_for_keys=False,
    )
    return client


def run(
    client: paramiko.SSHClient,
    command: str,
    *,
    check: bool = True,
    get_pty: bool = False,
) -> tuple[int, str, str]:
    print(f"[ssh] $ {command}")
    stdin, stdout, stderr = client.exec_command(command, get_pty=get_pty)
    out_chunks: list[str] = []
    err_chunks: list[str] = []
    while True:
        if stdout.channel.recv_ready():
            chunk = stdout.channel.recv(65536).decode("utf-8", errors="replace")
            out_chunks.append(chunk)
            print(chunk, end="", flush=True)
        if stderr.channel.recv_stderr_ready():
            chunk = stderr.channel.recv_stderr(65536).decode("utf-8", errors="replace")
            err_chunks.append(chunk)
            print(chunk, end="", file=sys.stderr, flush=True)
        if stdout.channel.exit_status_ready():
            # drain remaining
            while stdout.channel.recv_ready():
                chunk = stdout.channel.recv(65536).decode("utf-8", errors="replace")
                out_chunks.append(chunk)
                print(chunk, end="", flush=True)
            while stderr.channel.recv_stderr_ready():
                chunk = stderr.channel.recv_stderr(65536).decode(
                    "utf-8", errors="replace"
                )
                err_chunks.append(chunk)
                print(chunk, end="", file=sys.stderr, flush=True)
            break
        time.sleep(0.2)
    code = stdout.channel.recv_exit_status()
    out = "".join(out_chunks)
    err = "".join(err_chunks)
    if check and code != 0:
        raise RuntimeError(f"remote command failed ({code}): {command}\n{err}")
    return code, out, err


def sftp_mkdirs(sftp: paramiko.SFTPClient, remote_dir: str) -> None:
    parts = [p for p in remote_dir.strip("/").split("/") if p]
    cur = ""
    for part in parts:
        cur = f"{cur}/{part}"
        try:
            sftp.stat(cur)
        except FileNotFoundError:
            try:
                sftp.mkdir(cur)
            except OSError:
                pass


def put_text(sftp: paramiko.SFTPClient, remote_path: str, content: str) -> None:
    sftp_mkdirs(sftp, posixpath.dirname(remote_path))
    with sftp.file(remote_path, "w") as fh:
        fh.write(content)


def upload_tree(sftp: paramiko.SFTPClient, local: Path, remote: str) -> None:
    sftp_mkdirs(sftp, remote)
    for root, dirs, files in os.walk(local):
        dirs[:] = [
            d
            for d in dirs
            if d not in EXCLUDE_DIRS and not d.endswith(".egg-info")
        ]
        rel = os.path.relpath(root, local)
        remote_dir = (
            remote if rel == "." else posixpath.join(remote, rel.replace(os.sep, "/"))
        )
        sftp_mkdirs(sftp, remote_dir)
        for name in files:
            if name.endswith((".pyc", ".pyo")):
                continue
            local_path = Path(root) / name
            remote_path = posixpath.join(remote_dir, name)
            sftp.put(str(local_path), remote_path)
    print(f"[sftp] uploaded {local} -> {remote}")


def run_script(
    client: paramiko.SSHClient,
    sftp: paramiko.SFTPClient,
    name: str,
    content: str,
    *,
    check: bool = True,
) -> tuple[int, str, str]:
    remote_path = f"{REMOTE_SCRIPTS}/{name}"
    put_text(sftp, remote_path, content)
    run(client, f"chmod +x {remote_path}", check=False)
    return run(client, f"bash {remote_path}", check=check, get_pty=True)


PROBE_SCRIPT = """#!/bin/bash
set +e
echo "=== host ==="
uname -a
echo "=== gpu ==="
nvidia-smi -L
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "=== disk ==="
df -h /
df -h /root/autodl-tmp 2>/dev/null
echo "=== autodl-tmp ==="
ls -la /root/autodl-tmp
echo "=== conda ==="
command -v conda
ls /root/miniconda3/bin/conda 2>/dev/null
which python3
python3 --version
echo "=== existing envs ==="
/root/miniconda3/bin/conda env list 2>/dev/null || true
echo "=== turbo ==="
ls -la /etc/network_turbo 2>/dev/null || true
echo "PROBE_OK"
"""


SETUP_SCRIPT = f"""#!/bin/bash
set -euo pipefail
# Do NOT source /etc/network_turbo here — it breaks conda/pip mirrors.
# HuggingFace downloads use hf-mirror instead.

export MICROWAKEWORD_HF_ENDPOINT="${{MICROWAKEWORD_HF_ENDPOINT:-https://hf-mirror.com}}"
# Prefer official/pypi defaults; avoid broken turbo mirrors
export PIP_INDEX_URL="${{PIP_INDEX_URL:-https://pypi.org/simple}}"
unset PIP_EXTRA_INDEX_URL || true

ASSETS="{REMOTE_ASSETS}"
ROOT="{REMOTE_ROOT}"
ENV_PREFIX="{REMOTE_ENV}"
WORKSPACE="{REMOTE_WORKSPACE}"

mkdir -p "$ASSETS/piper-voices" "$ASSETS/negative-datasets" "$WORKSPACE" "$(dirname "$ENV_PREFIX")"

# shellcheck disable=SC1091
source /root/miniconda3/etc/profile.d/conda.sh
# Fix broken aliyun default_channels (404) for this session
cat > /root/.condarc <<'EOF'
channels:
  - defaults
default_channels:
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/msys2
show_channel_urls: true
EOF

if [[ ! -x "$ENV_PREFIX/bin/python" ]]; then
  conda create -y -p "$ENV_PREFIX" python=3.10
fi
conda activate "$ENV_PREFIX"

python -m pip install --upgrade pip
# 1) Torch cu124 first (must stay pinned)
python -m pip install --index-url https://download.pytorch.org/whl/cu124 \\
  torch==2.4.1+cu124 torchaudio==2.4.1+cu124
# 2) TF 2.17 + compatible numpy
PIP_MIRROR="https://pypi.tuna.tsinghua.edu.cn/simple"
python -m pip install -i "$PIP_MIRROR" 'numpy>=1.26,<2.0' 'tensorflow[and-cuda]==2.17.0' 'onnxruntime-gpu>=1.19.0'
# 3) Project/runtime deps (avoid torchcodec which upgrades torch)
python -m pip install -i "$PIP_MIRROR" \\
  'piper-phonemize-cross==1.2.1' \\
  'piper-tts>=1.3.0' \\
  'datasets[audio]==2.21.0' \\
  audiomentations \\
  mmap_ninja \\
  pymicro-features \\
  pyyaml \\
  webrtcvad-wheels \\
  ai-edge-litert \\
  pypinyin \\
  fastapi \\
  'uvicorn[standard]' \\
  httpx \\
  soundfile
python -m pip install -i "$PIP_MIRROR" 'nvidia-cudnn-cu12==8.9.7.29'
# Explicitly avoid torchcodec (incompatible with torch 2.4 + requires CUDA 13 libs)
python -m pip uninstall -y torchcodec >/dev/null 2>&1 || true
# 4) Re-pin torch/numpy after datasets may have pulled newer wheels
python -m pip install -i "$PIP_MIRROR" 'numpy>=1.26,<2.0'
python -m pip install --index-url https://download.pytorch.org/whl/cu124 \\
  --force-reinstall --no-deps torch==2.4.1+cu124 torchaudio==2.4.1+cu124
# 5) audio-metadata (optional GitHub pin; fall back to PyPI)
python -m pip install -i "$PIP_MIRROR" attrs || true
python -m pip install --no-build-isolation \\
  'git+https://github.com/whatsnowplaying/audio-metadata@d4ebb238e6a401bb1a5aaaac60c9e2b3cb30929f' \\
  || python -m pip install -i "$PIP_MIRROR" audio-metadata
cd "$ROOT"
python -m pip install --no-deps -e .

VOICE_URL="$MICROWAKEWORD_HF_ENDPOINT/rhasspy/piper-voices/resolve/main/zh/zh_CN/huayan/medium/zh_CN-huayan-medium.onnx?download=true"
VOICE_CFG_URL="$MICROWAKEWORD_HF_ENDPOINT/rhasspy/piper-voices/resolve/main/zh/zh_CN/huayan/medium/zh_CN-huayan-medium.onnx.json?download=true"
if [[ ! -f "$ASSETS/piper-voices/zh_CN-huayan-medium.onnx" ]]; then
  wget -O "$ASSETS/piper-voices/zh_CN-huayan-medium.onnx" "$VOICE_URL"
fi
if [[ ! -f "$ASSETS/piper-voices/zh_CN-huayan-medium.onnx.json" ]]; then
  wget -O "$ASSETS/piper-voices/zh_CN-huayan-medium.onnx.json" "$VOICE_CFG_URL"
fi

NEG_ROOT="$MICROWAKEWORD_HF_ENDPOINT/datasets/kahrendt/microwakeword/resolve/main"
cd "$ASSETS/negative-datasets"
for name in dinner_party dinner_party_eval no_speech speech; do
  if [[ ! -d "$name/training" ]]; then
    if [[ ! -f "$name.zip" ]]; then
      wget -O "$name.zip" "$NEG_ROOT/$name.zip"
    fi
    unzip -qo "$name.zip"
  fi
done

chmod +x "$ROOT/scripts/mww-service.sh"
echo "SETUP_OK"
python - <<'PY'
import torch, tensorflow as tf
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
print("tf", tf.__version__, "gpus", len(tf.config.list_physical_devices("GPU")))
PY
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="connect.westb.seetacloud.com")
    parser.add_argument("--port", type=int, default=32565)
    parser.add_argument("--user", default="root")
    parser.add_argument("--password", default=os.getenv("MWW_SSH_PASSWORD", ""))
    parser.add_argument(
        "--step",
        choices=["all", "probe", "upload", "setup", "start", "health", "test"],
        default="all",
    )
    parser.add_argument("--wakeword", default="嘿，小龙虾！")
    args = parser.parse_args()
    if not args.password:
        parser.error("--password or MWW_SSH_PASSWORD is required")

    client = connect(args.host, args.port, args.user, args.password)
    sftp = client.open_sftp()
    try:
        steps = (
            ["probe", "upload", "setup", "start", "health", "test"]
            if args.step == "all"
            else [args.step]
        )

        if "probe" in steps:
            run_script(client, sftp, "probe.sh", PROBE_SCRIPT)

        if "upload" in steps:
            # ensure data root exists
            run(client, "mkdir -p /root/autodl-tmp", check=False)
            upload_tree(sftp, PROJECT_ROOT, REMOTE_ROOT)
            run(client, f"chmod +x {REMOTE_ROOT}/scripts/mww-service.sh")

        if "setup" in steps:
            run_script(client, sftp, "setup.sh", SETUP_SCRIPT)

        if "start" in steps:
            run(client, f"bash {REMOTE_ROOT}/scripts/mww-service.sh restart")

        if "health" in steps:
            ok = False
            for attempt in range(1, 21):
                _, out, _ = run(
                    client,
                    "curl -fsS http://127.0.0.1:6006/healthz || true",
                    check=False,
                )
                if '"status"' in out and "ok" in out:
                    print("[health] OK:", out.strip())
                    ok = True
                    break
                print(f"[health] waiting ({attempt}/20)...")
                time.sleep(2)
            if not ok:
                run(
                    client,
                    f"tail -n 100 {REMOTE_WORKSPACE}/mww-service.log || true",
                    check=False,
                )
                raise RuntimeError("health check failed")

        if "test" in steps:
            payload = json.dumps(
                {
                    "wakeword": args.wakeword,
                    "training_steps": 10000,
                    "max_samples": 400,
                    "metadata": {"source": "deploy-test"},
                },
                ensure_ascii=False,
            )
            put_text(sftp, f"{REMOTE_SCRIPTS}/create_job.json", payload)
            _, out, _ = run(
                client,
                "curl -fsS -X POST http://127.0.0.1:6006/api/v1/jobs "
                "-H 'Content-Type: application/json' "
                f"--data-binary @{REMOTE_SCRIPTS}/create_job.json",
            )
            print("[test] created job:", out.strip())
    finally:
        sftp.close()
        client.close()


if __name__ == "__main__":
    main()
