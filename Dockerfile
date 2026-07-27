# RTX 4090D 默认 Dockerfile（兼兼容 2080/30/40 系列）
# GPU: CUDA 12.3 + PyTorch cu124 + TensorFlow 2.17
# CPU: 可通过 --build-arg DEVICE=cpu 构建
#
# 5090 请使用 Dockerfile.5090（专用编译的 TF，支持 sm_120）
#
# 目录配置：分离固定资源与动态工作区
# 固定资源（镜像内置）：
#   - /opt/piper-voices: Piper TTS 模型
#   - /opt/negative-datasets: 负样本数据集
# 动态工作区（挂载卷）：
#   - /samples: 输入语音样本
#   - /output: 输出 tflite 模型
#   - /cache: 训练中间文件缓存
#   - /negative-samples: 自定义负样本（可选）

ARG DEVICE=gpu

FROM nvidia/cuda:12.3.2-cudnn9-runtime-ubuntu22.04 AS base-gpu
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/cuda/compat${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

FROM ubuntu:22.04 AS base-cpu

FROM base-${DEVICE}
ARG DEVICE

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8

# Piper TTS 语音模型配置
ENV MICROWAKEWORD_VOICE_URL="https://huggingface.co/rhasspy/piper-voices/resolve/main/zh/zh_CN/huayan/medium/zh_CN-huayan-medium.onnx?download=true" \
    MICROWAKEWORD_VOICE_CONFIG_URL="https://huggingface.co/rhasspy/piper-voices/resolve/main/zh/zh_CN/huayan/medium/zh_CN-huayan-medium.onnx.json?download=true" \
    MICROWAKEWORD_VOICE_MODEL=/opt/piper-voices/zh_CN-huayan-medium.onnx \
    MICROWAKEWORD_VOICE_CONFIG=/opt/piper-voices/zh_CN-huayan-medium.onnx.json

# 目录配置
ENV MICROWAKEWORD_SAMPLES_DIR=/samples \
    MICROWAKEWORD_OUTPUT_DIR=/output \
    MICROWAKEWORD_CACHE_DIR=/cache \
    MICROWAKEWORD_NEGATIVE_DATASETS_DIR=/opt/negative-datasets

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        git \
        wget \
        unzip \
        python3 \
        python3-pip \
        python3-venv \
        python3-dev \
        libsndfile1 \
        ffmpeg \
        espeak-ng \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3 /usr/local/bin/python

WORKDIR /app
COPY . /app

# 安装依赖：先固定 GPU/CPU 栈，再装项目其余依赖
# --no-deps 安装项目本身，避免 setup.py 的 tensorflow>=2.18 覆盖 4090D 固定的 2.17
RUN python -m pip install --no-cache-dir --upgrade pip && \
    if [ "$DEVICE" = "gpu" ]; then \
        python -m pip install --no-cache-dir \
            --index-url https://download.pytorch.org/whl/cu124 \
            torch==2.4.1+cu124 \
            torchaudio==2.4.1+cu124 && \
        python -m pip install --no-cache-dir \
            'tensorflow[and-cuda]==2.17.0' \
            'onnxruntime-gpu>=1.19.0'; \
    else \
        python -m pip install --no-cache-dir \
            --index-url https://download.pytorch.org/whl/cpu \
            torch \
            torchaudio && \
        python -m pip install --no-cache-dir \
            'tensorflow-cpu==2.17.0' \
            'onnxruntime>=1.19.0'; \
    fi && \
    python -m pip install --no-cache-dir \
        'piper-phonemize-cross==1.2.1' \
        'piper-tts>=1.3.0' \
        'git+https://github.com/whatsnowplaying/audio-metadata@d4ebb238e6a401bb1a5aaaac60c9e2b3cb30929f' \
        'datasets[audio]' \
        torchcodec \
        audiomentations \
        mmap_ninja \
        pymicro-features \
        pyyaml \
        webrtcvad-wheels \
        ai-edge-litert && \
    python -m pip install --no-cache-dir --no-deps -e .

# 创建挂载点目录
RUN mkdir -p /samples /output /cache

# 下载固定资源：Piper TTS 语音模型
RUN mkdir -p /opt/piper-voices && \
    wget -q -O /opt/piper-voices/zh_CN-huayan-medium.onnx "${MICROWAKEWORD_VOICE_URL}" && \
    wget -q -O /opt/piper-voices/zh_CN-huayan-medium.onnx.json "${MICROWAKEWORD_VOICE_CONFIG_URL}"

# 下载固定资源：负样本数据集（约 2GB，加速首次运行）
RUN mkdir -p /opt/negative-datasets && \
    cd /opt/negative-datasets && \
    wget -q "https://huggingface.co/datasets/kahrendt/microwakeword/resolve/main/dinner_party.zip" && \
    wget -q "https://huggingface.co/datasets/kahrendt/microwakeword/resolve/main/dinner_party_eval.zip" && \
    wget -q "https://huggingface.co/datasets/kahrendt/microwakeword/resolve/main/no_speech.zip" && \
    wget -q "https://huggingface.co/datasets/kahrendt/microwakeword/resolve/main/speech.zip" && \
    unzip -q dinner_party.zip && \
    unzip -q dinner_party_eval.zip && \
    unzip -q no_speech.zip && \
    unzip -q speech.zip && \
    rm -f *.zip

ENTRYPOINT ["python", "-m", "microwakeword.docker_entrypoint"]
