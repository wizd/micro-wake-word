FROM nvidia/cuda:12.3.2-cudnn9-runtime-ubuntu22.04

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    MICROWAKEWORD_WORKDIR=/workspace \
    MICROWAKEWORD_VOICE_DIR=/opt/piper-voices \
    MICROWAKEWORD_VOICE_MODEL_NAME=zh_CN-huayan-medium.onnx \
    MICROWAKEWORD_VOICE_URL="https://huggingface.co/rhasspy/piper-voices/resolve/main/zh/zh_CN/huayan/medium/zh_CN-huayan-medium.onnx?download=true" \
    MICROWAKEWORD_VOICE_CONFIG_URL="https://huggingface.co/rhasspy/piper-voices/resolve/main/zh/zh_CN/huayan/medium/zh_CN-huayan-medium.onnx.json?download=true"
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/cuda/compat${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

ENV MICROWAKEWORD_VOICE_MODEL="${MICROWAKEWORD_VOICE_DIR}/${MICROWAKEWORD_VOICE_MODEL_NAME}" \
    MICROWAKEWORD_VOICE_CONFIG="${MICROWAKEWORD_VOICE_MODEL}.json"

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

# Install依赖顺序：
# 1. PyTorch GPU（cu121）
# 2. TensorFlow GPU（内置 CUDA/cuDNN）
# 3. onnxruntime-gpu + 音频依赖
# 4. 项目本身
RUN python -m pip install --no-cache-dir --upgrade pip && \
    python -m pip install --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cu124 \
        torch==2.4.1+cu124 \
        torchaudio==2.4.1+cu124 && \
    python -m pip install --no-cache-dir \
        'tensorflow[and-cuda]==2.17.0' \
        'onnxruntime-gpu>=1.19.0' \
        'piper-tts==1.2.0' \
        'piper-phonemize-cross==1.2.1' \
        'git+https://github.com/whatsnowplaying/audio-metadata@d4ebb238e6a401bb1a5aaaac60c9e2b3cb30929f' \
        'datasets[audio]' && \
    python -m pip install --no-cache-dir -e .

RUN mkdir -p ${MICROWAKEWORD_WORKDIR} /app/serve ${MICROWAKEWORD_VOICE_DIR} \
    && wget -O ${MICROWAKEWORD_VOICE_MODEL} ${MICROWAKEWORD_VOICE_URL} \
    && wget -O ${MICROWAKEWORD_VOICE_CONFIG} ${MICROWAKEWORD_VOICE_CONFIG_URL}

EXPOSE 8080

ENTRYPOINT ["python", "-m", "microwakeword.docker_entrypoint"]
