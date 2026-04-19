ARG DEVICE=gpu

FROM nvidia/cuda:12.3.2-cudnn9-runtime-ubuntu22.04 AS base-gpu
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/cuda/compat${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

FROM ubuntu:22.04 AS base-cpu

FROM base-${DEVICE}
ARG DEVICE

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    MICROWAKEWORD_VOICE_URL="https://huggingface.co/rhasspy/piper-voices/resolve/main/zh/zh_CN/huayan/medium/zh_CN-huayan-medium.onnx?download=true" \
    MICROWAKEWORD_VOICE_CONFIG_URL="https://huggingface.co/rhasspy/piper-voices/resolve/main/zh/zh_CN/huayan/medium/zh_CN-huayan-medium.onnx.json?download=true" \
    MICROWAKEWORD_VOICE_MODEL=/opt/piper-voices/zh_CN-huayan-medium.onnx \
    MICROWAKEWORD_VOICE_CONFIG=/opt/piper-voices/zh_CN-huayan-medium.onnx.json \
    MICROWAKEWORD_SAMPLES_DIR=/samples \
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
        torchcodec && \
    python -m pip install --no-cache-dir -e .

RUN mkdir -p /samples /output /cache /opt/piper-voices /opt/negative-datasets && \
    wget -q -O /opt/piper-voices/zh_CN-huayan-medium.onnx "${MICROWAKEWORD_VOICE_URL}" && \
    wget -q -O /opt/piper-voices/zh_CN-huayan-medium.onnx.json "${MICROWAKEWORD_VOICE_CONFIG_URL}" && \
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
