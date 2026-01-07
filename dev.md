# 1. 创建 conda 环境（Python 3.10）
conda create -n microwakeword python=3.10 -y
conda activate microwakeword

# 2. 安装 PyTorch GPU 版本
pip install --index-url https://download.pytorch.org/whl/cu124 \
    torch==2.4.1+cu124 \
    torchaudio==2.4.1+cu124

# 3. 安装 TensorFlow GPU 和其他依赖
pip install \
    'tensorflow[and-cuda]==2.17.0' \
    'onnxruntime-gpu>=1.19.0' \
    'piper-tts==1.2.0' \
    'piper-phonemize-cross==1.2.1' \
    'datasets[audio]' \
    'git+https://github.com/whatsnowplaying/audio-metadata@d4ebb238e6a401bb1a5aaaac60c9e2b3cb30929f'

# 4. 安装项目本身
cd /home/wizard/apps/micro-wake-word
pip install -e .

# 5. 下载 Piper voice 模型
mkdir -p /home/wizard/data/piper-voices
wget -O /home/wizard/data/piper-voices/zh_CN-huayan-medium.onnx \
    "https://huggingface.co/rhasspy/piper-voices/resolve/main/zh/zh_CN/huayan/medium/zh_CN-huayan-medium.onnx?download=true"
wget -O /home/wizard/data/piper-voices/zh_CN-huayan-medium.onnx.json \
    "https://huggingface.co/rhasspy/piper-voices/resolve/main/zh/zh_CN/huayan/medium/zh_CN-huayan-medium.onnx.json?download=true"

# other packages
conda install -n microwakeword pyyaml -y


# 6. 运行训练
cd /home/wizard/apps/micro-wake-word && \
MICROWAKEWORD_WORKDIR=/home/wizard/data/microwakeword-data \
MICROWAKEWORD_TRAIN_BATCH=256 \
MICROWAKEWORD_VOICE_MODEL=/home/wizard/data/piper-voices/zh_CN-huayan-medium.onnx \
MICROWAKEWORD_VOICE_CONFIG=/home/wizard/data/piper-voices/zh_CN-huayan-medium.onnx.json \
python -m microwakeword.docker_entrypoint -c "嘿！赛赛猫！"


# 5090 GPU

# 1. 创建 conda 环境
conda create -n microwakeword python=3.11 -y
conda activate microwakeword

# 2. 安装 PyTorch（使用最新 CUDA 13 支持版本）
pip install --index-url https://download.pytorch.org/whl/cu131 \
    torch \
    torchaudio

# 3. 安装 TensorFlow（最新版本应支持 CUDA 13）
pip install 'tensorflow[and-cuda]'

# 4. 安装其他依赖
pip install \
    'onnxruntime-gpu' \
    'piper-tts==1.2.0' \
    'piper-phonemize-cross==1.2.1' \
    'datasets[audio]' \
    'git+https://github.com/whatsnowplaying/audio-metadata@d4ebb238e6a401bb1a5aaaac60c9e2b3cb30929f'

# 5. 安装项目
cd /home/wizard/apps/micro-wake-word
pip install -e .


# 5090 GPU with docker, works, 24min
# 1. 重新构建镜像
docker build -f Dockerfile.5090 -t wakeword-5090 .

# 2. 运行（挂载到 /data，不是 /workspace）
docker run -it --rm \
  --gpus all \
  -p 8080:8080 \
  -e MICROWAKEWORD_TRAIN_BATCH=256 \
  -v /home/wizard/data/microwakeword-data:/data \
  wakeword-5090 -c "嘿！赛赛猫！"


# ============================================================
# 使用自定义语音样本训练（跳过 Piper TTS 合成）
# ============================================================

# 样本要求：
#   - 格式：WAV (16kHz, 单声道, 16-bit PCM)
#   - 命名：任意 *.wav 文件
#   - 数量：建议 200-500 个

# 3. 使用预生成的语音样本训练
# 假设你的语音样本在 /home/wizard/data/my-wakeword-samples/ 目录下
docker run -it --rm \
  --gpus all \
  -p 8080:8080 \
  -e MICROWAKEWORD_TRAIN_BATCH=256 \
  -v /home/wizard/data/microwakeword-data:/data \
  -v /home/wizard/data/my-wakeword-samples:/samples:ro \
  wakeword-5090 -c "嘿！赛赛猫！" -s /samples

# 参数说明：
#   -c, --wakeword        唤醒词文本（用于命名输出模型）
#   -s, --samples-dir     预生成语音样本的目录路径
#   --training-steps      训练步数（默认 10000）

# 示例：使用高质量 TTS 生成的样本
docker run -it --rm \
  --gpus all \
  -p 8080:8080 \
  -e MICROWAKEWORD_TRAIN_BATCH=256 \
  -v /home/wizard/data/microwakeword-data:/data \
  -v /home/wizard/data/voice-samples:/samples:ro \
  wakeword-5090 -c "嘿，赛赛猫！" -s /samples --training-steps 15000