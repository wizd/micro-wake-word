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


# ============================================================
# 5090 GPU with Docker（推荐方式）
# ============================================================
#
# 容器特性：
#   - 训练完成后自动退出（无 webserver）
#   - 固定资源（Piper TTS 模型、负样本数据集）内置于镜像
#   - 动态工作区通过挂载卷分离
#
# 挂载点设计：
#   /samples  - 输入：预生成的语音样本（只读）
#   /output   - 输出：训练好的 tflite 模型
#   /cache    - 缓存：训练中间文件（可选，加速重复训练）

# 1. 构建镜像（首次需要，约 10 分钟）
docker build -f Dockerfile.5090 -t wakeword-5090 .

# 2. 基本运行（使用预生成的语音样本）
docker run --rm --gpus all \
  -e MICROWAKEWORD_TRAIN_BATCH=256 \
  -v /home/wizard/data/voice-samples:/samples:ro \
  -v /home/wizard/data/output:/output \
  wakeword-5090 -c "嘿！赛赛猫！"

# 训练完成后：
#   - 容器自动退出
#   - 模型保存在 /home/wizard/data/output/<slug>.tflite

# 3. 带缓存的运行（加速重复训练）
docker run --rm --gpus all \
  -e MICROWAKEWORD_TRAIN_BATCH=256 \
  -v /home/wizard/data/voice-samples:/samples:ro \
  -v /home/wizard/data/output:/output \
  -v /home/wizard/data/cache:/cache \
  wakeword-5090 -c "嘿！赛赛猫！"

# ============================================================
# 自动化脚本示例
# ============================================================

#!/bin/bash
# train_wakeword.sh - 同步运行，训练完成后继续执行

WAKEWORD="嘿！赛赛猫！"
SAMPLES_DIR="./samples"
OUTPUT_DIR="./output"

mkdir -p "$OUTPUT_DIR"

docker run --rm --gpus all \
  -e MICROWAKEWORD_TRAIN_BATCH=256 \
  -v "$SAMPLES_DIR:/samples:ro" \
  -v "$OUTPUT_DIR:/output" \
  wakeword-5090 -c "$WAKEWORD"

if [ $? -eq 0 ]; then
  echo "训练成功！"
  ls -la "$OUTPUT_DIR"/*.tflite
else
  echo "训练失败，退出码: $?"
  exit 1
fi

# ============================================================
# 样本要求
# ============================================================
#
# 格式：WAV (16kHz, 单声道, 16-bit PCM)
# 命名：任意 *.wav 文件
# 数量：建议 200-500 个

# ============================================================
# 参数说明
# ============================================================
#
# -c, --wakeword        唤醒词文本（用于命名输出模型）
# -s, --samples-dir     预生成语音样本的目录路径（可选，默认使用 /samples）
# -o, --output-dir      输出目录路径（可选，默认使用 /output）
# --training-steps      训练步数（默认 10000）
# --max-samples         合成样本数量（仅当无预生成样本时使用，默认 400）

# ============================================================
# 高级示例
# ============================================================

# 使用更多训练步数
docker run --rm --gpus all \
  -e MICROWAKEWORD_TRAIN_BATCH=256 \
  -v /home/wizard/data/voice-samples:/samples:ro \
  -v /home/wizard/data/output:/output \
  -v /home/wizard/data/cache:/cache \
  wakeword-5090 -c "嘿，赛赛猫！" --training-steps 15000