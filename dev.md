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
# 通用 Docker（GPU / CPU）
# ============================================================
#
# 容器特性：
#   - 训练完成后自动退出（无 webserver）
#   - 固定资源（Piper TTS 模型、默认负样本数据集）内置于镜像
#   - 动态工作区通过挂载卷分离
#
# 挂载点设计：
#   /samples  - 输入：预生成语音样本（只读）
#   /negative-samples - 输入：自定义负样本（WAV，可选，追加到默认负样本）
#   /output   - 输出：训练好的 tflite 模型
#   /cache    - 缓存：训练中间文件（可选，加速重复训练）

# 1. 构建 GPU 镜像（默认；支持 RTX 2080/30/40 系列）
docker build -t wakeword .
# 或显式指定
docker build --build-arg DEVICE=gpu -t wakeword .

# 2. 构建 CPU-only 镜像
docker build --build-arg DEVICE=cpu -t wakeword-cpu .

# 3. GPU 基本运行（使用预生成语音样本）
docker run --rm --gpus all \
  -e MICROWAKEWORD_TRAIN_BATCH=256 \
  -v /home/wizard/data/voice-samples:/samples:ro \
  -v /home/wizard/data/output:/output \
  wakeword -c "嘿！赛赛猫！"

# 4. CPU-only 运行
docker run --rm \
  -e MICROWAKEWORD_USE_CUDA=false \
  -e MICROWAKEWORD_TRAIN_BATCH=128 \
  -v /home/wizard/data/voice-samples:/samples:ro \
  -v /home/wizard/data/output:/output \
  wakeword-cpu -c "嘿！赛赛猫！"

# 5. GPU 带缓存运行
docker run --rm --gpus all \
  -e MICROWAKEWORD_TRAIN_BATCH=256 \
  -v /home/wizard/data/voice-samples:/samples:ro \
  -v /home/wizard/data/output:/output \
  -v /home/wizard/data/cache:/cache \
  wakeword -c "嘿！赛赛猫！"

# 5b. CPU-only 带缓存运行
docker run --rm \
  -e MICROWAKEWORD_USE_CUDA=false \
  -e MICROWAKEWORD_TRAIN_BATCH=128 \
  -v /home/wizard/data/voice-samples:/samples:ro \
  -v /home/wizard/data/output:/output \
  -v /home/wizard/data/cache:/cache \
  wakeword-cpu -c "嘿！赛赛猫！"

# 6. 追加自定义负样本（不会替换默认 negative-datasets）
docker run --rm --gpus all \
  -e MICROWAKEWORD_TRAIN_BATCH=256 \
  -e MICROWAKEWORD_HARD_NEG_PENALTY=3.0 \
  -e MICROWAKEWORD_HARD_NEG_SAMPLING=10.0 \
  -v /home/wizard/data/voice-samples:/samples:ro \
  -v /home/wizard/data/negative-samples:/negative-samples:ro \
  -v /home/wizard/data/output:/output \
  -v /home/wizard/data/cache:/cache \
  wakeword -c "嘿！赛赛猫！"

# 也可显式指定自定义负样本目录（容器内路径）
docker run --rm --gpus all \
  -e MICROWAKEWORD_TRAIN_BATCH=256 \
  -v /home/wizard/data/voice-samples:/samples:ro \
  -v /home/wizard/data/custom-neg:/my-negatives:ro \
  -v /home/wizard/data/output:/output \
  -v /home/wizard/data/cache:/cache \
  wakeword -c "嘿！赛赛猫！" \
  --negative-samples-dir /my-negatives \
  --hard-negative-penalty-weight 3.0 \
  --hard-negative-sampling-weight 10.0

# 训练完成后：
#   - 容器自动退出
#   - 模型保存在 /home/wizard/data/output/<slug>.tflite

# ============================================================
# 样本要求
# ============================================================
#
# 格式：WAV (16kHz, 单声道, 16-bit PCM)
# 命名：任意 *.wav 文件
# 数量：建议 200-500 个
# 说明：自定义负样本会和内置 speech/dinner_party/no_speech/dinner_party_eval 一起训练

# ============================================================
# 参数说明
# ============================================================
#
# -c, --wakeword         唤醒词文本（用于命名输出模型）
# -s, --samples-dir      预生成语音样本目录（默认使用 /samples）
# --negative-samples-dir 自定义负样本目录（WAV，可选；会追加到默认负样本）
# --hard-negative-penalty-weight  自定义硬负样本的惩罚权重（默认 3.0）
# --hard-negative-sampling-weight 自定义硬负样本抽样权重（默认 10.0）
# -o, --output-dir       输出目录（默认使用 /output）
# --training-steps       训练步数（默认 10000）
# --max-samples          合成样本数量（仅当无预生成样本时使用，默认 400）
#
# 环境变量（可选）：
# MICROWAKEWORD_CUSTOM_NEGATIVE_DIR   自定义负样本目录（默认 /negative-samples）
# MICROWAKEWORD_HARD_NEG_PENALTY      自定义硬负样本的惩罚权重（默认 3.0）
# MICROWAKEWORD_HARD_NEG_SAMPLING     自定义硬负样本抽样权重（默认 10.0）