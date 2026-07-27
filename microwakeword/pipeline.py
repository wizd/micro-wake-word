"""Reusable wake-word training pipeline for CLI and HTTP service."""

from __future__ import annotations

import contextlib
import hashlib
import itertools
import logging
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import wave
import zipfile
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import Callable, Optional

import yaml

NEGATIVE_DATASETS = {
    "dinner_party.zip": "dinner_party",
    "dinner_party_eval.zip": "dinner_party_eval",
    "no_speech.zip": "no_speech",
    "speech.zip": "speech",
}

DEFAULT_SAMPLE_COUNT = int(os.getenv("MICROWAKEWORD_SAMPLE_COUNT", "400"))
DEFAULT_BATCH_SIZE = int(os.getenv("MICROWAKEWORD_SAMPLE_BATCH", "50"))
DEFAULT_TRAINING_STEPS = int(os.getenv("MICROWAKEWORD_TRAINING_STEPS", "10000"))
DEFAULT_TRAIN_BATCH = int(os.getenv("MICROWAKEWORD_TRAIN_BATCH", "256"))
DEFAULT_HARD_NEG_PENALTY = float(os.getenv("MICROWAKEWORD_HARD_NEG_PENALTY", "3.0"))
DEFAULT_HARD_NEG_SAMPLING = float(os.getenv("MICROWAKEWORD_HARD_NEG_SAMPLING", "10.0"))

DEFAULT_SAMPLES_DIR = Path(os.getenv("MICROWAKEWORD_SAMPLES_DIR", "/samples"))
DEFAULT_OUTPUT_DIR = Path(os.getenv("MICROWAKEWORD_OUTPUT_DIR", "/output"))
DEFAULT_CACHE_DIR = Path(os.getenv("MICROWAKEWORD_CACHE_DIR", "/cache"))
DEFAULT_NEGATIVE_DIR = Path(
    os.getenv("MICROWAKEWORD_NEGATIVE_DATASETS_DIR", "/opt/negative-datasets")
)
DEFAULT_CUSTOM_NEGATIVE_DIR = Path(
    os.getenv("MICROWAKEWORD_CUSTOM_NEGATIVE_DIR", "/negative-samples")
)

DEFAULT_VOICE_MODEL = Path(
    os.getenv("MICROWAKEWORD_VOICE_MODEL", "/opt/piper-voices/zh_CN-huayan-medium.onnx")
)
DEFAULT_VOICE_CONFIG = Path(
    os.getenv(
        "MICROWAKEWORD_VOICE_CONFIG",
        "/opt/piper-voices/zh_CN-huayan-medium.onnx.json",
    )
)

# HuggingFace endpoint can be mirrored via MICROWAKEWORD_HF_ENDPOINT
# e.g. https://hf-mirror.com
_HF_ENDPOINT = os.getenv("MICROWAKEWORD_HF_ENDPOINT", "https://huggingface.co").rstrip(
    "/"
)

DEFAULT_VOICE_URL = os.getenv(
    "MICROWAKEWORD_VOICE_URL",
    f"{_HF_ENDPOINT}/rhasspy/piper-voices/resolve/main/zh/zh_CN/huayan/medium/"
    "zh_CN-huayan-medium.onnx?download=true",
)
DEFAULT_VOICE_CONFIG_URL = os.getenv(
    "MICROWAKEWORD_VOICE_CONFIG_URL",
    f"{_HF_ENDPOINT}/rhasspy/piper-voices/resolve/main/zh/zh_CN/huayan/medium/"
    "zh_CN-huayan-medium.onnx.json?download=true",
)
NEGATIVE_DATASET_ROOT = os.getenv(
    "MICROWAKEWORD_NEGATIVE_DATASET_ROOT",
    f"{_HF_ENDPOINT}/datasets/kahrendt/microwakeword/resolve/main/",
)
if not NEGATIVE_DATASET_ROOT.endswith("/"):
    NEGATIVE_DATASET_ROOT += "/"


@dataclass
class TrainingRequest:
    """Parameters for a single wake-word training run."""

    wakeword: str
    job_id: Optional[str] = None
    samples_dir: Optional[Path] = None
    output_dir: Path = field(default_factory=lambda: DEFAULT_OUTPUT_DIR)
    cache_dir: Path = field(default_factory=lambda: DEFAULT_CACHE_DIR)
    negatives_dir: Path = field(default_factory=lambda: DEFAULT_NEGATIVE_DIR)
    custom_negative_dir: Optional[Path] = None
    max_samples: int = DEFAULT_SAMPLE_COUNT
    sample_batch_size: int = DEFAULT_BATCH_SIZE
    training_steps: int = DEFAULT_TRAINING_STEPS
    train_batch: int = DEFAULT_TRAIN_BATCH
    hard_negative_penalty_weight: float = DEFAULT_HARD_NEG_PENALTY
    hard_negative_sampling_weight: float = DEFAULT_HARD_NEG_SAMPLING


class PipelineCancelled(Exception):
    """Raised when a training job is cancelled mid-flight."""


def should_use_cuda() -> bool:
    """Decide whether to use CUDA based on env and availability."""
    flag = os.getenv("MICROWAKEWORD_USE_CUDA", "auto").lower()
    if flag in ("0", "false", "cpu", "no"):
        return False
    if flag in ("1", "true", "yes", "gpu", "cuda"):
        return True
    if flag == "auto":
        # Prefer nvidia-smi / TensorFlow: torch may fail to import (e.g. cudnn
        # soname mismatch) even when the GPU and TF CUDA stack are fine.
        try:
            import subprocess

            result = subprocess.run(
                ["nvidia-smi", "-L"],
                check=False,
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0 and "GPU" in (result.stdout or ""):
                return True
        except Exception:
            pass
        try:
            import torch

            return bool(torch.cuda.is_available())
        except Exception:
            pass
        try:
            import tensorflow as tf

            return bool(tf.config.list_physical_devices("GPU"))
        except Exception:
            return False
    return False


def slugify_phrase(wakeword: str) -> str:
    """Slugify a wake word, supporting Chinese via pypinyin."""
    text = wakeword.strip()
    if not text:
        return "wakeword"

    try:
        from pypinyin import Style, lazy_pinyin

        tokens: list[str] = []
        buf: list[str] = []

        def flush_buf() -> None:
            if buf:
                tokens.append("".join(buf).lower())
                buf.clear()

        for char in text:
            if "\u4e00" <= char <= "\u9fff":
                flush_buf()
                tokens.extend(lazy_pinyin(char, style=Style.NORMAL))
            elif char.isalnum():
                buf.append(char)
            else:
                flush_buf()
        flush_buf()

        slug = "_".join(t for t in tokens if t)
        slug = re.sub(r"_+", "_", slug).strip("_")
        if slug:
            return slug[:80]
    except Exception:
        pass

    # ASCII-only fallback
    slug = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    if slug:
        return slug[:80]

    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]
    return f"ww_{digest}"


def download_file(url: str, destination: Path) -> None:
    import urllib.request

    destination.parent.mkdir(parents=True, exist_ok=True)
    with contextlib.closing(urllib.request.urlopen(url)) as response, destination.open(
        "wb"
    ) as output:
        shutil.copyfileobj(response, output)


def ensure_voice_assets(
    model_path: Path = DEFAULT_VOICE_MODEL,
    config_path: Path = DEFAULT_VOICE_CONFIG,
) -> tuple[Path, Path]:
    if not model_path.exists():
        logging.info("downloading piper voice model from %s", DEFAULT_VOICE_URL)
        download_file(DEFAULT_VOICE_URL, model_path)

    if not config_path.exists():
        logging.info("downloading piper voice config from %s", DEFAULT_VOICE_CONFIG_URL)
        download_file(DEFAULT_VOICE_CONFIG_URL, config_path)

    return model_path, config_path


def _ort_cuda_lib_paths() -> list[str]:
    """Library dirs needed by onnxruntime-gpu (CUDA 12 + cuDNN 9).

    TensorFlow keeps using the env's nvidia-cudnn-cu12 8.9 packages; ORT 1.23+
    needs cuDNN 9, which we stage under MICROWAKEWORD_CUDNN9_DIR without
    replacing the TF stack (different sonames: libcudnn.so.8 vs .so.9).
    """
    paths: list[str] = []
    cudnn9 = Path(
        os.getenv(
            "MICROWAKEWORD_CUDNN9_DIR",
            "/root/autodl-tmp/assets/cudnn9",
        )
    )
    for sub in ("lib", "cublas"):
        lib = cudnn9 / sub
        if lib.is_dir():
            paths.append(str(lib))
    # CUDA 12 companion libs from the pip nvidia-* packages
    paths.extend(_nvidia_lib_paths())
    # Prefer cuDNN9 ahead of the env's cuDNN8 tree.
    deduped: list[str] = []
    seen: set[str] = set()
    for p in paths:
        if p not in seen:
            seen.add(p)
            deduped.append(p)
    return deduped


def ensure_ort_cuda_libraries() -> None:
    """Prepend ORT CUDA libs to LD_LIBRARY_PATH before onnxruntime loads CUDA EP."""
    ort_paths = _ort_cuda_lib_paths()
    if not ort_paths:
        return
    existing = [
        p
        for p in os.environ.get("LD_LIBRARY_PATH", "").split(":")
        if p and p not in ort_paths
    ]
    os.environ["LD_LIBRARY_PATH"] = ":".join(ort_paths + existing)


def load_voice(
    model_path: Path = DEFAULT_VOICE_MODEL,
    config_path: Path = DEFAULT_VOICE_CONFIG,
):
    from piper import PiperVoice  # type: ignore[import-not-found]

    model_path, config_path = ensure_voice_assets(model_path, config_path)
    use_cuda = should_use_cuda()
    if use_cuda:
        ensure_ort_cuda_libraries()
    logging.info(
        "loading Piper voice from %s (use_cuda=%s)", model_path, use_cuda
    )
    voice = PiperVoice.load(
        str(model_path), config_path=str(config_path), use_cuda=use_cuda
    )
    if use_cuda:
        providers = []
        try:
            providers = list(voice.session.get_providers())
        except Exception:
            pass
        if "CUDAExecutionProvider" not in providers:
            logging.warning(
                "Piper requested CUDA but session providers=%s; TTS will be CPU-bound. "
                "Install onnxruntime-gpu (not onnxruntime) and stage cuDNN9 under "
                "MICROWAKEWORD_CUDNN9_DIR.",
                providers,
            )
        else:
            logging.info("Piper ONNX providers: %s", providers)
    return voice


def synthesize_wakeword_samples(
    *, voice, wakeword: str, samples_dir: Path, max_samples: int
) -> None:
    logging.info(
        "generating %d synthetic samples for '%s' using Piper voice",
        max_samples,
        wakeword,
    )

    # Piper >=1.3 uses SynthesisConfig; older versions accept kwargs on synthesize().
    try:
        from piper.config import SynthesisConfig  # type: ignore[import-not-found]

        use_syn_config = True
    except Exception:
        SynthesisConfig = None  # type: ignore[misc, assignment]
        use_syn_config = False

    length_scales = [0.85, 0.95, 1.0, 1.1, 1.2]
    noise_scales = [0.55, 0.65, 0.75]
    noise_ws = [0.7, 0.8, 0.9]
    phrases = [wakeword, f"{wakeword}.", f"{wakeword}!", wakeword.title()]

    variation_iter = itertools.cycle(
        itertools.product(length_scales, noise_scales, noise_ws)
    )
    phrase_iter = itertools.cycle(phrases)

    for index in range(max_samples):
        length_scale, noise_scale, noise_w = next(variation_iter)
        phrase = next(phrase_iter)
        output_path = samples_dir / f"{index}.wav"

        with wave.open(str(output_path), "wb") as wav_file:
            if use_syn_config:
                syn_config = SynthesisConfig(
                    length_scale=length_scale,
                    noise_scale=noise_scale,
                    noise_w_scale=noise_w,
                )
                if hasattr(voice, "synthesize_wav"):
                    voice.synthesize_wav(phrase, wav_file, syn_config=syn_config)
                else:
                    voice.synthesize(phrase, wav_file, syn_config=syn_config)
            else:
                voice.synthesize(
                    phrase,
                    wav_file,
                    length_scale=length_scale,
                    noise_scale=noise_scale,
                    noise_w=noise_w,
                )


def ensure_wakeword_samples(
    *, wakeword: str, samples_dir: Path, max_samples: int, batch_size: int
) -> None:
    _ = batch_size  # retained for CLI compatibility

    samples_dir.mkdir(parents=True, exist_ok=True)
    if list(samples_dir.rglob("*.wav")):
        logging.info("wake word samples already exist; skipping synthesis")
        return

    voice = load_voice()
    synthesize_wakeword_samples(
        voice=voice,
        wakeword=wakeword,
        samples_dir=samples_dir,
        max_samples=max_samples,
    )


def ensure_negative_datasets(base_dir: Path) -> None:
    base_dir.mkdir(parents=True, exist_ok=True)
    for archive, folder in NEGATIVE_DATASETS.items():
        target_dir = base_dir / folder
        training_dir = target_dir / "training"
        if training_dir.exists() and any(training_dir.iterdir()):
            logging.info("negative dataset '%s' already present", folder)
            continue

        url = NEGATIVE_DATASET_ROOT + archive
        archive_path = base_dir / archive
        if not archive_path.exists():
            logging.info(
                "downloading negative dataset '%s' (this may take a while)", folder
            )
            download_file(url, archive_path)
        logging.info("extracting negative dataset '%s' (this may take a while)", folder)
        with zipfile.ZipFile(archive_path, "r") as zip_file:
            zip_file.extractall(base_dir)
        logging.info("completed extraction of '%s'", folder)


def generate_positive_feature_sets(samples_dir: Path, features_dir: Path) -> None:
    from mmap_ninja.ragged import RaggedMmap  # type: ignore[import-not-found]

    from microwakeword.audio.augmentation import Augmentation
    from microwakeword.audio.clips import Clips
    from microwakeword.audio.spectrograms import SpectrogramGeneration

    features_dir.mkdir(parents=True, exist_ok=True)
    clips = Clips(
        input_directory=str(samples_dir),
        file_pattern="**/*.wav",
        remove_silence=False,
        random_split_seed=10,
        split_count=0.1,
    )

    augmenter = Augmentation(
        augmentation_duration_s=3.2,
        augmentation_probabilities={
            "SevenBandParametricEQ": 0.05,
            "TanhDistortion": 0.05,
            "PitchShift": 0.05,
            "BandStopFilter": 0.05,
            "AddColorNoise": 0.05,
            "AddBackgroundNoise": 0.0,
            "Gain": 1.0,
            "RIR": 0.0,
        },
        impulse_paths=[],
        background_paths=[],
        background_min_snr_db=-5,
        background_max_snr_db=10,
        min_jitter_s=0.195,
        max_jitter_s=0.205,
    )

    for split in ("training", "validation", "testing"):
        split_dir = features_dir / split
        mmap_dir = split_dir / "wakeword_mmap"
        manifest = mmap_dir / "manifest.json"
        if manifest.exists():
            logging.info("positive features for %s already exist", split)
            continue

        split_dir.mkdir(parents=True, exist_ok=True)

        if split == "training":
            split_name = "train"
            repetition = 2
            spectrograms = SpectrogramGeneration(
                clips=clips, augmenter=augmenter, slide_frames=10, step_ms=10
            )
        elif split == "validation":
            split_name = "validation"
            repetition = 1
            spectrograms = SpectrogramGeneration(
                clips=clips, augmenter=augmenter, slide_frames=10, step_ms=10
            )
        else:
            split_name = "test"
            repetition = 1
            spectrograms = SpectrogramGeneration(
                clips=clips, augmenter=augmenter, slide_frames=1, step_ms=10
            )

        logging.info("creating positive feature set for %s", split)
        RaggedMmap.from_generator(
            out_dir=str(mmap_dir),
            sample_generator=spectrograms.spectrogram_generator(
                split=split_name, repeat=repetition
            ),
            batch_size=100,
            verbose=True,
        )


def generate_custom_negative_feature_sets(
    samples_dir: Path, features_dir: Path
) -> None:
    from mmap_ninja.ragged import RaggedMmap  # type: ignore[import-not-found]

    from microwakeword.audio.augmentation import Augmentation
    from microwakeword.audio.clips import Clips
    from microwakeword.audio.spectrograms import SpectrogramGeneration

    features_dir.mkdir(parents=True, exist_ok=True)
    clips = Clips(
        input_directory=str(samples_dir),
        file_pattern="**/*.wav",
        remove_silence=False,
        random_split_seed=10,
        split_count=0.1,
    )

    augmenter = Augmentation(
        augmentation_duration_s=3.2,
        augmentation_probabilities={
            "SevenBandParametricEQ": 0.05,
            "TanhDistortion": 0.05,
            "PitchShift": 0.05,
            "BandStopFilter": 0.05,
            "AddColorNoise": 0.05,
            "AddBackgroundNoise": 0.0,
            "Gain": 1.0,
            "RIR": 0.0,
        },
        impulse_paths=[],
        background_paths=[],
        background_min_snr_db=-5,
        background_max_snr_db=10,
        min_jitter_s=0.195,
        max_jitter_s=0.205,
    )

    for split in ("training", "validation", "testing"):
        split_dir = features_dir / split
        mmap_dir = split_dir / "custom_negative_mmap"
        manifest = mmap_dir / "manifest.json"
        if manifest.exists():
            logging.info("custom negative features for %s already exist", split)
            continue

        split_dir.mkdir(parents=True, exist_ok=True)

        if split == "training":
            split_name = "train"
            spectrograms = SpectrogramGeneration(
                clips=clips, augmenter=augmenter, slide_frames=10, step_ms=10
            )
        elif split == "validation":
            split_name = "validation"
            spectrograms = SpectrogramGeneration(
                clips=clips, augmenter=augmenter, slide_frames=10, step_ms=10
            )
        else:
            split_name = "test"
            spectrograms = SpectrogramGeneration(
                clips=clips, augmenter=augmenter, slide_frames=1, step_ms=10
            )

        logging.info("creating custom negative feature set for %s", split)
        RaggedMmap.from_generator(
            out_dir=str(mmap_dir),
            sample_generator=spectrograms.spectrogram_generator(split=split_name),
            batch_size=100,
            verbose=True,
        )


def write_training_config(
    *,
    session_dir: Path,
    slug: str,
    training_steps: int,
    negatives_dir: Path,
    train_batch: int = DEFAULT_TRAIN_BATCH,
    custom_negative_features_dir: Path | None = None,
    hard_negative_penalty_weight: float = DEFAULT_HARD_NEG_PENALTY,
    hard_negative_sampling_weight: float = DEFAULT_HARD_NEG_SAMPLING,
) -> tuple[Path, Path]:
    train_dir = Path("trained_models") / slug
    negatives_path = str(negatives_dir)

    config = {
        "window_step_ms": 10,
        "train_dir": str(train_dir),
        "features": [
            {
                "features_dir": "generated_augmented_features",
                "sampling_weight": 2.0,
                "penalty_weight": 1.0,
                "truth": True,
                "truncation_strategy": "truncate_start",
                "type": "mmap",
            },
            {
                "features_dir": f"{negatives_path}/speech",
                "sampling_weight": 10.0,
                "penalty_weight": 1.0,
                "truth": False,
                "truncation_strategy": "random",
                "type": "mmap",
            },
            {
                "features_dir": f"{negatives_path}/dinner_party",
                "sampling_weight": 10.0,
                "penalty_weight": 1.0,
                "truth": False,
                "truncation_strategy": "random",
                "type": "mmap",
            },
            {
                "features_dir": f"{negatives_path}/no_speech",
                "sampling_weight": 5.0,
                "penalty_weight": 1.0,
                "truth": False,
                "truncation_strategy": "random",
                "type": "mmap",
            },
            {
                "features_dir": f"{negatives_path}/dinner_party_eval",
                "sampling_weight": 0.0,
                "penalty_weight": 1.0,
                "truth": False,
                "truncation_strategy": "split",
                "type": "mmap",
            },
        ],
        "training_steps": [training_steps],
        "positive_class_weight": [1],
        "negative_class_weight": [20],
        "learning_rates": [0.001],
        "batch_size": train_batch,
        "time_mask_max_size": [0],
        "time_mask_count": [0],
        "freq_mask_max_size": [0],
        "freq_mask_count": [0],
        "eval_step_interval": min(500, max(50, training_steps // 4)),
        "clip_duration_ms": 1500,
        "target_minimization": 0.9,
        "minimization_metric": None,
        "maximization_metric": "average_viable_recall",
    }

    if custom_negative_features_dir:
        config["features"].append(
            {
                "features_dir": str(custom_negative_features_dir),
                "sampling_weight": hard_negative_sampling_weight,
                "penalty_weight": hard_negative_penalty_weight,
                "truth": False,
                "truncation_strategy": "random",
                "type": "mmap",
            }
        )

    config_path = session_dir / "training_parameters.yaml"
    with config_path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(config, file, sort_keys=False)

    return config_path, session_dir / train_dir


def locate_tflite_model(train_dir: Path) -> Path:
    candidate = (
        train_dir
        / "tflite_stream_state_internal_quant"
        / "stream_state_internal_quant.tflite"
    )
    if not candidate.exists():
        raise FileNotFoundError(
            f"Expected quantized streaming model at {candidate}, but it was not created."
        )
    return candidate


def _check_cancelled(cancel: Optional[threading.Event]) -> None:
    if cancel is not None and cancel.is_set():
        raise PipelineCancelled("job cancelled")


def _nvidia_lib_paths() -> list[str]:
    """Return pip-installed NVIDIA shared-library directories for TF/torch."""
    paths: list[str] = []
    try:
        import site
        from pathlib import Path as _P

        candidates = []
        for sp in site.getsitepackages() + ([site.getusersitepackages()] if site.getusersitepackages() else []):
            nvidia = _P(sp) / "nvidia"
            if nvidia.is_dir():
                candidates.append(nvidia)
        # Also check relative to tensorflow package
        try:
            import tensorflow as tf

            tf_nvidia = _P(tf.__file__).resolve().parent.parent / "nvidia"
            if tf_nvidia.is_dir():
                candidates.append(tf_nvidia)
        except Exception:
            pass

        seen: set[str] = set()
        for nvidia in candidates:
            for name in (
                "cudnn",
                "cublas",
                "cuda_runtime",
                "cufft",
                "curand",
                "cusolver",
                "cusparse",
                "nccl",
                "nvtx",
                "cuda_nvrtc",
                "cuda_cupti",
            ):
                lib = nvidia / name / "lib"
                key = str(lib)
                if lib.is_dir() and key not in seen:
                    seen.add(key)
                    paths.append(key)
    except Exception:
        pass
    return paths


def run_training_process(
    config_path: Path,
    *,
    workdir: Path,
    log_path: Optional[Path] = None,
    cancel: Optional[threading.Event] = None,
) -> None:
    env = os.environ.copy()
    if should_use_cuda():
        env.pop("CUDA_VISIBLE_DEVICES", None)
        # Prefer pip NVIDIA libs over the host's older system CuDNN (e.g. AutoDL 8.6).
        nv_paths = _nvidia_lib_paths()
        if nv_paths:
            existing = env.get("LD_LIBRARY_PATH", "")
            # Drop system cudnn paths that would win otherwise.
            filtered = [
                p
                for p in existing.split(":")
                if p and "/usr/lib" not in p and "x86_64-linux-gnu" not in p
            ]
            env["LD_LIBRARY_PATH"] = ":".join(nv_paths + filtered)
    else:
        env.setdefault("CUDA_VISIBLE_DEVICES", "-1")
    env.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")

    command = [
        sys.executable,
        "-m",
        "microwakeword.model_train_eval",
        f"--training_config={config_path.name}",
        "--train",
        "1",
        "--restore_checkpoint",
        "1",
        "--test_tf_nonstreaming",
        "0",
        "--test_tflite_nonstreaming",
        "0",
        "--test_tflite_nonstreaming_quantized",
        "0",
        "--test_tflite_streaming",
        "0",
        "--test_tflite_streaming_quantized",
        "1",
        "--use_weights",
        "best_weights",
        "mixednet",
        "--pointwise_filters",
        "64,64,64,64",
        "--repeat_in_block",
        "1,1,1,1",
        "--mixconv_kernel_sizes",
        "[5],[7,11],[9,15],[23]",
        "--residual_connection",
        "0,0,0,0",
        "--first_conv_filters",
        "32",
        "--first_conv_kernel_size",
        "5",
        "--stride",
        "3",
    ]

    logging.info("launching training process")
    log_file = None
    try:
        if log_path is not None:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_file = log_path.open("a", encoding="utf-8")
            log_file.write(f"\n===== training start {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} =====\n")
            log_file.flush()
            stdout = log_file
            stderr = subprocess.STDOUT
        else:
            stdout = None
            stderr = None

        proc = subprocess.Popen(
            command,
            cwd=workdir,
            env=env,
            stdout=stdout,
            stderr=stderr,
        )

        while True:
            _check_cancelled(cancel)
            ret = proc.poll()
            if ret is not None:
                if ret != 0:
                    raise subprocess.CalledProcessError(ret, command)
                return
            time.sleep(1)
    except PipelineCancelled:
        if "proc" in locals() and proc.poll() is None:
            logging.info("terminating training process due to cancel")
            proc.terminate()
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)
        raise
    finally:
        if log_file is not None:
            log_file.close()


def run_pipeline(
    req: TrainingRequest,
    *,
    log_path: Optional[Path] = None,
    cancel: Optional[threading.Event] = None,
    on_stage: Optional[Callable[[str], None]] = None,
) -> Path:
    """Run the full wake-word training pipeline.

    Returns path to the exported ``.tflite`` model.
    """

    def stage(name: str) -> None:
        logging.info("pipeline stage: %s", name)
        if on_stage is not None:
            on_stage(name)
        _check_cancelled(cancel)

    wakeword = req.wakeword.strip()
    if not wakeword:
        raise ValueError("Wake word must not be empty")

    slug = slugify_phrase(wakeword)
    job_key = req.job_id or slug

    output_dir = Path(req.output_dir)
    cache_dir = Path(req.cache_dir)
    negatives_dir = Path(req.negatives_dir)
    session_dir = cache_dir / "jobs" / job_key
    features_dir = session_dir / "generated_augmented_features"

    # Resolve samples
    use_external_samples = False
    if req.samples_dir:
        samples_dir = Path(req.samples_dir)
        if not samples_dir.exists():
            raise FileNotFoundError(f"Samples directory does not exist: {samples_dir}")
        wav_files = list(samples_dir.rglob("*.wav"))
        if not wav_files:
            raise FileNotFoundError(f"No WAV files found in samples directory: {samples_dir}")
        logging.info(
            "using %d pre-generated samples from '%s'", len(wav_files), samples_dir
        )
        use_external_samples = True
    else:
        samples_dir = session_dir / "generated_samples"

    custom_negative_dir = None
    if req.custom_negative_dir:
        custom_negative_dir = Path(req.custom_negative_dir)
        if not custom_negative_dir.exists():
            raise FileNotFoundError(
                f"Negative samples directory does not exist: {custom_negative_dir}"
            )
        if not list(custom_negative_dir.rglob("*.wav")):
            raise FileNotFoundError(
                f"No WAV files found in negative samples directory: {custom_negative_dir}"
            )

    session_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.time()

    stage("synthesize")
    if not use_external_samples:
        ensure_wakeword_samples(
            wakeword=wakeword,
            samples_dir=samples_dir,
            max_samples=req.max_samples,
            batch_size=req.sample_batch_size,
        )

    stage("negatives")
    ensure_negative_datasets(negatives_dir)

    stage("features")
    generate_positive_feature_sets(samples_dir, features_dir)
    custom_negative_features_dir = None
    if custom_negative_dir:
        custom_negative_features_dir = session_dir / "custom_negative_features"
        generate_custom_negative_feature_sets(
            custom_negative_dir, custom_negative_features_dir
        )

    stage("configure")
    config_path, train_dir = write_training_config(
        session_dir=session_dir,
        slug=slug,
        training_steps=req.training_steps,
        negatives_dir=negatives_dir,
        train_batch=req.train_batch,
        custom_negative_features_dir=custom_negative_features_dir,
        hard_negative_penalty_weight=req.hard_negative_penalty_weight,
        hard_negative_sampling_weight=req.hard_negative_sampling_weight,
    )

    stage("train")
    logging.info("starting training for '%s' (slug=%s, job=%s)", wakeword, slug, job_key)
    run_training_process(
        config_path, workdir=session_dir, log_path=log_path, cancel=cancel
    )

    stage("export")
    model_path = locate_tflite_model(train_dir)
    output_model = output_dir / f"{slug}.tflite"
    shutil.copy2(model_path, output_model)

    duration = timedelta(seconds=int(time.time() - start_time))
    logging.info(
        "training complete for '%s'; model saved to %s (took %s)",
        wakeword,
        output_model,
        duration,
    )
    stage("done")
    return output_model
