"""Reusable wake-word training pipeline for CLI and HTTP service."""

from __future__ import annotations

import contextlib
import hashlib
import itertools
import json
import importlib.metadata
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

# Feature-cache schema. Bump whenever positive-feature augmentation changes
# so restore_synthetic_asset_cache cannot reuse dry TTS spectrograms.
AUGMENTATION_REVISION = 2

TRAIN_AUGMENTATION_PROBABILITIES = {
    "SevenBandParametricEQ": 0.1,
    "TanhDistortion": 0.1,
    "PitchShift": 0.1,
    "BandStopFilter": 0.1,
    "AddColorNoise": 0.25,
    "AddBackgroundNoise": 0.75,
    "Gain": 1.0,
    "RIR": 0.5,
}

# Testing is intentionally harder than training: every clip is reverberated
# and mixed with noise so a 100% score actually means generalization.
TEST_AUGMENTATION_PROBABILITIES = {
    "SevenBandParametricEQ": 0.1,
    "TanhDistortion": 0.1,
    "PitchShift": 0.1,
    "BandStopFilter": 0.1,
    "AddColorNoise": 0.25,
    "AddBackgroundNoise": 1.0,
    "Gain": 1.0,
    "RIR": 1.0,
}

DEFAULT_SAMPLE_COUNT = int(os.getenv("MICROWAKEWORD_SAMPLE_COUNT", "400"))
DEFAULT_BATCH_SIZE = int(os.getenv("MICROWAKEWORD_SAMPLE_BATCH", "50"))
DEFAULT_TRAINING_STEPS = int(os.getenv("MICROWAKEWORD_TRAINING_STEPS", "10000"))
DEFAULT_TRAIN_BATCH = int(os.getenv("MICROWAKEWORD_TRAIN_BATCH", "256"))
DEFAULT_HARD_NEG_PENALTY = float(os.getenv("MICROWAKEWORD_HARD_NEG_PENALTY", "3.0"))
DEFAULT_HARD_NEG_SAMPLING = float(os.getenv("MICROWAKEWORD_HARD_NEG_SAMPLING", "10.0"))
DEFAULT_EARLY_STOP = os.getenv("MICROWAKEWORD_EARLY_STOP", "false").lower() in (
    "1",
    "true",
    "yes",
    "on",
)
DEFAULT_EARLY_STOP_MIN_STEPS = int(
    os.getenv("MICROWAKEWORD_EARLY_STOP_MIN_STEPS", "3000")
)
DEFAULT_EARLY_STOP_PATIENCE = int(
    os.getenv("MICROWAKEWORD_EARLY_STOP_PATIENCE", "5")
)
DEFAULT_EARLY_STOP_MIN_DELTA = float(
    os.getenv("MICROWAKEWORD_EARLY_STOP_MIN_DELTA", "1e-6")
)

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
DEFAULT_TEST_VOICE_MODEL = Path(
    os.getenv(
        "MICROWAKEWORD_TEST_VOICE_MODEL",
        "/opt/piper-voices/zh_CN-huayan-x_low.onnx",
    )
)
DEFAULT_TEST_VOICE_CONFIG = Path(
    os.getenv(
        "MICROWAKEWORD_TEST_VOICE_CONFIG",
        "/opt/piper-voices/zh_CN-huayan-x_low.onnx.json",
    )
)
DEFAULT_AUGMENTATION_DIR = Path(
    os.getenv(
        "MICROWAKEWORD_AUGMENTATION_DIR",
        "/opt/augmentation-datasets",
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
DEFAULT_TEST_VOICE_URL = os.getenv(
    "MICROWAKEWORD_TEST_VOICE_URL",
    f"{_HF_ENDPOINT}/rhasspy/piper-voices/resolve/main/zh/zh_CN/huayan/x_low/"
    "zh_CN-huayan-x_low.onnx?download=true",
)
DEFAULT_TEST_VOICE_CONFIG_URL = os.getenv(
    "MICROWAKEWORD_TEST_VOICE_CONFIG_URL",
    f"{_HF_ENDPOINT}/rhasspy/piper-voices/resolve/main/zh/zh_CN/huayan/x_low/"
    "zh_CN-huayan-x_low.onnx.json?download=true",
)
NEGATIVE_DATASET_ROOT = os.getenv(
    "MICROWAKEWORD_NEGATIVE_DATASET_ROOT",
    f"{_HF_ENDPOINT}/datasets/kahrendt/microwakeword/resolve/main/",
)
if not NEGATIVE_DATASET_ROOT.endswith("/"):
    NEGATIVE_DATASET_ROOT += "/"

DEFAULT_RIR_URL = os.getenv(
    "MICROWAKEWORD_RIR_URL",
    "https://mcdermottlab.mit.edu/Reverb/IRMAudio/Audio.zip",
)
DEFAULT_NOISE_URL = os.getenv("MICROWAKEWORD_NOISE_URL", "")
DEFAULT_NOISE_INDEX_URL = os.getenv(
    "MICROWAKEWORD_NOISE_INDEX_URL",
    "https://cdn.jsdelivr.net/gh/karolpiczak/ESC-50@master/meta/esc50.csv",
)
DEFAULT_NOISE_FILE_URL = os.getenv(
    "MICROWAKEWORD_NOISE_FILE_URL",
    "https://cdn.jsdelivr.net/gh/karolpiczak/ESC-50@master/audio/{filename}",
)
RIR_MIN_WAVS = int(os.getenv("MICROWAKEWORD_RIR_MIN_WAVS", "20"))
NOISE_MIN_WAVS = int(os.getenv("MICROWAKEWORD_NOISE_MIN_WAVS", "20"))
NOISE_FILE_COUNT = int(os.getenv("MICROWAKEWORD_NOISE_FILE_COUNT", "40"))
DOWNLOAD_TIMEOUT_S = int(os.getenv("MICROWAKEWORD_DOWNLOAD_TIMEOUT", "120"))


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
    request = urllib.request.Request(
        url, headers={"User-Agent": "microwakeword-pipeline/1.0"}
    )
    with contextlib.closing(
        urllib.request.urlopen(request, timeout=DOWNLOAD_TIMEOUT_S)
    ) as response, destination.open("wb") as output:
        shutil.copyfileobj(response, output)


def ensure_voice_assets(
    model_path: Path = DEFAULT_VOICE_MODEL,
    config_path: Path = DEFAULT_VOICE_CONFIG,
    model_url: str = DEFAULT_VOICE_URL,
    config_url: str = DEFAULT_VOICE_CONFIG_URL,
) -> tuple[Path, Path]:
    if not model_path.exists():
        logging.info("downloading piper voice model from %s", model_url)
        download_file(model_url, model_path)

    if not config_path.exists():
        logging.info("downloading piper voice config from %s", config_url)
        download_file(config_url, config_path)

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
    model_url: str = DEFAULT_VOICE_URL,
    config_url: str = DEFAULT_VOICE_CONFIG_URL,
):
    from piper import PiperVoice  # type: ignore[import-not-found]

    model_path, config_path = ensure_voice_assets(
        model_path, config_path, model_url=model_url, config_url=config_url
    )
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
    *,
    wakeword: str,
    samples_dir: Path,
    max_samples: int,
    batch_size: int,
    voice_model: Path = DEFAULT_VOICE_MODEL,
    voice_config: Path = DEFAULT_VOICE_CONFIG,
    voice_url: str = DEFAULT_VOICE_URL,
    voice_config_url: str = DEFAULT_VOICE_CONFIG_URL,
) -> None:
    _ = batch_size  # retained for CLI compatibility

    samples_dir.mkdir(parents=True, exist_ok=True)
    if list(samples_dir.rglob("*.wav")):
        logging.info("wake word samples already exist in %s; skipping synthesis", samples_dir)
        return

    voice = load_voice(
        voice_model,
        voice_config,
        model_url=voice_url,
        config_url=voice_config_url,
    )
    synthesize_wakeword_samples(
        voice=voice,
        wakeword=wakeword,
        samples_dir=samples_dir,
        max_samples=max_samples,
    )


def _path_fingerprint(path: Path) -> dict:
    try:
        return {
            "size": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    except OSError:
        return {"path": str(path)}


def _synthetic_asset_cache_key(wakeword: str, max_samples: int) -> str:
    """Hash every input that affects generated samples/features."""
    try:
        piper_version = importlib.metadata.version("piper-tts")
    except importlib.metadata.PackageNotFoundError:
        piper_version = "unknown"

    payload = {
        "version": 3,
        "wakeword": wakeword,
        "max_samples": int(max_samples),
        "voice_model": _path_fingerprint(DEFAULT_VOICE_MODEL),
        "voice_config": _path_fingerprint(DEFAULT_VOICE_CONFIG),
        "test_voice_model": _path_fingerprint(DEFAULT_TEST_VOICE_MODEL),
        "test_voice_config": _path_fingerprint(DEFAULT_TEST_VOICE_CONFIG),
        "piper_version": piper_version,
        "pipeline_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "audio_utils_sha256": hashlib.sha256(
            (Path(__file__).parent / "audio" / "audio_utils.py").read_bytes()
        ).hexdigest(),
        "length_scales": [0.85, 0.95, 1.0, 1.1, 1.2],
        "noise_scales": [0.55, 0.65, 0.75],
        "noise_ws": [0.7, 0.8, 0.9],
        "feature_config": {
            "augmentation_duration_s": 3.2,
            "slide_frames": {"training": 10, "validation": 10, "testing": 1},
            "step_ms": 10,
            "training_repetition": 2,
            "augmentation_revision": AUGMENTATION_REVISION,
            "train_augmentation": TRAIN_AUGMENTATION_PROBABILITIES,
            "test_augmentation": TEST_AUGMENTATION_PROBABILITIES,
        },
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _link_or_copy(source: str, destination: str) -> None:
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def _ragged_mmap_complete(path: Path) -> bool:
    return all(
        (path / relative).exists()
        for relative in (
            "data.ninja",
            "type.ninja",
            "dtype.ninja",
            "shape.ninja",
            "order.ninja",
            "starts/data.ninja",
            "ends/data.ninja",
        )
    )


def restore_synthetic_asset_cache(
    cache_root: Path,
    wakeword: str,
    max_samples: int,
    samples_dir: Path,
    features_dir: Path,
    heldout_dir: Path | None = None,
) -> bool:
    key = _synthetic_asset_cache_key(wakeword, max_samples)
    entry = cache_root / key
    cached_samples = entry / "samples"
    cached_features = entry / "features"
    cached_heldout = entry / "heldout"
    ready = entry / "READY"
    heldout_ok = heldout_dir is None or (
        cached_heldout.exists() and len(list(cached_heldout.glob("*.wav"))) > 0
    )
    if not (
        ready.exists()
        and len(list(cached_samples.glob("*.wav"))) >= max_samples
        and heldout_ok
        and all(
            _ragged_mmap_complete(cached_features / split / "wakeword_mmap")
            for split in ("training", "validation", "testing")
        )
    ):
        return False

    shutil.copytree(cached_samples, samples_dir, copy_function=_link_or_copy)
    shutil.copytree(cached_features, features_dir, copy_function=_link_or_copy)
    if heldout_dir is not None and cached_heldout.exists():
        shutil.copytree(cached_heldout, heldout_dir, copy_function=_link_or_copy)
    logging.info("restored synthetic samples/features cache key=%s", key[:12])
    return True


def publish_synthetic_asset_cache(
    cache_root: Path,
    wakeword: str,
    max_samples: int,
    samples_dir: Path,
    features_dir: Path,
    heldout_dir: Path | None = None,
) -> None:
    key = _synthetic_asset_cache_key(wakeword, max_samples)
    entry = cache_root / key
    if (entry / "READY").exists():
        return
    cache_root.mkdir(parents=True, exist_ok=True)
    temporary = cache_root / f".{key}.{os.getpid()}.tmp"
    shutil.rmtree(temporary, ignore_errors=True)
    temporary.mkdir(parents=True)
    try:
        shutil.copytree(
            samples_dir, temporary / "samples", copy_function=_link_or_copy
        )
        shutil.copytree(
            features_dir, temporary / "features", copy_function=_link_or_copy
        )
        if heldout_dir is not None and heldout_dir.exists():
            shutil.copytree(
                heldout_dir, temporary / "heldout", copy_function=_link_or_copy
            )
        (temporary / "READY").write_text(key, encoding="utf-8")
        try:
            temporary.replace(entry)
        except OSError:
            if not entry.exists():
                raise
    finally:
        shutil.rmtree(temporary, ignore_errors=True)


def ensure_negative_datasets(base_dir: Path) -> None:
    base_dir.mkdir(parents=True, exist_ok=True)
    for archive, folder in NEGATIVE_DATASETS.items():
        target_dir = base_dir / folder
        mmap_directories = (
            [
                path
                for path in target_dir.rglob("*_mmap")
                if path.is_dir() and _ragged_mmap_complete(path)
            ]
            if target_dir.exists()
            else []
        )
        if mmap_directories:
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


def _wav_count(directory: Path) -> int:
    if not directory.exists():
        return 0
    return sum(1 for _ in directory.rglob("*.wav"))


def _extract_archive(archive_path: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path, "r") as zip_file:
        zip_file.extractall(destination)


def _download_audio_archive(
    *, url: str, archive_path: Path, extract_dir: Path, min_wavs: int, label: str
) -> Path:
    if _wav_count(extract_dir) >= min_wavs:
        logging.info("%s dataset already present (%d wavs)", label, _wav_count(extract_dir))
        return extract_dir
    if not archive_path.exists():
        logging.info("downloading %s dataset from %s", label, url)
        download_file(url, archive_path)
    logging.info("extracting %s dataset", label)
    _extract_archive(archive_path, extract_dir)
    count = _wav_count(extract_dir)
    if count < min_wavs:
        raise FileNotFoundError(
            f"{label} dataset at {extract_dir} has {count} wav files; need >= {min_wavs}"
        )
    logging.info("completed %s dataset extraction (%d wavs)", label, count)
    return extract_dir


def _download_esc50_files(destination: Path, count: int = NOISE_FILE_COUNT) -> Path:
    """Fetch a class-strided ESC-50 subset via jsDelivr (avoids a 600MB GitHub zip)."""
    destination.mkdir(parents=True, exist_ok=True)
    index_path = destination / "esc50.csv"
    if not index_path.exists():
        logging.info("downloading ESC-50 index from %s", DEFAULT_NOISE_INDEX_URL)
        download_file(DEFAULT_NOISE_INDEX_URL, index_path)
    names: list[str] = []
    for line in index_path.read_text(encoding="utf-8").splitlines()[1:]:
        filename = line.split(",")[0].strip()
        if filename.endswith(".wav"):
            names.append(filename)
    if not names:
        raise FileNotFoundError(f"ESC-50 index {index_path} contained no wav names")
    step = max(1, len(names) // max(1, count))
    chosen = names[::step][:count]
    audio_dir = destination / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    for name in chosen:
        dest = audio_dir / name
        if dest.exists() and dest.stat().st_size > 0:
            continue
        download_file(DEFAULT_NOISE_FILE_URL.format(filename=name), dest)
        if dest.exists() and dest.stat().st_size == 0:
            dest.unlink()
            download_file(DEFAULT_NOISE_FILE_URL.format(filename=name), dest)
    found = _wav_count(destination)
    if found < min(count, NOISE_MIN_WAVS):
        raise FileNotFoundError(
            f"ESC-50 subset at {destination} has {found} wav files; need >= {NOISE_MIN_WAVS}"
        )
    logging.info("completed ESC-50 subset download (%d wavs)", found)
    return destination


def ensure_augmentation_datasets(
    base_dir: Path = DEFAULT_AUGMENTATION_DIR,
) -> tuple[list[str], list[str]]:
    """Download MIT IR Survey + ESC-50 and return (impulse_paths, background_paths)."""
    base_dir.mkdir(parents=True, exist_ok=True)
    impulse_dir = _download_audio_archive(
        url=DEFAULT_RIR_URL,
        archive_path=base_dir / "mit_ir_survey.zip",
        extract_dir=base_dir / "rir",
        min_wavs=RIR_MIN_WAVS,
        label="RIR",
    )
    noise_dir = base_dir / "noise"
    if DEFAULT_NOISE_URL:
        background_dir = _download_audio_archive(
            url=DEFAULT_NOISE_URL,
            archive_path=base_dir / "noise.zip",
            extract_dir=noise_dir,
            min_wavs=NOISE_MIN_WAVS,
            label="noise",
        )
    else:
        if _wav_count(noise_dir) >= NOISE_MIN_WAVS:
            logging.info(
                "noise dataset already present (%d wavs)", _wav_count(noise_dir)
            )
            background_dir = noise_dir
        else:
            background_dir = _download_esc50_files(noise_dir)
    return [str(impulse_dir)], [str(background_dir)]


def _build_augmenter(
    probabilities: dict,
    impulse_paths: list[str],
    background_paths: list[str],
):
    from microwakeword.audio.augmentation import Augmentation

    if not impulse_paths or not background_paths:
        raise ValueError(
            "acoustic augmentation requires both impulse_paths and background_paths"
        )
    return Augmentation(
        augmentation_duration_s=3.2,
        augmentation_probabilities=probabilities,
        impulse_paths=impulse_paths,
        background_paths=background_paths,
        background_min_snr_db=-5,
        background_max_snr_db=10,
        min_jitter_s=0.195,
        max_jitter_s=0.205,
    )


def generate_positive_feature_sets(
    samples_dir: Path,
    features_dir: Path,
    *,
    heldout_samples_dir: Path | None = None,
    impulse_paths: list[str] | None = None,
    background_paths: list[str] | None = None,
) -> None:
    from mmap_ninja.ragged import RaggedMmap  # type: ignore[import-not-found]

    from microwakeword.audio.clips import Clips
    from microwakeword.audio.spectrograms import SpectrogramGeneration

    if not impulse_paths or not background_paths:
        impulse_paths, background_paths = ensure_augmentation_datasets()

    features_dir.mkdir(parents=True, exist_ok=True)
    clips = Clips(
        input_directory=str(samples_dir),
        file_pattern="**/*.wav",
        remove_silence=False,
        random_split_seed=10,
        split_count=0.1,
    )
    train_augmenter = _build_augmenter(
        TRAIN_AUGMENTATION_PROBABILITIES, impulse_paths, background_paths
    )
    test_augmenter = _build_augmenter(
        TEST_AUGMENTATION_PROBABILITIES, impulse_paths, background_paths
    )
    heldout_clips = None
    if heldout_samples_dir is not None and list(Path(heldout_samples_dir).rglob("*.wav")):
        heldout_clips = Clips(
            input_directory=str(heldout_samples_dir),
            file_pattern="**/*.wav",
            remove_silence=False,
            random_split_seed=None,
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
            spectrograms = SpectrogramGeneration(
                clips=clips, augmenter=train_augmenter, slide_frames=10, step_ms=10
            )
            generator = spectrograms.spectrogram_generator(split="train", repeat=2)
        elif split == "validation":
            spectrograms = SpectrogramGeneration(
                clips=clips, augmenter=train_augmenter, slide_frames=10, step_ms=10
            )
            generator = spectrograms.spectrogram_generator(split="validation", repeat=1)
        elif heldout_clips is not None:
            spectrograms = SpectrogramGeneration(
                clips=heldout_clips,
                augmenter=test_augmenter,
                slide_frames=1,
                step_ms=10,
            )
            generator = spectrograms.spectrogram_generator(repeat=1)
        else:
            spectrograms = SpectrogramGeneration(
                clips=clips, augmenter=test_augmenter, slide_frames=1, step_ms=10
            )
            generator = spectrograms.spectrogram_generator(split="test", repeat=1)

        logging.info(
            "creating positive feature set for %s (heldout=%s)",
            split,
            heldout_clips is not None and split == "testing",
        )
        RaggedMmap.from_generator(
            out_dir=str(mmap_dir),
            sample_generator=generator,
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
        "early_stop_enabled": DEFAULT_EARLY_STOP,
        "early_stop_min_steps": DEFAULT_EARLY_STOP_MIN_STEPS,
        "early_stop_patience_evals": DEFAULT_EARLY_STOP_PATIENCE,
        "early_stop_min_delta": DEFAULT_EARLY_STOP_MIN_DELTA,
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

    heldout_dir = session_dir / "heldout_samples"
    heldout_count = max(40, int(req.max_samples * 0.1))

    start_time = time.time()
    synthetic_cache_hit = False
    synthetic_cache_root = cache_dir / "synthetic_asset_cache"
    if not use_external_samples:
        synthetic_cache_hit = restore_synthetic_asset_cache(
            synthetic_cache_root,
            wakeword,
            req.max_samples,
            samples_dir,
            features_dir,
            heldout_dir=heldout_dir,
        )

    stage("synthesize")
    if not use_external_samples:
        ensure_wakeword_samples(
            wakeword=wakeword,
            samples_dir=samples_dir,
            max_samples=req.max_samples,
            batch_size=req.sample_batch_size,
        )
        ensure_wakeword_samples(
            wakeword=wakeword,
            samples_dir=heldout_dir,
            max_samples=heldout_count,
            batch_size=req.sample_batch_size,
            voice_model=DEFAULT_TEST_VOICE_MODEL,
            voice_config=DEFAULT_TEST_VOICE_CONFIG,
            voice_url=DEFAULT_TEST_VOICE_URL,
            voice_config_url=DEFAULT_TEST_VOICE_CONFIG_URL,
        )

    stage("negatives")
    ensure_negative_datasets(negatives_dir)

    stage("augmentation")
    impulse_paths, background_paths = ensure_augmentation_datasets(
        Path(os.getenv("MICROWAKEWORD_AUGMENTATION_DIR", str(DEFAULT_AUGMENTATION_DIR)))
    )

    stage("features")
    generate_positive_feature_sets(
        samples_dir,
        features_dir,
        heldout_samples_dir=heldout_dir if not use_external_samples else None,
        impulse_paths=impulse_paths,
        background_paths=background_paths,
    )
    if not use_external_samples and not synthetic_cache_hit:
        publish_synthetic_asset_cache(
            synthetic_cache_root,
            wakeword,
            req.max_samples,
            samples_dir,
            features_dir,
            heldout_dir=heldout_dir,
        )
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

    stage("probe")
    from microwakeword.domain_probe import list_wavs, probe_exported_model

    probe_samples = list_wavs(samples_dir, limit=32)
    quality_path = session_dir / "quality_report.json"
    quality = probe_exported_model(
        model_path=model_path,
        sample_wavs=probe_samples,
        impulse_dir=Path(impulse_paths[0]),
        noise_dir=Path(background_paths[0]),
        output_path=quality_path,
    )
    shutil.copy2(quality_path, output_dir / f"{slug}.quality.json")
    if quality["quality_warning"]:
        logging.warning("quality gate: %s", quality["quality_message"])

    duration = timedelta(seconds=int(time.time() - start_time))
    logging.info(
        "training complete for '%s'; model saved to %s (took %s)",
        wakeword,
        output_model,
        duration,
    )
    stage("done")
    return output_model
