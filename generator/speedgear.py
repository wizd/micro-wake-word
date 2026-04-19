#!/usr/bin/env python3
"""Generate speed-augmented wav datasets for micro-wake-word training."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import Iterable

RATES = [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.4, 1.6]
TARGET_SAMPLE_RATE = 16000


def format_rate(rate: float) -> str:
    """Keep directory names stable as sp_0.7, sp_1.0, etc."""
    return f"{rate:.1f}"


def list_wavs(src_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in src_dir.rglob("*.wav")
        if path.is_file() and not any(part.startswith("sp_") for part in path.relative_to(src_dir).parts)
    )


def process_directory(
    src_dir: Path,
    rates: Iterable[float],
    overwrite: bool,
    librosa_module,
    sf_module,
    tqdm_module,
) -> None:
    wav_files = list_wavs(src_dir)
    if not wav_files:
        print(f"[WARN] No wav files in: {src_dir}")
        return

    print(f"[INFO] Processing {src_dir} ({len(wav_files)} wav files)")
    for rate in rates:
        rate_name = format_rate(rate)
        out_dir = src_dir / f"sp_{rate_name}"
        out_dir.mkdir(parents=True, exist_ok=True)

        for wav_file in tqdm_module(
            wav_files,
            desc=f"{src_dir.name} -> sp_{rate_name}",
            unit="file",
        ):
            relative_wav_path = wav_file.relative_to(src_dir)
            out_file = out_dir / relative_wav_path
            out_file.parent.mkdir(parents=True, exist_ok=True)
            if out_file.exists() and not overwrite:
                continue

            if rate == 1.0:
                # Keep 1.0 output deterministic and faster by direct copy.
                shutil.copy2(wav_file, out_file)
                continue

            audio, _ = librosa_module.load(wav_file, sr=TARGET_SAMPLE_RATE, mono=True)
            stretched = librosa_module.effects.time_stretch(audio, rate=rate)
            sf_module.write(out_file, stretched, TARGET_SAMPLE_RATE, subtype="PCM_16")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate speed-augmented wav files.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("~/data/nihao_saisai").expanduser(),
        help="Dataset root containing negative-samples and voice-samples.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing augmented wav files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.expanduser().resolve()

    try:
        import librosa
        import soundfile as sf
        from tqdm import tqdm
    except ModuleNotFoundError as err:
        missing_module = err.name or "dependency"
        raise SystemExit(
            f"Missing Python dependency: {missing_module}. "
            f"Current interpreter: {sys.executable}. "
            "Please install with: python -m pip install librosa soundfile tqdm "
            "or run this script with python3.10."
        ) from err

    target_dirs = [
        data_dir / "negative-samples",
        data_dir / "voice-samples" / "nihaosaisai",
    ]

    missing_dirs = [str(path) for path in target_dirs if not path.is_dir()]
    if missing_dirs:
        raise FileNotFoundError(f"Missing required directories: {', '.join(missing_dirs)}")

    for src_dir in target_dirs:
        process_directory(
            src_dir,
            RATES,
            overwrite=args.overwrite,
            librosa_module=librosa,
            sf_module=sf,
            tqdm_module=tqdm,
        )

    print("[DONE] Speed augmentation completed.")


if __name__ == "__main__":
    main()
