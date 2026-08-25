# coding=utf-8
"""Offline domain-gap probe for exported microWakeWord models.

Scores the same positive clips in three acoustic conditions (clean / RIR /
RIR+noise) so a model that only memorized dry TTS is visible before device
flash. Also proposes a probability_cutoff from the enhanced-positive and
ambient peak distributions.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from scipy.io import wavfile

DEFAULT_STRIDE = 3
DEFAULT_SLIDING_WINDOW = 5
DEFAULT_SNR_DB = 10.0
ENHANCED_MEDIAN_MIN = 0.15
CUTOFF_FLOOR = 0.05
CUTOFF_CEILING = 0.85


def sliding_window_peaks(
    probabilities: Sequence[float], window: int = DEFAULT_SLIDING_WINDOW
) -> float:
    """Return the max moving-average of a streaming probability track."""
    values = np.asarray(list(probabilities), dtype=np.float64).reshape(-1)
    if values.size == 0:
        return 0.0
    if values.size < window:
        return float(np.mean(values))
    kernel = np.ones(window, dtype=np.float64) / window
    moving = np.convolve(values, kernel, mode="valid")
    return float(np.max(moving))


def summarize_peaks(peaks: Sequence[float]) -> dict[str, float]:
    values = np.asarray(list(peaks), dtype=np.float64).reshape(-1)
    if values.size == 0:
        return {
            "n": 0.0,
            "min": 0.0,
            "mean": 0.0,
            "p10": 0.0,
            "p25": 0.0,
            "p50": 0.0,
            "p90": 0.0,
            "max": 0.0,
        }
    percentiles = np.percentile(values, [10, 25, 50, 90])
    return {
        "n": float(values.size),
        "min": float(np.min(values)),
        "mean": float(np.mean(values)),
        "p10": float(percentiles[0]),
        "p25": float(percentiles[1]),
        "p50": float(percentiles[2]),
        "p90": float(percentiles[3]),
        "max": float(np.max(values)),
    }


def select_cutoff(
    enhanced_peaks: Sequence[float],
    ambient_peaks: Sequence[float] | None = None,
    *,
    median_min: float = ENHANCED_MEDIAN_MIN,
) -> dict:
    """Choose a device cutoff from enhanced-positive vs ambient peaks.

    The cutoff sits above ambient traffic and below most enhanced positives.
    A warning is raised when the enhanced median is too low to be usable.
    """
    enhanced = summarize_peaks(enhanced_peaks)
    ambient = summarize_peaks(ambient_peaks or [])
    raw = min(enhanced["p25"] * 0.6, enhanced["p50"] * 0.4)
    if raw <= 0.0:
        raw = enhanced["p50"] * 0.5
    ambient_floor = ambient["p90"] + 0.05 if ambient["n"] else 0.0
    cutoff = float(np.clip(max(raw, ambient_floor), CUTOFF_FLOOR, CUTOFF_CEILING))
    warning = bool(enhanced["n"] == 0 or enhanced["p50"] < median_min)
    if enhanced["n"] == 0:
        message = "no enhanced positive scores; cannot certify the model"
    elif warning:
        message = (
            f"enhanced-positive median {enhanced['p50']:.3f} "
            f"< {median_min:.2f}; model is unlikely to fire on real audio"
        )
    else:
        message = (
            f"cutoff {cutoff:.3f} from enhanced p25={enhanced['p25']:.3f} "
            f"p50={enhanced['p50']:.3f} ambient_p90={ambient['p90']:.3f}"
        )
    return {
        "suggested_cutoff": cutoff,
        "quality_warning": warning,
        "quality_message": message,
        "enhanced": enhanced,
        "ambient": ambient,
    }


def load_mono_wav(path: Path, sample_rate: int = 16000) -> np.ndarray:
    rate, data = wavfile.read(str(path))
    if data.ndim > 1:
        data = data[:, 0]
    if data.dtype == np.int16:
        samples = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        samples = data.astype(np.float32) / 2147483648.0
    elif data.dtype in (np.float32, np.float64):
        samples = data.astype(np.float32)
    else:
        samples = data.astype(np.float32)
        peak = np.max(np.abs(samples))
        if peak > 1.5:
            samples = samples / 32768.0
    if rate != sample_rate and samples.size > 1:
        duration = samples.size / float(rate)
        target = max(1, int(round(duration * sample_rate)))
        samples = np.interp(
            np.linspace(0.0, samples.size - 1, target),
            np.arange(samples.size),
            samples,
        ).astype(np.float32)
    return np.clip(samples, -1.0, 1.0)


def list_wavs(directory: Path, limit: int | None = None) -> list[Path]:
    if not directory or not Path(directory).exists():
        return []
    paths = sorted(Path(directory).rglob("*.wav"))
    if limit is not None:
        return paths[: int(limit)]
    return paths


def apply_impulse_response(samples: np.ndarray, impulse: np.ndarray) -> np.ndarray:
    if impulse.size == 0:
        return samples.astype(np.float32)
    ir = impulse.astype(np.float32)
    ir_peak = np.max(np.abs(ir))
    if ir_peak > 1e-8:
        ir = ir / ir_peak
    wet = np.convolve(samples.astype(np.float32), ir, mode="full")
    source_peak = float(np.max(np.abs(samples))) if samples.size else 0.0
    wet_peak = float(np.max(np.abs(wet)))
    if wet_peak > 1e-8 and source_peak > 0.0:
        wet = wet * (source_peak / wet_peak)
    keep = max(samples.size, min(wet.size, samples.size + int(0.4 * 16000)))
    return np.clip(wet[:keep], -1.0, 1.0).astype(np.float32)


def mix_noise(
    samples: np.ndarray, noise: np.ndarray, snr_db: float = DEFAULT_SNR_DB
) -> np.ndarray:
    if noise.size == 0:
        return samples.astype(np.float32)
    tiled = noise.astype(np.float32)
    if tiled.size < samples.size:
        repeats = int(np.ceil(samples.size / tiled.size))
        tiled = np.tile(tiled, repeats)
    if tiled.size > samples.size:
        offset = int(np.random.randint(0, tiled.size - samples.size + 1))
        tiled = tiled[offset : offset + samples.size]
    signal_power = float(np.mean(np.square(samples)))
    noise_power = float(np.mean(np.square(tiled)))
    if noise_power < 1e-12:
        return samples.astype(np.float32)
    if signal_power < 1e-12:
        peak = float(np.max(np.abs(tiled)))
        return (tiled / peak).astype(np.float32) if peak > 1e-8 else tiled
    scale = np.sqrt(signal_power / (noise_power * (10.0 ** (snr_db / 10.0))))
    mixed = samples.astype(np.float32) + (scale * tiled)
    peak = float(np.max(np.abs(mixed)))
    if peak > 1.0:
        mixed = mixed / peak
    return mixed.astype(np.float32)


def interpret_domain_gap(clean: dict, rir: dict, rir_noise: dict) -> str:
    """Human-readable verdict used by Phase 0 logs and quality reports."""
    if clean["n"] == 0:
        return "no clean scores; probe did not run"
    if clean["p50"] < ENHANCED_MEDIAN_MIN and rir_noise["p50"] < ENHANCED_MEDIAN_MIN:
        return (
            "clean and enhanced medians are both near zero; "
            "suspect export/quantization, not just domain gap"
        )
    if clean["p50"] >= 0.5 and rir_noise["p50"] < ENHANCED_MEDIAN_MIN:
        return (
            "domain gap confirmed: dry positives score high, "
            "RIR+noise positives collapse"
        )
    if rir_noise["p50"] >= ENHANCED_MEDIAN_MIN:
        return "enhanced positives remain separable; domain gap is not catastrophic"
    return (
        f"partial gap: clean p50={clean['p50']:.3f} "
        f"rir p50={rir['p50']:.3f} rir_noise p50={rir_noise['p50']:.3f}"
    )


def _peak_for_clip(model, samples: np.ndarray, stride: int, window: int) -> float:
    predictions = model.predict_clip(samples, step_ms=10)
    return sliding_window_peaks(predictions, window=window)


def probe_exported_model(
    *,
    model_path: Path,
    sample_wavs: Sequence[Path],
    impulse_dir: Path | None,
    noise_dir: Path | None,
    output_path: Path | None = None,
    stride: int = DEFAULT_STRIDE,
    sliding_window: int = DEFAULT_SLIDING_WINDOW,
    snr_db: float = DEFAULT_SNR_DB,
    max_clips: int = 32,
) -> dict:
    """Score clean / RIR / RIR+noise variants and propose a cutoff."""
    from microwakeword.inference import Model

    wavs = [Path(p) for p in sample_wavs if Path(p).exists()][:max_clips]
    if not wavs:
        raise FileNotFoundError("probe requires at least one positive WAV")

    impulses = list_wavs(Path(impulse_dir), limit=32) if impulse_dir else []
    noises = list_wavs(Path(noise_dir), limit=32) if noise_dir else []
    if not impulses:
        raise FileNotFoundError(f"no impulse responses in {impulse_dir}")
    if not noises:
        raise FileNotFoundError(f"no noise clips in {noise_dir}")

    model = Model(str(model_path), stride=stride)
    rng = np.random.default_rng(7)
    clean_peaks: list[float] = []
    rir_peaks: list[float] = []
    enhanced_peaks: list[float] = []
    ambient_peaks: list[float] = []

    for index, wav_path in enumerate(wavs):
        clip = load_mono_wav(wav_path)
        ir = load_mono_wav(impulses[index % len(impulses)])
        noise = load_mono_wav(noises[index % len(noises)])
        wet = apply_impulse_response(clip, ir)
        enhanced = mix_noise(wet, noise, snr_db=snr_db)
        clean_peaks.append(_peak_for_clip(model, clip, stride, sliding_window))
        rir_peaks.append(_peak_for_clip(model, wet, stride, sliding_window))
        enhanced_peaks.append(
            _peak_for_clip(model, enhanced, stride, sliding_window)
        )
        ambient = mix_noise(
            np.zeros_like(clip),
            noise * (0.4 + 0.4 * float(rng.random())),
            snr_db=0.0,
        )
        ambient_peaks.append(_peak_for_clip(model, ambient, stride, sliding_window))

    clean_summary = summarize_peaks(clean_peaks)
    rir_summary = summarize_peaks(rir_peaks)
    enhanced_summary = summarize_peaks(enhanced_peaks)
    decision = select_cutoff(enhanced_peaks, ambient_peaks)
    report = {
        "model_path": str(model_path),
        "clip_count": len(wavs),
        "stride": stride,
        "sliding_window": sliding_window,
        "snr_db": snr_db,
        "variants": {
            "clean": clean_summary,
            "rir": rir_summary,
            "rir_noise": enhanced_summary,
        },
        "ambient": decision["ambient"],
        "suggested_cutoff": decision["suggested_cutoff"],
        "quality_warning": decision["quality_warning"],
        "quality_message": decision["quality_message"],
        "verdict": interpret_domain_gap(clean_summary, rir_summary, enhanced_summary),
    }
    logging.info(
        "domain probe clean p50=%.3f rir p50=%.3f rir_noise p50=%.3f cutoff=%.3f warning=%s verdict=%s",
        clean_summary["p50"],
        rir_summary["p50"],
        enhanced_summary["p50"],
        report["suggested_cutoff"],
        report["quality_warning"],
        report["verdict"],
    )
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
        )
    return report


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--samples", required=True, type=Path)
    parser.add_argument("--impulse-dir", required=True, type=Path)
    parser.add_argument("--noise-dir", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--max-clips", type=int, default=32)
    args = parser.parse_args(list(argv) if argv is not None else None)
    report = probe_exported_model(
        model_path=args.model,
        sample_wavs=list_wavs(args.samples, limit=args.max_clips),
        impulse_dir=args.impulse_dir,
        noise_dir=args.noise_dir,
        output_path=args.output,
        max_clips=args.max_clips,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if not report["quality_warning"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
