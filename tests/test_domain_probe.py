# coding=utf-8
from pathlib import Path

import numpy as np
from scipy.io import wavfile

from microwakeword.domain_probe import (
    ENHANCED_MEDIAN_MIN,
    apply_impulse_response,
    interpret_domain_gap,
    load_mono_wav,
    mix_noise,
    select_cutoff,
    sliding_window_peaks,
    summarize_peaks,
)


def test_sliding_window_peaks_uses_max_moving_average():
    track = [0.0, 0.0, 1.0, 1.0, 1.0, 0.0]
    assert sliding_window_peaks(track, window=3) == 1.0
    assert sliding_window_peaks([], window=5) == 0.0
    assert sliding_window_peaks([0.4], window=5) == 0.4


def test_summarize_peaks_percentiles():
    peaks = [0.1, 0.2, 0.3, 0.4, 0.5]
    summary = summarize_peaks(peaks)
    assert summary["n"] == 5
    assert summary["min"] == 0.1
    assert summary["max"] == 0.5
    assert abs(summary["p50"] - 0.3) < 1e-9


def test_select_cutoff_warns_when_enhanced_median_collapses():
    decision = select_cutoff([0.0, 0.01, 0.02], [0.0, 0.0, 0.01])
    assert decision["quality_warning"] is True
    assert decision["enhanced"]["p50"] < ENHANCED_MEDIAN_MIN
    assert 0.05 <= decision["suggested_cutoff"] <= 0.85


def test_select_cutoff_from_separable_distributions():
    enhanced = [0.55, 0.62, 0.70, 0.81, 0.90]
    ambient = [0.01, 0.02, 0.03, 0.04]
    decision = select_cutoff(enhanced, ambient)
    assert decision["quality_warning"] is False
    assert decision["suggested_cutoff"] < min(enhanced)
    assert decision["suggested_cutoff"] > max(ambient)


def test_apply_impulse_response_lengthens_and_keeps_energy():
    clip = np.zeros(1600, dtype=np.float32)
    clip[200:400] = 0.4
    ir = np.array([1.0, 0.4, 0.1], dtype=np.float32)
    wet = apply_impulse_response(clip, ir)
    assert wet.size >= clip.size
    assert float(np.max(np.abs(wet))) > 0.0


def test_mix_noise_at_10db_raises_energy():
    rng = np.random.default_rng(0)
    clip = np.zeros(1600, dtype=np.float32)
    clip[100:500] = 0.3
    noise = rng.normal(0, 0.4, size=1600).astype(np.float32)
    mixed = mix_noise(clip, noise, snr_db=10.0)
    assert mixed.shape == clip.shape
    assert float(np.mean(np.square(mixed))) > float(np.mean(np.square(clip)))


def test_interpret_domain_gap_confirms_clean_vs_enhanced_collapse():
    clean = summarize_peaks([0.9, 0.95, 1.0])
    rir = summarize_peaks([0.02, 0.03, 0.04])
    enhanced = summarize_peaks([0.0, 0.01, 0.02])
    verdict = interpret_domain_gap(clean, rir, enhanced)
    assert "domain gap confirmed" in verdict


def test_interpret_export_failure_when_all_near_zero():
    dead = summarize_peaks([0.0, 0.01, 0.02])
    verdict = interpret_domain_gap(dead, dead, dead)
    assert "export/quantization" in verdict


def test_load_mono_wav_resamples(tmp_path: Path):
    path = tmp_path / "tone.wav"
    samples = (np.linspace(-0.2, 0.2, 8000) * 32767).astype(np.int16)
    wavfile.write(path, 8000, samples)
    loaded = load_mono_wav(path, sample_rate=16000)
    assert loaded.dtype == np.float32
    assert 15900 <= loaded.size <= 16100
