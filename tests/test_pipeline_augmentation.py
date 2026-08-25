# coding=utf-8
from pathlib import Path

from microwakeword.pipeline import (
    AUGMENTATION_REVISION,
    TEST_AUGMENTATION_PROBABILITIES,
    TRAIN_AUGMENTATION_PROBABILITIES,
    _download_audio_archive,
    _synthetic_asset_cache_key,
    _wav_count,
)


def test_augmentation_probabilities_are_enabled():
    assert AUGMENTATION_REVISION >= 2
    assert TRAIN_AUGMENTATION_PROBABILITIES["RIR"] > 0
    assert TRAIN_AUGMENTATION_PROBABILITIES["AddBackgroundNoise"] > 0
    assert TRAIN_AUGMENTATION_PROBABILITIES["AddColorNoise"] > 0
    assert TEST_AUGMENTATION_PROBABILITIES["RIR"] == 1.0
    assert TEST_AUGMENTATION_PROBABILITIES["AddBackgroundNoise"] == 1.0


def test_cache_key_includes_revision_and_is_stable():
    first = _synthetic_asset_cache_key("御前卫", 400)
    second = _synthetic_asset_cache_key("御前卫", 400)
    other = _synthetic_asset_cache_key("御前卫", 401)
    assert first == second
    assert first != other
    assert len(first) == 64


def test_download_skips_when_enough_wavs(tmp_path: Path):
    extract = tmp_path / "rir"
    extract.mkdir()
    for index in range(25):
        (extract / f"{index}.wav").write_bytes(b"RIFF")
    result = _download_audio_archive(
        url="http://invalid.example/missing.zip",
        archive_path=tmp_path / "missing.zip",
        extract_dir=extract,
        min_wavs=20,
        label="RIR",
    )
    assert result == extract
    assert _wav_count(extract) == 25
    assert not (tmp_path / "missing.zip").exists()
