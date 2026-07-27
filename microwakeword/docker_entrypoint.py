"""Container entrypoint for training microWakeWord models.

Thin CLI wrapper around :mod:`microwakeword.pipeline`.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

from microwakeword.pipeline import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_CACHE_DIR,
    DEFAULT_CUSTOM_NEGATIVE_DIR,
    DEFAULT_HARD_NEG_PENALTY,
    DEFAULT_HARD_NEG_SAMPLING,
    DEFAULT_NEGATIVE_DIR,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_SAMPLE_COUNT,
    DEFAULT_SAMPLES_DIR,
    DEFAULT_TRAINING_STEPS,
    TrainingRequest,
    run_pipeline,
    slugify_phrase,
)


class UTCFormatter(logging.Formatter):
    """Formatter that forces UTC timestamps."""

    def formatTime(self, record, datefmt=None):  # noqa: D401 (inherits docstring)
        dt = time.gmtime(record.created)
        if datefmt:
            return time.strftime(datefmt, dt) + "Z"
        return time.strftime("%Y-%m-%dT%H:%M:%S", dt) + "Z"


def configure_logging() -> None:
    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(UTCFormatter("%(asctime)s %(levelname)s %(message)s"))
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers.clear()
    root.addHandler(handler)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a wake word model inside Docker")
    parser.add_argument(
        "-c",
        "--wakeword",
        required=True,
        help="Wake word phrase to synthesise and train",
    )
    parser.add_argument(
        "-s",
        "--samples-dir",
        type=str,
        default=None,
        help="Path to directory containing pre-generated WAV samples (16kHz, mono, 16-bit). "
        "If not provided, uses MICROWAKEWORD_SAMPLES_DIR or generates samples using Piper TTS.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        default=None,
        help="Path to output directory for the trained model. "
        "Defaults to MICROWAKEWORD_OUTPUT_DIR env var or /output.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=DEFAULT_SAMPLE_COUNT,
        help="Number of synthetic wake word samples to generate (ignored if samples are provided)",
    )
    parser.add_argument(
        "--sample-batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Sample generation batch size (unused placeholder for compatibility)",
    )
    parser.add_argument(
        "--training-steps",
        type=int,
        default=DEFAULT_TRAINING_STEPS,
        help="Training steps per iteration",
    )
    parser.add_argument(
        "--negative-samples-dir",
        type=str,
        default=None,
        help="Path to directory containing custom negative WAV samples (16kHz, mono, 16-bit). "
        "These will be added alongside the default negative datasets.",
    )
    parser.add_argument(
        "--hard-negative-penalty-weight",
        type=float,
        default=DEFAULT_HARD_NEG_PENALTY,
        help=(
            "Penalty multiplier for custom hard-negative samples. "
            "Final loss weight = penalty_weight * negative_class_weight."
        ),
    )
    parser.add_argument(
        "--hard-negative-sampling-weight",
        type=float,
        default=DEFAULT_HARD_NEG_SAMPLING,
        help="Sampling weight for how frequently custom hard-negative samples are drawn.",
    )
    args = parser.parse_args()

    configure_logging()

    wakeword = args.wakeword.strip()
    if not wakeword:
        parser.error("Wake word must not be empty")

    output_dir = Path(args.output_dir) if args.output_dir else DEFAULT_OUTPUT_DIR
    samples_dir = Path(args.samples_dir) if args.samples_dir else None
    custom_negative_dir = (
        Path(args.negative_samples_dir) if args.negative_samples_dir else None
    )

    # Fall back to default mount points if they contain WAV files
    if samples_dir is None and DEFAULT_SAMPLES_DIR.exists() and list(
        DEFAULT_SAMPLES_DIR.rglob("*.wav")
    ):
        samples_dir = DEFAULT_SAMPLES_DIR

    if custom_negative_dir is None and DEFAULT_CUSTOM_NEGATIVE_DIR.exists() and list(
        DEFAULT_CUSTOM_NEGATIVE_DIR.rglob("*.wav")
    ):
        custom_negative_dir = DEFAULT_CUSTOM_NEGATIVE_DIR

    slug = slugify_phrase(wakeword)
    req = TrainingRequest(
        wakeword=wakeword,
        job_id=slug,
        samples_dir=samples_dir,
        output_dir=output_dir,
        cache_dir=DEFAULT_CACHE_DIR,
        negatives_dir=DEFAULT_NEGATIVE_DIR,
        custom_negative_dir=custom_negative_dir,
        max_samples=args.max_samples,
        sample_batch_size=args.sample_batch_size,
        training_steps=args.training_steps,
        hard_negative_penalty_weight=args.hard_negative_penalty_weight,
        hard_negative_sampling_weight=args.hard_negative_sampling_weight,
    )

    try:
        run_pipeline(req)
    except Exception as exc:  # pragma: no cover - to aid manual diagnosis
        logging.exception("failed to train wake word model: %s", exc)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
