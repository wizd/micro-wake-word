#!/usr/bin/env python3
"""Phase 0: score an exported job model on clean / RIR / RIR+noise clips."""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from microwakeword.domain_probe import list_wavs, probe_exported_model
from microwakeword.pipeline import (
    DEFAULT_SAMPLE_COUNT,
    ensure_augmentation_datasets,
    ensure_wakeword_samples,
)


def _download(url: str, destination: Path, api_key: str | None) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url)
    if api_key:
        request.add_header("X-API-Key", api_key)
    with urllib.request.urlopen(request, timeout=120) as response, destination.open(
        "wb"
    ) as output:
        output.write(response.read())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-id", default="8b01c7a84a68")
    parser.add_argument("--wakeword", default="御前卫")
    parser.add_argument(
        "--base-url",
        default=os.getenv(
            "WAKE_WORD_TRAINING_URL",
            "https://u900415-8ab3-1acaf514.westb.seetacloud.com:8443",
        ),
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv(
            "WAKE_WORD_TRAINING_API_KEY",
            "kIcOmheLPbQAdV9Nr0hLIYLGcSqWIXiUad0-NHHS-_I",
        ),
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / ".probe",
    )
    parser.add_argument("--max-clips", type=int, default=16)
    args = parser.parse_args()

    work = args.work_dir
    model_path = work / f"{args.job_id}.tflite"
    samples_dir = work / "samples"
    report_path = work / f"{args.job_id}.quality.json"
    if not model_path.exists():
        url = f"{args.base_url.rstrip('/')}/api/v1/jobs/{args.job_id}/model"
        print(f"downloading model from {url}")
        _download(url, model_path, args.api_key)

    print("ensuring RIR + noise datasets")
    impulse_paths, background_paths = ensure_augmentation_datasets(work / "augmentation")

    if not list_wavs(samples_dir):
        print(f"synthesizing {args.max_clips} '{args.wakeword}' samples")
        ensure_wakeword_samples(
            wakeword=args.wakeword,
            samples_dir=samples_dir,
            max_samples=max(args.max_clips, 8),
            batch_size=DEFAULT_SAMPLE_COUNT,
        )

    report = probe_exported_model(
        model_path=model_path,
        sample_wavs=list_wavs(samples_dir, limit=args.max_clips),
        impulse_dir=Path(impulse_paths[0]),
        noise_dir=Path(background_paths[0]),
        output_path=report_path,
        max_clips=args.max_clips,
    )
    print(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False))
    return 0 if not report["quality_warning"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
