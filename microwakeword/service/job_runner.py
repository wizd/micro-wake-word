"""Isolated job runner process entrypoint."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import threading
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request-json", required=True)
    parser.add_argument("--log-path", required=True)
    parser.add_argument("--stage-file", required=True)
    parser.add_argument("--result-file", required=True)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        stream=sys.stdout,
    )

    # Prefer TF memory growth if TF gets imported later.
    os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")

    from microwakeword.pipeline import (  # noqa: WPS433
        PipelineCancelled,
        TrainingRequest,
        run_pipeline,
    )

    payload = json.loads(Path(args.request_json).read_text(encoding="utf-8"))
    req = TrainingRequest(
        wakeword=payload["wakeword"],
        job_id=payload["job_id"],
        output_dir=Path(payload["output_dir"]),
        cache_dir=Path(payload["cache_dir"]),
        negatives_dir=Path(payload["negatives_dir"]),
        max_samples=int(payload["max_samples"]),
        training_steps=int(payload["training_steps"]),
        train_batch=int(payload["train_batch"]),
    )
    cancel = threading.Event()
    cancel_flag = Path(payload.get("cancel_flag", ""))

    def on_stage(stage: str) -> None:
        Path(args.stage_file).write_text(stage, encoding="utf-8")
        if cancel_flag and cancel_flag.exists():
            cancel.set()

    try:
        model_path = run_pipeline(
            req,
            log_path=Path(args.log_path),
            cancel=cancel,
            on_stage=on_stage,
        )
        metadata = {}
        metadata_files = list(
            Path(args.request_json).parent.glob(
                "trained_models/*/training_metadata.json"
            )
        )
        if metadata_files:
            metadata = json.loads(metadata_files[0].read_text(encoding="utf-8"))
        Path(args.result_file).write_text(
            json.dumps(
                {
                    "ok": True,
                    "model_path": str(model_path),
                    "actual_training_steps": metadata.get("actual_training_steps"),
                    "early_stopped": metadata.get("early_stopped", False),
                    "stop_reason": metadata.get("stop_reason"),
                    "timings": metadata.get("timings"),
                }
            ),
            encoding="utf-8",
        )
    except PipelineCancelled:
        Path(args.result_file).write_text(
            json.dumps({"ok": False, "cancelled": True, "error": "cancelled"}),
            encoding="utf-8",
        )
        raise SystemExit(2)
    except Exception as exc:
        Path(args.result_file).write_text(
            json.dumps({"ok": False, "cancelled": False, "error": str(exc)}),
            encoding="utf-8",
        )
        logging.exception("job runner failed")
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
