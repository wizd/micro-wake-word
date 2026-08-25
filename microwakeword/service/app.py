"""FastAPI application for the serial wake-word training service."""

from __future__ import annotations

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import Depends, FastAPI, Header, HTTPException, Query
from fastapi.responses import FileResponse

from microwakeword.pipeline import should_use_cuda
from microwakeword.service.config import ServiceConfig, load_config
from microwakeword.service.schemas import (
    CreateJobRequest,
    CreateJobResponse,
    HealthResponse,
    JobListResponse,
    JobResponse,
    LogsResponse,
)
from microwakeword.service.store import JobStore
from microwakeword.service.worker import TrainingWorker

logger = logging.getLogger(__name__)


def _duration_seconds(record: dict) -> Optional[int]:
    started = record.get("started_at")
    finished = record.get("finished_at")
    if not started:
        return None
    try:
        start_dt = datetime.fromisoformat(started)
        end_dt = (
            datetime.fromisoformat(finished)
            if finished
            else datetime.now(start_dt.tzinfo)
        )
        return max(0, int((end_dt - start_dt).total_seconds()))
    except Exception:
        return None


def _to_response(
    record: dict, worker: TrainingWorker, *, public_base: str = ""
) -> JobResponse:
    job_id = record["job_id"]
    status = record["status"]
    queue_position = None
    if status == "queued":
        queue_position = worker.queue_position(job_id)
    elif status == "running":
        queue_position = 0

    model_url = None
    if record.get("model_path") and status == "succeeded":
        model_url = f"{public_base}/api/v1/jobs/{job_id}/model"

    return JobResponse(
        job_id=job_id,
        wakeword=record["wakeword"],
        slug=record["slug"],
        status=status,
        stage=record.get("stage"),
        queue_position=queue_position,
        created_at=datetime.fromisoformat(record["created_at"]),
        started_at=(
            datetime.fromisoformat(record["started_at"])
            if record.get("started_at")
            else None
        ),
        finished_at=(
            datetime.fromisoformat(record["finished_at"])
            if record.get("finished_at")
            else None
        ),
        duration_seconds=_duration_seconds(record),
        error=record.get("error"),
        model_path=record.get("model_path"),
        model_url=model_url,
        webhook_url=record.get("webhook_url"),
        metadata=record.get("metadata") or None,
        training_steps=int(record["training_steps"]),
        max_samples=int(record["max_samples"]),
        actual_training_steps=record.get("actual_training_steps"),
        early_stopped=bool(record.get("early_stopped", False)),
        stop_reason=record.get("stop_reason"),
        timings=record.get("timings"),
        suggested_cutoff=record.get("suggested_cutoff"),
        quality_warning=bool(record.get("quality_warning", False)),
        quality_message=record.get("quality_message"),
        score_report=record.get("score_report"),
        quality_verdict=record.get("quality_verdict"),
    )


def create_app(config: Optional[ServiceConfig] = None) -> FastAPI:
    config = config or load_config()
    config.workspace.mkdir(parents=True, exist_ok=True)
    config.jobs_dir.mkdir(parents=True, exist_ok=True)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    config.assets_dir.mkdir(parents=True, exist_ok=True)
    config.piper_dir.mkdir(parents=True, exist_ok=True)
    config.negatives_dir.mkdir(parents=True, exist_ok=True)
    config.augmentation_dir.mkdir(parents=True, exist_ok=True)

    # Ensure pipeline env points at shared assets
    os.environ.setdefault("MICROWAKEWORD_VOICE_MODEL", str(config.voice_model))
    os.environ.setdefault("MICROWAKEWORD_VOICE_CONFIG", str(config.voice_config))
    os.environ.setdefault(
        "MICROWAKEWORD_NEGATIVE_DATASETS_DIR", str(config.negatives_dir)
    )
    os.environ.setdefault(
        "MICROWAKEWORD_AUGMENTATION_DIR", str(config.augmentation_dir)
    )
    os.environ.setdefault("MICROWAKEWORD_TEST_VOICE_MODEL", str(config.test_voice_model))
    os.environ.setdefault(
        "MICROWAKEWORD_TEST_VOICE_CONFIG", str(config.test_voice_config)
    )

    store = JobStore(config.store_path)
    worker = TrainingWorker(config, store)

    app = FastAPI(
        title="microWakeWord Customization Service",
        version="1.0.0",
        description="Serial wake-word training API",
    )
    app.state.config = config
    app.state.store = store
    app.state.worker = worker

    def require_api_key(
        x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
    ) -> None:
        if config.api_key and x_api_key != config.api_key:
            raise HTTPException(status_code=401, detail="invalid API key")

    @app.on_event("startup")
    def _startup() -> None:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(name)s %(message)s",
        )
        try:
            import tensorflow as tf

            for gpu in tf.config.list_physical_devices("GPU"):
                try:
                    tf.config.experimental.set_memory_growth(gpu, True)
                except Exception:
                    pass
        except Exception:
            pass
        worker.start()
        logger.info(
            "service ready on %s:%s workspace=%s",
            config.host,
            config.port,
            config.workspace,
        )

    @app.on_event("shutdown")
    def _shutdown() -> None:
        worker.stop()

    @app.get("/healthz", response_model=HealthResponse)
    def healthz() -> HealthResponse:
        gpu_available = False
        gpu_name = None
        try:
            import tensorflow as tf

            gpus = tf.config.list_physical_devices("GPU")
            gpu_available = bool(gpus)
            if gpu_available:
                try:
                    details = tf.config.experimental.get_device_details(gpus[0])
                    gpu_name = details.get("device_name") or str(gpus[0].name)
                except Exception:
                    gpu_name = str(gpus[0].name)
        except Exception:
            try:
                import subprocess as _sp

                out = _sp.check_output(
                    [
                        "nvidia-smi",
                        "--query-gpu=name",
                        "--format=csv,noheader",
                    ],
                    text=True,
                    timeout=5,
                ).strip()
                if out:
                    gpu_available = True
                    gpu_name = out.splitlines()[0].strip()
            except Exception:
                gpu_available = should_use_cuda()

        return HealthResponse(
            status="ok",
            gpu_available=gpu_available,
            gpu_name=gpu_name,
            queue_length=worker.queue_length(),
            current_job_id=worker.current_job_id,
            workspace=str(config.workspace),
        )

    @app.post(
        "/api/v1/jobs",
        response_model=CreateJobResponse,
        dependencies=[Depends(require_api_key)],
    )
    def create_job(body: CreateJobRequest) -> CreateJobResponse:
        wakeword = body.wakeword.strip()
        if not wakeword:
            raise HTTPException(status_code=400, detail="wakeword must not be empty")

        record = worker.submit(
            wakeword=wakeword,
            webhook_url=str(body.webhook_url) if body.webhook_url else None,
            training_steps=body.training_steps,
            max_samples=body.max_samples,
            metadata=body.metadata,
        )
        return CreateJobResponse(
            job_id=record["job_id"],
            status=record["status"],
            queue_position=record.get("queue_position") or worker.queue_position(
                record["job_id"]
            )
            or 1,
            slug=record["slug"],
        )

    @app.get(
        "/api/v1/jobs",
        response_model=JobListResponse,
        dependencies=[Depends(require_api_key)],
    )
    def list_jobs() -> JobListResponse:
        records = store.list()
        jobs = [_to_response(r, worker) for r in records]
        return JobListResponse(jobs=jobs, total=len(jobs))

    @app.get(
        "/api/v1/jobs/{job_id}",
        response_model=JobResponse,
        dependencies=[Depends(require_api_key)],
    )
    def get_job(job_id: str) -> JobResponse:
        record = store.get(job_id)
        if not record:
            raise HTTPException(status_code=404, detail="job not found")
        return _to_response(record, worker)

    @app.get(
        "/api/v1/jobs/{job_id}/logs",
        response_model=LogsResponse,
        dependencies=[Depends(require_api_key)],
    )
    def get_logs(job_id: str, tail: int = Query(default=100, ge=1, le=5000)) -> LogsResponse:
        record = store.get(job_id)
        if not record:
            raise HTTPException(status_code=404, detail="job not found")
        log_path = config.jobs_dir / job_id / "train.log"
        lines: list[str] = []
        if log_path.exists():
            content = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
            lines = content[-tail:]
        return LogsResponse(job_id=job_id, lines=lines)

    @app.get(
        "/api/v1/jobs/{job_id}/model",
        dependencies=[Depends(require_api_key)],
    )
    def download_model(job_id: str):
        record = store.get(job_id)
        if not record:
            raise HTTPException(status_code=404, detail="job not found")
        if record.get("status") != "succeeded" or not record.get("model_path"):
            raise HTTPException(status_code=409, detail="model not ready")
        path = Path(record["model_path"])
        if not path.exists():
            raise HTTPException(status_code=404, detail="model file missing")
        return FileResponse(
            path,
            media_type="application/octet-stream",
            filename=path.name,
        )

    @app.delete(
        "/api/v1/jobs/{job_id}",
        response_model=JobResponse,
        dependencies=[Depends(require_api_key)],
    )
    def cancel_job(job_id: str) -> JobResponse:
        record = worker.cancel(job_id)
        if not record:
            raise HTTPException(status_code=404, detail="job not found")
        # re-read for latest state
        latest = store.get(job_id) or record
        return _to_response(latest, worker)

    return app


# Lazily constructed for `uvicorn microwakeword.service.app:app`.
# Prefer `python -m microwakeword.service.main` (factory) in production.
def __getattr__(name: str):
    if name == "app":
        return create_app()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
