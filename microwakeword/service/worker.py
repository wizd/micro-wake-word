"""Single-threaded serial training worker."""

from __future__ import annotations

import json
import logging
import os
import queue
import subprocess
import sys
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from microwakeword.pipeline import slugify_phrase
from microwakeword.service.config import ServiceConfig
from microwakeword.service.store import JobStore
from microwakeword.service.webhook import deliver_webhook

logger = logging.getLogger(__name__)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class TrainingWorker:
    """FIFO queue drained by a single background thread."""

    def __init__(self, config: ServiceConfig, store: JobStore) -> None:
        self.config = config
        self.store = store
        self._queue: queue.Queue[str] = queue.Queue()
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._current_proc: Optional[subprocess.Popen] = None
        self._current_job_id: Optional[str] = None
        self._lock = threading.Lock()

    @property
    def current_job_id(self) -> Optional[str]:
        return self._current_job_id

    def queue_length(self) -> int:
        return self._queue.qsize()

    def queue_position(self, job_id: str) -> Optional[int]:
        with self._lock:
            items = list(self._queue.queue)
        try:
            return items.index(job_id) + 1
        except ValueError:
            if self._current_job_id == job_id:
                return 0
            return None

    def start(self) -> None:
        interrupted = self.store.mark_interrupted_running()
        if interrupted:
            logger.warning("marked interrupted jobs: %s", interrupted)

        for record in reversed(self.store.list()):
            if record.get("status") == "queued":
                self._queue.put(record["job_id"])

        self._thread = threading.Thread(
            target=self._run_loop, name="mww-worker", daemon=True
        )
        self._thread.start()
        logger.info("training worker started")

    def stop(self) -> None:
        self._stop.set()
        self._queue.put("__stop__")
        proc = self._current_proc
        if proc and proc.poll() is None:
            proc.terminate()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)

    def submit(
        self,
        *,
        wakeword: str,
        webhook_url: Optional[str] = None,
        training_steps: Optional[int] = None,
        max_samples: Optional[int] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        slug = slugify_phrase(wakeword)
        job_id = uuid.uuid4().hex[:12]
        steps = training_steps or self.config.default_training_steps
        samples = max_samples or self.config.default_max_samples

        record = self.store.create(
            job_id=job_id,
            wakeword=wakeword,
            slug=slug,
            training_steps=steps,
            max_samples=samples,
            webhook_url=webhook_url,
            metadata=metadata,
        )
        job_dir = self.config.jobs_dir / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        self._queue.put(job_id)
        record["queue_position"] = self.queue_position(job_id)
        return record

    def cancel(self, job_id: str) -> Optional[dict[str, Any]]:
        record = self.store.request_cancel(job_id)
        if not record:
            return None
        if record.get("status") == "cancelled":
            with self._lock:
                items = list(self._queue.queue)
                if job_id in items:
                    self._queue.queue.clear()
                    for item in items:
                        if item != job_id:
                            self._queue.queue.append(item)
            return record

        # Signal running job runner
        cancel_flag = self.config.jobs_dir / job_id / "CANCEL"
        cancel_flag.parent.mkdir(parents=True, exist_ok=True)
        cancel_flag.write_text("1", encoding="utf-8")
        proc = self._current_proc
        if self._current_job_id == job_id and proc and proc.poll() is None:
            proc.terminate()
        return record

    def _run_loop(self) -> None:
        while not self._stop.is_set():
            try:
                job_id = self._queue.get(timeout=1.0)
            except queue.Empty:
                continue
            if job_id == "__stop__":
                break

            record = self.store.get(job_id)
            if not record:
                continue
            if record.get("status") == "cancelled" or record.get("cancel_requested"):
                self.store.update(
                    job_id,
                    status="cancelled",
                    finished_at=_utcnow(),
                )
                continue

            self._execute_job(job_id, record)

    def _execute_job(self, job_id: str, record: dict[str, Any]) -> None:
        self._current_job_id = job_id
        job_dir = self.config.jobs_dir / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        log_path = job_dir / "train.log"
        stage_file = job_dir / "stage.txt"
        result_file = job_dir / "result.json"
        request_file = job_dir / "request.json"
        cancel_flag = job_dir / "CANCEL"
        if cancel_flag.exists():
            cancel_flag.unlink()
        if result_file.exists():
            result_file.unlink()

        self.store.update(
            job_id,
            status="running",
            stage="starting",
            started_at=_utcnow(),
            error=None,
        )

        request_payload = {
            "wakeword": record["wakeword"],
            "job_id": job_id,
            "output_dir": str(self.config.output_dir),
            "cache_dir": str(self.config.workspace),
            "negatives_dir": str(self.config.negatives_dir),
            "max_samples": int(record["max_samples"]),
            "training_steps": int(record["training_steps"]),
            "train_batch": self.config.train_batch,
            "cancel_flag": str(cancel_flag),
        }
        request_file.write_text(
            json.dumps(request_payload, ensure_ascii=False), encoding="utf-8"
        )

        env = os.environ.copy()
        env.setdefault("MICROWAKEWORD_VOICE_MODEL", str(self.config.voice_model))
        env.setdefault("MICROWAKEWORD_VOICE_CONFIG", str(self.config.voice_config))
        env.setdefault(
            "MICROWAKEWORD_NEGATIVE_DATASETS_DIR", str(self.config.negatives_dir)
        )
        env.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
        env.setdefault("PYTHONUNBUFFERED", "1")

        cmd = [
            sys.executable,
            "-m",
            "microwakeword.service.job_runner",
            f"--request-json={request_file}",
            f"--log-path={log_path}",
            f"--stage-file={stage_file}",
            f"--result-file={result_file}",
        ]

        logger.info("launching job runner for %s", job_id)
        proc = subprocess.Popen(
            cmd,
            cwd=str(Path(__file__).resolve().parents[2]),
            env=env,
            stdout=open(job_dir / "runner.out", "a", encoding="utf-8"),
            stderr=subprocess.STDOUT,
        )
        self._current_proc = proc

        model_path: Optional[Path] = None
        result_metadata: dict[str, Any] = {}
        error: Optional[str] = None
        final_status = "succeeded"

        try:
            while True:
                # Update stage from file
                if stage_file.exists():
                    try:
                        stage = stage_file.read_text(encoding="utf-8").strip()
                        if stage:
                            self.store.update(job_id, stage=stage)
                    except Exception:
                        pass

                latest = self.store.get(job_id) or {}
                if latest.get("cancel_requested"):
                    cancel_flag.write_text("1", encoding="utf-8")
                    proc.terminate()

                ret = proc.poll()
                if ret is not None:
                    break
                time.sleep(1)

            if result_file.exists():
                result = json.loads(result_file.read_text(encoding="utf-8"))
                if result.get("ok"):
                    model_path = Path(result["model_path"])
                    result_metadata = {
                        "actual_training_steps": result.get("actual_training_steps"),
                        "early_stopped": bool(result.get("early_stopped", False)),
                        "stop_reason": result.get("stop_reason"),
                        "timings": result.get("timings"),
                        "suggested_cutoff": result.get("suggested_cutoff"),
                        "quality_warning": bool(result.get("quality_warning", False)),
                        "quality_message": result.get("quality_message"),
                        "score_report": result.get("score_report"),
                        "quality_verdict": result.get("quality_verdict"),
                    }
                    final_status = "succeeded"
                elif result.get("cancelled"):
                    final_status = "cancelled"
                    error = "cancelled by user"
                else:
                    final_status = "failed"
                    error = result.get("error") or f"job runner exited {proc.returncode}"
            else:
                final_status = "failed"
                error = f"job runner exited {proc.returncode} without result"
        except Exception as exc:
            final_status = "failed"
            error = str(exc)
            logger.exception("job %s failed: %s", job_id, exc)
            if proc.poll() is None:
                proc.terminate()
        finally:
            self._current_proc = None

        finished = _utcnow()
        current = self.store.get(job_id) or record
        started_raw = current.get("started_at")
        duration = None
        if started_raw:
            try:
                started_dt = datetime.fromisoformat(started_raw)
                duration = int((finished - started_dt).total_seconds())
            except Exception:
                duration = None

        updated = self.store.update(
            job_id,
            status=final_status,
            stage="done" if final_status == "succeeded" else current.get("stage"),
            finished_at=finished,
            error=error,
            model_path=str(model_path) if model_path else None,
            **result_metadata,
        )

        self._current_job_id = None

        webhook_url = (updated or current).get("webhook_url")
        if webhook_url:
            payload = {
                "event": f"job.{final_status}",
                "job_id": job_id,
                "wakeword": record["wakeword"],
                "slug": record["slug"],
                "status": final_status,
                "error": error,
                "model_path": str(model_path) if model_path else None,
                "duration_seconds": duration,
                "metadata": record.get("metadata") or {},
                "suggested_cutoff": (updated or current).get("suggested_cutoff"),
                "quality_warning": bool(
                    (updated or current).get("quality_warning", False)
                ),
                "quality_message": (updated or current).get("quality_message"),
                "quality_verdict": (updated or current).get("quality_verdict"),
            }
            deliver_webhook(
                webhook_url,
                payload,
                secret=self.config.webhook_secret,
                max_retries=self.config.webhook_max_retries,
                timeout_s=self.config.webhook_timeout_s,
            )
