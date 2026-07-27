"""Persistent job store backed by a JSON file."""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _iso(dt: Optional[datetime]) -> Optional[str]:
    if dt is None:
        return None
    return dt.astimezone(timezone.utc).isoformat()


def _parse(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    return datetime.fromisoformat(value)


class JobStore:
    """Thread-safe JSON-backed job registry."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.RLock()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self._write({})

    def _read(self) -> dict[str, dict[str, Any]]:
        with self.path.open("r", encoding="utf-8") as fh:
            return json.load(fh)

    def _write(self, data: dict[str, dict[str, Any]]) -> None:
        tmp = self.path.with_suffix(".tmp")
        with tmp.open("w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False, indent=2, sort_keys=True)
        tmp.replace(self.path)

    def create(
        self,
        *,
        job_id: str,
        wakeword: str,
        slug: str,
        training_steps: int,
        max_samples: int,
        webhook_url: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        record = {
            "job_id": job_id,
            "wakeword": wakeword,
            "slug": slug,
            "status": "queued",
            "stage": None,
            "created_at": _iso(_utcnow()),
            "started_at": None,
            "finished_at": None,
            "error": None,
            "model_path": None,
            "webhook_url": webhook_url,
            "metadata": metadata or {},
            "training_steps": training_steps,
            "max_samples": max_samples,
            "cancel_requested": False,
        }
        with self._lock:
            data = self._read()
            data[job_id] = record
            self._write(data)
        return dict(record)

    def get(self, job_id: str) -> Optional[dict[str, Any]]:
        with self._lock:
            data = self._read()
            record = data.get(job_id)
            return dict(record) if record else None

    def list(self) -> list[dict[str, Any]]:
        with self._lock:
            data = self._read()
            records = [dict(v) for v in data.values()]
        records.sort(key=lambda r: r.get("created_at") or "", reverse=True)
        return records

    def update(self, job_id: str, **fields: Any) -> Optional[dict[str, Any]]:
        with self._lock:
            data = self._read()
            record = data.get(job_id)
            if not record:
                return None
            for key, value in fields.items():
                if key.endswith("_at") and isinstance(value, datetime):
                    record[key] = _iso(value)
                else:
                    record[key] = value
            data[job_id] = record
            self._write(data)
            return dict(record)

    def mark_interrupted_running(self) -> list[str]:
        """Mark any previously running jobs as interrupted (after restart)."""
        changed: list[str] = []
        with self._lock:
            data = self._read()
            for job_id, record in data.items():
                if record.get("status") == "running":
                    record["status"] = "interrupted"
                    record["finished_at"] = _iso(_utcnow())
                    record["error"] = "service restarted while job was running"
                    changed.append(job_id)
            if changed:
                self._write(data)
        return changed

    def request_cancel(self, job_id: str) -> Optional[dict[str, Any]]:
        with self._lock:
            data = self._read()
            record = data.get(job_id)
            if not record:
                return None
            status = record.get("status")
            if status == "queued":
                record["status"] = "cancelled"
                record["finished_at"] = _iso(_utcnow())
                record["cancel_requested"] = True
            elif status == "running":
                record["cancel_requested"] = True
            else:
                return dict(record)
            data[job_id] = record
            self._write(data)
            return dict(record)
