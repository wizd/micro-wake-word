"""Pydantic request/response schemas."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, HttpUrl

JobStatus = Literal[
    "queued",
    "running",
    "succeeded",
    "failed",
    "cancelled",
    "interrupted",
]


class CreateJobRequest(BaseModel):
    wakeword: str = Field(..., min_length=1, max_length=64)
    webhook_url: Optional[HttpUrl] = None
    training_steps: Optional[int] = Field(default=None, ge=100, le=200000)
    max_samples: Optional[int] = Field(default=None, ge=20, le=5000)
    metadata: Optional[dict[str, Any]] = None


class JobResponse(BaseModel):
    job_id: str
    wakeword: str
    slug: str
    status: JobStatus
    stage: Optional[str] = None
    queue_position: Optional[int] = None
    created_at: datetime
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    duration_seconds: Optional[int] = None
    error: Optional[str] = None
    model_path: Optional[str] = None
    model_url: Optional[str] = None
    webhook_url: Optional[str] = None
    metadata: Optional[dict[str, Any]] = None
    training_steps: int
    max_samples: int
    actual_training_steps: Optional[int] = None
    early_stopped: bool = False
    stop_reason: Optional[str] = None
    timings: Optional[dict[str, float]] = None
    suggested_cutoff: Optional[float] = None
    quality_warning: bool = False
    quality_message: Optional[str] = None
    score_report: Optional[dict[str, Any]] = None
    quality_verdict: Optional[str] = None


class JobListResponse(BaseModel):
    jobs: list[JobResponse]
    total: int


class CreateJobResponse(BaseModel):
    job_id: str
    status: JobStatus
    queue_position: int
    slug: str


class HealthResponse(BaseModel):
    status: str
    gpu_available: bool
    gpu_name: Optional[str] = None
    queue_length: int
    current_job_id: Optional[str] = None
    workspace: str


class LogsResponse(BaseModel):
    job_id: str
    lines: list[str]
