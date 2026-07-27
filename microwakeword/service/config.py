"""Service configuration from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _env_path(name: str, default: str) -> Path:
    return Path(os.getenv(name, default)).expanduser().resolve()


@dataclass(frozen=True)
class ServiceConfig:
    host: str = "0.0.0.0"
    port: int = 6006
    workspace: Path = Path("/root/autodl-tmp/mww")
    assets_dir: Path = Path("/root/autodl-tmp/assets")
    api_key: str | None = None
    webhook_secret: str | None = None
    webhook_max_retries: int = 3
    webhook_timeout_s: float = 15.0
    default_training_steps: int = 10000
    default_max_samples: int = 400
    train_batch: int = 256

    @property
    def jobs_dir(self) -> Path:
        return self.workspace / "jobs"

    @property
    def output_dir(self) -> Path:
        return self.workspace / "output"

    @property
    def store_path(self) -> Path:
        return self.workspace / "store" / "jobs.json"

    @property
    def piper_dir(self) -> Path:
        return self.assets_dir / "piper-voices"

    @property
    def negatives_dir(self) -> Path:
        return self.assets_dir / "negative-datasets"

    @property
    def voice_model(self) -> Path:
        return self.piper_dir / "zh_CN-huayan-medium.onnx"

    @property
    def voice_config(self) -> Path:
        return self.piper_dir / "zh_CN-huayan-medium.onnx.json"


def load_config() -> ServiceConfig:
    return ServiceConfig(
        host=os.getenv("MWW_HOST", "0.0.0.0"),
        port=int(os.getenv("MWW_PORT", "6006")),
        workspace=_env_path("MWW_WORKSPACE", "/root/autodl-tmp/mww"),
        assets_dir=_env_path("MWW_ASSETS_DIR", "/root/autodl-tmp/assets"),
        api_key=os.getenv("MWW_API_KEY") or None,
        webhook_secret=os.getenv("MWW_WEBHOOK_SECRET") or None,
        webhook_max_retries=int(os.getenv("MWW_WEBHOOK_MAX_RETRIES", "3")),
        webhook_timeout_s=float(os.getenv("MWW_WEBHOOK_TIMEOUT_S", "15")),
        default_training_steps=int(
            os.getenv("MICROWAKEWORD_TRAINING_STEPS", "10000")
        ),
        default_max_samples=int(os.getenv("MICROWAKEWORD_SAMPLE_COUNT", "400")),
        train_batch=int(os.getenv("MICROWAKEWORD_TRAIN_BATCH", "256")),
    )
