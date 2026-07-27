"""ASGI entrypoint: python -m microwakeword.service.main"""

from __future__ import annotations

import uvicorn

from microwakeword.service.config import load_config


def main() -> None:
    config = load_config()
    uvicorn.run(
        "microwakeword.service.app:create_app",
        host=config.host,
        port=config.port,
        log_level="info",
        factory=True,
    )


if __name__ == "__main__":
    main()
