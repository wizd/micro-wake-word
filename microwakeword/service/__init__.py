"""Service package for the serial wake-word training API."""

__all__ = ["create_app"]


def __getattr__(name: str):
    if name == "create_app":
        from microwakeword.service.app import create_app as _create_app

        return _create_app
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
