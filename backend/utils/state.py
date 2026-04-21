from datetime import datetime
from threading import Lock


_lock = Lock()
_state = {
    "status": "idle",
    "error": None,
    "started_at": None,
    "completed_at": None,
}


def get_training_state() -> dict:
    with _lock:
        return dict(_state)


def set_training_state(status: str, error: str | None = None) -> None:
    with _lock:
        _state["status"] = status
        _state["error"] = error

        if status == "training":
            _state["started_at"] = datetime.utcnow().isoformat() + "Z"
            _state["completed_at"] = None
        elif status in {"completed", "failed"}:
            _state["completed_at"] = datetime.utcnow().isoformat() + "Z"
