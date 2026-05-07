import json
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

from config.settings import BASE_DIR


LOG_FILE = BASE_DIR / "database" / "scanner_logs.jsonl"
MAX_LOGS = 120
_memory_logs: deque[dict] = deque(maxlen=MAX_LOGS)


def add_log(event: str, message: str, level: str = "info", symbol: str | None = None) -> dict:
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "level": level,
        "event": event,
        "symbol": symbol,
        "message": message,
    }
    _memory_logs.append(entry)
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with LOG_FILE.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return entry


def get_logs(limit: int = 60) -> list[dict]:
    file_logs: list[dict] = []
    if LOG_FILE.exists():
        lines = LOG_FILE.read_text(encoding="utf-8").splitlines()[-MAX_LOGS:]
        for line in lines:
            try:
                file_logs.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    combined = file_logs or list(_memory_logs)
    if not combined:
        combined = [
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "level": "info",
                "event": "dashboard online",
                "symbol": None,
                "message": "Waiting for scanner activity.",
            }
        ]

    return combined[-limit:][::-1]

