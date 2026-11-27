"""本地事件埋点存储。

以 JSONL 方式追加写入事件，每个事件一行，适合 MVP 阶段本地验证指标口径。
"""
from __future__ import annotations

import json
import os
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_EVENTS_PATH = Path(
    os.environ.get("LEARNING_ASSISTANT_EVENTS_PATH")
    or str(PROJECT_ROOT / "data" / "analytics" / "events.jsonl")
)

_LOCK = threading.Lock()
_SESSION_ID = uuid.uuid4().hex[:12]
_STORE: Optional["AnalyticsStore"] = None


class AnalyticsStore:
    """追加式事件存储。"""

    def __init__(self, path: Optional[str] = None):
        self.path = Path(path) if path else DEFAULT_EVENTS_PATH
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def track(
        self,
        event: str,
        user_id: str = "default",
        properties: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        now = datetime.now().astimezone()
        iso = now.isocalendar()
        payload = {
            "event_id": uuid.uuid4().hex[:16],
            "session_id": session_id or _SESSION_ID,
            "event": event,
            "user_id": str(user_id or "default"),
            "timestamp": now.isoformat(timespec="seconds"),
            "date": now.strftime("%Y-%m-%d"),
            "week": f"{iso.year}-W{iso.week:02d}",
            "properties": properties or {},
        }
        with _LOCK:
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")
        return payload

    def read_events(self) -> Iterable[Dict[str, Any]]:
        if not self.path.exists():
            return []
        events = []
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                    if isinstance(item, dict):
                        events.append(item)
                except json.JSONDecodeError:
                    continue
        return events


def _get_store() -> AnalyticsStore:
    global _STORE
    if _STORE is None:
        _STORE = AnalyticsStore()
    return _STORE


def track_event(
    event: str,
    user_id: str = "default",
    properties: Optional[Dict[str, Any]] = None,
    session_id: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """写一条事件；埋点失败不能影响主流程。"""
    try:
        return _get_store().track(event, user_id, properties, session_id)
    except Exception:
        return None


def read_events() -> Iterable[Dict[str, Any]]:
    return _get_store().read_events()
