"""会话级上下文：按 (user_id, conversation_id) 隔离。

保存两类信息：
1) 最近的用户消息（只取用户说的，不读助手回答，不扫描其它会话）；
2) 待完成的动作（如 create_plan）及其已知参数。

用途：解决"解释一下agent → 制定一个学习计划 → 上面不是说了agent"的连续对话，
澄清时记住动作，下一轮用户补充主题后恢复执行，而不是重新做独立意图分类。
"""
from __future__ import annotations

import threading
import time
from typing import Any, Dict, List, Optional, Tuple


class ConversationContextManager:
    def __init__(self, max_recent: int = 8) -> None:
        self.max_recent = max_recent
        self._lock = threading.RLock()
        # key -> {"recent": [str, ...], "pending": dict|None, "last_topic": str|None, "updated_at": float}
        self._store: Dict[Tuple[str, str], Dict[str, Any]] = {}

    @staticmethod
    def make_key(user_id: str, conversation_id: Optional[str]) -> Tuple[str, str]:
        return (str(user_id or "default"), str(conversation_id or "default"))

    def _ensure(self, key: Tuple[str, str]) -> Dict[str, Any]:
        if key not in self._store:
            self._store[key] = {
                "recent": [],
                "pending": None,
                "last_topic": None,
                "updated_at": time.time(),
            }
        return self._store[key]

    def add_user_message(self, user_id: str, conversation_id: Optional[str], text: str) -> None:
        """记录用户本轮消息（调用方在处理开始时调用）。"""
        if not text:
            return
        with self._lock:
            ctx = self._ensure(self.make_key(user_id, conversation_id))
            ctx["recent"].append(str(text))
            if len(ctx["recent"]) > self.max_recent:
                ctx["recent"] = ctx["recent"][-self.max_recent:]
            ctx["updated_at"] = time.time()

    def get_recent_user_messages(self, user_id: str, conversation_id: Optional[str], limit: int = 5) -> List[str]:
        """返回最近用户消息（最近在前），不含当前消息由调用方自行排除。"""
        with self._lock:
            ctx = self._ensure(self.make_key(user_id, conversation_id))
            return list(reversed(ctx["recent"][-limit:]))

    def set_pending_action(self, user_id: str, conversation_id: Optional[str], action: str, **params: Any) -> None:
        """保存待完成动作及参数。"""
        with self._lock:
            ctx = self._ensure(self.make_key(user_id, conversation_id))
            ctx["pending"] = {"action": action, "params": dict(params), "created_at": time.time()}
            ctx["updated_at"] = time.time()

    def get_pending_action(self, user_id: str, conversation_id: Optional[str]) -> Optional[Dict[str, Any]]:
        with self._lock:
            ctx = self._ensure(self.make_key(user_id, conversation_id))
            pending = ctx.get("pending")
            return dict(pending) if pending else None

    def clear_pending_action(self, user_id: str, conversation_id: Optional[str]) -> None:
        with self._lock:
            ctx = self._ensure(self.make_key(user_id, conversation_id))
            ctx["pending"] = None
            ctx["updated_at"] = time.time()

    def set_last_topic(self, user_id: str, conversation_id: Optional[str], topic: Optional[str]) -> None:
        with self._lock:
            ctx = self._ensure(self.make_key(user_id, conversation_id))
            ctx["last_topic"] = topic
            ctx["updated_at"] = time.time()

    def get_last_topic(self, user_id: str, conversation_id: Optional[str]) -> Optional[str]:
        with self._lock:
            ctx = self._ensure(self.make_key(user_id, conversation_id))
            return ctx.get("last_topic")

    def reset(self, user_id: str, conversation_id: Optional[str]) -> None:
        """切换/清空会话时重置上下文。"""
        with self._lock:
            self._store.pop(self.make_key(user_id, conversation_id), None)


conversation_context = ConversationContextManager()
