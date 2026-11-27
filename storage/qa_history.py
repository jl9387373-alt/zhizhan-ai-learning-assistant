from typing import Dict, Any, List
from datetime import datetime
from ._base import JsonStoreBase


class QAHistoryStore(JsonStoreBase):
    def __init__(self, data_dir: str = "data", max_history: int = 1000):
        super().__init__(data_dir, "qa_history.json", {"history": []})
        self.max_history = max_history

    def append(self, record: Dict[str, Any]) -> None:
        data = self._read() or {"history": []}
        record.setdefault("timestamp", datetime.now().isoformat())
        data.setdefault("history", []).append(record)
        if len(data["history"]) > self.max_history:
            data["history"] = data["history"][-self.max_history :]
        self._write(data)

    def list_by_user(self, user_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        data = self._read() or {"history": []}
        items = [h for h in data.get("history", []) if h.get("user_id") == user_id]
        return items[-limit:]

    def list_all(self, limit: int = 50) -> List[Dict[str, Any]]:
        """获取本机全部问答历史（跨账号，按记录顺序取最新）。"""
        data = self._read() or {"history": []}
        return list(data.get("history", []))[-limit:]
    
    def delete_by_timestamp_any(self, timestamp: str, question: str = None) -> bool:
        """跨账号按时间戳删除记录（不校验账号，用于全部历史视图）"""
        try:
            data = self._read() or {"history": []}
            history = data.get("history", [])
            original_count = len(history)
            if question:
                data["history"] = [
                    h for h in history
                    if not (h.get("timestamp") == timestamp and h.get("question") == question)
                ]
            else:
                data["history"] = [
                    h for h in history if h.get("timestamp") != timestamp
                ]
            if len(data["history"]) < original_count:
                self._write(data)
                return True
            return False
        except Exception as e:
            print(f"跨账号删除问答历史失败: {str(e)}")
            return False

    def delete_by_timestamp(self, user_id: str, timestamp: str, question: str = None) -> bool:
        """根据时间戳删除记录"""
        try:
            data = self._read() or {"history": []}
            history = data.get("history", [])
            original_count = len(history)
            
            # 过滤掉匹配的记录
            # 如果提供了question，使用question和时间戳匹配；否则只使用时间戳匹配
            if question:
                data["history"] = [
                    h for h in history
                    if not (h.get("user_id") == user_id and 
                           h.get("timestamp") == timestamp and
                           h.get("question") == question)
                ]
            else:
                # 如果没有提供question，只根据时间戳和user_id匹配（可能删除多条，但至少匹配一条）
                data["history"] = [
                    h for h in history
                    if not (h.get("user_id") == user_id and 
                           h.get("timestamp") == timestamp)
                ]
            
            if len(data["history"]) < original_count:
                self._write(data)
                return True
            return False
        except Exception as e:
            # 记录错误但不抛出异常
            import traceback
            print(f"删除问答历史失败: {str(e)}")
            print(traceback.format_exc())
            return False
