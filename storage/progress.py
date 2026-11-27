from typing import Dict, Any, Optional
from datetime import datetime
from ._base import JsonStoreBase
import threading
import os
import json
import tempfile


class ProgressStore(JsonStoreBase):
    _locks_guard = threading.Lock()
    _locks = {}
    def __init__(self, data_dir: str = "data"):
        key = os.path.normcase(os.path.abspath(os.path.join(data_dir, 'progress.json')))
        with self._locks_guard:
            self._lock = self._locks.setdefault(key, threading.RLock())
        with self._lock:
            super().__init__(data_dir, "progress.json", {"progress": {}})

    def _write(self, data):
        descriptor, temporary = tempfile.mkstemp(prefix='.progress-', suffix='.json', dir=self.data_dir)
        try:
            with os.fdopen(descriptor, 'w', encoding='utf-8') as handle:
                json.dump(data, handle, ensure_ascii=False, indent=2)
            os.replace(temporary, self.path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    def save(self, user_id: str, subject: str, progress: Dict[str, Any]) -> None:
        with self._lock:
            return self._save(user_id, subject, progress)

    def _save(self, user_id: str, subject: str, progress: Dict[str, Any]):
        data = self._read() or {"progress": {}}
        key = f"{user_id}_{subject}"
        progress["updated_at"] = datetime.now().isoformat()
        data.setdefault("progress", {})[key] = progress
        self._write(data)

    def get(self, user_id: str, subject: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            return self._get(user_id, subject)

    def _get(self, user_id: str, subject: str):
        data = self._read()
        return data.get("progress", {}).get(f"{user_id}_{subject}")

    def get_all(self, user_id: str) -> Dict[str, Any]:
        with self._lock:
            return self._get_all(user_id)

    def _get_all(self, user_id: str):
        data = self._read()
        result = {}
        for key, item in data.get("progress", {}).items():
            if item.get("user_id") == user_id:
                result[key] = item
        return result

    def delete(self, user_id: str, subject: str) -> bool:
        with self._lock:
            return self._delete(user_id, subject)

    def _delete(self, user_id: str, subject: str):
        data = self._read()
        # 处理键名：如果subject已经包含user_id前缀，直接使用；否则添加前缀
        if subject.startswith(f"{user_id}_"):
            key = subject
        else:
            key = f"{user_id}_{subject}"
        
        # 查找键名，支持两种格式：直接键名和包含在字典中的键名
        progress_data = data.get("progress", {})
        if key in progress_data:
            del progress_data[key]
            self._write(data)
            return True
        
        # 如果直接键名找不到，尝试查找匹配的user_id和subject
        for stored_key, item in progress_data.items():
            if item.get("user_id") == user_id and item.get("subject") == subject:
                del progress_data[stored_key]
                self._write(data)
                return True
        
        return False



