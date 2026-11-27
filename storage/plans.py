from typing import Dict, Any, Optional
from datetime import datetime
from ._base import JsonStoreBase
import json
import os
import tempfile
import threading
from copy import deepcopy
from core.plan_progress import summarize_plan_progress


class PlansStore(JsonStoreBase):
    _locks_guard = threading.Lock()
    _locks = {}

    def __init__(self, data_dir: str = "data"):
        lock_key = os.path.normcase(os.path.abspath(os.path.join(data_dir, 'plans.json')))
        with self._locks_guard:
            self._lock = self._locks.setdefault(lock_key, threading.RLock())
        with self._lock:
            super().__init__(data_dir, "plans.json", {"plans": {}})

    def _write(self, data):
        # 原子替换避免并发读取读到只写了一半的JSON。
        descriptor, temporary = tempfile.mkstemp(prefix='.plans-', suffix='.json', dir=self.data_dir)
        try:
            with os.fdopen(descriptor, 'w', encoding='utf-8') as handle:
                json.dump(data, handle, ensure_ascii=False, indent=2)
            os.replace(temporary, self.path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    def save(self, user_id: str, subject: str, plan: Dict[str, Any]) -> None:
        with self._lock:
            self._save(user_id, subject, plan)

    def _save(self, user_id, subject, plan):
        data = self._read() or {"plans": {}}
        key = f"{user_id}_{subject}"
        plan["user_id"] = user_id
        plan["updated_at"] = datetime.now().isoformat()
        data.setdefault("plans", {})[key] = plan
        self._write(data)

    def get(self, user_id: str, subject: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            data = self._read()
            return data.get("plans", {}).get(f"{user_id}_{subject}")

    def get_all(self, user_id: str) -> Dict[str, Any]:
        with self._lock:
            data = self._read()
            return {k: v for k, v in data.get("plans", {}).items() if v.get("user_id") == user_id}

    def delete(self, user_id: str, subject: str) -> bool:
        with self._lock:
            return self._delete(user_id, subject)

    def _delete(self, user_id, subject):
        data = self._read() or {"plans": {}}
        key = f"{user_id}_{subject}"
        if key in data.get("plans", {}):
            del data["plans"][key]
            self._write(data)
            return True
        return False

    def set_task_completed(self, user_id, subject, week, task_index, completed, expected_created_at=None):
        if type(task_index) is not int or type(completed) is not bool:
            raise ValueError('任务索引须为整数，完成状态须为布尔值。')
        with self._lock:
            plan = self.get(user_id, subject)
            if not plan:
                return None
            if expected_created_at and expected_created_at != plan.get('created_at'):
                raise ValueError('计划已重新生成，请刷新后再勾选。')
            item = next((w for idx, w in enumerate(plan.get('weekly_plan') or [])
                         if str(w.get('week', idx + 1)) == str(week)), None)
            if item is None or not 0 <= task_index < len(item.get('tasks') or []):
                raise ValueError('任务不存在，请刷新计划后再勾选。')
            if not isinstance(plan.get('task_completion'), dict):
                plan['task_completion'] = {}
            flags = plan['task_completion'].get(str(week))
            flags = flags if isinstance(flags, list) else []
            flags = [bool(flags[i]) if i < len(flags) else False for i in range(len(item['tasks']))]
            flags[task_index] = completed
            plan['task_completion'][str(week)] = flags
            summary = summarize_plan_progress(plan)
            plan['progress'] = summary['detailed_task_percentage'] or 0
            plan['progress_summary'] = summary
            self._save(user_id, subject, plan)
            return deepcopy(plan)

    def save_if_unchanged(self, user_id, subject, expected, plan):
        with self._lock:
            if self.get(user_id, subject) != expected:
                return False
            self._save(user_id, subject, plan)
            return True



