import uuid
from datetime import datetime
from typing import Dict, Any, List, Optional

from ._base import JsonStoreBase


class FeedbackStore(JsonStoreBase):
    def __init__(self, data_dir: str = "data"):
        super().__init__(data_dir, "feedback.json", {"items": []})

    def append(self, record: Dict[str, Any]) -> Dict[str, Any]:
        data = self._read() or {"items": []}
        item = dict(record)
        item.setdefault("id", str(uuid.uuid4()))
        item.setdefault("created_at", datetime.now().isoformat())
        data.setdefault("items", []).append(item)
        self._write(data)
        return item

    def list(
        self,
        limit: int = 50,
        offset: int = 0,
        user_id: Optional[str] = None,
        rating: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        data = self._read() or {"items": []}
        items = list(data.get("items", []))
        items.sort(key=lambda x: x.get("created_at", ""), reverse=True)
        if user_id:
            items = [x for x in items if str(x.get("user_id")) == str(user_id)]
        if rating is not None:
            items = [x for x in items if int(x.get("rating", 0)) == int(rating)]
        return items[offset: offset + limit]

    def stats(self) -> Dict[str, Any]:
        data = self._read() or {"items": []}
        items = data.get("items", [])
        total = len(items)
        pos = sum(1 for x in items if int(x.get("rating", 0)) > 0)
        neg = sum(1 for x in items if int(x.get("rating", 0)) < 0)
        by_source: Dict[str, int] = {}
        for x in items:
            src = str(x.get("source") or x.get("mode") or "unknown")
            by_source[src] = by_source.get(src, 0) + 1
        return {
            "total": total,
            "positive": pos,
            "negative": neg,
            "by_source": by_source
        }

