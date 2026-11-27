"""
RAG 语料来源元数据存储
"""
import json
import os
from datetime import datetime
from typing import Dict, Any, List

from core.config import config

"""
RAG 元数据存储类
"""
class RAGMetadataStore:
    def __init__(self):
        data_dir = config.get("storage.data_dir", "data")
        os.makedirs(data_dir, exist_ok=True)
        self.path = os.path.join(data_dir, "metadata.json")
        if not os.path.exists(self.path):
            self._write({"sources": []})
    """
    读取元数据文件内容
    """
    def _read(self) -> Dict[str, Any]:
        if os.path.exists(self.path):
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return {"sources": []}
    """
    写入元数据文件内容
    """
    def _write(self, data: Dict[str, Any]) -> None:
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    """
    追加元数据记录
    """
    def append(self, records: List[Dict[str, Any]]) -> None:
        data = self._read()
        sources = data.setdefault("sources", [])
        now = datetime.now().isoformat()
        for record in records:
            rec = {**record}
            rec.setdefault("ingested_at", now)
            sources.append(rec)
        self._write(data)
    """
    列出所有元数据记录
    """
    def list_all(self) -> List[Dict[str, Any]]:
        return self._read().get("sources", [])


