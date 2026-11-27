import os
import json
from typing import Dict, Any


class JsonStoreBase:
    def __init__(self, data_dir: str, file_name: str, default_data: Dict[str, Any]):
        self.data_dir = data_dir
        os.makedirs(self.data_dir, exist_ok=True)
        self.path = os.path.join(self.data_dir, file_name)
        if not os.path.exists(self.path):
            self._write(default_data)

    def _read(self) -> Dict[str, Any]:
        try:
            if os.path.exists(self.path):
                with open(self.path, "r", encoding="utf-8") as f:
                    return json.load(f)
        except Exception:
            pass
        return {}

    def _write(self, data: Dict[str, Any]):
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)



