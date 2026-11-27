from typing import Dict, Any
from ._base import JsonStoreBase


class ResourcesStore(JsonStoreBase):
    def __init__(self, data_dir: str = "data"):
        super().__init__(data_dir, "resources.json", {"resources": {}})

    def upsert(self, key: str, resource: Dict[str, Any]) -> None:
        data = self._read() or {"resources": {}}
        data.setdefault("resources", {})[key] = resource
        self._write(data)

    def all(self) -> Dict[str, Any]:
        data = self._read()
        return data.get("resources", {})



