from typing import Dict, Any, List
from ._base import JsonStoreBase


class TagsStore(JsonStoreBase):
    def __init__(self, data_dir: str = "data"):
        super().__init__(data_dir, "tags.json", {"tags": {}})

    def add_tag(self, user_id: str, tag_name: str, resource_type: str, resource_id: str) -> Dict[str, Any]:
        data = self._read() or {"tags": {}}
        
        # Structure: {user_id: {tag_name: [{type: 'note', id: '123'}, ...]}}
        user_tags = data.get("tags", {}).setdefault(user_id, {})
        tag_items = user_tags.setdefault(tag_name, [])
        
        # Avoid duplicates
        item = {"type": resource_type, "id": resource_id}
        if item not in tag_items:
            tag_items.append(item)
            self._write(data)
            
        return {"tag": tag_name, "count": len(tag_items)}

    def get_tags(self, user_id: str) -> List[Dict[str, Any]]:
        data = self._read()
        user_tags = data.get("tags", {}).get(user_id, {})
        
        result = []
        for tag, items in user_tags.items():
            result.append({
                "name": tag,
                "count": len(items),
                "items": items
            })
        
        # Sort by count desc
        result.sort(key=lambda x: x["count"], reverse=True)
        return result

    def get_resources_by_tag(self, user_id: str, tag_name: str) -> List[Dict[str, str]]:
        data = self._read()
        user_tags = data.get("tags", {}).get(user_id, {})
        return user_tags.get(tag_name, [])

    def remove_tag(self, user_id: str, tag_name: str, resource_type: str, resource_id: str) -> bool:
        data = self._read() or {"tags": {}}
        user_tags = data.get("tags", {}).get(user_id, {})
        
        if tag_name in user_tags:
            items = user_tags[tag_name]
            # Filter out the item
            new_items = [item for item in items if not (item["type"] == resource_type and item["id"] == resource_id)]
            
            if len(new_items) != len(items):
                user_tags[tag_name] = new_items
                # If tag is empty, remove it
                if not new_items:
                    del user_tags[tag_name]
                self._write(data)
                return True
                
        return False
