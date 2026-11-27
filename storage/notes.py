from typing import Dict, Any, Optional, List
from datetime import datetime
import uuid
from ._base import JsonStoreBase


class NotesStore(JsonStoreBase):
    def __init__(self, data_dir: str = "data"):
        super().__init__(data_dir, "notes.json", {"notes": {}})

    def create(self, user_id: str, subject: str, title: str, content: str) -> Dict[str, Any]:
        data = self._read() or {"notes": {}}
        note_id = str(uuid.uuid4())
        
        note = {
            "id": note_id,
            "user_id": user_id,
            "subject": subject,
            "title": title,
            "content": content,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat()
        }
        
        data.setdefault("notes", {})[note_id] = note
        self._write(data)
        return note

    def get(self, user_id: str, note_id: str) -> Optional[Dict[str, Any]]:
        data = self._read()
        note = data.get("notes", {}).get(note_id)
        if note and note.get("user_id") == user_id:
            return note
        return None

    def get_all(self, user_id: str, subject: str = "") -> List[Dict[str, Any]]:
        data = self._read()
        notes = []
        for note in data.get("notes", {}).values():
            if note.get("user_id") == user_id:
                if subject and note.get("subject") != subject:
                    continue
                notes.append(note)
        
        # Sort by updated_at desc
        notes.sort(key=lambda x: x.get("updated_at", ""), reverse=True)
        return notes

    def update(self, user_id: str, note_id: str, updates: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        data = self._read() or {"notes": {}}
        note = data.get("notes", {}).get(note_id)
        
        if not note or note.get("user_id") != user_id:
            return None
            
        # Update allowed fields
        if "title" in updates:
            note["title"] = updates["title"]
        if "content" in updates:
            note["content"] = updates["content"]
        if "subject" in updates:
            note["subject"] = updates["subject"]
            
        note["updated_at"] = datetime.now().isoformat()
        data["notes"][note_id] = note
        self._write(data)
        return note

    def delete(self, user_id: str, note_id: str) -> bool:
        data = self._read() or {"notes": {}}
        note = data.get("notes", {}).get(note_id)
        
        if note and note.get("user_id") == user_id:
            del data["notes"][note_id]
            self._write(data)
            return True
        return False
