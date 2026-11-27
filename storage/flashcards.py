from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import uuid
from ._base import JsonStoreBase


class FlashcardsStore(JsonStoreBase):
    def __init__(self, data_dir: str = "data"):
        super().__init__(data_dir, "flashcards.json", {"cards": {}})

    def create_card(self, user_id: str, subject: str, front: str, back: str) -> Dict[str, Any]:
        data = self._read() or {"cards": {}}
        card_id = str(uuid.uuid4())
        
        card = {
            "id": card_id,
            "user_id": user_id,
            "subject": subject,
            "front": front,
            "back": back,
            "box": 0,  # Leitner system box (0-5)
            "next_review": datetime.now().isoformat(),
            "created_at": datetime.now().isoformat(),
            "last_reviewed": None
        }
        
        data.setdefault("cards", {})[card_id] = card
        self._write(data)
        return card

    def get_due_cards(self, user_id: str) -> List[Dict[str, Any]]:
        data = self._read() or {"cards": {}}
        cards = []
        now = datetime.now()
        
        for card in data.get("cards", {}).values():
            if card.get("user_id") == user_id:
                next_review = datetime.fromisoformat(card.get("next_review"))
                if next_review <= now:
                    cards.append(card)
        
        return cards

    def list_cards(self, user_id: str) -> List[Dict[str, Any]]:
        data = self._read() or {"cards": {}}
        cards = []
        for card in data.get("cards", {}).values():
            if card.get("user_id") == user_id:
                cards.append(card)
        cards.sort(key=lambda c: c.get("next_review") or "")
        return cards

    def review_card(self, user_id: str, card_id: str, success: bool) -> Optional[Dict[str, Any]]:
        data = self._read() or {"cards": {}}
        card = data.get("cards", {}).get(card_id)
        
        if not card or card.get("user_id") != user_id:
            return None
            
        # Leitner System Implementation
        # Box intervals (in days): 0->1, 1->3, 2->7, 3->14, 4->30, 5->60
        intervals = [1, 3, 7, 14, 30, 60]
        current_box = card.get("box", 0)
        
        if success:
            new_box = min(current_box + 1, len(intervals) - 1)
        else:
            new_box = 0 # Reset to box 0 on failure
            
        days_to_add = intervals[new_box]
        next_review = datetime.now() + timedelta(days=days_to_add)
        
        card["box"] = new_box
        card["last_reviewed"] = datetime.now().isoformat()
        card["next_review"] = next_review.isoformat()
        
        self._write(data)
        return card

    def delete_card(self, user_id: str, card_id: str) -> bool:
        data = self._read() or {"cards": {}}
        card = data.get("cards", {}).get(card_id)
        
        if card and card.get("user_id") == user_id:
            del data["cards"][card_id]
            self._write(data)
            return True
        return False
