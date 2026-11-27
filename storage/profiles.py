from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import uuid
from ._base import JsonStoreBase


class ProfilesStore(JsonStoreBase):
    def __init__(self, data_dir: str = "data"):
        super().__init__(data_dir, "profiles.json", {"profiles": {}})

    @staticmethod
    def _default_profile(user_id: str) -> Dict[str, Any]:
        return {
            "user_id": user_id,
            "nickname": f"User_{user_id[:6]}",
            "avatar": "default",
            "level": 1,
            "xp": 0,
            "streak_days": 0,
            "last_login": datetime.now().isoformat(),
            "preferences": {"dark_mode": False},
            "achievements": [],
            "major": "",
            "grade": "",
            "goal": "",
            "known_skills": [],
            "weak_areas": [],
            "daily_hours": 2,
            "profile_completed": False,
        }

    @staticmethod
    def _as_non_negative_int(value: Any, default: int = 0) -> int:
        try:
            number = int(value)
        except (TypeError, ValueError):
            return default
        return max(0, number)

    @classmethod
    def _normalize_profile(cls, user_id: str, profile: Any) -> Dict[str, Any]:
        """补齐旧资料缺失字段，同时保留已有数据。"""
        defaults = cls._default_profile(user_id)
        if not isinstance(profile, dict):
            return defaults

        normalized = dict(defaults)
        normalized.update(profile)
        normalized["user_id"] = user_id

        nickname = str(normalized.get("nickname") or "").strip()
        normalized["nickname"] = nickname or defaults["nickname"]
        normalized["avatar"] = str(normalized.get("avatar") or "default")
        normalized["level"] = max(1, cls._as_non_negative_int(normalized.get("level"), 1))
        normalized["xp"] = cls._as_non_negative_int(normalized.get("xp"), 0)
        normalized["streak_days"] = cls._as_non_negative_int(normalized.get("streak_days"), 0)
        normalized["last_login"] = str(normalized.get("last_login") or defaults["last_login"])

        preferences = normalized.get("preferences")
        normalized["preferences"] = dict(preferences) if isinstance(preferences, dict) else {"dark_mode": False}
        normalized["preferences"].setdefault("dark_mode", False)

        for key in ("achievements", "known_skills", "weak_areas"):
            value = normalized.get(key)
            normalized[key] = list(value) if isinstance(value, list) else []

        for key in ("major", "grade", "goal"):
            normalized[key] = str(normalized.get(key) or "")

        try:
            daily_hours = float(normalized.get("daily_hours", 2))
        except (TypeError, ValueError):
            daily_hours = 2
        normalized["daily_hours"] = daily_hours if daily_hours > 0 else 2
        normalized["profile_completed"] = bool(normalized.get("profile_completed", False))
        return normalized

    def get_profile(self, user_id: str) -> Dict[str, Any]:
        data = self._read() or {"profiles": {}}
        profile = data.get("profiles", {}).get(user_id, {})
        normalized = self._normalize_profile(user_id, profile)

        if normalized != profile:
            self.save_profile(user_id, normalized)
        return normalized

    def save_profile(self, user_id: str, profile_data: Dict[str, Any]) -> None:
        data = self._read() or {"profiles": {}}
        data.setdefault("profiles", {})[user_id] = self._normalize_profile(user_id, profile_data)
        self._write(data)

    def update_xp(self, user_id: str, xp_gain: int) -> Dict[str, Any]:
        profile = self.get_profile(user_id)
        current_xp = profile.get("xp", 0)
        current_level = profile.get("level", 1)
        
        new_xp = current_xp + xp_gain
        
        # 简单升级逻辑: 等级 * 100 XP 升级
        xp_needed = current_level * 100
        leveled_up = False
        
        while new_xp >= xp_needed:
            new_xp -= xp_needed
            current_level += 1
            xp_needed = current_level * 100
            leveled_up = True
            
        profile["xp"] = new_xp
        profile["level"] = current_level
        
        if leveled_up:
            profile["achievements"].append(f"升级达人 Lv.{current_level}")
            
        self.save_profile(user_id, profile)
        return {"xp": new_xp, "level": current_level, "leveled_up": leveled_up}

    def check_in(self, user_id: str) -> Dict[str, Any]:
        profile = self.get_profile(user_id)
        last_login_str = profile.get("last_login", "")
        today = datetime.now().date()
        
        try:
            last_login_date = datetime.fromisoformat(last_login_str).date()
        except:
            last_login_date = today - timedelta(days=2) # 默认断签
            
        streak = profile.get("streak_days", 0)
        
        already = False
        if last_login_date == today:
            # 今天已签到：不改数值，但要让前端知道
            already = True
        elif last_login_date == today - timedelta(days=1):
            # 连续登录
            streak += 1
        else:
            # 断签
            streak = 1
            
        # ---- 签到奖励：基础 10 XP，连续 7 天额外 50 ----
        xp_gain = 0
        new_achievements = []
        if not already:
            xp_gain = 10
            if streak > 0 and streak % 7 == 0:
                xp_gain += 50
                badge = f"连续签到 {streak} 天"
                if badge not in profile.setdefault("achievements", []):
                    profile["achievements"].append(badge)
                    new_achievements.append(badge)

            new_xp = int(profile.get("xp", 0)) + xp_gain
            level = int(profile.get("level", 1) or 1)
            need = level * 100
            leveled = False
            while new_xp >= need:
                new_xp -= need
                level += 1
                need = level * 100
                leveled = True
            profile["xp"] = new_xp
            profile["level"] = level
            if leveled:
                lv_badge = f"升级达人 Lv.{level}"
                if lv_badge not in profile.setdefault("achievements", []):
                    profile["achievements"].append(lv_badge)
                    new_achievements.append(lv_badge)

        profile["streak_days"] = streak
        profile["last_login"] = datetime.now().isoformat()
        self.save_profile(user_id, profile)

        return {
            "streak_days": streak,
            "already_checked_in": already,
            "xp_gain": xp_gain,
            "xp": profile.get("xp", 0),
            "level": profile.get("level", 1),
            "new_achievements": new_achievements,
        }
