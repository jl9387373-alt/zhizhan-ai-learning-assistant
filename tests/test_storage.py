import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from storage.plans import PlansStore
from storage.profiles import ProfilesStore
from storage.progress import ProgressStore


class TestPlansStore(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.store = PlansStore(data_dir=self.tmp)

    def test_save_and_get(self):
        plan = {"weeks": 4, "tasks": ["a", "b"]}
        self.store.save("u1", "python", plan)
        got = self.store.get("u1", "python")
        self.assertIsNotNone(got)
        self.assertEqual(got["user_id"], "u1")
        self.assertIn("updated_at", got)
        self.assertEqual(got["tasks"], ["a", "b"])

    def test_get_missing(self):
        self.assertIsNone(self.store.get("u1", "nope"))

    def test_get_all_filters_by_user(self):
        self.store.save("u1", "python", {"x": 1})
        self.store.save("u2", "java", {"x": 2})
        all_u1 = self.store.get_all("u1")
        self.assertEqual(len(all_u1), 1)
        self.assertIn("u1_python", all_u1)

    def test_delete(self):
        self.store.save("u1", "python", {})
        self.assertTrue(self.store.delete("u1", "python"))
        self.assertIsNone(self.store.get("u1", "python"))
        self.assertFalse(self.store.delete("u1", "python"))


class TestProgressStore(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.store = ProgressStore(data_dir=self.tmp)

    def test_save_get_delete(self):
        self.store.save("u1", "python", {"done": 3})
        got = self.store.get("u1", "python")
        self.assertEqual(got["done"], 3)
        self.assertTrue(self.store.delete("u1", "python"))
        self.assertIsNone(self.store.get("u1", "python"))

    def test_delete_missing(self):
        self.assertFalse(self.store.delete("u1", "missing"))


class TestProfilesStore(unittest.TestCase):
    def test_profile_persists_after_restart(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = ProfilesStore(data_dir=tmp)
            store.save_profile("u1", {
                "nickname": "坚持学习",
                "avatar": "rocket",
                "level": 2,
                "xp": 37,
                "streak_days": 6,
                "profile_completed": True,
            })

            restarted_store = ProfilesStore(data_dir=tmp)
            profile = restarted_store.get_profile("u1")
            self.assertEqual(profile["nickname"], "坚持学习")
            self.assertEqual(profile["avatar"], "rocket")
            self.assertEqual(profile["level"], 2)
            self.assertEqual(profile["xp"], 37)
            self.assertEqual(profile["streak_days"], 6)
            self.assertTrue(profile["profile_completed"])

    def test_legacy_profile_fields_are_repaired_without_data_loss(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "profiles.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump({"profiles": {"u1": {"xp_gain": 80}}}, f)

            profile = ProfilesStore(data_dir=tmp).get_profile("u1")
            self.assertEqual(profile["user_id"], "u1")
            self.assertEqual(profile["xp_gain"], 80)
            self.assertEqual(profile["xp"], 0)
            self.assertEqual(profile["streak_days"], 0)
            self.assertEqual(profile["nickname"], "User_u1")
            self.assertEqual(profile["avatar"], "default")


if __name__ == "__main__":
    unittest.main()
