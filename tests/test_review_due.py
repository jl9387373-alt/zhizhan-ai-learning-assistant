"""测试复习到期过滤逻辑：get_due_reviews 正确区分到期/未到期/已掌握项。"""
import unittest
import tempfile
import shutil
from datetime import datetime, timedelta

from storage.review import ReviewStore


class TestReviewDue(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.store = ReviewStore(data_dir=self.tmp)
        self.user = "test_user"

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _make(self, rid, next_delta_days, status="active", subject="S", content="C"):
        now = datetime.now()
        data = {
            "review_id": rid,
            "user_id": self.user,
            "subject": subject,
            "content": content,
            "importance": "medium",
            "scheduled_at": now.isoformat(),
            "next_review_at": (now + timedelta(days=next_delta_days)).isoformat(),
            "review_count": 0,
            "current_interval_index": 0,
            "status": status,
            "last_reviewed_at": None,
            "mastery_level": 0.0,
            "performance_history": [],
        }
        self.store.save_plan(self.user, data)
        return data

    def test_due_filters_correctly(self):
        self._make("r_overdue", -1 / 24)              # 过去1小时 -> 到期
        self._make("r_upcoming", 2)                   # 未来2天 -> 未到期
        self._make("r_mastered", -1, status="mastered")  # 已掌握 -> 应排除

        result = self.store.get_due_reviews(self.user)

        overdue_ids = [r.get("review_id") for r in result["overdue"]]
        self.assertEqual(overdue_ids, ["r_overdue"])
        # 到期项应包含完整内容（来自文件读取）
        self.assertEqual(result["overdue"][0].get("content"), "C")

        self.assertIsNotNone(result["next"])
        self.assertEqual(result["next"].get("review_id"), "r_upcoming")

    def test_empty_returns_none(self):
        result = self.store.get_due_reviews(self.user)
        self.assertEqual(result["overdue"], [])
        self.assertIsNone(result["next"])


if __name__ == "__main__":
    unittest.main()
