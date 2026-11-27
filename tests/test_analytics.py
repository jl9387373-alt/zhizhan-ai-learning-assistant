import os
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.compute_metrics import compute_metrics
from storage import analytics
from storage.analytics import AnalyticsStore


class TestAnalytics(unittest.TestCase):
    def test_track_and_compute_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = AnalyticsStore(os.path.join(tmp, "events.jsonl"))
            store.track("app_opened", "u1", {"path": "/"})
            store.track("plan_created", "u1", {"plan_key": "u1_python", "task_total": 2})
            store.track("task_completed", "u1", {
                "plan_key": "u1_python",
                "week": "1",
                "task_index": 0,
                "task_key": "u1_python|1|0",
            })
            store.track("qa_asked", "u1", {"qa_id": "q1"})
            store.track("qa_answered", "u1", {"qa_id": "q1", "source": "knowledge_base"})
            store.track("qa_helpful", "u1", {"feedback_key": "f1"})
            store.track("review_due", "u1", {"review_ids": ["r1"]})
            store.track("review_completed", "u1", {"review_id": "r1"})

            result = compute_metrics(store.read_events())
            metrics = result["metrics"]

            self.assertEqual(result["event_total"], 8)
            self.assertEqual(metrics["plan_completion_rate"]["value"], 0.5)
            self.assertEqual(metrics["activation_rate"]["value"], 1.0)
            self.assertEqual(metrics["qa_answer_rate"]["value"], 1.0)
            self.assertEqual(metrics["qa_resolution_rate"]["value"], 1.0)
            self.assertEqual(metrics["review_completion_rate"]["value"], 1.0)


class TestClientTracking(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = AnalyticsStore(os.path.join(self.tmp.name, "events.jsonl"))
        self.previous_store = analytics._STORE
        analytics._STORE = self.store
        from app import app
        self.client = app.test_client()

    def tearDown(self):
        analytics._STORE = self.previous_store
        self.tmp.cleanup()

    def test_track_requires_real_user_id(self):
        self.assertEqual(self.client.get('/', buffered=True).status_code, 200)
        self.assertEqual(list(self.store.read_events()), [])

        response = self.client.post('/api/track', json={
            'event': 'app_opened',
            'user_id': 'u_real',
            'session_id': 's1',
            'properties': {'path': '/'},
        })
        self.assertEqual(response.status_code, 200)

        events = list(self.store.read_events())
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]['event'], 'app_opened')
        self.assertEqual(events[0]['user_id'], 'u_real')
        self.assertEqual(events[0]['session_id'], 's1')
        self.assertEqual(events[0]['properties']['path'], '/')

        rejected = self.client.post('/api/track', json={
            'event': 'app_opened',
            'user_id': 'default',
        })
        self.assertEqual(rejected.status_code, 400)
        self.assertEqual(len(list(self.store.read_events())), 1)


if __name__ == "__main__":
    unittest.main()
