"""多主题目标生成、模型输出校验与逐日安排回归；所有模型响应均模拟。"""
import json
import os
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.plan_generation import generate_goal_path, validate_goal_path
from agents.planner_service import PlannerService


def model_response(topic, outcome):
    return {"outcome": outcome, "goal_feasible": True, "assumptions": ["已有入门基础"],
            "milestones": [
                {"topic": f"{topic}基础", "task": f"梳理{topic}的必要概念并写出3个例子", "hours": .75, "deliverable": "概念笔记"},
                {"topic": f"{topic}实践", "task": f"对照具体案例完成{topic}的实践成果", "hours": .75, "deliverable": outcome},
                {"topic": f"{topic}验收", "task": f"检查并展示{topic}实践成果，记录与目标的差距", "hours": .75, "deliverable": "成果验收记录"},
            ]}


class TestGoalResponse(unittest.TestCase):
    def test_valid_and_budget(self):
        payload = model_response("英语口语", "完成3分钟英文自我介绍")
        validated = validate_goal_path(payload, 3, .75)
        self.assertEqual(validated["outcome"], payload["outcome"])
        with self.assertRaises(ValueError):
            validate_goal_path(payload, 1, .5)

    def test_invalid_fields(self):
        for field, value in [("hours", -1), ("hours", float('nan')), ("task", "学习"), ("deliverable", "")]:
            payload = model_response("经济学", "案例分析")
            payload["milestones"][0][field] = value
            with self.assertRaises(ValueError):
                validate_goal_path(payload, 10, 1)

    def test_model_receives_all_constraints(self):
        payload = model_response("产品经理面试", "完成一次模拟面试")
        response = Mock()
        response.__enter__ = Mock(return_value=response)
        response.__exit__ = Mock(return_value=False)
        response.read.return_value = json.dumps({"choices": [{"message": {"content": json.dumps(payload)}}]}).encode()
        cfg = {"rag.api_key": "test-only-key", "rag.base_url": "https://example.invalid/v1",
               "rag.model_name": "fake-model", "rag.request_timeout": 30}
        with patch("agents.plan_generation.config.get", lambda key, default=None: cfg.get(key, default)), \
             patch("agents.plan_generation.urllib.request.urlopen", return_value=response) as request:
            result = generate_goal_path("产品经理面试", "完成一次模拟面试", 10, .75, {"known_skills": ["PRD"]})
        self.assertEqual(result["generation_source"], "model")
        body = json.loads(request.call_args.args[0].data)
        requirements = json.loads(body["messages"][1]["content"])
        self.assertEqual(requirements["subject"], "产品经理面试")
        self.assertEqual(requirements["goal"], "完成一次模拟面试")
        self.assertEqual(requirements["days"], 10)
        self.assertEqual(requirements["daily_hours"], .75)
        self.assertEqual(requirements["known_skills"], ["PRD"])


class TestMultipleTopics(unittest.TestCase):
    def planner(self):
        p = PlannerService.__new__(PlannerService)
        p.profile_store = SimpleNamespace(get_profile=lambda uid: {"daily_hours": 2})
        p.storage = SimpleNamespace(save_plan=Mock())
        p._attach_dynamic_fields = lambda plan, *args: plan
        p._enhance_plan_with_rag = lambda *args: self.fail("原文增强不应覆盖目标路径")
        return p

    def test_all_topics_use_the_same_generator(self):
        examples = [("RAG", "搭建问答应用"), ("Python", "编写文件整理脚本"),
                    ("经济学", "分析一个供需案例"), ("英语口语", "完成英文自我介绍"),
                    ("产品经理面试", "完成模拟面试")]
        for topic, goal in examples:
            with self.subTest(topic=topic), patch("agents.planner_service.track_event"):
                p = self.planner()
                path = validate_goal_path(model_response(topic, goal), 10, .75)
                p._generate_goal_path = Mock(return_value=path)
                plan = p.create_learning_plan(topic, "isolated", "2周", {
                    "goal": goal, "duration_days": 10, "requested_duration": "10天", "daily_hours": .75
                })
                p._generate_goal_path.assert_called_once()
                self.assertEqual(p._generate_goal_path.call_args.args[:4], (topic, goal, 10, .75))
                self.assertEqual(len(plan["daily_plan"]), 10)
                self.assertEqual(plan["generation_source"], "model")
                self.assertEqual(plan["outcome"], goal)
                self.assertTrue(all(d["estimated_hours"] <= d["available_hours"] for d in plan["daily_plan"]))
                text = str(plan["daily_plan"])
                self.assertIn(topic, text)
                self.assertIn(goal, text)
                self.assertIn("验收", text)
                self.assertFalse(plan["deferred_tasks"])

    def test_unavailable_model_is_marked_for_every_topic(self):
        for topic in ["Python", "经济学", "RAG"]:
            with self.subTest(topic=topic), patch("agents.planner_service.track_event"):
                p = self.planner()
                p._generate_goal_path = lambda *args: None
                plan = p.create_learning_plan(topic, "isolated", "2周", {
                    "goal": "完成实践", "duration_days": 10, "daily_hours": .75
                })
                self.assertEqual(plan["generation_source"], "draft")
                self.assertFalse(plan["goal_feasible"])
                self.assertIn("通用草案", str(plan["assumptions"]))
                self.assertEqual(len(plan["daily_plan"]), 10)


if __name__ == '__main__':
    unittest.main()
