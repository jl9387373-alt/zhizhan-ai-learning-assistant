"""计划参数、多轮澄清和时间预算回归；隔离存储，不调用模型。"""
import math
import os
import sys
import tempfile
import unittest
from datetime import date, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import Mock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.conversation_context import conversation_context
from core.coordinator import AgentCoordinator
from core.message import Message, MessageType
from agents.planner_service import PlannerService
from agents.service_agent import ServiceAgent
from core.storage import DataStorage
from storage.profiles import ProfilesStore
from nlp.intent_detection import _detect_intent_rules
from nlp.plan_constraints import extract_plan_constraints, extract_plan_subject


class TestPlanExtraction(unittest.TestCase):
    def test_combined(self):
        params, errors = extract_plan_constraints("帮我制定RAG学习计划，目标：能搭建问答应用，周期10天，每天45分钟")
        self.assertFalse(errors)
        self.assertEqual(params["goal"], "搭建问答应用")
        self.assertEqual(params["duration_days"], 10)
        self.assertEqual(params["duration"], "2周")
        self.assertEqual(params["daily_hours"], .75)

    def test_cn_and_decimal(self):
        for query, hours in [("每日半小时", .5), ("每天1.5小时", 1.5), ("每天一小时半", 1.5),
                             ("每天1小时30分钟", 1.5), ("每天可用时间：两小时", 2), ("每天能学1小时", 1)]:
            self.assertEqual(extract_plan_constraints(query)[0]["daily_hours"], hours)
        self.assertEqual(extract_plan_constraints("周期一个月", reference_date=date(2026, 11, 25))[0]["duration_days"], 30)

    def test_absent_and_invalid(self):
        self.assertEqual(extract_plan_constraints("RAG"), ({}, {}))
        for query in ["每天0小时", "每天25小时", "每天-1小时", "周期0天", "周期-1周", "周期999周"]:
            self.assertTrue(extract_plan_constraints(query)[1], query)

    def test_goal_does_not_override_topic(self):
        self.assertEqual(extract_plan_subject("制定RAG学习计划，目标：用Python做AI问答应用，4周，每天1小时"), "RAG")
        self.assertEqual(extract_plan_subject("主题：经济学"), "经济学")


class Recorder:
    def __init__(self):
        self.messages = []

    def process(self, msg):
        self.messages.append(msg.content)
        return Message("planner", msg.sender, MessageType.RESPONSE,
                       {"status": "success", "plan": {"subject": msg.content["subject"]}})


class TestPlanConversation(unittest.TestCase):
    def setUp(self):
        self.user, self.cid = "plan_constraint_test", self.id()
        conversation_context.reset(self.user, self.cid)
        self.c = AgentCoordinator.__new__(AgentCoordinator)
        self.c.storage = SimpleNamespace(profiles_store=SimpleNamespace(get_profile=lambda uid: {"profile_completed": False}),
                                         get_conversation=lambda cid: {}, save_qa_history=Mock())
        self.recorder = Recorder()
        self.c.get_agent = lambda name: self.recorder
        self.c.detect_intent = lambda query: _detect_intent_rules(query)
        self.memory_patch = patch("core.coordinator.conversation_memory", SimpleNamespace(
            load_from_database=lambda uid: None, get_recent_user_questions=lambda uid, limit=5: [],
            get_relevant_long_term_memory=lambda uid, query: None))
        self.memory_patch.start()
        self.addCleanup(self.memory_patch.stop)
        self.addCleanup(conversation_context.reset, self.user, self.cid)

    def run_stream(self, text, cid=None):
        return list(self.c.process_request_stream(text, self.user, conversation_id=cid or self.cid))[-1]["data"]

    def test_one_message(self):
        result = self.run_stream("帮我制定RAG学习计划，目标：用Python搭建AI问答应用，10天，每天1小时")
        self.assertIn("plan", result)
        received = self.recorder.messages[-1]
        self.assertEqual(received["subject"], "RAG")
        self.assertEqual(received["constraints"]["duration_days"], 10)
        self.assertEqual(received["constraints"]["daily_hours"], 1)

    def test_multiple_turns_preserve_values(self):
        self.assertEqual(self.run_stream("帮我制定学习计划，周期10天")["type"], "plan_clarification")
        self.assertEqual(self.run_stream("RAG")["type"], "plan_clarification")
        self.assertEqual(self.run_stream("能用Python搭建问答应用")["missing_fields"], ["daily_hours"])
        self.assertIn("plan", self.run_stream("每天30分钟"))
        params = self.recorder.messages[-1]["constraints"]
        self.assertEqual(params["subject"], "RAG")
        self.assertEqual(params["duration_days"], 10)
        self.assertEqual(params["daily_hours"], .5)
        self.assertIn("Python", params["goal"])
        self.assertIsNone(conversation_context.get_pending_action(self.user, self.cid))

    def test_invalid_then_corrected(self):
        self.run_stream("制定RAG学习计划，目标：搭建问答应用，4周，每天0小时")
        self.assertFalse(self.recorder.messages)
        self.assertIn("plan", self.run_stream("每天1小时"))
        self.assertEqual(self.recorder.messages[-1]["constraints"]["duration_days"], 28)

    def test_explicit_defaults_and_isolation(self):
        self.run_stream("制定RAG学习计划，10天")
        other = self.run_stream("制定学习计划，其余用默认值", "another-conversation")
        self.assertIn("subject", other["missing_fields"])
        self.assertIn("plan", self.run_stream("其余用默认值"))
        self.assertEqual(self.recorder.messages[-1]["constraints"]["duration_days"], 10)

    def test_profile_is_overridden(self):
        self.c.storage.profiles_store.get_profile = lambda uid: {"profile_completed": True, "goal": "学会RAG", "daily_hours": 2}
        self.run_stream("制定RAG学习计划，目标：完成作品，周期4周，每天半小时")
        self.assertEqual(self.recorder.messages[-1]["constraints"]["daily_hours"], .5)
        self.assertEqual(self.recorder.messages[-1]["constraints"]["goal"], "完成作品")

    def test_nonstream_followup(self):
        first = self.c.process_request("制定RAG学习计划，目标：完成作品，周期10天", self.user,
                                       intent="plan", conversation_id=self.cid)
        self.assertEqual(first["missing_fields"], ["daily_hours"])
        result = self.c.process_request("每天45分钟", self.user, conversation_id=self.cid)
        self.assertIn("plan", result)
        self.assertEqual(self.recorder.messages[-1]["constraints"]["duration_days"], 10)

    def test_interleaved_qa_does_not_create(self):
        self.run_stream("制定RAG学习计划，周期10天")
        before = len(self.recorder.messages)
        result = self.run_stream("什么是Python装饰器")
        self.assertEqual(self.recorder.messages[-1]["action"], "generate")
        self.assertIsNotNone(conversation_context.get_pending_action(self.user, self.cid))
        self.assertEqual(len(self.recorder.messages), before + 1)
        self.run_stream("目标：完成问答应用，每天1小时")
        self.assertEqual(self.recorder.messages[-1]["subject"], "RAG")

    def test_plan_query_does_not_generate(self):
        self.c._query_plan_exists = lambda *args: {"type": "plan_query", "exists": False}
        result = self.run_stream("有没有RAG的学习计划")
        self.assertEqual(result["type"], "plan_query")
        self.assertFalse(self.recorder.messages)

    def test_rest_parameters_reach_generator(self):
        result = self.c.create_plan(self.user, "RAG", 10, "完成作品", .75)
        self.assertIn("subject", result)
        params = self.recorder.messages[-1]["constraints"]
        self.assertEqual(params["duration_days"], 10)
        self.assertEqual(params["daily_hours"], .75)


class TestGeneratedBudget(unittest.TestCase):
    def setUp(self):
        self.p = PlannerService.__new__(PlannerService)
        self.profile = {"daily_hours": 2, "goal": "无关目标", "known_skills": [], "weak_areas": []}
        self.p.profile_store = SimpleNamespace(get_profile=lambda uid: self.profile)
        self.p.storage = SimpleNamespace(save_plan=Mock())
        self.p.plan_templates = self.p._load_plan_templates()
        self.p._enhance_plan_with_rag = lambda source, *args: source
        self.p._generate_goal_path = lambda *args: None
        self.p._attach_dynamic_fields = lambda plan, *args: plan
        self.event_patch = patch("agents.planner_service.track_event")
        self.event_patch.start()
        self.addCleanup(self.event_patch.stop)

    def create(self, topic, days, hours, goal="搭建问答应用"):
        return self.p.create_learning_plan(topic, "isolated", f"{math.ceil(days / 7)}周",
                                           {"duration_days": days, "daily_hours": hours, "goal": goal,
                                            "requested_duration": f"{days}天"})

    def test_exact_days_and_goal(self):
        plan = self.create("RAG", 10, 1)
        self.assertEqual((datetime.fromisoformat(plan["end_date"]) - datetime.fromisoformat(plan["start_date"])).days, 10)
        self.assertEqual(len(plan["weekly_plan"]), 2)
        self.assertEqual(plan["weekly_plan"][-1]["available_days"], 3)
        self.assertEqual(len(plan["daily_plan"]), 10)
        self.assertEqual([d["day"] for d in plan["daily_plan"]], list(range(1, 11)))
        self.assertTrue(all(w["estimated_hours"] <= w["available_hours"] for w in plan["weekly_plan"]))
        self.assertTrue(all(w["goals"] == ["搭建问答应用"] for w in plan["weekly_plan"]))
        self.assertEqual(self.profile["goal"], "无关目标")
        self.assertEqual(self.profile["daily_hours"], 2)
        self.assertEqual(self.p.storage.save_plan.call_args.args[2]["daily_hours"], 1)

    def test_hours_change_task_load(self):
        low, high = self.create("RAG", 14, .25), self.create("RAG", 14, 2)
        hours = lambda p: sum(d['estimated_hours'] for d in p['daily_plan'])
        self.assertLess(hours(low), hours(high))
        self.assertTrue(low["deferred_tasks"])
        self.assertTrue(any("搭建问答应用" in t for w in high["weekly_plan"] for t in w["tasks"]))

    def test_python_does_not_expand_period(self):
        plan = self.create("python", 10, 1, "掌握Python基础")
        self.assertEqual(plan["duration_days"], 10)
        self.assertEqual(len(plan["weekly_plan"]), 2)
        self.assertEqual(len(plan["daily_plan"]), 10)
        self.assertTrue(all(w["estimated_hours"] <= w["available_hours"] for w in plan["weekly_plan"]))

    def test_template_respects_period(self):
        plan = self.create("机器学习", 7, 1)
        self.assertEqual(len(plan["weekly_plan"]), 1)
        self.assertTrue(plan["deferred_tasks"])

    def test_zero_hours_rejected(self):
        with self.assertRaises(ValueError):
            self.create("RAG", 10, 0)

    def test_small_budget_can_start_and_timeline_is_clamped(self):
        plan = self.create("RAG", 1, .5)
        self.assertTrue(plan["weekly_plan"][0]["tasks"])
        self.assertLessEqual(plan["weekly_plan"][0]["estimated_hours"], .5)
        self.assertIn("后续继续", plan["weekly_plan"][0]["tasks"][0])
        plan = self.create("RAG", 10, 1)
        timeline = self.p._build_timeline(plan)
        last = datetime.fromisoformat(timeline[-1]["end"])
        start = datetime.fromisoformat(plan["start_date"])
        self.assertEqual((last.date() - start.date()).days, 9)

    def test_longer_period_spreads_learning(self):
        plan = self.create("RAG", 28, 2)
        self.assertEqual(len(plan['daily_plan']), 7)
        self.assertEqual(len(plan['stage_plan']), 4)
        self.assertTrue(plan['weekly_plan'][0]['tasks'])
        self.assertFalse(plan['weekly_plan'][1]['tasks'])
        self.assertTrue(all(day['estimated_hours'] < day['available_hours'] for day in plan['daily_plan']))
        self.assertFalse(plan["deferred_tasks"])

    def test_screenshot_ten_days_45_minutes_reaches_application(self):
        self.p._enhance_plan_with_rag = lambda *args: self.fail("最小应用链路不应被额外理论任务挤占")
        plan = self.create("RAG", 10, .75)
        daily = plan["daily_plan"]
        self.assertEqual(len(daily), 10)
        self.assertEqual([d["date"] for d in daily], [
            (datetime.fromisoformat(plan["start_date"]) + timedelta(days=i)).date().isoformat() for i in range(10)
        ])
        self.assertTrue(all(d["tasks"] and d["deliverables"] for d in daily))
        self.assertTrue(all(d["estimated_hours"] == .75 and d["available_hours"] == .75 for d in daily))
        self.assertFalse(plan["deferred_tasks"])
        self.assertIn("验收", str(daily[-1]))
        self.assertNotIn("第 1/", str(daily))
        self.assertEqual(plan["generation_source"], "draft")
        self.assertIn("通用草案", str(plan["assumptions"]))

    def test_daily_task_refs_preserve_completion_api(self):
        plan = self.create("RAG", 10, .75)
        weekly = {str(w["week"]): w for w in plan["weekly_plan"]}
        for day in plan["daily_plan"]:
            for task, ref in zip(day["tasks"], day["task_refs"]):
                self.assertEqual(weekly[ref["week"]]["tasks"][ref["task_index"]], f"第{day['day']}天：{task}")
                self.assertFalse(plan["task_completion"][ref["week"]][ref["task_index"]])
        self.assertEqual(plan["daily_plan"][7]["task_refs"][0], {"week": "2", "task_index": 0})
        updated, _ = self.p._adapt_plan_with_progress(plan, {"weekly_progress": {"1": {"completed_tasks": 0}}})
        self.assertEqual(updated["weekly_plan"], plan["weekly_plan"])
        self.assertEqual(updated["daily_plan"], plan["daily_plan"])

    def test_insufficient_budget_keeps_goal_honest(self):
        plan = self.create("RAG", 1, .5)
        self.assertTrue(plan["deferred_tasks"])
        self.assertIn("时间不足", plan["outcome"])

    def test_real_service_saves_chat_constraints(self):
        with tempfile.TemporaryDirectory() as data_dir:
            self.p.storage = DataStorage(data_dir)
            self.p.profile_store = ProfilesStore(data_dir)
            self.p.agent_id = "planner"
            self.p._attach_dynamic_fields = lambda plan, uid, topic: PlannerService._attach_dynamic_fields(self.p, plan, uid, topic)
            service = ServiceAgent.__new__(ServiceAgent)
            service._get_component = lambda capability: self.p
            coordinator = AgentCoordinator.__new__(AgentCoordinator)
            coordinator.storage = self.p.storage
            coordinator.get_agent = lambda name: service
            coordinator.detect_intent = lambda query: _detect_intent_rules(query)
            uid, cid = "integration", self.id()
            self.addCleanup(conversation_context.reset, uid, cid)
            with patch("core.coordinator.conversation_memory", SimpleNamespace(load_from_database=lambda uid: None)):
                result = list(coordinator.process_request_stream(
                    "制定RAG学习计划，目标：搭建问答应用，周期10天，每天45分钟", uid, conversation_id=cid
                ))[-1]["data"]
            self.assertEqual(result["status"], "success")
            saved = self.p.storage.get_plan(uid, "RAG")
            self.assertEqual(saved["goal"], "搭建问答应用")
            self.assertEqual(saved["duration_days"], 10)
            self.assertEqual(saved["daily_hours"], .75)
            self.assertEqual(result["plan"]["timeline"][-1]["end"],
                             (datetime.fromisoformat(saved["start_date"]) + timedelta(days=9)).date().isoformat())


if __name__ == "__main__":
    unittest.main()
