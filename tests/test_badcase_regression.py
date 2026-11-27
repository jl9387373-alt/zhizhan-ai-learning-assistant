# -*- coding: utf-8 -*-
"""Bad Case 回归用例：把 5 条真实缺陷固化为测试。

对应《智能学习助手 Bad Case 复盘报告》第三节：
  BC-01 学习路径答非所问
  BC-02 学习目标被答成项目介绍
  BC-03 连续对话丢失任务
  BC-04 来源提示重复
  BC-05 问 agent 答成候选人评价

全部离线运行，不调用真实模型，不产生费用。
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.conversation_context import ConversationContextManager
from rag.prompting import build_general_prompt
from rag.service import RAGService


class FakeDoc:
    """最小文档块，只保留过滤逻辑需要的字段。"""

    def __init__(self, content, file_name, chunk_title=""):
        self.page_content = content
        self.metadata = {"file_name": file_name, "chunk_title": chunk_title or file_name}


class FakePromptTemplate:
    captured = None

    @classmethod
    def from_template(cls, template):
        cls.captured = template
        return template


def make_service():
    svc = RAGService()
    svc.cfg = dict(svc.cfg)
    svc.cfg["api_key"] = "fake-key-for-test"
    svc._ready = True
    return svc


class TestBC01LearnPath(unittest.TestCase):
    """BC-01：问"RAG 怎么学"，不能拿"原理/本项目做法"充当学习路径。"""

    PRINCIPLE_DOC = (
        "# RAG 技术原理与本项目实践\n"
        "## 什么是 RAG？\nRAG 是一种把检索与生成结合的技术架构。\n"
        "## 本项目的优化\n- 混合检索：结合语义向量与关键词。\n- 动态切分：按内容类型调整切片。"
    )
    PATH_DOC = (
        "# RAG 学习路径：从原理到可评测的检索系统\n"
        "## 学习路径：先学什么，再学什么\n"
        "第一阶段跑通最小链路；第二阶段提升检索质量；第三阶段提升生成质量。\n"
        "## 每个阶段练什么\n1. 把 3 篇文档切块后检索 10 个问题。"
    )

    @classmethod
    def setUpClass(cls):
        cls.svc = make_service()

    def test_intent_is_learn(self):
        self.assertEqual(self.svc._question_intent("RAG怎么学"), "learn")

    def test_principle_doc_rejected_for_learn(self):
        doc = FakeDoc(self.PRINCIPLE_DOC, "RAG技术原理简述.md", "本项目的优化")
        self.assertFalse(self.svc._doc_answers_intent("RAG怎么学", doc, "RAG"))

    def test_path_doc_accepted_for_learn(self):
        doc = FakeDoc(self.PATH_DOC, "RAG 学习路径.md", "学习路径：先学什么，再学什么")
        self.assertTrue(self.svc._doc_answers_intent("RAG怎么学", doc, "RAG"))


class TestBC02LearnGoal(unittest.TestCase):
    """BC-02：说"我想学习 Agent"，不能拿本项目功能介绍作答。"""

    PROJECT_DOC = (
        "# 本项目的优化\n"
        "## 混合检索\n结合语义向量匹配与关键词检索。\n"
        "## 动态切分\n根据内容类型自动调整切片策略。\n"
        "## Agent 协作\n不同 Agent 根据检索置信度决定是否追问。"
    )

    @classmethod
    def setUpClass(cls):
        cls.svc = make_service()

    def test_want_to_learn_is_learn_intent(self):
        self.assertEqual(self.svc._question_intent("我想学习agent"), "learn")

    def test_project_doc_rejected(self):
        doc = FakeDoc(self.PROJECT_DOC, "项目介绍.md", "本项目的优化")
        self.assertFalse(self.svc._doc_answers_intent("我想学习agent", doc, "agent"))


class TestBC03ConversationState(unittest.TestCase):
    """BC-03：澄清后要记住"待创建计划"，补充信息时补参而不是重新解释概念。"""

    def setUp(self):
        self.ctx = ConversationContextManager()

    def test_pending_action_saved_and_recovered(self):
        self.ctx.set_pending_action("u1", "c1", "create_plan", duration="4周")
        pending = self.ctx.get_pending_action("u1", "c1")
        self.assertIsNotNone(pending)
        self.assertEqual(pending["action"], "create_plan")
        self.assertEqual(pending["params"]["duration"], "4周")

    def test_pending_action_cleared_after_use(self):
        self.ctx.set_pending_action("u1", "c1", "create_plan")
        self.ctx.clear_pending_action("u1", "c1")
        self.assertIsNone(self.ctx.get_pending_action("u1", "c1"))

    def test_conversations_are_isolated(self):
        self.ctx.set_pending_action("u1", "c1", "create_plan", topic="agent")
        self.assertIsNone(self.ctx.get_pending_action("u1", "c2"))
        self.assertIsNone(self.ctx.get_pending_action("u2", "c1"))


class TestBC04SourceBanner(unittest.TestCase):
    """BC-04：答案正文不再出现重复的来源声明。"""

    BANNED = "基于模型通用知识"

    def test_general_prompt_has_no_banner(self):
        build_general_prompt({"PromptTemplate": FakePromptTemplate})
        template = FakePromptTemplate.captured or ""
        self.assertNotIn(self.BANNED, template)
        self.assertNotIn("未经知识库核实", template)

    def test_hybrid_prompt_has_no_banner(self):
        from rag.prompting import build_hybrid_prompt

        build_hybrid_prompt({"PromptTemplate": FakePromptTemplate})
        template = FakePromptTemplate.captured or ""
        self.assertNotIn(self.BANNED, template)


class TestBC05CrossTopicDoc(unittest.TestCase):
    """BC-05：问"解释 agent"，不能拿《求职面试指南》里的"软技能"作答。"""

    INTERVIEW_DOC = (
        "## 3. 常见软技能 (Soft Skills)\n"
        "- **沟通能力**: 解释技术概念时的清晰度。\n"
        "- **学习能力**: 如何快速上手新技术。\n"
        "- **抗压能力**: 面对死线或线上问题的态度。"
    )
    INTRO_DOC = (
        "# Agent 入门：从概念到第一个可用智能体\n"
        "## 1. Agent 是什么\n"
        "Agent 指具备目标、工具和行动能力的智能体。判断标准有三条。"
    )

    @classmethod
    def setUpClass(cls):
        cls.svc = make_service()

    def test_intent_is_define(self):
        self.assertEqual(self.svc._question_intent("解释agent"), "define")

    def test_interview_doc_rejected(self):
        doc = FakeDoc(self.INTERVIEW_DOC, "程序员求职面试与简历指南.md", "3. 常见软技能 (Soft Skills)")
        self.assertFalse(self.svc._doc_answers_intent("解释agent", doc, "agent"))

    def test_agent_intro_doc_accepted(self):
        doc = FakeDoc(self.INTRO_DOC, "Agent 入门.md", "1. Agent 是什么")
        self.assertTrue(self.svc._doc_answers_intent("解释agent", doc, "agent"))

    def test_scaffold_word_not_topic(self):
        """问句虚词"解释"不能作为主题词命中别的文档。

        若"解释"被当成主题词，这篇标题含"解释"、正文含"解释技术概念"的文档
        就会被判为与"解释agent"相关——这正是 BC-05 的原始缺陷。
        """
        doc = FakeDoc(
            "## 解释方法\n解释是很重要的能力，解释技术概念要清晰。",
            "写作指南.md",
            "解释方法",
        )
        self.assertFalse(self.svc._doc_answers_intent("解释agent", doc, "agent"))


if __name__ == "__main__":
    unittest.main()


class TestFixThreeRegression(unittest.TestCase):
    """针对重放暴露的三个问题补的回归用例。"""

    PATH_DOC_NO_TOPIC_IN_BODY = (
        "# RAG 学习路径：从原理到可评测的检索系统\n"
        "## 2. 学习路径：先学什么，再学什么\n"
        "第一阶段 · 跑通最小链路：加载文档、切分、向量化。\n"
        "第二阶段 · 提升检索质量。\n"
    )

    @classmethod
    def setUpClass(cls):
        cls.svc = make_service()

    def test_learn_accepts_path_doc_without_topic_in_body(self):
        """学习路径文档正文不重复主题词，也应被接受（原先被判死）。"""
        doc = FakeDoc(self.PATH_DOC_NO_TOPIC_IN_BODY, "RAG 学习路径.md", "2. 学习路径：先学什么，再学什么")
        self.assertTrue(self.svc._doc_answers_intent("RAG怎么学", doc, "RAG"))

    def test_define_prefers_chunk_title_match(self):
        """定义类问题应优先选片段标题含主题的那一段，而不是「常见误区」。"""
        wrong = FakeDoc("## 5. 常见误区\n- 一上来就做多 Agent。", "Agent 入门.md", "5. 常见误区")
        right = FakeDoc("## 1. Agent 是什么\nAgent 指具备目标、工具和行动能力的智能体。", "Agent 入门.md", "1. Agent 是什么")
        ordered = self.svc._prefer_topic_chunks([wrong, right], "解释agent", "agent")
        self.assertIs(ordered[0], right)


if __name__ == "__main__":
    unittest.main()
