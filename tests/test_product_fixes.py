"""回归测试：修复三类真实问题的自动化覆盖。

1) 主题识别：RAG / 经济学等自由主题，以及"学完RAG检索部分"的识别
2) 进度计算：无计划依据不默认 10%；溯源追问读业务数据而非知识库
3) 通用知识回答：无资料/部分覆盖/指定资料缺失/模型不可用/来源标记

所有模型调用均为模拟，不访问真实 API，不产生费用。
"""
import os
import sys
import types
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nlp.intent_detection import extract_subject, is_progress_inquiry
from rag import service as rag_service_module
from rag.service import RAGService, is_document_specific_question

# 用户要求：这句文案不得再出现在任何答案/提示词/标签里
BANNED_TAG = "基于模型通用知识"


# ---------- 模拟模型调用的基础设施 ----------

class FakeDoc:
    def __init__(self, content, title="资料片段"):
        self.page_content = content
        self.metadata = {"chunk_title": title, "file_name": title + ".md"}


class FakeResponse:
    def __init__(self, content):
        self.content = content


class FakeChain:
    def __init__(self, answer):
        self.answer = answer

    def invoke(self, variables):
        # 让模型答案带上用户问题关键词，保证 _is_answer_usable 通过
        return FakeResponse(self.answer)


class FakePromptTemplate:
    """替代 langchain PromptTemplate：from_template 返回支持 | 的对象。"""
    answer = None

    @classmethod
    def from_template(cls, template):
        return cls()

    def __or__(self, llm):
        return FakeChain(self.__class__.answer)


def make_service(answer, docs, model_available=True):
    """构造一个完全离线的 RAGService：检索与模型都可控。"""
    svc = RAGService()
    svc.cfg = dict(svc.cfg)
    svc.cfg["api_key"] = "fake-key-for-test"
    svc.PromptTemplate = FakePromptTemplate
    FakePromptTemplate.answer = answer
    svc._ready = True
    svc.imports = {"PromptTemplate": FakePromptTemplate}
    svc._query_cache.clear()

    svc.ensure_ready = lambda: True
    svc._retrieve = lambda q, s, h: (docs, [])
    svc._build_llm = (lambda key: object()) if model_available else (lambda key: None)
    return svc


# ---------- 1. 主题识别 ----------

class TestSubjectExtraction(unittest.TestCase):
    def test_rag_topic_from_completion(self):
        self.assertEqual(extract_subject("RAG的检索部分我已经学完了"), "RAG")

    def test_rag_topic_from_plan(self):
        self.assertEqual(extract_subject("为我制定RAG学习计划"), "RAG")

    def test_free_topic_economics(self):
        self.assertEqual(extract_subject("帮我制定经济学8周学习计划"), "经济学")

    def test_keyword_topic_still_wins(self):
        self.assertEqual(extract_subject("学完机器学习的回归部分"), "机器学习")

    def test_no_topic(self):
        self.assertEqual(extract_subject("今天天气怎么样"), "通用")

    def test_query_form_topic(self):
        """查询式'有没有X的学习计划'也要能提取主题（截图 bug）。"""
        self.assertEqual(extract_subject("那有没有rag的学习计划"), "rag")

    def test_topic_before_instruction(self):
        """'我想学习X，为我制定一个学习路线'只能提取X，不能把整句当主题（截图 bug）。"""
        self.assertEqual(extract_subject("我想学习智能体，为我制定一个学习路线"), "agent")
        self.assertEqual(extract_subject("我想学习RAG，帮我制定一个4周学习计划"), "RAG")
        subject = extract_subject("我想学习智能体，为我制定一个学习路线")
        self.assertNotIn("制定", subject)
        self.assertNotIn("学习路线", subject)
        self.assertNotIn("，", subject)

    def test_plan_modifier_is_not_topic(self):
        """'制定一个详细/本周/完整计划'里的修饰词不是主题（截图 bug）。"""
        self.assertEqual(extract_subject("为我制定一个详细计划"), "通用")
        self.assertEqual(extract_subject("帮我制定本周的学习计划"), "通用")
        self.assertEqual(extract_subject("制定一份完整的学习计划"), "通用")

    def test_interview_role_topic(self):
        """'我想面试产品经理实习生'应识别为面试主题，而不是整句或裸岗位名。"""
        self.assertEqual(
            extract_subject("我想面试产品经理实习生，不知道从何开始"),
            "产品经理实习生面试",
        )
        self.assertEqual(extract_subject("产品经理面试怎么准备"), "产品经理面试")

    def test_agent_alias_and_elliptical_followup(self):
        """agent 与 智能体 归一到同一主题 agent；'怎么学''详细点'应被识别为依赖上文的追问。"""
        from nlp.intent_detection import is_elliptical_query
        self.assertEqual(extract_subject("解释一下agent"), "agent")
        self.assertEqual(extract_subject("agent的学习计划"), "agent")
        self.assertTrue(is_elliptical_query("怎么学"))
        self.assertTrue(is_elliptical_query("详细点"))
        self.assertFalse(is_elliptical_query("今天天气怎么样"))


class TestInterviewPlan(unittest.TestCase):
    def test_interview_plan_ignores_unrelated_profile_goal(self):
        """面试计划不能用画像里无关的"春招后端开发"当目标（截图 bug）。"""
        from agents.planner_service import PlannerService
        p = PlannerService()
        plan = p._build_topic_plan("产品经理实习生面试", "4周", {"goal": "准备春招后端开发"})
        text = str(plan)
        self.assertNotIn("准备春招后端开发", text)
        self.assertNotIn("详细", text)
        self.assertIn("岗位与能力认知", text)
        self.assertTrue(all("面试" in week["goals"][0] for week in plan))

    def test_topic_goal_kept_when_relevant(self):
        """画像目标与主题相关时仍要保留个性化。"""
        from agents.planner_service import PlannerService
        p = PlannerService()
        plan = p._build_topic_plan("后端开发", "4周", {"goal": "准备春招后端开发"})
        self.assertEqual(plan[0]["goals"][0], "准备春招后端开发")


class TestPlanQueryVsCreate(unittest.TestCase):
    def test_query_does_not_create_plan(self):
        """'有没有X计划'是查询，不应创建新计划（截图 bug）。"""
        from core.coordinator import AgentCoordinator
        c = AgentCoordinator()
        c.get_all_plans = lambda u: {"plans": []}
        c.get_plan_detail = lambda u, s: None
        c.route_message = lambda m: types.SimpleNamespace(content={"plan": None})
        result = c.handle_plan_request("那有没有rag的学习计划", "default")
        self.assertEqual(result.get("type"), "plan_query")
        self.assertFalse(result.get("exists"))
        self.assertIn("没有找到", result["message"])
        self.assertIn("rag", result["message"].lower())


class TestPlanRelevance(unittest.TestCase):
    def test_irrelevant_rag_tasks_filtered(self):
        """画像引入的软件工程/Git 任务不能写进 RAG 计划（截图 bug）。"""
        from agents.planner_service import PlannerService
        p = PlannerService()
        answer = (
            "1. 梳理RAG的定义与检索流程\n"
            "2. 软件工程(SE): 软件生命周期、设计模式\n"
            "3. 学习Git版本控制\n"
            "4. 掌握RAG的向量检索与重排序"
        )
        tasks = p._extract_task_suggestions(answer, "RAG核心概念")
        self.assertTrue(any("RAG" in t for t in tasks))
        self.assertFalse(any("软件工程" in t or "Git" in t for t in tasks))

    def test_generic_subject_skips_rag_enhance(self):
        """通用主题不应触发 RAG 增强（避免混入无关资料）。"""
        from agents.planner_service import PlannerService
        p = PlannerService()
        p._enhance_plan_with_rag = lambda *a, **k: self.fail("通用主题不应调用RAG增强")
        plan = p._build_topic_plan("通用", "4周", {"goal": "准备春招后端"})
        # 直接验证：create 路径中 subject=通用 被跳过（此处仅确认生成器产出不含无关词）
        self.assertTrue(all("软件工程" not in t for w in plan for t in w["tasks"]))


# ---------- 2. 进度溯源 ----------

class TestProgressInquiry(unittest.TestCase):
    def test_inquiry_detected(self):
        self.assertTrue(is_progress_inquiry("你根据什么更新的，剩下的90%是什么"))
        self.assertTrue(is_progress_inquiry("剩下的任务有哪些"))
        self.assertTrue(is_progress_inquiry("10%是怎么算出来的"))

    def test_normal_question_not_inquiry(self):
        self.assertFalse(is_progress_inquiry("什么是向量数据库"))

    def test_no_basis_not_verifiable(self):
        """没有可信 total_tasks 依据时，进度不可核实。"""
        from agents.progress_service import ProgressService
        svc = ProgressService()
        fake_storage = types.SimpleNamespace(
            get_progress=lambda u, s: None,
            save_progress=lambda u, s, p: None,
        )
        svc.storage = fake_storage
        result = svc.update_progress("default", "RAG", {
            "completed_tasks": 1,
            "total_tasks": 0,
            "note": "RAG的检索部分我已经学完了",
            "calculation_basis": {"verifiable": False, "plan_subject": "RAG"},
        })
        self.assertFalse(result.get("percentage_verifiable"))
        self.assertIn("last_update_basis", result)
        self.assertFalse(result["last_update_basis"]["verifiable"])

    def test_with_basis_verifiable(self):
        from agents.progress_service import ProgressService
        svc = ProgressService()
        svc.storage = types.SimpleNamespace(
            get_progress=lambda u, s: None,
            save_progress=lambda u, s, p: None,
        )
        result = svc.update_progress("default", "RAG", {
            "completed_tasks": 1,
            "total_tasks": 8,
            "calculation_basis": {"verifiable": True, "plan_subject": "RAG", "explain": "依据计划"},
        })
        self.assertTrue(result.get("percentage_verifiable"))
        self.assertEqual(result.get("progress_percentage"), 12)


# ---------- 3. 通用知识回答 ----------

class TestGeneralKnowledge(unittest.TestCase):
    def test_no_docs_uses_general_llm(self):
        svc = make_service(
            answer="RAG 是检索增强生成，先检索再生成。",
            docs=[],
        )
        result = svc.ask_with_details("什么是RAG", subject="RAG")
        self.assertEqual(result["mode"], "general_llm")
        self.assertEqual(result["source"], "general_llm")
        self.assertNotIn(BANNED_TAG, result["answer"])
        self.assertEqual(result.get("docs_count"), 0)

    def test_no_docs_model_unavailable_error(self):
        svc = make_service(answer="", docs=[], model_available=False)
        result = svc.ask_with_details("什么是RAG", subject="RAG")
        self.assertEqual(result["mode"], "error")
        self.assertIn("不可用", result["answer"])

    def test_doc_specific_missing_no_fabrication(self):
        """指定资料找不到时明确说明，不编造。"""
        svc = make_service(answer="随便编的答案", docs=[])
        result = svc.ask_with_details("根据我上传的资料，RAG是什么", subject="RAG")
        self.assertIn("未找到", result["answer"])
        self.assertNotEqual(result["mode"], "general_llm")

    def test_partial_coverage_keeps_supplement(self):
        """资料覆盖部分问题时，模型补充内容不被剪掉，且 mode 由内部返回、答案不含来源文案。"""
        supplemented = "向量检索用于相似度匹配。RAG 还会对结果做重排序以提升精度。"
        svc = make_service(
            answer=supplemented,
            docs=[FakeDoc("向量检索用于相似度匹配。", "检索资料")],
        )
        answer, mode = svc.generate_answer_with_postcheck(
            question="向量检索和重排序是怎么回事",
            subject="RAG",
            docs=[FakeDoc("向量检索用于相似度匹配。", "检索资料")],
            context="[片段1] 向量检索用于相似度匹配。",
            allow_llm=True,
        )
        self.assertEqual(mode, "rag_hybrid")
        self.assertIn("重排序", answer)
        self.assertNotIn(BANNED_TAG, answer)

    def test_document_specific_detection(self):
        self.assertTrue(is_document_specific_question("根据我上传的资料回答"))
        self.assertTrue(is_document_specific_question("这一章讲了什么"))
        self.assertFalse(is_document_specific_question("什么是RAG"))


# ---------- 4. 流式/非流式来源标记 ----------

class TestSourceMarkers(unittest.TestCase):
    def test_stream_meta_has_source(self):
        svc = make_service(answer="RAG 是检索增强生成。", docs=[])
        events = list(svc.ask_with_details_stream("什么是RAG", subject="RAG"))
        meta = next(e["data"] for e in events if e["type"] == "meta")
        self.assertEqual(meta["mode"], "general_llm")
        self.assertEqual(meta["source"], "general_llm")
        self.assertTrue(meta["is_general"])

    def test_stream_meta_context_mode(self):
        """有资料命中时，mode 不应为 general_llm。"""
        doc = FakeDoc("RAG 是检索增强生成的缩写。", "RAG资料")
        svc = make_service(answer="", docs=[doc])
        events = list(svc.ask_with_details_stream("RAG是什么", subject="RAG"))
        meta = next(e["data"] for e in events if e["type"] == "meta")
        self.assertNotEqual(meta["mode"], "general_llm")

    def test_qa_agent_stream_uses_real_source(self):
        """qa_agent 流式历史记录的来源应来自 meta，而非写死 rag_llm。"""
        from agents.qa_agent import QAAgent
        agent = QAAgent()
        captured = {}
        agent.storage = types.SimpleNamespace(
            save_qa_history=lambda *a, **kw: captured.update(kw),
        )

        def fake_stream(*args, **kwargs):
            yield {"type": "meta", "data": {"source": "general_llm", "is_general": True}}
            yield {"type": "token", "data": "RAG 是检索增强生成，先检索再生成。"}

        import agents.qa_agent as qa_module
        original = qa_module.get_rag_helper
        qa_module.get_rag_helper = lambda: types.SimpleNamespace(ask_with_details_stream=fake_stream)
        try:
            msg = types.SimpleNamespace(content={
                "question": "什么是RAG", "subject": "RAG", "user_id": "default", "action": "generate",
            })
            list(agent.process_stream(msg))
        finally:
            qa_module.get_rag_helper = original

        self.assertEqual(captured["metadata"]["source"], "general_llm")


class TestRagHowToLearn(unittest.TestCase):
    """回归：'rag怎么学'不能把集成测试片段当答案（用户实测 bug）。"""

    LEARNING_PATH_ANSWER = (
        "学习 RAG 的合理顺序：\n"
        "1. 先理解 RAG 的概念：检索增强生成，先检索资料再让模型回答。\n"
        "2. 再学习文档加载与切分：如何把资料切成合适的知识块。\n"
        "3. 然后学习向量化与检索：把文本转成向量并做相似度检索。\n"
        "4. 最后学习生成与优化：把检索结果交给模型生成并做质量优化。"
    )

    def test_irrelevant_chunk_filtered(self):
        """检索命中的集成测试段落（顺带提到RAG）必须被资料意图过滤剔除。"""
        svc = make_service(
            answer=self.LEARNING_PATH_ANSWER,
            docs=[FakeDoc(
                "## 2.2 集成测试 验证多个模块协同工作。例如，在本系统中，"
                "意图识别、任务路由、RAG 检索和结果返回之间的协同，就是典型的集成测试对象。",
                title="2.2 集成测试",
            )],
        )
        self.assertFalse(svc._doc_answers_intent(
            "rag怎么学",
            svc._retrieve("rag怎么学", "RAG", "")[0][0],
            "RAG",
        ))

    def test_answer_no_off_topic_content(self):
        """最终答案不能出现集成测试/意图识别、任务路由，且要给出学习顺序。"""
        svc = make_service(
            answer=self.LEARNING_PATH_ANSWER,
            docs=[FakeDoc(
                "意图识别、任务路由、RAG 检索和结果返回之间的协同，是典型的集成测试对象。",
                title="2.2 集成测试",
            )],
        )
        result = svc.ask_with_details("rag怎么学", subject="RAG")
        answer = result["answer"]
        self.assertNotIn("集成测试", answer)
        self.assertNotIn("意图识别、任务路由", answer)
        self.assertNotIn("主要内容是", answer)
        self.assertNotIn(BANNED_TAG, answer)
        self.assertEqual(result["mode"], "general_llm")
        self.assertIn("RAG", answer)
        # 含学习顺序信号
        self.assertTrue(any(sig in answer for sig in ("先", "再", "然后", "最后", "步骤", "顺序")))

    def test_off_topic_answer_rejected(self):
        """即使生成阶段产出了跑题答案，也要被检测出来并重走通用知识。"""
        svc = make_service(
            answer=self.LEARNING_PATH_ANSWER,
            docs=[FakeDoc("RAG 核心流程：加载、切分、向量化、检索、生成。", title="RAG 技术原理")],
        )
        self.assertFalse(svc._answer_on_topic(
            "rag怎么学", "集成测试的步骤是：意图识别、任务路由、RAG 检索。"
        ))
        self.assertTrue(svc._answer_on_topic("rag怎么学", self.LEARNING_PATH_ANSWER))

    def test_learn_intent_no_extract_candidates(self):
        """'怎么学'不能用资料原文抽取候选（'主要内容是：...'），必须让 LLM/通用知识生成。"""
        svc = make_service(answer=self.LEARNING_PATH_ANSWER, docs=[])
        docs = [FakeDoc("RAG 是一种技术架构，检索增强生成。", title="什么是 RAG？")]
        candidates = svc._build_answer_candidates("rag怎么学", docs, ["rag"])
        self.assertEqual(candidates, [])
        # 非学习意图仍可用抽取候选
        others = svc._build_answer_candidates("什么是RAG", docs, ["rag"])
        self.assertTrue(any(mode.endswith("extract") for _, mode in others))

    def test_relevant_doc_still_used(self):
        """真正讲 RAG 学习顺序的资料应被保留，不能被一律过滤掉。"""
        svc = make_service(answer=self.LEARNING_PATH_ANSWER, docs=[])
        good = FakeDoc(
            "# RAG 技术原理 ## 核心流程 1. 加载 2. 切分 3. 向量化 4. 检索 5. 生成",
            title="RAG 技术原理与本项目实践",
        )
        self.assertTrue(svc._doc_answers_intent("rag怎么学", good, "RAG"))


class TestNoSourceBanner(unittest.TestCase):
    """用户要求：固定文案不得出现在提示词/答案/来源标签/前端。"""

    def test_prompts_have_no_banned_text(self):
        from rag import prompting as p
        imports = {"PromptTemplate": FakePromptTemplate}
        for fn in (p.build_qa_prompt, p.build_hybrid_prompt, p.build_general_prompt, p.build_learning_path_prompt):
            prompt = fn(imports)
            template = getattr(prompt, "template", "") or getattr(prompt, "_template", "") or ""
            self.assertNotIn(BANNED_TAG, str(template), f"{fn.__name__} 含被禁文案")

    def test_general_answer_no_banner_but_has_internal_mode(self):
        svc = make_service(answer="RAG 是检索增强生成，先检索再生成。", docs=[])
        result = svc.ask_with_details("什么是RAG", subject="RAG")
        self.assertNotIn(BANNED_TAG, result["answer"])
        self.assertEqual(result["mode"], "general_llm")
        self.assertEqual(result["source"], "general_llm")

    def test_qa_agent_source_label_no_banner(self):
        from agents.qa_agent import SOURCE_LABELS
        for label in SOURCE_LABELS.values():
            self.assertNotIn(BANNED_TAG, label)
        self.assertEqual(SOURCE_LABELS.get("general_llm"), "模型知识回答")


class TestContinuousPlanTopic(unittest.TestCase):
    """连续两轮对话：主题继承 / 新主题覆盖 / 无历史追问。"""

    def setUp(self):
        import core.coordinator as coord_module
        self.coord = coord_module.AgentCoordinator()
        self.mem = coord_module.conversation_memory
        # 用独立用户隔离，避免污染真实数据
        self.user = "test_continuous_topic_user"
        from core.conversation_context import conversation_context
        conversation_context.reset(self.user, None)
        self.mem._short_term_memory[self.user] = __import__("collections").deque(maxlen=20)

    def _say(self, question, answer="好的"):
        self.mem.append(self.user, question, answer, save_to_db=False)

    def test_second_turn_inherits_rag(self):
        self._say("rag怎么学", "RAG 是检索增强生成。")
        self.assertEqual(self.coord.resolve_plan_subject("制定一个计划", self.user), "rag")

    def test_recent_topic_overrides_old(self):
        self._say("rag怎么学", "RAG 是检索增强生成。")
        self._say("帮我制定经济学计划", "好的")
        self.assertEqual(self.coord.resolve_plan_subject("制定一个计划", self.user), "经济学")

    def test_explicit_new_topic_wins(self):
        self._say("rag怎么学", "RAG 是检索增强生成。")
        self.assertEqual(self.coord.resolve_plan_subject("制定Python计划", self.user), "python")

    def test_no_history_returns_none_for_clarification(self):
        fresh = "test_no_history_user_" + str(id(self))
        self.mem._short_term_memory[fresh] = __import__("collections").deque(maxlen=20)
        self.assertIsNone(self.coord.resolve_plan_subject("制定一个计划", fresh))
        result = self.coord.handle_plan_request("制定一个计划", fresh)
        self.assertEqual(result.get("type"), "plan_clarification")
        self.assertIn("你想学习哪个主题", result.get("message", ""))

    def test_plan_request_creates_rag_plan_from_history(self):
        """端到端：第二轮'制定一个计划'应创建 RAG 计划而非通用。"""
        self._say("rag怎么学", "RAG 是检索增强生成。")
        captured = {}
        self.coord.route_message = lambda m: (
            captured.update(m.content) or types.SimpleNamespace(
                content={"status": "success", "plan": {"subject": "rag"}, "message": "已创建"}
            )
        )
        result = self.coord.handle_plan_request("制定一个计划，目标：完成问答应用，4周，每天1小时", self.user)
        self.assertEqual(captured.get("subject"), "rag")
        self.assertNotEqual(captured.get("subject"), "通用")


class TestAgentLearningFlow(unittest.TestCase):
    """复现并回归：'我想学习agent' 给学习路径；下一轮'制定一个计划'沿用 Agent 主题。"""

    AGENT_LEARNING_ANSWER = (
        "学习 Agent 的入门顺序：\n"
        "1. 先理解 Agent 的概念：能感知环境、调用工具并完成目标的智能体。\n"
        "2. 再学习工具调用：让模型按格式调用外部函数并拿回结果。\n"
        "3. 然后学习记忆与规划：让 Agent 记住上下文并拆分任务。\n"
        "4. 最后动手练习：用几行代码搭建一个能查天气的最小 Agent。"
        "需要我为你制定一份详细的学习计划吗？"
    )
    PROJECT_DOC_CONTENT = (
        "本项目采用混合检索与动态切分，通过 Agent协作 完成意图识别、任务路由与结果返回。"
    )

    def setUp(self):
        import core.coordinator as coord_module
        self.coord = coord_module.AgentCoordinator()
        self.mem = coord_module.conversation_memory
        self.user = "test_agent_learning_user_" + str(id(self))
        self.mem._short_term_memory[self.user] = __import__("collections").deque(maxlen=20)

    def test_want_to_learn_is_learn_intent(self):
        svc = make_service(answer=self.AGENT_LEARNING_ANSWER, docs=[])
        self.assertEqual(svc._question_intent("我想学习agent"), "learn")
        self.assertEqual(svc._question_intent("agent怎么学"), "learn")

    def test_learning_inquiry_routes_to_qa_not_plan(self):
        from nlp.intent_detection import is_learning_inquiry, detect_intent
        from core.coordinator import AgentCoordinator
        self.assertTrue(is_learning_inquiry("我想学习agent"))
        self.assertFalse(is_learning_inquiry("制定一个计划"))
        self.assertEqual(detect_intent("我想学习agent"), "qa")
        self.assertEqual(AgentCoordinator().detect_intent("我想学习agent"), "qa")

    def test_project_intro_doc_filtered_for_learn(self):
        svc = make_service(answer=self.AGENT_LEARNING_ANSWER, docs=[])
        doc = FakeDoc(self.PROJECT_DOC_CONTENT, title="Agent协作与系统架构")
        self.assertFalse(svc._doc_answers_intent("我想学习agent", doc, "agent"))
        self.assertFalse(svc._answer_on_topic("我想学习agent", self.PROJECT_DOC_CONTENT))

    def test_end_to_end_learning_path_no_project_details(self):
        """mock：命中项目介绍资料也不能污染答案，最终给入门顺序+练习+追问。"""
        docs = [FakeDoc(self.PROJECT_DOC_CONTENT, title="Agent协作模块设计")]
        svc = make_service(answer=self.AGENT_LEARNING_ANSWER, docs=docs)
        result = svc.ask_with_details("我想学习agent", subject="agent")
        answer = result["answer"]
        for banned in ("混合检索", "动态切分", "Agent协作", "本系统"):
            self.assertNotIn(banned, answer)
        self.assertIn("agent", answer.lower())
        self.assertTrue(any(s in answer for s in ("先", "再", "然后", "最后")))
        self.assertIn("练习", answer)
        self.assertIn("计划", answer)
        self.assertEqual(result["mode"], "general_llm")
        self.assertEqual(result.get("docs_count"), 0)

    def test_followup_plan_uses_agent_subject(self):
        """连续两轮：第一轮学习路径，第二轮'制定一个计划'生成 Agent 计划。"""
        self.mem.append(self.user, "我想学习agent", self.AGENT_LEARNING_ANSWER, save_to_db=False)
        subject = self.coord.resolve_plan_subject("制定一个计划", self.user)
        self.assertEqual((subject or "").lower(), "agent")
        captured = {}
        self.coord.route_message = lambda m: (
            captured.update(m.content) or types.SimpleNamespace(
                content={"status": "success", "plan": {"subject": subject}, "message": "已创建"}
            )
        )
        result = self.coord.handle_plan_request("制定一个计划，目标：完成智能体应用，4周，每天1小时", self.user)
        self.assertEqual((captured.get("subject") or "").lower(), "agent")
        self.assertEqual(result.get("status"), "success")


if __name__ == "__main__":
    unittest.main(verbosity=2)
