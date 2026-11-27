"""连续对话回归：主题继承 + 澄清后继续执行（全 mock，不调用真实模型 API）。

覆盖用户复现：
    用户：解释一下agent
    助手：解释 Agent
    用户：制定一个学习计划
    助手：（修复后应直接创建 Agent 计划；若澄清则保存待办）
    用户：上面不是说了agent
    助手：（若上轮澄清，应恢复创建 Agent 计划，而不是再次解释 Agent）
"""
import os
import sys
import unittest
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.coordinator import AgentCoordinator
from core.conversation_context import conversation_context
from nlp.intent_detection import _detect_intent_rules


class FakeQaAgent:
    """模拟问答 Agent：产出解释文本，不调用 RAG/模型。"""

    def process_stream(self, msg):
        subject = msg.content.get('subject', '该主题')
        yield {"type": "meta", "data": {"mode": "general_llm", "source": "general_llm"}}
        yield {"type": "token", "data": f"解释 {subject}：这是一个概念说明。"}
        yield {"type": "complete", "data": {"status": "success", "answer": f"解释 {subject}", "source": "general_llm"}}


class FakePlannerAgent:
    """模拟计划 Agent：记录收到的参数并产出计划完成事件。"""

    def __init__(self, log):
        self.log = log

    def process_stream(self, msg):
        content = dict(msg.content)
        self.log.append(content)
        subject = content.get('subject', '通用')
        yield {"type": "complete", "data": {
            "status": "success",
            "type": "plan_created",
            "plan": {"subject": subject},
            "message": f"已创建{subject}的学习计划",
        }}


class TestContinuousConversation(unittest.TestCase):
    def setUp(self):
        self.uid = "mock_user_" + str(id(self))
        self.cid = "conv_" + str(id(self))
        conversation_context.reset(self.uid, self.cid)
        self.c = AgentCoordinator()
        # 强制走规则意图识别，绝不调用 LLM API
        self.c.detect_intent = lambda q: _detect_intent_rules(q)
        self.planner_log = []

        def fake_get_agent(name):
            if name == "planner":
                return FakePlannerAgent(self.planner_log)
            if name == "qa":
                return FakeQaAgent()
            return FakeQaAgent()

        self.c.get_agent = fake_get_agent

    def _run(self, question, conversation_id=None):
        outputs = []
        cid = self.cid if conversation_id is None else conversation_id
        for chunk in self.c.process_request_stream(question, self.uid, conversation_id=cid):
            outputs.append(chunk)
        return outputs

    @staticmethod
    def _summarize(outputs):
        result = []
        for chunk in outputs:
            data = chunk.get("data", {})
            if isinstance(data, dict):
                result.append({
                    "type": chunk.get("type"),
                    "result_type": data.get("type", ""),
                    "message": data.get("message") or data.get("answer") or "",
                    "subject": (data.get("plan") or {}).get("subject") if isinstance(data.get("plan"), dict) else None,
                })
        return result

    def test_three_turns_inherit_agent_and_create_plan(self):
        """完整三轮：解释 agent → 制定计划（直接创建 agent）→ 不再重复解释。"""
        turns = ["解释一下agent", "制定一个学习计划，目标：完成智能体应用，4周，每天1小时", "上面不是说了agent"]
        printed = []
        for q in turns:
            out = self._run(q)
            printed.append((q, self._summarize(out)))

        # 第二轮就直接创建 Agent 计划，主题来自第一轮用户消息
        second = printed[1][1]
        self.assertTrue(any(s.get("result_type") == "plan_created" for s in second), f"第二轮未创建计划: {second}")
        self.assertTrue(any((s.get("subject") or "").lower() == "agent" for s in second), f"第二轮主题非 agent: {second}")

        # planner 收到的创建请求主题必须是 agent
        create_msgs = [m for m in self.planner_log if m.get('action') == 'create']
        self.assertTrue(create_msgs, "没有创建计划请求")
        self.assertTrue(all((m.get('subject') or '').lower() == 'agent' for m in create_msgs))

        print("\n===== 三轮实际输出 =====")
        for q, summary in printed:
            print(f"用户: {q}")
            for s in summary:
                print(f"  助手: type={s['result_type'] or s['type']} message={s['message']} subject={s['subject']}")
        print("===== 结束 =====\n")

    def test_plan_modifier_inherits_interview_topic(self):
        """用户复现：聊过面试话题后说“为我制定一个详细计划”，
        主题应继承，不能把“详细”当成学习主题（截图 bug）。"""
        self._run("我想面试产品经理实习生，不知道从何开始")
        self._run("详细点")
        self._run("为我制定一个详细计划，目标：通过岗位面试，4周，每天1小时")
        created = [m for m in self.planner_log if m.get('action') == 'create']
        self.assertTrue(created, "没有创建计划")
        self.assertEqual(created[-1].get("subject"), "产品经理实习生面试")

    def test_confirmation_uses_previous_plan_offer(self):
        """用户回“需要”时，若上一轮助手邀请过制定计划，应直接创建计划而不是泛泛回答。"""
        conversation_context.reset(self.uid, self.cid)
        self.planner_log.clear()
        self.c._get_stored_conversation = lambda cid: {"messages": [
            {"role": "user", "content": "解释一下agent"},
            {"role": "assistant", "content": "Agent 是能够感知环境并采取行动的实体。需要我为你制定一份详细学习计划吗？"},
        ]}
        self._run("需要")
        self._run("目标：完成智能体应用，4周，每天1小时")
        created = [m for m in self.planner_log if m.get('action') == 'create']
        self.assertTrue(created, "“需要”没有触发创建计划")
        self.assertEqual(created[-1].get("subject"), "agent")

    def test_clarification_resume_create_agent_plan(self):
        """澄清后继续执行：计划缺主题 → 澄清并保存待办 → 用户答 agent → 恢复创建计划。"""
        conversation_context.reset(self.uid, self.cid)
        self.planner_log.clear()

        # 第一轮：无主题的计划请求 → 澄清
        out1 = self._run("制定一个学习计划，目标：完成实践作品，4周，每天1小时")
        s1 = self._summarize(out1)
        self.assertTrue(any(x.get("result_type") == "plan_clarification" for x in s1), s1)
        pending = conversation_context.get_pending_action(self.uid, self.cid)
        self.assertIsNotNone(pending)
        self.assertEqual(pending.get("action"), "create_plan")

        # 第二轮：用户回指主题 → 应恢复创建 Agent 计划，而非重新解释
        out2 = self._run("上面不是说了agent")
        s2 = self._summarize(out2)
        self.assertTrue(any(x.get("result_type") == "plan_created" for x in s2), f"未恢复创建计划: {s2}")
        self.assertTrue(any((x.get("subject") or "").lower() == "agent" for x in s2), f"恢复的主题非 agent: {s2}")
        # 待办已清空
        self.assertIsNone(conversation_context.get_pending_action(self.uid, self.cid))
        # 没有走成解释型回答
        self.assertFalse(any("解释" in (x.get("message") or "") for x in s2), f"又解释了 Agent: {s2}")

        print("\n===== 澄清恢复两轮实际输出 =====")
        print("用户: 制定一个学习计划")
        for x in s1:
            print(f"  助手: type={x['result_type'] or x['type']} message={x['message']}")
        print("用户: 上面不是说了agent")
        for x in s2:
            print(f"  助手: type={x['result_type'] or x['type']} message={x['message']} subject={x['subject']}")
        print("===== 结束 =====\n")

    def test_topic_inheritance_is_generic(self):
        """主题继承通用性：多个不同主题都能从第一轮继承到第二轮计划。"""
        topics = ['RAG', '经济学', '机器学习', '量子计算', 'Vue', '区块链']
        for i, topic in enumerate(topics):
            uid = 'generic_user_' + str(i)
            cid = 'generic_conv_' + str(i)
            conversation_context.reset(uid, cid)
            c = AgentCoordinator()
            c.detect_intent = lambda q: _detect_intent_rules(q)
            planner_log = []
            c.get_agent = lambda name, log=planner_log: (
                FakePlannerAgent(log) if name == 'planner' else FakeQaAgent()
            )

            # 第一轮：解释主题 → 第二轮：制定计划（应继承该主题）
            # 必须完全消费生成器，process_request_stream 的函数体才会执行
            for _ in c.process_request_stream('解释一下' + topic, uid, conversation_id=cid):
                pass
            out = []
            for chunk in c.process_request_stream('制定一个学习计划，目标：完成实践作品，4周，每天1小时', uid, conversation_id=cid):
                out.append(chunk)

            created = [m for m in planner_log if m.get('action') == 'create']
            self.assertTrue(created, f'{topic}: 未创建计划')
            self.assertEqual(
                (created[0].get('subject') or '').lower(), topic.lower(),
                f'{topic}: 继承的主题错误: {created[0].get("subject")}',
            )
            print(f'  {topic}: 继承成功 → 创建 {created[0].get("subject")} 计划')

    def test_new_conversation_does_not_inherit_topic(self):
        """切换新会话不继承旧主题。"""
        # 旧会话建立 agent 主题
        self._run("解释一下agent", conversation_id="conv-old")
        # 新会话请求计划 → 应澄清，不能继承 agent
        new_cid = "conv-brand-new-" + str(id(self))
        conversation_context.reset(self.uid, new_cid)
        out = self._run("制定一个学习计划", conversation_id=new_cid)
        s = self._summarize(out)
        self.assertTrue(any(x.get("result_type") == "plan_clarification" for x in s), f"新会话错误继承了主题: {s}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
