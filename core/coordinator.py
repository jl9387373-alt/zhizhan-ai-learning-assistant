"""
Agent协调器
负责协调各个Agent之间的通信和协作
支持多智能体协作：通过消息总线实现 Agent 间直接通信
"""

from typing import Dict, Optional, List, Any
from datetime import datetime
import re
from agents.base_agent import BaseAgent
from agents.qa_agent import QAAgent
from agents.service_agent import ServiceAgent
from agents.message_protocol import message_bus, MessageAction, MessagePriority, AgentMessage
from core.message import Message, MessageType
from core.logger import logger
from memory.chat_memory import get_conversation_memory
from core.storage import DataStorage
from core.conversation_context import conversation_context
from nlp.plan_constraints import extract_plan_constraints, extract_plan_subject
from core.plan_dialog import build_plan_clarification, goal_choices

# 初始化对话记忆
conversation_memory = get_conversation_memory()
from nlp.intent_detection import (
    detect_intent as nlp_detect_intent,
    extract_subject as nlp_extract_subject,
    extract_duration as nlp_extract_duration,
    is_progress_inquiry as nlp_is_progress_inquiry,
    is_elliptical_query as nlp_is_elliptical,
    SUBJECT_KEYWORDS,
)


_AFFIRMATION_RE = re.compile(
    r"^(?:需要|要|好|好的|可以|行|嗯+|是的|对|继续|当然|好呀|要的|需要呀)[。.!！?？~～]*$"
)


class AgentCoordinator:
    """Agent协调器"""
    
    SERVICE_ALIASES = {
        "planner": "service",
        "resource": "service",
        "progress": "service",
        "quiz": "service",
        "review": "service",
    }

    # ---- 统一意图配置（集中管理关键词/规则，避免多处硬编码、规则不一致）----
    PROGRESS_UPDATE_PATTERNS = [
        "完成", "完成了", "学完了", "学完", "学习了", "学了", "更新",
        "已经完成", "已经学完", "学习完成", "完成学习", "完成了.*学习",
        "我.*学完", "我.*完成",
    ]
    REVIEW_TRIGGER_PATTERNS = [
        r"安排", r"计划", r"清单", r"重点", r"提纲", r"复习", r"回顾", r"巩固",
    ]

    def __init__(self):
        self.agents: Dict[str, BaseAgent] = {}
        self.agent_aliases = dict(self.SERVICE_ALIASES)
        self.storage = DataStorage()
        self.register_agents()
        
        # 多智能体协作：协作链路追踪
        self.collaboration_trace: List[Dict[str, Any]] = []
        self._collaboration_enabled = True
    
    def register_agents(self):
        """注册所有Agent"""
        import time
        agent_initialization_times = {}
        
        # 记录每个Agent的初始化时间
        start = time.time()
        self.agents["service"] = ServiceAgent()
        agent_initialization_times["service"] = time.time() - start
        
        start = time.time()
        self.agents["qa"] = QAAgent()
        agent_initialization_times["qa"] = time.time() - start
        
        # 注册QuizAgent（如果可用）
        logger.info("ServiceAgent handles planner/resource/progress/quiz/review capabilities")
        logger.info("Active agent initialization timings:")
        for agent_name, duration in agent_initialization_times.items():
            logger.info(f"  {agent_name}: {duration:.2f}秒")
    
    def get_agent(self, agent_id: str) -> Optional[BaseAgent]:
        """获取Agent"""
        resolved_agent_id = self.agent_aliases.get(agent_id, agent_id)
        return self.agents.get(resolved_agent_id)
    
    def route_message(self, message: Message) -> Message:
        """路由消息到对应的Agent"""
        receiver = message.receiver
        agent = self.get_agent(receiver)
        
        if agent is None:
            # 如果找不到Agent，返回错误消息
            return Message(
                sender="coordinator",
                receiver=message.sender,
                msg_type=MessageType.RESPONSE,
                content={
                    "status": "error",
                    "message": f"未找到Agent: {receiver}"
                }
            )
        
        # 处理消息
        return agent.process(message)
    
    def process_request(self, query: str, user_id: str = "default", 
                       intent: Optional[str] = None, action: Optional[str] = None,
                       conversation_id: Optional[str] = None) -> Dict:
        """处理用户请求"""
        # 从数据库加载用户的对话历史
        try:
            loaded = conversation_memory.load_from_database(user_id)
            if loaded:
                logger.info(f"从数据库加载了用户 {user_id} 的对话历史")
        except Exception as e:
            logger.error(f"加载用户 {user_id} 的对话历史失败: {e}")
        
        # 1. 会话开始：读取长期记忆（用户画像/偏好）
        user_long_term_memory = conversation_memory.get_relevant_long_term_memory(user_id, query)
        if user_long_term_memory:
            logger.info(f"读取了用户 {user_id} 的长期记忆")
            # 可以在这里将长期记忆用于构建用户上下文或个性化处理
        
        normalized_query = (query or "").replace("：", ":")
        
        # 兼容前端“通过聊天指令”触发的计划详情/删除
        if not action:
            if normalized_query.startswith("查看计划详情:"):
                subject = normalized_query.split(":", 1)[1].strip()
                return self.handle_view_plan_details(user_id, subject)
            if normalized_query.startswith("删除计划:"):
                subject = normalized_query.split(":", 1)[1].strip()
                return self.handle_delete_plan(user_id, subject)

        # 优先根据action处理请求
        if action == "view_plan_details":
            # 添加调试信息
            logger.debug(f"DEBUG - 接收到的query: {repr(query)}")
            logger.debug(f"DEBUG - query.startswith('查看计划详情:'): {query.startswith('查看计划详情:')}")
            
            # 从查询中提取主题（格式：查看计划详情:主题）
            if normalized_query.startswith("查看计划详情:"):
                subject = normalized_query.split(":", 1)[1].strip()
                return self.handle_view_plan_details(user_id, subject)
            else:
                return {"status": "error", "message": f"查看计划详情请求格式错误，实际接收: {repr(query)}"}
        elif action == "get_all_plans":
            return self.get_all_plans(user_id)
        elif action == "delete":
            # 从查询中提取主题（格式：删除计划:主题）
            if normalized_query.startswith("删除计划:"):
                subject = normalized_query.split(":", 1)[1].strip()
                return self.handle_delete_plan(user_id, subject)
            else:
                return {"status": "error", "message": f"删除计划请求格式错误，实际接收: {repr(query)}"}
        
        conversation_context.add_user_message(user_id, conversation_id, query)
        if not action and self._is_plan_followup(query, user_id, conversation_id):
            return self.handle_plan_request(query, user_id, conversation_id)
        if not intent and not action:
            confirmation, _ = self._resolve_plan_confirmation(query, user_id, conversation_id)
            if confirmation:
                intent = "plan"
        # 目标里的“完成作品”等措辞不是“更新学习进度”指令。
        intent_query = re.sub(r"(?:学习目标|目标)\s*[:：=]?\s*[^，,。；;\n]+", "", query)
        compound_intents = self._detect_compound_intents(intent_query)
        if len(compound_intents) >= 2:
            return self.handle_complex_request(query, user_id)

        # 意图识别
        if intent is None:
            intent = self.detect_intent(query)
        
        # 根据意图路由到相应的Agent
        # 注意：意图识别返回的是"qa"，但这里检查的是"question"，需要兼容两者
        if intent == "plan":
            return self.handle_plan_request(query, user_id, conversation_id)
        elif intent == "question" or intent == "qa":  # 兼容两种命名
            return self.handle_question_request(query, user_id)
        elif intent == "resource":
            return self.handle_resource_request(query, user_id)
        elif intent == "progress":
            return self.handle_progress_request(query, user_id)
        elif intent == "quiz":
            return self.handle_quiz_request(query, user_id)
        elif intent == "review":
            return self.handle_review_request(query, user_id)
        else:
            # 默认处理：尝试多个Agent
            # 但在进入复杂处理前，优先检查是否是进度更新意图
            query_lower = query.lower()
            if any(re.search(p, query, re.IGNORECASE) for p in self.PROGRESS_UPDATE_PATTERNS):
                # 检测到进度更新关键词，优先作为进度更新处理
                subject = self.extract_subject(query)
                if subject and subject != "通用":
                    return self.handle_progress_request(query, user_id)
            
            # 检查是否是问答意图
            qa_keywords = ['解释一下', '什么是', '什么叫做', '什么意思', '定义', '为什么', '如何', '怎样', '怎么', '？', '?',
                           '我想学', '我要学', '想学习', '想学', '怎么学', '如何学', '从哪学', '入门']
            if any(keyword in query_lower for keyword in qa_keywords):
                # 明确是问答意图，直接处理（handle_question_request会检查并更新进度）
                return self.handle_question_request(query, user_id)
            return self.handle_complex_request(query, user_id)
    
    def detect_intent(self, query: str) -> str:
        """检测用户意图：学习路径询问优先，LLM 次之，规则兜底"""
        from nlp.intent_detection import detect_intent_llm, _detect_intent_rules, is_learning_inquiry
        if is_learning_inquiry(query):
            logger.info(f"意图识别(学习路径): '{query[:30]}' → qa")
            return "qa"
        llm_result = detect_intent_llm(query)
        if llm_result:
            logger.info(f"意图识别(LLM): '{query[:30]}' → {llm_result}")
            return llm_result
        rule_result = _detect_intent_rules(query)
        logger.info(f"意图识别(规则): '{query[:30]}' → {rule_result}")
        return rule_result

    def process_request_stream(self, query: str, user_id: str = "default", 
                              intent: Optional[str] = None, action: Optional[str] = None,
                              image: Optional[str] = None,
                              conversation_id: Optional[str] = None):
        """流式处理用户请求（上下文按会话隔离）"""
        try:
            # 1. 加载对话历史；记录本轮用户消息到会话上下文
            try:
                conversation_memory.load_from_database(user_id)
            except Exception:
                pass
            conversation_context.add_user_message(user_id, conversation_id, query)
            # 记录本轮可识别的主题（只从用户消息提取，不读助手回答）
            turn_subject = extract_plan_subject(query)
            if turn_subject:
                conversation_context.set_last_topic(user_id, conversation_id, turn_subject)

            # 2. 澄清后继续执行：若有待办动作，且当前消息给出了所需参数，则恢复原动作，
            #    不再重新做独立意图分类（例如"上面不是说了agent"→继续创建计划）。
            if not action and self._is_plan_followup(query, user_id, conversation_id):
                intent = "plan"

            # 2.5 短确认（"需要"/"好的"）：承接上一轮"要不要帮你制定计划"的提问
            if not intent:
                is_confirmation, confirmation_subject = self._resolve_plan_confirmation(
                    query, user_id, conversation_id
                )
                if is_confirmation:
                    intent = "plan"

            # 3. 意图识别（无待办恢复时才做全新分类）
            if not intent:
                intent = self.detect_intent(query)
            
            # 4. 确定接收Agent
            agent_map = {
                "plan": "planner",
                "question": "qa",
                "qa": "qa",
                "resource": "resource",
                "progress": "progress",
                "quiz": "quiz",
                "review": "review"
            }
            receiver = agent_map.get(intent, "qa")
            
            # 5. 业务数据类请求（进度更新/查询/溯源追问）：流式与非流式走同一套业务逻辑
            if receiver == "progress" or nlp_is_progress_inquiry(query):
                result = self.process_request(query, user_id, action=action, intent="progress")
                yield {"type": "complete", "data": result}
                return

            # 6. 构建消息；计划主题与非流式共用统一解析
            plan_subject = self.extract_subject(query)
            if receiver == "planner" and (not action or action == "create"):
                if self._is_plan_query(query):
                    yield {"type": "complete", "data": self._query_plan_exists(
                        user_id, self.extract_subject(query), query
                    )}
                    return
                plan_params, clarification = self._prepare_plan_request(query, user_id, conversation_id)
                if clarification:
                    yield {"type": "complete", "data": clarification}
                    return
                plan_subject = plan_params["subject"]
            elif receiver == "qa" and plan_subject == "通用" and nlp_is_elliptical(query):
                # 省略式追问（"怎么学"/"详细点"）继承上一轮主题，而不是答成通用建议
                plan_subject = (
                    conversation_context.get_last_topic(user_id, conversation_id)
                    or self.resolve_plan_subject(query, user_id, conversation_id)
                    or "通用"
                )
                # 关键：把上文补回问句本身，否则检索拿到的只是"详细点"三个字。
                # 优先用上一句原话（如"解释一下什么是PRD"），因为 extract_subject
                # 会把 PRD 归到"产品经理"这一大类，主题太粗会导致检索跑偏。
                prev_question = ""
                try:
                    for item in conversation_context.get_recent_user_messages(user_id, conversation_id, limit=4):
                        if item and item.strip() and item.strip() != str(query).strip():
                            prev_question = item.strip()
                            break
                except Exception:
                    prev_question = ""
                if prev_question:
                    query = f"{prev_question} {query}"
                elif plan_subject and plan_subject != "通用" and plan_subject not in query:
                    query = f"{plan_subject} {query}"

            msg = Message(
                sender="user",
                receiver=receiver,
                msg_type=MessageType.REQUEST,
                content={
                    "question": query,
                    "user_id": user_id,
                    "action": action or "generate",
                    "image": image,
                    "conversation_id": conversation_id,
                    "subject": plan_subject or "通用"
                }
            )

            # 如果是planner，补充duration
            if receiver == "planner" and (not action or action == "create"):
                msg.content["action"] = "create"
                msg.content["duration"] = plan_params["duration"]
                msg.content["constraints"] = plan_params

            agent = self.get_agent(receiver)
            if agent:
                if hasattr(agent, "process_stream"):
                    for chunk in agent.process_stream(msg):
                        if receiver == "planner" and chunk.get("type") == "complete" and (chunk.get("data") or {}).get("plan"):
                            conversation_context.clear_pending_action(user_id, conversation_id)
                        yield chunk
                else:
                    # 伪流式：同步处理后一次性返回
                    res = agent.process(msg)
                    data = res.content if isinstance(res, Message) else res
                    if receiver == "planner" and data.get("plan"):
                        conversation_context.clear_pending_action(user_id, conversation_id)
                    yield {"type": "complete", "data": data}
            else:
                yield {"type": "error", "message": f"未找到Agent: {receiver}"}
                
        except Exception as e:
            logger.error(f"流式处理请求失败: {e}", exc_info=True)
            yield {"type": "error", "message": str(e)}
    
    def handle_view_plan_details(self, user_id: str, subject: str) -> Dict:
        """处理查看学习计划详情的请求"""
        message = Message(
            sender="coordinator",
            receiver="planner",
            msg_type=MessageType.REQUEST,
            content={
                "action": "get",
                "user_id": user_id,
                "subject": subject
            }
        )
        
        response = self.route_message(message)
        return response.content

    def recommend_resources(self, user_id: str, subject: str, difficulty: str = "all", count: int = 5) -> Dict:
        """推荐指定主题的学习资源"""
        message = Message(
            sender="coordinator",
            receiver="resource",
            msg_type=MessageType.REQUEST,
            content={
                "action": "recommend",
                "user_id": user_id,
                "subject": subject,
                "difficulty": difficulty,
                "count": count
            }
        )
        response = self.route_message(message)
        return response.content

    def handle_delete_plan(self, user_id: str, subject: str) -> Dict:
        """处理删除学习计划的请求"""
        message = Message(
            sender="coordinator",
            receiver="planner",
            msg_type=MessageType.REQUEST,
            content={
                "action": "delete",
                "user_id": user_id,
                "subject": subject
            }
        )
        
        response = self.route_message(message)
        return response.content
    
    def get_plan_detail(self, user_id: str, subject: str) -> Dict:
        """获取学习计划详情"""
        message = Message(
            sender="coordinator",
            receiver="planner",
            msg_type=MessageType.REQUEST,
            content={
                "action": "get",
                "user_id": user_id,
                "subject": subject
            }
        )
        
        response = self.route_message(message)
        return response.content
    
    def update_plan(self, user_id: str, subject: str, plan_data: Dict) -> Dict:
        """更新学习计划"""
        message = Message(
            sender="coordinator",
            receiver="planner",
            msg_type=MessageType.REQUEST,
            content={
                "action": "update",
                "user_id": user_id,
                "subject": subject,
                "plan_data": plan_data
            }
        )
        
        response = self.route_message(message)
        return response.content
    
    def delete_plan(self, user_id: str, subject: str) -> bool:
        """删除学习计划"""
        message = Message(
            sender="coordinator",
            receiver="planner",
            msg_type=MessageType.REQUEST,
            content={
                "action": "delete",
                "user_id": user_id,
                "subject": subject
            }
        )
        
        response = self.route_message(message)
        return response.content.get("status") == "success"
    
    def _query_plan_exists(self, user_id: str, subject: str, query: str) -> Dict[str, Any]:
        """查询式请求：只读取已有学习计划，绝不创建新计划。"""
        if subject == "通用":
            all_plans = self.get_all_plans(user_id) or {}
            plans = all_plans.get("plans") or all_plans.get("data") or []
            if isinstance(plans, dict):
                plans = list(plans.values())
            subjects = [p.get("subject") for p in plans if isinstance(p, dict) and p.get("subject")]
            if not subjects:
                return {
                    "status": "success", "type": "plan_query", "exists": False, "source": "business_data",
                    "message": "你目前还没有任何学习计划。可以说「为我制定 RAG 学习计划」来创建一个。",
                }
            return {
                "status": "success", "type": "plan_query", "exists": True, "source": "business_data",
                "subjects": subjects,
                "message": "你现有的学习计划：" + "、".join(subjects) + "。可以说「查看计划详情:主题」查看详情。",
            }

        plan = self.get_plan_detail(user_id, subject) if hasattr(self, "get_plan_detail") else None
        if not plan:
            try:
                message = Message(
                    sender="coordinator", receiver="planner", msg_type=MessageType.REQUEST,
                    content={"action": "get", "user_id": user_id, "subject": subject},
                )
                response = self.route_message(message)
                plan = response.content.get("plan") if response.content else None
            except Exception:
                plan = None
        if plan:
            return {
                "status": "success", "type": "plan_query", "exists": True, "source": "business_data",
                "subject": subject, "plan": plan,
                "message": f"你有《{subject}》的学习计划。",
            }
        return {
            "status": "success", "type": "plan_query", "exists": False, "source": "business_data",
            "subject": subject,
            "message": f"目前没有找到《{subject}》的学习计划。可以说「为我制定{subject}学习计划」来创建一个。",
        }

    @staticmethod
    def _is_plan_query(query: str) -> bool:
        return bool(re.search(r"(有没有|是否有|有么|查看|看看|我的|显示).{0,10}(计划|路线|规划)", query, re.IGNORECASE))

    def _is_plan_followup(self, query: str, user_id: str, conversation_id: Optional[str]) -> bool:
        pending = self._get_pending_plan(user_id, conversation_id)
        if not pending or pending.get("action") != "create_plan":
            return False
        if re.fullmatch(r"(?:取消|算了|不做了|不用了)[。.!！]?", query.strip()):
            conversation_context.clear_pending_action(user_id, conversation_id)
            return False
        values, errors = extract_plan_constraints(query)
        if values or errors or "默认" in query or re.search(r"沿用|修改.*(?:每天|时间)|自定义|不用之前|不使用之前", query):
            return True
        missing = pending.get("params", {}).get("missing_fields", [])
        if "daily_hours" in missing and re.fullmatch(r"\s*[\d.一二两三四五六七八九十半]+\s*(?:小时|分钟|分|h)\s*", query, re.I):
            return True
        if "subject" in missing and extract_plan_subject(query, allow_bare=True):
            return True
        # 缺目标时允许直接回答目标；提问仍交给问答能力。
        return "goal" in missing and not re.search(r"[?？]|什么|怎么|为什么|如何|取消|查看|解释", query)

    def _get_pending_plan(self, user_id: str, conversation_id: Optional[str]):
        pending = conversation_context.get_pending_action(user_id, conversation_id)
        if pending or not conversation_id:
            return pending
        # 重启后恢复当前会话最后一次追问；生成过计划或用户取消后不再恢复。
        messages = self._get_stored_conversation(conversation_id).get("messages", [])
        for idx in range(len(messages) - 1, -1, -1):
            message = messages[idx]
            if message.get("role") == "user" and re.fullmatch(r"(?:取消|算了|不做了|不用了)[。.!！]?", str(message.get("content") or "").strip()):
                return None
            if message.get("role") != "assistant":
                continue
            meta = message.get("metadata") or {}
            if meta.get("plan"):
                return None
            clarification = meta.get("plan_clarification")
            if clarification:
                params = dict(clarification.get("plan_params") or {})
                conversation_context.set_pending_action(user_id, conversation_id, "create_plan", **params)
                return conversation_context.get_pending_action(user_id, conversation_id)
            # 兼容旧版已保存的文字追问：从原用户需求恢复，不继承旧回复擅自采用的画像时间。
            content = str(message.get("content") or "")
            if content.startswith("已记住：") and "请补充" in content:
                previous = next((m for m in reversed(messages[:idx]) if m.get("role") == "user"), {})
                query = str(previous.get("content") or "")
                params, _ = extract_plan_constraints(query)
                params["subject"] = extract_plan_subject(query)
                params["missing_fields"] = [key for key in ("subject", "goal", "duration", "daily_hours") if not params.get(key)]
                conversation_context.set_pending_action(user_id, conversation_id, "create_plan", **params)
                return conversation_context.get_pending_action(user_id, conversation_id)
        return None

    def _prepare_plan_request(self, query: str, user_id: str,
                              conversation_id: Optional[str] = None):
        pending = self._get_pending_plan(user_id, conversation_id) or {}
        params = dict(pending.get("params") or {}) if pending.get("action") == "create_plan" else {}
        missing_before = params.pop("missing_fields", [])
        values, errors = extract_plan_constraints(query)
        if "daily_hours" in missing_before and "daily_hours" not in values and re.fullmatch(
            r"\s*[\d.一二两三四五六七八九十半]+\s*(?:小时|分钟|分|h)\s*", query, re.I
        ):
            time_values, time_errors = extract_plan_constraints("每天" + query)
            values.update(time_values)
            errors.update(time_errors)
        if "duration" in values:
            for field in ("duration_months", "planned_start_date", "planned_end_date"):
                params.pop(field, None)
        if "daily_hours" in values:
            params.pop("suggested_daily_hours", None)
            params.pop("daily_hours_declined", None)
            params["daily_hours_source"] = "message"
        elif re.search(r"修改.*(?:每天|时间)|不沿用|不用之前|不使用之前", query):
            params.pop("daily_hours", None)
            params["daily_hours_declined"] = True
        elif re.search(r"沿用", query) and params.get("suggested_daily_hours") is not None:
            params["daily_hours"] = params["suggested_daily_hours"]
            params["daily_hours_source"] = "profile_confirmed"
        if "goal" in missing_before and params.get("subject") and "subject" not in missing_before and not values.get("goal"):
            for choice in goal_choices(params.get("subject") or ""):
                if re.search(rf"(?:^|[，,。；;\s]){re.escape(choice['label'])}(?:$|[，,。；;\s])", query):
                    values["goal"] = choice["message"].split("：", 1)[1]
                    break
            if not values.get("goal") and re.search(r"自定义|修改.*(?:每天|时间)|沿用|默认|不用之前|不使用之前", query):
                pass
            elif not values.get("goal") and not values and not errors and not re.search(r"[?？]|什么|怎么|为什么|如何", query):
                values["goal"] = query.strip()
        current_subject = extract_plan_subject(query, allow_bare="subject" in missing_before)
        if params.get("subject") and "subject" not in missing_before and not re.search(
            r"主题|学习内容|制定|创建|生成|计划", query
        ):
            current_subject = None
        if current_subject and params.get("subject") and current_subject.lower() != params["subject"].lower():
            params.pop("goal", None)
        params["subject"] = current_subject or params.get("subject") or self.resolve_plan_subject(query, user_id, conversation_id)
        params.update(values)
        if not params.get("subject") and params.get("goal"):
            goal_subject = self.extract_subject(params["goal"])
            if goal_subject in SUBJECT_KEYWORDS or re.match(r"(?:学习|学会|掌握|入门)", params["goal"]):
                if goal_subject == "通用":
                    goal_subject = None
            else:
                goal_subject = None
            if goal_subject:
                params["subject"] = goal_subject
        for field in errors:
            params.pop(field, None)
        if "daily_hours" in errors:
            params["daily_hours_declined"] = True
        if "duration" in errors:
            params.pop("duration_days", None)
            params.pop("requested_duration", None)
            params.pop("duration_months", None)
            params.pop("planned_start_date", None)
            params.pop("planned_end_date", None)
        profile = self.storage.profiles_store.get_profile(user_id)
        if profile.get("profile_completed"):
            if "daily_hours" not in errors and not params.get("daily_hours"):
                params.setdefault("suggested_daily_hours", profile.get("daily_hours", 2))
            profile_goal = str(profile.get("goal") or "")
            if params.get("subject") and params["subject"].lower() in profile_goal.lower():
                params.setdefault("goal", profile_goal)
        if "默认" in query:
            params.setdefault("duration", "4周")
            params.setdefault("duration_days", 28)
            params.setdefault("requested_duration", "4周")
            if not params.get("daily_hours"):
                params["daily_hours"] = 2
                params["daily_hours_source"] = "default_confirmed"
            if params.get("subject"):
                params.setdefault("goal", f"掌握{params['subject']}的核心知识并完成实践")
        if params.get("duration_months"):
            calendar_params, _ = extract_plan_constraints(f"{params['duration_months']}个月")
            for field in ("duration", "duration_days", "planned_start_date", "planned_end_date"):
                params[field] = calendar_params[field]
        missing = [key for key in ("subject", "goal", "duration", "daily_hours") if not params.get(key)]
        if missing or errors:
            params["missing_fields"] = missing
            conversation_context.set_pending_action(user_id, conversation_id, "create_plan", **params)
            return params, build_plan_clarification(params, errors)
        conversation_context.set_pending_action(user_id, conversation_id, "create_plan", **params)
        return params, None

    def handle_plan_request(self, query: str, user_id: str,
                            conversation_id: Optional[str] = None) -> Dict:
        """处理学习计划请求"""
        # 0. 查询式："有没有/是否有 X 的学习计划" → 查询已有计划，不创建新计划
        if self._is_plan_query(query):
            subject = self.extract_subject(query)
            return self._query_plan_exists(user_id, subject, query)

        # 统一主题解析：当前主题优先 → 最近相关提问 → 无法确定则追问
        params, clarification = self._prepare_plan_request(query, user_id, conversation_id)
        if clarification:
            return clarification
        subject, duration = params["subject"], params["duration"]
        
        message = Message(
            sender="coordinator",
            receiver="planner",
            msg_type=MessageType.REQUEST,
            content={
                "action": "create",
                "user_id": user_id,
                "subject": subject,
                "duration": duration,
                "constraints": params,
            }
        )
        
        response = self.route_message(message)
        
        # 将完整学习计划数据保存到QA历史记录，以便历史会话能显示完整内容
        if response.content.get("status") == "success" and "plan" in response.content:
            conversation_context.clear_pending_action(user_id, conversation_id)
            plan = response.content["plan"]
            message_text = response.content.get("message", f"已为您创建{subject}的{duration}学习计划")
            

            # 将学习计划结构化存入 QA 历史，前端历史渲染时复用 renderPlan（不再由后端拼 HTML）
            self.storage.save_qa_history(
                user_id,
                query,
                message_text,
                subject,
                metadata={
                    "source": "planner",
                    "source_label": "学习计划",
                    "plan": plan,
                }
            )
            
            # 更新响应消息，使其更友好
            response.content["message"] = message_text
        
        return response.content
    
    def extract_duration(self, query: str) -> str:
        """从查询中提取学习时长"""
        return nlp_extract_duration(query)

    def create_plan(self, user_id: str, subject: str, duration: Any = 30,
                    goal: str = "", daily_hours: Optional[float] = None) -> Dict:
        """REST 入口与对话使用同一个生成器；数字周期沿用接口的天数语义。"""
        period = f"{duration}天" if isinstance(duration, (int, float)) else str(duration)
        params, errors = extract_plan_constraints(f"周期：{period}")
        if errors or not params.get("duration"):
            raise ValueError("请提供有效的计划周期，例如 14 或 4周")
        params["goal"] = goal
        if daily_hours is not None:
            params["daily_hours"] = float(daily_hours)
        message = Message(sender="coordinator", receiver="planner", msg_type=MessageType.REQUEST,
                          content={"action": "create", "user_id": user_id, "subject": subject,
                                   "duration": params["duration"], "constraints": params})
        response = self.route_message(message).content
        if response.get("status") != "success":
            raise ValueError(response.get("message") or "计划创建失败")
        return response["plan"]
    
    def get_plans(self, user_id: str = "default", subject: str = "") -> List[Dict[str, Any]]:
        """获取用户的学习计划列表
        
        Args:
            user_id: 用户ID，默认为"default"
            subject: 可选，指定主题过滤
            
        Returns:
            学习计划列表
        """
        try:
            # 获取用户的所有学习计划
            all_plans = self.storage.get_all_plans(user_id)
            
            # 转换为列表格式
            plans_list = []
            for key, plan_data in all_plans.items():
                # 从键中解析主题名 (格式: user_id_subject)
                _, plan_subject = key.rsplit("_", 1)
                
                if subject and plan_subject != subject:
                    continue
                    
                plan = plan_data.copy()
                plan["subject"] = plan_subject
                plans_list.append(plan)
            
            return plans_list
        except Exception as e:
            logger.error(f"获取用户 {user_id} 的学习计划失败: {e}")
            return []
    
    def handle_question_request(self, query: str, user_id: str) -> Dict:
        """处理问答请求"""
        subject = self.extract_subject(query)
        if subject == "通用" and nlp_is_elliptical(query):
            subject = self.resolve_plan_subject(query, user_id) or "通用"

        # 进度溯源追问：优先读真实业务数据，不能交给知识库RAG
        if nlp_is_progress_inquiry(query):
            return self._progress_inquiry_answer(query, user_id, subject)

        # 检查是否是进度更新请求（学完了、完成了等）
        query_lower = query.lower()
        is_progress_update = any(re.search(p, query, re.IGNORECASE) for p in self.PROGRESS_UPDATE_PATTERNS)
        
        if is_progress_update and subject and subject != "通用":
            # 先获取当前进度，以便累加任务数
            try:
                get_message = Message(
                    sender="coordinator",
                    receiver="progress",
                    msg_type=MessageType.REQUEST,
                    content={
                        "action": "get",
                        "user_id": user_id,
                        "subject": subject
                    }
                )
                get_response = self.route_message(get_message)
                current_progress = get_response.content.get("progress") if get_response.content else None
            except:
                current_progress = None
            
            # 累加已完成任务数
            current_completed = current_progress.get("completed_tasks", 0) if current_progress else 0
            current_total = current_progress.get("total_tasks", 0) if current_progress else 0

            # 从真实计划取依据，杜绝默认 1/10
            basis = self._compute_progress_basis(user_id, subject, query)
            total_tasks = max(basis["total_tasks"], current_total) if basis["verifiable"] else current_total

            # 更新进度
            progress_data = {
                "completed_tasks": current_completed + 1,
                "total_tasks": total_tasks,
                "study_time": 30,
                "note": query,
                "date": datetime.now().strftime("%Y-%m-%d"),
                "calculation_basis": basis,
            }
            
            progress_message = Message(
                sender="coordinator",
                receiver="progress",
                msg_type=MessageType.REQUEST,
                content={
                    "action": "update",
                    "user_id": user_id,
                    "subject": subject,
                    "progress_data": progress_data
                }
            )
            
            try:
                progress_response = self.route_message(progress_message)
                progress_result = progress_response.content
                
                if progress_result.get("status") == "success":
                    # 进度更新成功，返回祝贺消息和进度信息
                    updated_progress = progress_result.get("progress", {})
                    percentage = updated_progress.get("progress_percentage", 0)
                    
                    return {
                        "status": "success",
                        "message": f"太好了！我已经为您记录了{subject}的学习进度。当前进度：{percentage}%。继续加油！",
                        "progress": updated_progress,
                        "type": "progress_update"
                    }
            except Exception as e:
                logger.error(f"更新进度失败: {e}", exc_info=True)
                # 即使更新失败，也继续回答用户问题
                pass
        
        # 正常处理问答请求
        message = Message(
            sender="coordinator",
            receiver="qa",
            msg_type=MessageType.REQUEST,
            content={
                "action": "generate",
                "question": query,
                "subject": subject,
                "user_id": user_id
            }
        )
        
        response = self.route_message(message)
        result = response.content or {}
        if "subject" not in result:
            result["subject"] = subject
        return result
    
    def handle_resource_request(self, query: str, user_id: str) -> Dict:
        """处理资源推荐请求"""
        subject = self.extract_subject(query)
        
        message = Message(
            sender="coordinator",
            receiver="resource",
            msg_type=MessageType.REQUEST,
            content={
                "action": "recommend",
                "subject": subject,
                "user_id": user_id,
                "count": 5
            }
        )
        
        response = self.route_message(message)
        return response.content
    
    def _get_plan_tasks(self, user_id: str, subject: str) -> List[str]:
        """从真实学习计划读取所有任务描述；无计划或无任务返回空列表。"""
        try:
            plan_message = Message(
                sender="coordinator",
                receiver="planner",
                msg_type=MessageType.REQUEST,
                content={"action": "get", "user_id": user_id, "subject": subject},
            )
            plan_response = self.route_message(plan_message)
            plan = plan_response.content.get("plan") if plan_response.content else None
            if not plan or not plan.get("weekly_plan"):
                return []
            tasks: List[str] = []
            for week in plan.get("weekly_plan", []):
                for task in week.get("tasks", []) or []:
                    if isinstance(task, str) and task.strip():
                        tasks.append(task.strip())
                    elif isinstance(task, dict) and task.get("name"):
                        tasks.append(str(task["name"]).strip())
            return tasks
        except Exception as e:
            logger.warning(f"读取学习计划任务失败: {e}")
            return []

    def _match_task_from_query(self, query: str, tasks: List[str]) -> Optional[str]:
        """把用户话语匹配到计划里的具体任务；用主题词重叠打分，取最高者。"""
        if not tasks:
            return None
        query_lower = (query or "").lower()
        best_task, best_score = None, 0
        for task in tasks:
            score = 0
            for token in re.findall(r"[\u4e00-\u9fa5a-zA-Z]{2,}", task.lower()):
                if token in query_lower:
                    score += len(token)
            if score > best_score:
                best_task, best_score = task, score
        return best_task if best_score > 0 else None

    def _compute_progress_basis(self, user_id: str, subject: str, query: str) -> Dict[str, Any]:
        """计算进度更新依据：只有找到计划及其任务，才给出可核实的 total_tasks。"""
        plan_tasks = self._get_plan_tasks(user_id, subject)
        matched_task = self._match_task_from_query(query, plan_tasks)
        if plan_tasks:
            return {
                "verifiable": True,
                "plan_subject": subject,
                "total_tasks": len(plan_tasks),
                "total_tasks_source": f"学习计划《{subject}》的任务总数",
                "completed_tasks_source": "本次学习语句匹配到计划任务" if matched_task else "按一次学习记 1 个任务",
                "matched_task": matched_task,
                "all_tasks": plan_tasks,
                "explain": f"进度按学习计划《{subject}》的 {len(plan_tasks)} 个任务计算。",
            }
        return {
            "verifiable": False,
            "plan_subject": subject,
            "total_tasks": 0,
            "total_tasks_source": None,
            "completed_tasks_source": "仅记录学习内容，无计划任务可对应",
            "matched_task": None,
            "all_tasks": [],
            "explain": f"暂无计算{subject}进度的计划依据：未找到对应学习计划及其任务，因此本次只记录学习内容，不计算百分比。",
        }

    def _progress_inquiry_answer(self, query: str, user_id: str, subject: str) -> Dict[str, Any]:
        """回答进度溯源追问：读取真实进度与计划，列出依据和未完成任务。"""
        progress_message = Message(
            sender="coordinator", receiver="progress", msg_type=MessageType.REQUEST,
            content={"action": "get", "user_id": user_id, "subject": subject, "detail": True},
        )
        response = self.route_message(progress_message)
        progress = response.content.get("progress") if response.content else None
        basis = (progress or {}).get("last_update_basis") or {}
        plan_tasks = self._get_plan_tasks(user_id, subject)
        completed_tasks = (progress or {}).get("completed_tasks", 0) or 0
        done_count = min(completed_tasks, len(plan_tasks)) if plan_tasks else completed_tasks
        remaining = plan_tasks[done_count:] if plan_tasks else []

        lines: List[str] = []
        if basis.get("verifiable"):
            lines.append(basis.get("explain") or f"进度依据学习计划《{subject}》计算。")
            lines.append(
                f"计算方式：已完成任务数 {done_count} ÷ 计划任务总数 {len(plan_tasks)} = "
                f"{int(done_count / len(plan_tasks) * 100)}%。"
            )
            if basis.get("matched_task"):
                lines.append(f"本次更新匹配到的任务：{basis['matched_task']}。")
        elif basis:
            lines.append(basis.get("explain") or "此前进度缺少可核实依据。")
            lines.append("因此此前显示的百分比无法核实，建议制定学习计划后再计算进度。")
        else:
            lines.append(
                f"当前没有找到《{subject}》的进度更新记录，无法说明百分比的计算依据。"
                if subject != "通用" else "当前没有找到可核实的进度更新记录，无法说明百分比的计算依据。"
            )

        if remaining:
            lines.append("尚未完成的任务：" + "；".join(f"{i + 1}. {t}" for i, t in enumerate(remaining[:12])))
            if len(remaining) > 12:
                lines.append(f"（还有 {len(remaining) - 12} 个任务未列出）")
        else:
            lines.append("计划中没有可列出的未完成任务。")

        answer = "\n".join(lines)
        return {
            "status": "success",
            "answer": answer,
            "message": answer,
            "source": "business_data",
            "type": "progress_inquiry",
            "subject": subject,
            "progress": progress,
            "verifiable": bool(basis.get("verifiable")),
        }

    def handle_progress_request(self, query: str, user_id: str) -> Dict:
        """处理进度查询和更新请求"""
        subject = self.extract_subject(query)

        # 进度溯源追问（根据什么更新/剩下的是什么）优先读真实业务数据，不走知识库
        if nlp_is_progress_inquiry(query):
            return self._progress_inquiry_answer(query, user_id, subject)
        
        # 检查是否包含更新关键词（统一意图配置）
        query_lower = query.lower()
        is_update = any(re.search(pattern, query, re.IGNORECASE) for pattern in self.PROGRESS_UPDATE_PATTERNS)
        
        if is_update:
            # 这是一个进度更新请求
            # 先获取当前进度，以便累加任务数
            try:
                get_message = Message(
                    sender="coordinator",
                    receiver="progress",
                    msg_type=MessageType.REQUEST,
                    content={
                        "action": "get",
                        "user_id": user_id,
                        "subject": subject,
                        "detail": True  # 获取详细任务数据
                    }
                )
                get_response = self.route_message(get_message)
                current_progress = get_response.content.get("progress") if get_response.content else None
            except Exception as e:
                logger.warning(f"获取当前进度失败: {e}")
                current_progress = None
            
            # 累加已完成任务数，而不是重置
            current_completed = current_progress.get("completed_tasks", 0) if current_progress else 0
            current_total = current_progress.get("total_tasks", 0) if current_progress else 0

            # 计算新的任务数：只有找到真实计划及任务才得到可核实的 total_tasks
            new_completed = current_completed + 1
            basis = self._compute_progress_basis(user_id, subject, query)
            if basis["verifiable"]:
                new_total = max(basis["total_tasks"], current_total)
            else:
                # 无计划依据：不虚构总数；沿用已有可信总数，否则 total=0
                new_total = current_total

            # 构建详细进度数据，包含任务详情与计算依据
            progress_data = {
                "completed_tasks": new_completed,
                "total_tasks": new_total,
                "study_time": 30,
                "note": query,
                "date": datetime.now().strftime("%Y-%m-%d"),
                "detail": True,
                "calculation_basis": basis,
            }
            
            # 尝试从当前进度中获取已有任务列表，如果存在则传递给更新
            if current_progress and "tasks" in current_progress:
                progress_data["tasks"] = current_progress["tasks"]
            
            message = Message(
                sender="coordinator",
                receiver="progress",
                msg_type=MessageType.REQUEST,
                content={
                    "action": "update",
                    "user_id": user_id,
                    "subject": subject,
                    "progress_data": progress_data,
                    "detail": True  # 请求返回详细数据
                }
            )
            
            response = self.route_message(message)
            result = response.content
            
            # 如果更新成功，依据是否可核实给出不同话术
            if result and result.get("status") == "success":
                updated_progress = result.get("progress", {})
                if basis["verifiable"]:
                    percentage = updated_progress.get("progress_percentage", 0)
                    result["message"] = (
                        f"已记录《{subject}》的学习进度。"
                        f"根据学习计划共 {basis['total_tasks']} 个任务，已完成 {new_completed} 个，当前进度 {percentage}%。"
                    )
                    suggestions = ["定期复习巩固知识点", "按计划继续完成剩余任务"]
                    if "progress" in result:
                        result["progress"]["suggestions"] = suggestions
                else:
                    result["message"] = (
                        f"已记录本次《{subject}》的学习内容，但暂无计算进度的计划依据："
                        "未找到对应学习计划及其任务，因此不计算百分比。"
                        "此前若显示过百分比，因缺少计划依据无法核实。"
                    )
                    if "progress" in result:
                        result["progress"]["suggestions"] = ["建议先制定该主题的学习计划，之后即可按任务计算进度"]
                result["type"] = "progress_update"
                result["source"] = "business_data"

            return result if result else {"status": "error", "message": "更新进度失败"}
        else:
            # 这是一个进度查询请求
            message = Message(
                sender="coordinator",
                receiver="progress",
                msg_type=MessageType.REQUEST,
                content={
                    "action": "get",
                    "user_id": user_id,
                    "subject": subject,
                    "detail": True  # 默认返回详细任务数据
                }
            )
            
            response = self.route_message(message)
            result = response.content
            
            # 确保返回的消息包含所有必要的进度详情和学习建议
            if result and result.get("status") == "success" and "progress" in result:
                progress = result["progress"]
                percentage = progress.get("progress_percentage", 0)
                
                # 添加学习建议
                if "suggestions" not in progress:
                    suggestions = []
                    if percentage < 30:
                        suggestions = ["建议制定详细学习计划", "每天安排固定学习时间"]
                    elif percentage < 70:
                        suggestions = ["学习进度良好，继续保持", "定期复习巩固知识点"]
                    else:
                        suggestions = ["学习即将完成，建议全面复习", "应用所学知识解决实际问题"]
                    progress["suggestions"] = suggestions
            
            return result if result else {"status": "error", "message": "获取进度失败"}
    
    def _matches_any(self, query: str, patterns: List[str]) -> bool:
        """Return whether the query matches any regex pattern."""
        return any(re.search(pattern, query, re.IGNORECASE) for pattern in patterns)

    def _detect_compound_intents(self, query: str) -> List[str]:
        """Detect likely multi-intent requests for orchestration."""
        intent_patterns = {
            "plan": [
                r"学习计划",
                r"复习计划",
                r"制定.*计划",
                r"做一个.*计划",
                r"列.*计划",
                r"规划.*(天|周|学习|路线|安排)",
                r"安排.*(天|周|学习|路线|计划)",
                r"排.*(计划|安排|路线|每天学什么)",
                r"先.*(做|排|列).*(计划|安排)",
                r"\d+\s*(天|周|个月).*(计划|安排|路线)",
                r"学习路线",
                r"学习路径",
                r"学习安排",
                r"做安排",
                r"(在|用).*(天|周|个月)内.*(学完|补完|复习|掌握)",
                r"过一遍",
                r"系统复习",
            ],
            "qa": [
                r"什么是",
                r"解释",
                r"说明",
                r"原理",
                r"区别",
                r"关系",
                r"为什么",
                r"作用",
                r"怎么理解",
                r"讲清楚",
            ],
            "resource": [
                r"推荐",
                r"资料",
                r"教程",
                r"视频",
                r"网站",
                r"课程",
                r"资源",
                r"文章",
                r"书",
                r"配几份",
            ],
            "progress": [
                r"进度",
                r"更新",
                r"记录",
                r"学完",
                r"完成",
                r"看完",
                r"搞定",
                r"同步.*进度",
                r"查看.*进度",
                r"当前.*进度",
            ],
            "quiz": [
                r"出\s*\d*\s*道.*题",
                r"出几道.*题",
                r"出题",
                r"练习题",
                r"测试题",
                r"测验",
                r"考我",
                r"自测",
                r"小测",
                r"小题",
                r"检验",
                r"检测",
                r"检测题",
                r"来一组.*题",
                r"热身",
                r"掌握情况",
                r"题目",
                r"习题",
                r"判断题",
            ],
            "review": [
                r"复习",
                r"复盘",
                r"回顾",
                r"巩固",
                r"提纲",
                r"重点",
                r"清单",
                r"总结",
                r"记忆曲线",
            ],
        }

        detected: List[str] = []
        for intent, patterns in intent_patterns.items():
            if self._matches_any(query, patterns):
                detected.append(intent)

        ordered = ["qa", "plan", "progress", "resource", "quiz", "review"]
        return [intent for intent in ordered if intent in detected]

    def _schedule_review_request(
        self,
        user_id: str,
        subject: str,
        content_text: str = "",
        importance: str = "medium",
    ) -> Dict:
        """Create or refresh a review plan for the given subject."""
        if not self.get_agent("review"):
            return {
                "status": "error",
                "message": "复习功能暂不可用",
            }

        message = Message(
            sender="coordinator",
            receiver="review",
            msg_type=MessageType.REQUEST,
            content={
                "action": "schedule",
                "user_id": user_id,
                "subject": subject or "通用",
                "content": content_text or f"{subject or '当前'}相关知识点复习",
                "importance": importance,
            },
        )

        try:
            response = self.route_message(message)
            return response.content
        except Exception as e:
            logger.error(f"安排复习请求失败: {str(e)}", exc_info=True)
            return {
                "status": "error",
                "message": f"安排复习请求失败: {str(e)}",
            }

    def _merge_complex_results(
        self,
        query: str,
        subject: str,
        intents: List[str],
        qa_result: Optional[Dict] = None,
        plan_result: Optional[Dict] = None,
        progress_result: Optional[Dict] = None,
        resource_result: Optional[Dict] = None,
        quiz_result: Optional[Dict] = None,
        review_result: Optional[Dict] = None,
    ) -> Dict:
        """Merge sub-agent outputs into one evaluation-friendly response."""
        result: Dict[str, Any] = {
            "status": "success",
            "type": "complex",
            "subject": subject,
            "intent_chain": intents,
            "query": query,
        }
        completed_parts: List[str] = []

        if qa_result and qa_result.get("status") == "success":
            if qa_result.get("answer"):
                result["answer"] = qa_result.get("answer")
            if qa_result.get("source"):
                result["source"] = qa_result.get("source")
            if qa_result.get("context_snippets"):
                result["context_snippets"] = qa_result.get("context_snippets")
            completed_parts.append("问答")

        if plan_result and plan_result.get("status") == "success" and plan_result.get("plan"):
            result["plan"] = plan_result.get("plan")
            completed_parts.append("学习计划")

        if progress_result and progress_result.get("status") == "success":
            if progress_result.get("progress"):
                result["progress"] = progress_result.get("progress")
            if progress_result.get("type") == "progress_update":
                result["type"] = "progress_update"
            completed_parts.append("进度管理")

        if resource_result and resource_result.get("status") == "success" and resource_result.get("resources"):
            result["resources"] = resource_result.get("resources")
            completed_parts.append("资源推荐")

        if quiz_result and quiz_result.get("status") == "success":
            if quiz_result.get("questions"):
                result["questions"] = quiz_result.get("questions")
            if quiz_result.get("quiz_id"):
                result["quiz_id"] = quiz_result.get("quiz_id")
            completed_parts.append("测验生成")

        if review_result and review_result.get("status") == "success":
            if review_result.get("review"):
                result["review"] = review_result.get("review")
            if review_result.get("review_id"):
                result["review_id"] = review_result.get("review_id")
            if review_result.get("next_review_at"):
                result["next_review_at"] = review_result.get("next_review_at")
            completed_parts.append("复习安排")

        if not completed_parts:
            return {
                "status": "error",
                "message": "未能完成复合任务编排",
                "type": "complex",
                "subject": subject,
                "intent_chain": intents,
            }

        result["message"] = "已完成" + "、".join(completed_parts)
        return result

    def extract_subject(self, query: str) -> str:
        """从查询中提取主题"""
        return nlp_extract_subject(query)

    def resolve_plan_subject(self, query: str, user_id: str, conversation_id: Optional[str] = None) -> Optional[str]:
        """统一的计划主题解析（流式与非流式共用），按会话隔离。
        1) 当前请求明确提到主题 → 优先使用；
        2) 同一会话待办动作里已记录的 subject → 次之；
        3) 同一会话最近用户消息（不含当前）中最近一个明确主题；
        4) 同一会话记录的最近主题；
        5) 兼容旧的用户级短期记忆；
        6) 仍无法确定 → 返回 None，由调用方追问，不创建"通用"计划。
        不读助手回答，不扫描全部历史。"""
        current = extract_plan_subject(query)
        if current:
            return current

        # 2) 待办动作里保存的已知主题
        pending = conversation_context.get_pending_action(user_id, conversation_id)
        if pending:
            pending_subject = (pending.get("params") or {}).get("subject")
            if pending_subject and pending_subject != "通用":
                return pending_subject

        # 3) 同一会话最近的用户消息（只取用户说的，最近在前）
        for question in conversation_context.get_recent_user_messages(user_id, conversation_id, limit=5):
            if question.strip() == query.strip():
                continue
            subject = extract_plan_subject(question)
            if subject:
                return subject

        # 3.5) 服务重启后内存上下文会丢失，用已落库的会话消息兜底
        stored_conversation = self._get_stored_conversation(conversation_id)
        for msg in reversed(stored_conversation.get("messages", [])[-20:]):
            if msg.get("role") != "user":
                continue
            question = (msg.get("content") or "").strip()
            if not question or question == query.strip():
                continue
            subject = extract_plan_subject(question)
            if subject:
                return subject

        # 4) 会话上下文里记录的最近主题（由用户消息提取）
        last_topic = conversation_context.get_last_topic(user_id, conversation_id)
        if last_topic and last_topic != "通用":
            return last_topic

        # 5) 兼容旧的用户级短期记忆（无会话上下文时）
        for question in (conversation_memory.get_recent_user_questions(user_id, limit=5) if not conversation_id else []):
            if question.strip() == query.strip():
                continue
            subject = extract_plan_subject(question)
            if subject:
                return subject
        return None

    def _get_stored_conversation(self, conversation_id: Optional[str]) -> Dict[str, Any]:
        if not conversation_id:
            return {}
        try:
            return self.storage.get_conversation(conversation_id) or {}
        except Exception:
            return {}

    def _resolve_plan_confirmation(self, query: str, user_id: str,
                                   conversation_id: Optional[str] = None) -> tuple:
        """识别"需要/好的"这类确认，并判断上一轮助手是否在询问"要不要制定计划"。
        返回 (是否确认, 主题或None)。"""
        text = (query or "").strip()
        if not text or not _AFFIRMATION_RE.match(text):
            return False, None
        conversation = self._get_stored_conversation(conversation_id)
        assistant_messages = [
            msg.get("content") or ""
            for msg in conversation.get("messages", [])
            if msg.get("role") == "assistant" and (msg.get("content") or "").strip()
        ]
        offer = ""
        for content in reversed(assistant_messages[-3:]):
            if "计划" in content and any(key in content for key in (
                "需要我", "要不要", "是否要", "帮你制定", "为你制定", "制定一份", "制定一个",
            )):
                offer = content
                break
        if not offer:
            return False, None
        return True, self.resolve_plan_subject("", user_id, conversation_id)

    def handle_quiz_request(self, query: str, user_id: str) -> Dict:
        """处理测验请求"""
        subject = self.extract_subject(query)
        
        # 检查quiz agent是否存在
        if not self.get_agent("quiz"):
            return {
                "status": "error",
                "message": "测验功能正在开发中，暂不可用"
            }
        
        # 从查询中提取参数
        import re
        difficulty_match = re.search(r'(简单|easy|初级|中等|medium|中级|困难|hard|高级)', query, re.IGNORECASE)
        difficulty = difficulty_match.group(1) if difficulty_match else "medium"
        
        num_match = re.search(r'(\d+)\s*道', query)
        num_questions = int(num_match.group(1)) if num_match else 5
        
        message = Message(
            sender="coordinator",
            receiver="quiz",
            msg_type=MessageType.REQUEST,
            content={
                "action": "generate",
                "query": query,
                "subject": subject,
                "difficulty": difficulty,
                "num_questions": num_questions,
                "user_id": user_id
            }
        )
        
        try:
            response = self.route_message(message)
            return response.content
        except Exception as e:
            logger.error(f"处理测验请求失败: {str(e)}", exc_info=True)
            return {
                "status": "error",
                "message": f"处理测验请求时发生错误: {str(e)}"
            }
    
    def handle_complex_request(self, query: str, user_id: str) -> Dict:
        """Handle compound requests by orchestrating multiple capabilities."""
        subject = self.extract_subject(query)
        intents = self._detect_compound_intents(query)

        if len(intents) >= 2:
            qa_result = self.handle_question_request(query, user_id) if "qa" in intents else None
            plan_result = self.handle_plan_request(query, user_id) if "plan" in intents else None
            progress_result = self.handle_progress_request(query, user_id) if "progress" in intents else None
            resource_result = self.handle_resource_request(query, user_id) if "resource" in intents else None
            quiz_result = self.handle_quiz_request(query, user_id) if "quiz" in intents else None
            review_result = (
                self._schedule_review_request(
                    user_id=user_id,
                    subject=subject,
                    content_text=(qa_result or {}).get("answer") or query,
                    importance="medium",
                )
                if "review" in intents
                else None
            )

            return self._merge_complex_results(
                query=query,
                subject=subject,
                intents=intents,
                qa_result=qa_result,
                plan_result=plan_result,
                progress_result=progress_result,
                resource_result=resource_result,
                quiz_result=quiz_result,
                review_result=review_result,
            )

        if "plan" in intents:
            return self.handle_plan_request(query, user_id)

        if "resource" in intents:
            return self.handle_resource_request(query, user_id)

        if "quiz" in intents:
            return self.handle_quiz_request(query, user_id)

        if "progress" in intents:
            return self.handle_progress_request(query, user_id)

        if "review" in intents:
            return self.handle_review_request(query, user_id)

        return self.handle_question_request(query, user_id)

    def handle_review_request(self, query: str, user_id: str) -> Dict:
        """Handle review requests with scheduling-first behavior."""
        if not self.get_agent("review"):
            return {
                "status": "error",
                "message": "复习功能暂不可用",
            }

        query_lower = query.lower()
        subject = self.extract_subject(query)

        if self._matches_any(query, self.REVIEW_TRIGGER_PATTERNS):
            return self._schedule_review_request(
                user_id=user_id,
                subject=subject,
                content_text=query,
                importance="medium",
            )

        action = "complete" if ("完成" in query or "complete" in query_lower) else "next"
        message = Message(
            sender="coordinator",
            receiver="review",
            msg_type=MessageType.REQUEST,
            content={
                "action": action,
                "query": query,
                "user_id": user_id,
                "subject": subject,
            }
        )

        try:
            response = self.route_message(message)
            return response.content
        except Exception as e:
            logger.error(f"处理复习请求失败: {str(e)}", exc_info=True)
            return {
                "status": "error",
                "message": f"处理复习请求失败: {str(e)}"
            }

    def get_agent_trace(self) -> List[Dict[str, Any]]:
        """获取最近的 Agent 协作链路（来自真实运行记录）"""
        return self.get_collaboration_trace()

    def get_agent_status(self) -> Dict:
        """获取所有Agent的状态"""
        status = {
            "agents": [
                {
                    "id": agent_id,
                    "name": agent.name,
                    "available": True
                }
                for agent_id, agent in self.agents.items()
            ],
            "timestamp": datetime.now().isoformat()
        }
        return status
    
    # ========== 多智能体协作编排方法 ==========
    
    def process_collaboration(self):
        """
        处理 Agent 间的协作消息
        由主循环定期调用，确保消息被及时处理
        """
        for agent_id, agent in self.agents.items():
            if hasattr(agent, 'receive_collaboration_messages'):
                processed = agent.receive_collaboration_messages()
                for msg in processed:
                    self._record_collaboration_trace(agent_id, msg)
    
    def _record_collaboration_trace(self, agent_id: str, msg: AgentMessage):
        """记录协作链路"""
        self.collaboration_trace.append({
            "step": len(self.collaboration_trace) + 1,
            "agent": agent_id,
            "action": msg.action.value if hasattr(msg.action, 'value') else str(msg.action),
            "sender": msg.sender,
            "receiver": msg.receiver,
            "content_type": msg.content.get("type", "unknown") if isinstance(msg.content, dict) else "unknown",
            "timestamp": msg.timestamp
        })
        
        # 保持最近 50 条记录
        if len(self.collaboration_trace) > 50:
            self.collaboration_trace = self.collaboration_trace[-50:]
    
    def get_collaboration_trace(self) -> List[Dict[str, Any]]:
        """获取协作链路记录"""
        return self.collaboration_trace.copy()
    
    def get_message_bus_stats(self) -> Dict[str, Any]:
        """获取消息总线统计"""
        return message_bus.get_statistics()
    
    def broadcast_task(self, task: Dict[str, Any], target_agents: Optional[List[str]] = None) -> bool:
        """
        广播任务给指定的 Agent 或所有 Agent
        
        Args:
            task: 任务内容
            target_agents: 目标 Agent 列表，None 表示所有 Agent
        
        Returns:
            是否发送成功
        """
        if target_agents is None:
            # 发送给所有 Agent
            for agent_id in self.agents:
                msg = AgentMessage(
                    sender="coordinator",
                    receiver=agent_id,
                    action=MessageAction.TASK,
                    content=task
                )
                message_bus.send(msg)
        else:
            for agent_id in target_agents:
                if agent_id in self.agents:
                    msg = AgentMessage(
                        sender="coordinator",
                        receiver=agent_id,
                        action=MessageAction.TASK,
                        content=task
                    )
                    message_bus.send(msg)
        
        return True

    # 笔记相关功能
    def create_note(self, user_id: str, subject: str, title: str, content: str) -> Dict:
        return self.storage.create_note(user_id, subject, title, content)

    def get_notes(self, user_id: str, subject: str = "") -> List[Dict]:
        return self.storage.get_notes(user_id, subject)

    def get_note_detail(self, user_id: str, note_id: str) -> Optional[Dict]:
        return self.storage.get_note_detail(user_id, note_id)

    def update_note(self, user_id: str, note_id: str, updates: Dict) -> Optional[Dict]:
        return self.storage.update_note(user_id, note_id, updates)

    def delete_note(self, user_id: str, note_id: str) -> bool:
        return self.storage.delete_note(user_id, note_id)

    # 用户画像
    def get_user_profile(self, user_id: str) -> Dict:
        return self.storage.get_profile(user_id)

    def update_user_profile(self, user_id: str, profile_data: Dict) -> None:
        self.storage.save_profile(user_id, profile_data)

    def check_in(self, user_id: str) -> Dict:
        return self.storage.check_in(user_id)

    # 错题本 - 已移除

    # 闪卡管理
    def create_flashcard(self, user_id: str, subject: str, front: str, back: str) -> Dict:
        return self.storage.create_flashcard(user_id, subject, front, back)

    def get_due_flashcards(self, user_id: str) -> List[Dict]:
        return self.storage.get_due_flashcards(user_id)

    def get_all_flashcards(self, user_id: str) -> List[Dict]:
        return self.storage.get_all_flashcards(user_id)

    def review_flashcard(self, user_id: str, card_id: str, success: bool) -> Optional[Dict]:
        # 记录复习行为增加XP
        if success:
            self.storage.update_xp(user_id, 10)
        return self.storage.review_flashcard(user_id, card_id, success)

    def delete_flashcard(self, user_id: str, card_id: str) -> bool:
        return self.storage.delete_flashcard(user_id, card_id)
    
    def get_resources(self, user_id: str, subject: str = "") -> Dict:
        """获取学习资源列表"""
        message = Message(
            sender="coordinator",
            receiver="resource",
            msg_type=MessageType.REQUEST,
            content={
                "action": "get_all",
                "user_id": user_id,
                "subject": subject
            }
        )
        response = self.route_message(message)
        return response.content
    
    def get_qa_history(self, user_id: str, limit: int = 50) -> Dict:
        """获取问答历史记录"""
        message = Message(
            sender="coordinator",
            receiver="qa",
            msg_type=MessageType.REQUEST,
            content={
                "action": "get_history",
                "user_id": user_id,
                "limit": limit
            }
        )
        response = self.route_message(message)
        return response.content
    
    def get_next_review(self, user_id: str) -> Dict:
        """获取下一个复习内容"""
        message = Message(
            sender="coordinator",
            receiver="review",
            msg_type=MessageType.REQUEST,
            content={
                "action": "next",
                "user_id": user_id
            }
        )
        response = self.route_message(message)
        return response.content
    
    def get_quiz_history(self, user_id: str, limit: int = 5) -> Dict:
        """获取测验历史记录"""
        message = Message(
            sender="coordinator",
            receiver="quiz",
            msg_type=MessageType.REQUEST,
            content={
                "action": "get_history",
                "user_id": user_id,
                "limit": limit
            }
        )
        response = self.route_message(message)
        return response.content

    def delete_qa_history(self, user_id: str, timestamp: str, question: str) -> Dict:
        """删除问答历史记录"""
        message = Message(
            sender="coordinator",
            receiver="qa",
            msg_type=MessageType.REQUEST,
            content={
                "action": "delete_history",
                "user_id": user_id,
                "timestamp": timestamp,
                "question": question
            }
        )
        response = self.route_message(message)
        return response.content

