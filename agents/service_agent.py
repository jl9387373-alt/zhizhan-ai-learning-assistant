"""
Unified service agent.
Routes plan/resource/progress/quiz/review capabilities behind one entrypoint.
支持多智能体协作：接收 QAAgent 的通知并响应
"""

from typing import Any, Dict, Optional, List

from agents.base_agent import BaseAgent
from agents.message_protocol import MessageAction, MessagePriority, AgentMessage
from agents.planner_service import PlannerService
from agents.progress_service import ProgressService
from agents.resource_service import ResourceService
from core.logger import logger
from core.message import Message, MessageType

try:
    from agents.quiz_service import QuizService
except ImportError:
    QuizService = None

try:
    from agents.review_service import ReviewService
except ImportError:
    ReviewService = None


class ServiceAgent(BaseAgent):
    """Single gateway for non-QA capabilities."""

    CAPABILITY_ALIASES = {
        "service": "service",
        "planner": "planner",
        "plan": "planner",
        "resource": "resource",
        "resources": "resource",
        "progress": "progress",
        "quiz": "quiz",
        "review": "review",
    }

    ACTION_CAPABILITY_MAP = {
        "create": "planner",
        "timeline": "planner",
        "adjust": "planner",
        "recommend": "resource",
        "dashboard": "progress",
        "report": "progress",
        "generate": "quiz",
        "submit": "quiz",
        "schedule": "review",
        "next": "review",
        "complete": "review",
    }

    def __init__(self):
        super().__init__("service", "Unified Service Agent")
        self._components: Dict[str, BaseAgent] = {}
        self._factories = {
            "planner": PlannerService,
            "resource": ResourceService,
            "progress": ProgressService,
        }
        if QuizService is not None:
            self._factories["quiz"] = QuizService
        if ReviewService is not None:
            self._factories["review"] = ReviewService

    @property
    def capabilities(self):
        return sorted(self._factories.keys())

    def handles(self, capability: str) -> bool:
        capability = self.CAPABILITY_ALIASES.get(capability, capability)
        return capability in self._factories

    def _get_component(self, capability: str) -> Optional[BaseAgent]:
        capability = self.CAPABILITY_ALIASES.get(capability, capability)
        if capability not in self._factories:
            return None
        if capability not in self._components:
            try:
                self._components[capability] = self._factories[capability]()
                logger.info(f"ServiceAgent loaded capability: {capability}")
            except Exception as e:
                logger.warning(f"ServiceAgent failed to load capability {capability}: {e}")
                return None
        return self._components.get(capability)

    def _resolve_capability(self, message: Message) -> Optional[str]:
        content = message.content or {}

        for key in ("service", "capability", "module"):
            explicit = content.get(key)
            if explicit:
                resolved = self.CAPABILITY_ALIASES.get(explicit, explicit)
                if resolved != "service":
                    return resolved

        receiver = self.CAPABILITY_ALIASES.get(message.receiver, message.receiver)
        if receiver != "service":
            return receiver

        action = content.get("action")
        if action in self.ACTION_CAPABILITY_MAP:
            return self.ACTION_CAPABILITY_MAP[action]

        if "progress_data" in content:
            return "progress"
        if "difficulty" in content and "num_questions" in content:
            return "quiz"
        if "importance" in content or "review_id" in content:
            return "review"
        if "plan_data" in content or "duration" in content:
            return "planner"
        if "count" in content or "type" in content:
            return "resource"

        return None

    def process(self, message: Message) -> Message:
        capability = self._resolve_capability(message)
        if not capability:
            return self.send_message(
                receiver=message.sender,
                msg_type=MessageType.ERROR,
                content={
                    "status": "error",
                    "message": "Unable to determine service capability"
                }
            )

        component = self._get_component(capability)
        if component is None:
            return self.send_message(
                receiver=message.sender,
                msg_type=MessageType.ERROR,
                content={
                    "status": "error",
                    "message": f"Capability unavailable: {capability}"
                }
            )

        proxy_content = dict(message.content or {})
        proxy_content.pop("service", None)
        proxy_content.pop("capability", None)
        proxy_content.pop("module", None)
        proxy_message = Message(
            sender=message.sender,
            receiver=capability,
            msg_type=message.msg_type,
            content=proxy_content,
            timestamp=message.timestamp,
        )
        return component.process(proxy_message)

    def _invoke(self, capability: str, method_name: str, fallback: Any, *args, **kwargs):
        component = self._get_component(capability)
        if component is None or not hasattr(component, method_name):
            return fallback
        return getattr(component, method_name)(*args, **kwargs)

    def create_learning_plan(self, subject: str, user_id: str, duration: str = "4周",
                             constraints: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return self._invoke("planner", "create_learning_plan", {}, subject, user_id, duration, constraints)

    def get_learning_plan(self, user_id: str, subject: str) -> Dict[str, Any]:
        return self._invoke("planner", "get_learning_plan", {}, user_id, subject)

    def update_learning_plan(self, user_id: str, subject: str, plan_data: Dict[str, Any]) -> Dict[str, Any]:
        return self._invoke("planner", "update_learning_plan", {}, user_id, subject, plan_data)

    def delete_learning_plan(self, user_id: str, subject: str) -> bool:
        return self._invoke("planner", "delete_learning_plan", False, user_id, subject)

    def get_all_plans(self, user_id: str) -> Dict[str, Any]:
        return self._invoke("planner", "get_all_plans", {}, user_id)

    def recommend_resources(
        self,
        subject: str,
        difficulty: str = "all",
        resource_type: str = "all",
        count: int = 5,
        user_id: str = "default",
        personalized: bool = True,
    ):
        return self._invoke(
            "resource",
            "recommend_resources",
            [],
            subject,
            difficulty,
            resource_type,
            count,
            user_id,
            personalized,
        )

    def get_progress(self, user_id: str, subject: str) -> Dict[str, Any]:
        return self._invoke("progress", "get_progress", {}, user_id, subject)

    def update_progress(self, user_id: str, subject: str, progress_data: Dict[str, Any]) -> Dict[str, Any]:
        return self._invoke("progress", "update_progress", {}, user_id, subject, progress_data)

    def get_all_progress(self, user_id: str) -> Dict[str, Any]:
        return self._invoke("progress", "get_all_progress", {}, user_id)

    def generate_quiz(
        self,
        subject: str,
        difficulty: str = "medium",
        num_questions: int = 5,
        user_id: str = "default",
    ) -> Dict[str, Any]:
        return self._invoke(
            "quiz",
            "generate_quiz",
            {"status": "error", "message": "Quiz capability unavailable"},
            subject,
            difficulty,
            num_questions,
            user_id,
        )

    def submit_quiz(self, quiz_id: str, answers, user_id: str = "default") -> Dict[str, Any]:
        return self._invoke(
            "quiz",
            "submit_quiz",
            {"status": "error", "message": "Quiz capability unavailable"},
            quiz_id,
            answers,
            user_id,
        )

    def get_quiz(self, quiz_id: str, user_id: str = "default") -> Dict[str, Any]:
        return self._invoke(
            "quiz",
            "get_quiz",
            {"status": "error", "message": "Quiz capability unavailable"},
            quiz_id,
            user_id,
        )

    def get_quiz_history(self, user_id: str, subject: Optional[str] = None, limit: int = 50) -> Dict[str, Any]:
        return self._invoke(
            "quiz",
            "get_quiz_history",
            {"status": "error", "message": "Quiz capability unavailable"},
            user_id,
            subject,
            limit,
        )

    def schedule_review(
        self,
        user_id: str,
        subject: str,
        content_text: str = "",
        importance: str = "medium",
        first_review_at=None,
    ) -> Dict[str, Any]:
        return self._invoke(
            "review",
            "schedule_review",
            {"status": "error", "message": "Review capability unavailable"},
            user_id,
            subject,
            content_text,
            importance,
            first_review_at,
        )

    def reschedule_review(
        self,
        user_id: str,
        review_id: str,
        next_review_at=None,
        postpone_days=None,
    ) -> Dict[str, Any]:
        return self._invoke(
            "review",
            "reschedule_review",
            {"status": "error", "message": "Review capability unavailable"},
            user_id,
            review_id,
            next_review_at,
            postpone_days,
        )

    def get_next_review(self, user_id: str) -> Dict[str, Any]:
        return self._invoke(
            "review",
            "get_next_review",
            {"status": "error", "message": "Review capability unavailable"},
            user_id,
        )

    def complete_review(self, user_id: str, review_id: str, performance: Dict[str, Any]) -> Dict[str, Any]:
        return self._invoke(
            "review",
            "complete_review",
            {"status": "error", "message": "Review capability unavailable"},
            user_id,
            review_id,
            performance,
        )

    def get_review(self, user_id: str, review_id: str) -> Dict[str, Any]:
        return self._invoke(
            "review",
            "get_review",
            {"status": "error", "message": "Review capability unavailable"},
            user_id,
            review_id,
        )

    def get_review_history(self, user_id: str, limit: int = 50) -> Dict[str, Any]:
        return self._invoke(
            "review",
            "get_review_history",
            {"status": "error", "message": "Review capability unavailable"},
            user_id,
            limit,
        )

    def delete_review(self, user_id: str, review_id: str) -> Dict[str, Any]:
        return self._invoke(
            "review",
            "delete_review",
            {"status": "error", "message": "Review capability unavailable"},
            user_id,
            review_id,
        )
    
    # ========== 多智能体协作方法 ==========
    
    def _handle_notify(self, msg: AgentMessage):
        """处理来自其他 Agent 的通知消息"""
        content = msg.content
        msg_type = content.get("type", "")
        
        if msg_type == "learning_progress":
            # QAAgent 通知：用户在学习新知识
            self._handle_learning_progress_notification(content)
        elif msg_type == "create_review":
            # QAAgent 通知：创建复习计划
            self._handle_create_review_request(content)
    
    def _handle_request(self, msg: AgentMessage):
        """处理来自其他 Agent 的请求消息"""
        content = msg.content
        msg_type = content.get("type", "")
        
        if msg_type == "create_review":
            # QAAgent 请求创建复习计划
            self._handle_create_review_request(content)
    
    def _handle_learning_progress_notification(self, content: Dict[str, Any]):
        """
        处理 QAAgent 的学习进度通知
        自动检测并更新用户的学习进度
        """
        question = content.get("question", "")
        subject = content.get("subject", "通用")
        user_id = content.get("user_id", "default")
        
        # 简单实现：检测学习行为关键词
        learning_keywords = ["什么是", "解释", "学习", "了解", "概念"]
        if any(kw in question for kw in learning_keywords):
            logger.info(f"协作：检测到学习行为 -> {subject} / {user_id}")
            
            # 可选：自动创建学习记录
            # self._invoke("progress", "update_progress", {}, user_id, subject, {"auto_detected": True})
    
    def _handle_create_review_request(self, content: Dict[str, Any]):
        """
        处理 QAAgent 的复习计划创建请求
        根据问答内容自动创建复习计划
        """
        question = content.get("question", "")
        answer = content.get("answer", "")
        subject = content.get("subject", "通用")
        user_id = content.get("user_id", "default")
        key_concepts = content.get("key_concepts", [])
        
        logger.info(f"协作：创建自动复习计划 -> {subject} / {user_id}")
        
        # 使用现有能力创建复习计划
        try:
            # 优先使用 ScheduleAgent
            review_agent = self._get_component("review")
            if review_agent:
                result = review_agent.schedule_review(
                    user_id=user_id,
                    subject=subject,
                    content_text=f"Q: {question}\nA: {answer[:200]}...",
                    importance="medium"
                )
                logger.info(f"协作复习计划创建结果: {result.get('status', 'unknown')}")
            else:
                # 回退：使用 ServiceAgent 内置方法
                result = self.schedule_review(
                    user_id=user_id,
                    subject=subject,
                    content_text=f"Q: {question}\nA: {answer[:200]}...",
                    importance="medium"
                )
                logger.info(f"协作复习计划创建结果: {result.get('status', 'unknown')}")
        except Exception as e:
            logger.warning(f"协作复习计划创建失败: {e}")
    
    def receive_collaboration_messages(self) -> List[AgentMessage]:
        """
        接收并处理所有待处理的协作消息
        由 Coordinator 定期调用
        """
        messages = self.receive_messages()
        processed = self.process_inbox()
        return processed
