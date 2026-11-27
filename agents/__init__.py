# Agents module for Multi-Agent Learning Assistant System

from agents.base_agent import BaseAgent
from agents.qa_agent import QAAgent
from agents.service_agent import ServiceAgent

# Service 模块（内部组件，不再是独立 Agent）
from agents.planner_service import PlannerService
from agents.resource_service import ResourceService
from agents.progress_service import ProgressService

# 尝试导入 QuizService（如果存在）
try:
    from agents.quiz_service import QuizService
    QUIZ_SERVICE_AVAILABLE = True
except ImportError:
    QuizService = None
    QUIZ_SERVICE_AVAILABLE = False

# 尝试导入 ReviewService（如果存在）
try:
    from agents.review_service import ReviewService
    REVIEW_SERVICE_AVAILABLE = True
except ImportError:
    ReviewService = None
    REVIEW_SERVICE_AVAILABLE = False

# 构建 __all__ 列表
__all__ = [
    'BaseAgent',
    'QAAgent',
    'ServiceAgent',
    'PlannerService',
    'ResourceService',
    'ProgressService',
]

if QUIZ_SERVICE_AVAILABLE:
    __all__.append('QuizService')

if REVIEW_SERVICE_AVAILABLE:
    __all__.append('ReviewService')
