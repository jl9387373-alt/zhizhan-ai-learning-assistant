"""
测验Agent
负责生成、管理和分析学习测验
"""

from agents.base_agent import BaseAgent
from core.message import Message, MessageType
from core.storage import DataStorage
from core.logger import logger
from core.bkt import tracer as bkt_tracer
from typing import Dict, Any, List, Optional
from datetime import datetime
import uuid
import json
import os
import re
from difflib import SequenceMatcher
from rag.rag_chain import get_rag_helper


class QuizService(BaseAgent):
    """测验Agent"""
    
    def __init__(self):
        super().__init__("quiz", "测验Agent")
        self.storage = DataStorage()
        # 延迟初始化RAG helper，只在真正需要时才加载
        self._rag_helper = None
        self._rag_initialized = False
        self.difficulty_mapping = {
            "简单": "easy",
            "easy": "easy",
            "初级": "easy",
            "beginner": "easy",
            "中等": "medium",
            "medium": "medium",
            "中级": "medium",
            "intermediate": "medium",
            "困难": "hard",
            "hard": "hard",
            "高级": "hard",
            "advanced": "hard",
        }
    
    @property
    def rag_helper(self):
        if not self._rag_initialized:
            # 安全地初始化RAG helper
            try:
                self._rag_helper = get_rag_helper()
                if self._rag_helper is None:
                    logger.warning("RAG helper初始化失败，将尝试模型生成题目")
                    self._rag_helper = None
            except Exception as e:
                logger.warning("RAG helper初始化失败，将尝试模型生成题目: %s", type(e).__name__)
                self._rag_helper = None
            finally:
                self._rag_initialized = True
        return self._rag_helper
    
    def process(self, message: Message) -> Message:
        """处理测验相关请求"""
        # 确保 action 字段存在
        if "action" not in message.content:
            message.content["action"] = "generate"
            
        # 验证消息
        valid, error_msg = self.validate_message(message, ["action"])
        if not valid:
            return self.send_message(
                receiver=message.sender,
                msg_type=MessageType.ERROR,
                content={"status": "error", "message": error_msg}
            )
        
        content = message.content
        action = content.get("action", "generate")
        user_id = content.get("user_id", "default")
        
        try:
            if action == "generate":
                result = self.generate_quiz(
                    subject=content.get("subject", "通用"),
                    difficulty=content.get("difficulty", "medium"),
                    num_questions=int(content.get("num_questions", 5)),
                    user_id=user_id
                )
            elif action == "submit":
                result = self.submit_quiz(
                    quiz_id=content.get("quiz_id"),
                    answers=content.get("answers", []),
                    user_id=user_id
                )
            elif action == "get":
                result = self.get_quiz(
                    quiz_id=content.get("quiz_id"),
                    user_id=user_id
                )
            elif action == "analyze":
                result = self.analyze_performance(
                    user_id=user_id,
                    subject=content.get("subject")
                )
            elif action in ["history", "get_history"]:
                result = self.get_quiz_history(
                    user_id=user_id,
                    subject=content.get("subject"),
                    limit=content.get("limit", 50)
                )
            else:
                result = {
                    "status": "error",
                    "message": f"未知的操作: {action}"
                }
            
            return self.send_message(
                receiver=message.sender,
                msg_type=MessageType.RESPONSE,
                content=result
            )
        except Exception as e:
            logger.error(f"QuizAgent处理失败: {str(e)}", exc_info=True)
            return self.send_message(
                receiver=message.sender,
                msg_type=MessageType.ERROR,
                content={
                    "status": "error",
                    "message": f"处理测验请求失败: {str(e)}"
                }
            )
    
    def generate_quiz(self, subject: str, difficulty: str = "medium", 
                     num_questions: int = 5, user_id: str = "default") -> Dict[str, Any]:
        """生成测验题目"""
        try:
            # 标准化难度
            difficulty_normalized = self.difficulty_mapping.get(difficulty.lower(), "medium")
            
            # 优先使用模型生成主题相关题；知识库可用时补足有效题目。
            questions = self._generate_questions_with_llm(subject, difficulty_normalized, num_questions)
            questions = [q for q in questions if self._valid_generated_question(q)]
            questions = list({q["question"]: q for q in questions}.values())
            if len(questions) < num_questions and self.rag_helper:
                try:
                    if self.rag_helper.ensure_ready():
                        extra = self._generate_questions_with_rag(subject, difficulty_normalized,
                                                                  num_questions - len(questions))
                        seen = {q["question"] for q in questions}
                        for item in extra:
                            if self._valid_generated_question(item) and item["question"] not in seen:
                                questions.append(item)
                                seen.add(item["question"])
                                if len(questions) >= num_questions:
                                    break
                except Exception as exc:
                    logger.warning("补充知识库题目失败: %s", type(exc).__name__)
            questions = questions[:num_questions]
            if not questions:
                return {"status": "error", "message": "暂时无法生成与该主题相关的测验，请检查模型服务后重试。"}

            # 创建测验记录
            quiz_id = str(uuid.uuid4())
            title = f"{subject} - {difficulty_normalized} - {datetime.now().strftime('%m-%d %H:%M')}"
            quiz_record = {
                "quiz_id": quiz_id,
                "user_id": user_id,
                "subject": subject,
                "title": title,
                "difficulty": difficulty_normalized,
                "num_questions": len(questions),
                "questions": questions,
                "created_at": datetime.now().isoformat(),
                "status": "pending",
                "score": None,
                "answers": None,
                "completed_at": None
            }
            
            # 保存测验（即使保存失败，也返回题目）
            try:
                self.storage.save_quiz_record(user_id, quiz_record)
                logger.info(f"成功生成并保存测验: {quiz_id}, 主题: {subject}, 题目数: {len(questions)}")
            except Exception as save_error:
                logger.warning(f"保存测验记录失败，但题目已生成: {str(save_error)}")
                # 即使保存失败，也返回题目，用户可以继续使用
            
            return {
                "status": "success",
                "quiz_id": quiz_id,
                "subject": subject,
                "difficulty": difficulty_normalized,
                "questions": questions,
                "num_questions": len(questions)
            }
        except Exception as e:
            logger.error(f"生成测验失败: {str(e)}", exc_info=True)
            return {
                "status": "error",
                "message": f"生成测验失败: {str(e)}"
            }
    
    def _generate_questions_with_rag(self, subject: str, difficulty: str, 
                                    num_questions: int) -> List[Dict[str, Any]]:
        """使用RAG系统生成题目"""
        questions = []
        
        # 如果RAG helper不可用，直接返回空列表
        if not self.rag_helper:
            logger.info("RAG helper不可用，仅使用模型生成题目")
            return questions
        
        try:
            # 根据难度生成不同类型的题目
            question_types = self._get_question_types_for_difficulty(difficulty)
            
            for i in range(num_questions):
                try:
                    question_type = question_types[i % len(question_types)]
                    
                    # 构建查询
                    query = self._build_question_query(subject, difficulty, question_type, i + 1)
                    
                    # 使用RAG获取相关内容
                    rag_result = self.rag_helper.ask_with_details(query, subject=subject)
                    
                    if rag_result and rag_result.get("answer"):
                        # 从RAG答案中提取题目
                        question = self._extract_question_from_rag(
                            rag_result.get("answer", ""),
                            subject,
                            difficulty,
                            question_type,
                            i + 1
                        )
                        if question:
                            questions.append(question)
                except Exception as e:
                    logger.warning("生成第%s道题目失败: %s", i + 1, type(e).__name__)
                    # 继续生成下一道题
                    continue
        except Exception as e:
            logger.warning("RAG生成题目失败: %s", type(e).__name__)
        
        return questions
    
    def _generate_questions_with_llm(self, subject: str, difficulty: str,
                                     num_questions: int) -> List[Dict[str, Any]]:
        """用原项目已有模型连接生成主题相关题目。"""
        questions: List[Dict[str, Any]] = []
        if num_questions <= 0:
            return questions
        try:
            from core.config import config
            from langchain_openai import ChatOpenAI
        except Exception as exc:
            logger.warning("模型直出题目依赖不可用: %s", type(exc).__name__)
            return questions

        api_key = str(config.get("rag.api_key", "") or os.getenv("OPENAI_API_KEY") or os.getenv("DASHSCOPE_API_KEY") or "")
        if not api_key:
            return questions
        try:
            llm = ChatOpenAI(
                model=str(config.get("rag.model_name", "gpt-3.5-turbo")),
                openai_api_key=api_key,
                openai_api_base=str(config.get("rag.base_url", "")),
                temperature=0.3,
                max_tokens=800,
                timeout=int(config.get("rag.llm_timeout", 20)),
            )
        except Exception as exc:
            logger.warning("模型直出题目初始化失败: %s", type(exc).__name__)
            return questions

        level = {"easy": "入门", "medium": "中等", "hard": "进阶"}.get(difficulty, "中等")
        question_types = self._get_question_types_for_difficulty(difficulty)

        max_attempts = min(max(num_questions * 3, num_questions), 15)
        for attempt in range(max_attempts):
            if len(questions) >= num_questions:
                break
            question_type = question_types[len(questions) % len(question_types)]
            previous_stems = [q["question"] for q in questions]
            prompt = (
                f"你是出题老师。请围绕「{subject}」出一道{level}难度的单选题。\n"
                f"本次第{len(questions) + 1}/{num_questions}道；请与已出题目考查不同知识点。\n"
                f"已出题目：{json.dumps(previous_stems, ensure_ascii=False)}\n"
                "只输出一个 JSON 对象，字段：\n"
                "- question：题干（一句话）\n"
                "- options：4 个选项的数组，不要带 A/B/C/D 前缀\n"
                "- answer：正确选项的下标（0-3 的整数）\n"
                "- explanation：一句话解析，说明为什么\n"
                "要求：不要编造具体教材页码或出处；不要输出 JSON 以外的任何文字。"
            )
            try:
                resp = llm.invoke(prompt)
                content = getattr(resp, "content", "") or ""
                parsed = self._extract_question_from_rag(
                    content, subject, difficulty, question_type, len(questions) + 1
                )
                if not parsed or not self._valid_generated_question(parsed):
                    continue
                if any(self._questions_too_similar(parsed["question"], old["question"]) for old in questions):
                    logger.info("测验题目与已出题目过于相似，重新生成")
                    continue
                parsed["question_id"] = len(questions) + 1
                parsed.setdefault("type", "multiple_choice")
                questions.append(parsed)
            except Exception as exc:
                logger.warning("模型直出第 %s 次尝试失败: %s", attempt + 1, type(exc).__name__)
                continue
        logger.info(f"模型直出题目 {len(questions)}/{num_questions} 道（{subject}）")
        return questions

    @staticmethod
    def _questions_too_similar(first: str, second: str) -> bool:
        """Reject near-identical stems while keeping distinct questions."""
        normalize = lambda text: re.sub(r"\W+", "", str(text or "").lower())
        left, right = normalize(first), normalize(second)
        if not left or not right:
            return False
        return SequenceMatcher(None, left, right).ratio() >= 0.82

    @staticmethod
    def _valid_generated_question(question: Dict[str, Any]) -> bool:
        text = str(question.get("question") or "").strip()
        options = question.get("options")
        answer = question.get("correct_answer", question.get("answer"))
        if not text or not isinstance(options, list) or len(options) != 4:
            return False
        if not all(isinstance(option, str) and option.strip() for option in options):
            return False
        if len({option.strip() for option in options}) != 4:
            return False
        if isinstance(answer, str):
            label = answer.strip().upper().removesuffix(".").removesuffix("、")
            if label in {"A", "B", "C", "D"}:
                answer = ord(label) - ord("A")
            elif label in {"0", "1", "2", "3"}:
                answer = int(label)
        if isinstance(answer, bool) or not isinstance(answer, int) or not 0 <= answer < 4:
            return False
        if any(str(option).startswith("选项") for option in options):
            return False
        generic = ("是一种重要的概念", "是学习的基础", "是进阶内容", "是高级主题", "以上都是")
        if any(phrase in option for option in options for phrase in generic):
            return False
        question["correct_answer"] = answer
        question.setdefault("type", "multiple_choice")
        return True

    def _generate_questions_from_template(self, subject: str, difficulty: str, 
                                        num_questions: int) -> List[Dict[str, Any]]:
        """从模板生成题目（备用方案）"""
        questions = []
        
        # 基础题目模板（扩展版，确保有足够题目）
        templates = {
            "easy": [
                {
                    "question": f"什么是{subject}？",
                    "type": "multiple_choice",
                    "options": [
                        f"{subject}是一种重要的概念",
                        f"{subject}是学习的基础",
                        f"{subject}是进阶内容",
                        f"{subject}是高级主题"
                    ],
                    "correct_answer": 0,
                    "explanation": f"{subject}是学习的基础概念，需要掌握其基本定义和特点。"
                },
                {
                    "question": f"学习{subject}的第一步是什么？",
                    "type": "multiple_choice",
                    "options": [
                        "理解基本概念",
                        "直接开始实践",
                        "阅读高级资料",
                        "跳过基础知识"
                    ],
                    "correct_answer": 0,
                    "explanation": "学习任何主题都应该从理解基本概念开始。"
                },
                {
                    "question": f"{subject}的主要特点是什么？",
                    "type": "multiple_choice",
                    "options": [
                        "简单易学",
                        "功能强大",
                        "应用广泛",
                        "以上都是"
                    ],
                    "correct_answer": 3,
                    "explanation": f"{subject}具有多个优点，包括简单易学、功能强大和应用广泛。"
                },
                {
                    "question": f"学习{subject}需要什么基础？",
                    "type": "multiple_choice",
                    "options": [
                        "不需要基础",
                        "需要一定的基础知识",
                        "需要高级知识",
                        "需要专业背景"
                    ],
                    "correct_answer": 1,
                    "explanation": f"学习{subject}通常需要一定的基础知识，但不需要太高的起点。"
                },
                {
                    "question": f"{subject}的应用领域包括哪些？",
                    "type": "multiple_choice",
                    "options": [
                        "单一领域",
                        "多个领域",
                        "所有领域",
                        "不适用"
                    ],
                    "correct_answer": 1,
                    "explanation": f"{subject}可以应用于多个不同的领域。"
                }
            ],
            "medium": [
                {
                    "question": f"在{subject}中，最重要的概念是什么？",
                    "type": "multiple_choice",
                    "options": [
                        "核心原理",
                        "细节技巧",
                        "历史背景",
                        "无关内容"
                    ],
                    "correct_answer": 0,
                    "explanation": f"核心原理是理解{subject}的关键。"
                },
                {
                    "question": f"如何更好地掌握{subject}？",
                    "type": "multiple_choice",
                    "options": [
                        "只学习理论",
                        "只进行实践",
                        "理论与实践结合",
                        "不需要学习"
                    ],
                    "correct_answer": 2,
                    "explanation": f"掌握{subject}需要理论与实践相结合。"
                },
                {
                    "question": f"{subject}的核心技能包括什么？",
                    "type": "multiple_choice",
                    "options": [
                        "基础操作",
                        "高级应用",
                        "问题解决",
                        "以上都是"
                    ],
                    "correct_answer": 3,
                    "explanation": f"{subject}的核心技能包括基础操作、高级应用和问题解决能力。"
                },
                {
                    "question": f"学习{subject}时应该注意什么？",
                    "type": "multiple_choice",
                    "options": [
                        "只关注细节",
                        "只关注整体",
                        "兼顾整体和细节",
                        "不需要注意"
                    ],
                    "correct_answer": 2,
                    "explanation": f"学习{subject}时应该兼顾整体框架和具体细节。"
                },
                {
                    "question": f"{subject}的进阶学习方向是什么？",
                    "type": "multiple_choice",
                    "options": [
                        "深入理解原理",
                        "掌握高级技巧",
                        "实际项目应用",
                        "以上都是"
                    ],
                    "correct_answer": 3,
                    "explanation": f"{subject}的进阶学习包括深入理解原理、掌握高级技巧和实际项目应用。"
                }
            ],
            "hard": [
                {
                    "question": f"如何深入理解{subject}的高级应用？",
                    "type": "multiple_choice",
                    "options": [
                        "掌握基础后逐步深入",
                        "直接学习高级内容",
                        "忽略基础知识",
                        "只做表面了解"
                    ],
                    "correct_answer": 0,
                    "explanation": "深入理解需要建立在扎实的基础之上。"
                },
                {
                    "question": f"{subject}的高级特性包括哪些？",
                    "type": "multiple_choice",
                    "options": [
                        "基础功能",
                        "高级功能",
                        "扩展功能",
                        "以上都是"
                    ],
                    "correct_answer": 3,
                    "explanation": f"{subject}的高级特性包括基础功能、高级功能和扩展功能。"
                },
                {
                    "question": f"掌握{subject}的高级应用需要什么？",
                    "type": "multiple_choice",
                    "options": [
                        "扎实的基础",
                        "丰富的经验",
                        "持续的学习",
                        "以上都是"
                    ],
                    "correct_answer": 3,
                    "explanation": f"掌握{subject}的高级应用需要扎实的基础、丰富的经验和持续的学习。"
                },
                {
                    "question": f"{subject}的复杂场景如何处理？",
                    "type": "multiple_choice",
                    "options": [
                        "简化处理",
                        "分步解决",
                        "系统分析",
                        "以上都是"
                    ],
                    "correct_answer": 3,
                    "explanation": f"处理{subject}的复杂场景需要简化处理、分步解决和系统分析。"
                },
                {
                    "question": f"如何优化{subject}的应用效果？",
                    "type": "multiple_choice",
                    "options": [
                        "改进方法",
                        "提升技能",
                        "积累经验",
                        "以上都是"
                    ],
                    "correct_answer": 3,
                    "explanation": f"优化{subject}的应用效果需要改进方法、提升技能和积累经验。"
                }
            ]
        }
        
        template_list = templates.get(difficulty, templates["medium"])
        
        # 确保有足够的模板，如果不够就循环使用
        for i in range(num_questions):
            template = template_list[i % len(template_list)]
            question = dict(template)
            question["question"] = question["question"].format(subject=subject)
            question["explanation"] = question["explanation"].format(subject=subject)
            question["question_id"] = i + 1
            questions.append(question)
        
        logger.info(f"使用模板生成了{len(questions)}道题目")
        return questions
    
    def _get_question_types_for_difficulty(self, difficulty: str) -> List[str]:
        """根据难度获取题目类型"""
        if difficulty == "easy":
            return ["definition", "basic_concept", "simple_application"]
        elif difficulty == "hard":
            return ["advanced_concept", "complex_application", "analysis"]
        else:
            return ["concept", "application", "comparison"]
    
    def _build_question_query(self, subject: str, difficulty: str, 
                            question_type: str, question_num: int) -> str:
        """构建题目查询"""
        type_mapping = {
            "definition": "定义",
            "basic_concept": "基本概念",
            "simple_application": "简单应用",
            "concept": "概念理解",
            "application": "实际应用",
            "comparison": "对比分析",
            "advanced_concept": "高级概念",
            "complex_application": "复杂应用",
            "analysis": "深入分析"
        }
        
        type_cn = type_mapping.get(question_type, "概念")
        difficulty_cn = {"easy": "简单", "medium": "中等", "hard": "困难"}.get(difficulty, "中等")
        
        return f"请生成一道关于{subject}的{difficulty_cn}难度的{type_cn}类选择题，要求有4个选项和正确答案"
    
    def _extract_question_from_rag(self, rag_answer: str, subject: str, 
                                  difficulty: str, question_type: str, 
                                  question_num: int) -> Optional[Dict[str, Any]]:
        """从RAG答案中提取题目"""
        try:
            # 尝试解析RAG返回的结构化内容
            # 如果RAG返回的是JSON格式
            if "{" in rag_answer and "}" in rag_answer:
                try:
                    # 尝试提取JSON
                    start = rag_answer.find("{")
                    end = rag_answer.rfind("}") + 1
                    json_str = rag_answer[start:end]
                    parsed = json.loads(json_str)
                    if "question" in parsed:
                        parsed["question_id"] = question_num
                        return parsed
                except:
                    pass
            
            # 如果RAG返回的是文本格式，尝试解析
            lines = rag_answer.split("\n")
            question_text = ""
            options = []
            correct_answer = None
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                # 识别题目
                if "问题" in line or "题目" in line or "?" in line or "？" in line:
                    question_text = line.replace("问题:", "").replace("题目:", "").strip()
                # 识别选项
                elif line.startswith(("A.", "B.", "C.", "D.", "a.", "b.", "c.", "d.")):
                    option_text = line[2:].strip()
                    options.append(option_text)
                elif line.startswith(("A、", "B、", "C、", "D、", "a、", "b、", "c、", "d、")):
                    option_text = line[2:].strip()
                    options.append(option_text)
                # 识别正确答案
                elif "答案" in line or "正确" in line:
                    if "A" in line or "a" in line:
                        correct_answer = 0
                    elif "B" in line or "b" in line:
                        correct_answer = 1
                    elif "C" in line or "c" in line:
                        correct_answer = 2
                    elif "D" in line or "d" in line:
                        correct_answer = 3
            
            # 缺少选项或正确答案时拒绝保存。
            if question_text and len(options) == 4 and correct_answer is not None:
                
                return {
                    "question_id": question_num,
                    "question": question_text,
                    "type": "multiple_choice",
                    "options": options[:4],
                    "correct_answer": correct_answer,
                    "explanation": f"这是关于{subject}的{difficulty}难度题目。"
                }
        except Exception as e:
            logger.warning(f"解析RAG题目失败: {str(e)}")
        
        return None
    
    def submit_quiz(self, quiz_id: str, answers: List[Any], user_id: str) -> Dict[str, Any]:
        """提交测验答案"""
        try:
            quiz_record = self.storage.get_quiz_by_id(user_id, quiz_id)
            
            if not quiz_record:
                return {
                    "status": "error",
                    "message": "测验不存在"
                }
            
            questions = quiz_record.get("questions", [])
            
            if len(answers) != len(questions):
                return {
                    "status": "error",
                    "message": f"答案数量不匹配，期望{len(questions)}道题，收到{len(answers)}道题"
                }
            
            # 评分
            correct_count = 0
            feedback = []
            
            # BKT (贝叶斯知识追踪) 更新记录
            bkt_updates = {}
            
            for i, (user_answer, question) in enumerate(zip(answers, questions)):
                correct_answer = question.get("correct_answer")
                is_correct = False
                
                # 处理不同类型的答案
                if isinstance(correct_answer, int):
                    is_correct = user_answer == correct_answer
                elif isinstance(correct_answer, str):
                    is_correct = str(user_answer).lower() == str(correct_answer).lower()
                else:
                    is_correct = user_answer == correct_answer
                
                if is_correct:
                    correct_count += 1
                
                # 提取题目的知识点（如果没有则使用测验的主题）
                concept = question.get("concept") or quiz_record.get("subject", "通用")
                
                # 调用 BKT 算法动态更新知识点掌握度
                new_mastery = bkt_tracer.record_interaction(
                    user_id=user_id,
                    concept=concept,
                    is_correct=is_correct,
                    difficulty=quiz_record.get("difficulty", "default")
                )
                bkt_updates[concept] = new_mastery
                
                feedback.append({
                    "question_index": i,
                    "question_id": question.get("question_id", i + 1),
                    "is_correct": is_correct,
                    "user_answer": user_answer,
                    "correct_answer": correct_answer,
                    "explanation": question.get("explanation", "")
                })
            
            score = (correct_count / len(questions)) * 100 if questions else 0
            
            # 更新测验记录
            quiz_record["answers"] = answers
            quiz_record["score"] = round(score, 2)
            quiz_record["feedback"] = feedback
            quiz_record["completed_at"] = datetime.now().isoformat()
            quiz_record["status"] = "completed"
            
            self.storage.save_quiz_record(user_id, quiz_record)
            
            logger.info(f"测验提交成功: {quiz_id}, 得分: {score}")
            
            return {
                "status": "success",
                "quiz_id": quiz_id,
                "score": round(score, 2),
                "correct_count": correct_count,
                "total_questions": len(questions),
                "feedback": feedback
            }
        except Exception as e:
            logger.error(f"提交测验失败: {str(e)}", exc_info=True)
            return {
                "status": "error",
                "message": f"提交测验失败: {str(e)}"
            }
    
    def get_quiz(self, quiz_id: str, user_id: str) -> Dict[str, Any]:
        """获取测验详情"""
        try:
            quiz_record = self.storage.get_quiz_by_id(user_id, quiz_id)
            
            if not quiz_record:
                return {
                    "status": "error",
                    "message": "测验不存在"
                }
            
            # 如果测验已完成，返回完整信息（包括答案）
            # 如果未完成，只返回题目（不包含答案）
            result = {
                "status": "success",
                "quiz_id": quiz_id,
                "subject": quiz_record.get("subject"),
                "difficulty": quiz_record.get("difficulty"),
                "num_questions": quiz_record.get("num_questions"),
                "created_at": quiz_record.get("created_at"),
                "status": quiz_record.get("status")
            }
            
            if quiz_record.get("status") == "completed":
                result["score"] = quiz_record.get("score")
                result["feedback"] = quiz_record.get("feedback")
            
            # 返回题目（不包含正确答案，除非已完成）
            questions = []
            for q in quiz_record.get("questions", []):
                question_data = {
                    "question_id": q.get("question_id"),
                    "question": q.get("question"),
                    "type": q.get("type"),
                    "options": q.get("options", [])
                }
                # 只有已完成时才包含正确答案
                if quiz_record.get("status") == "completed":
                    question_data["correct_answer"] = q.get("correct_answer")
                    question_data["explanation"] = q.get("explanation")
                questions.append(question_data)
            
            result["questions"] = questions
            
            return result
        except Exception as e:
            logger.error(f"获取测验失败: {str(e)}", exc_info=True)
            return {
                "status": "error",
                "message": f"获取测验失败: {str(e)}"
            }
    
    def get_quiz_history(self, user_id: str, subject: Optional[str] = None, limit: int = 50) -> Dict[str, Any]:
        """获取测验历史"""
        try:
            if subject:
                history = self.storage.get_subject_quiz_history(user_id, subject, limit)
            else:
                history = self.storage.get_quiz_history(user_id, limit)
            
            return {
                "status": "success",
                "history": history,
                "count": len(history)
            }
        except Exception as e:
            logger.error(f"获取测验历史失败: {str(e)}", exc_info=True)
            return {
                "status": "error",
                "message": f"获取测验历史失败: {str(e)}"
            }
    
    def analyze_performance(self, user_id: str, subject: Optional[str] = None) -> Dict[str, Any]:
        """分析测验表现"""
        try:
            # 获取测验历史
            if subject:
                history = self.storage.get_subject_quiz_history(user_id, subject, limit=100)
            else:
                history = self.storage.get_quiz_history(user_id, limit=100)
            
            if not history:
                return {
                    "status": "success",
                    "message": "暂无测验记录",
                    "statistics": {}
                }
            
            # 计算统计信息
            total_quizzes = len(history)
            completed_quizzes = [q for q in history if q.get("status") == "completed"]
            total_questions = sum(q.get("num_questions", 0) for q in completed_quizzes)
            total_correct = 0
            
            scores = []
            for quiz in completed_quizzes:
                score = quiz.get("score", 0)
                if score is not None:
                    scores.append(score)
                    # 估算正确题目数
                    num_q = quiz.get("num_questions", 0)
                    if num_q > 0:
                        total_correct += int((score / 100) * num_q)
            
            avg_score = sum(scores) / len(scores) if scores else 0
            
            # 按主题统计
            subject_stats = {}
            for quiz in completed_quizzes:
                subj = quiz.get("subject", "未知")
                if subj not in subject_stats:
                    subject_stats[subj] = {"count": 0, "scores": []}
                subject_stats[subj]["count"] += 1
                score = quiz.get("score", 0)
                if score is not None:
                    subject_stats[subj]["scores"].append(score)
            
            for subj in subject_stats:
                scores_list = subject_stats[subj]["scores"]
                subject_stats[subj]["avg_score"] = sum(scores_list) / len(scores_list) if scores_list else 0
            
            return {
                "status": "success",
                "statistics": {
                    "total_quizzes": total_quizzes,
                    "completed_quizzes": len(completed_quizzes),
                    "total_questions": total_questions,
                    "total_correct": total_correct,
                    "accuracy_rate": round((total_correct / total_questions * 100) if total_questions > 0 else 0, 2),
                    "average_score": round(avg_score, 2),
                    "subject_statistics": subject_stats
                }
            }
        except Exception as e:
            logger.error(f"分析测验表现失败: {str(e)}", exc_info=True)
            return {
                "status": "error",
                "message": f"分析测验表现失败: {str(e)}"
            }







