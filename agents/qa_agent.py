"""
答疑Agent
负责回答用户的学习问题 - 支持多智能体协作
"""

from agents.base_agent import BaseAgent
from agents.message_protocol import MessageAction, MessagePriority, AgentMessage
from core.message import Message, MessageType
from core.storage import DataStorage
from core.logger import logger
from typing import Dict, Any, Tuple, List, Optional
import re
import uuid

from storage.analytics import track_event
from rag.rag_chain import get_rag_helper
from memory.chat_memory import get_conversation_memory

# 初始化对话记忆
conversation_memory = get_conversation_memory()

SOURCE_LABELS = {
    "knowledge_base": "直接知识解答",
    "knowledge_base_cross": "跨主题知识整合",
    "rag_llm": "引导式学习回答",
    "rag_hybrid": "资料结合模型知识",
    "general_llm": "模型知识回答",
    "rag_context": "知识要点提取",
    "multi_hop": "深入理解回答",
    "business_data": "学习记录数据",
    "default": "学习建议"
}

class QAAgent(BaseAgent):
    """答疑Agent"""
    
    def __init__(self):
        super().__init__("qa", "引导式学习助手")
        self.storage = DataStorage()
        self.knowledge_base = self._load_knowledge_base()
    
    def _load_knowledge_base(self) -> Dict[str, Dict[str, str]]:
        """加载知识库"""
        return {
            "python": {
                "什么是Python": "Python是一种高级、解释型的编程语言，由Guido van Rossum在1991年创建。Python以简洁、可读性强而著称，支持多种编程范式（面向对象、函数式、过程式）。它广泛应用于Web开发、数据分析、人工智能、科学计算等领域。",
                "如何安装Python": "安装Python有多种方式：1) 从python.org官网下载安装包；2) 使用包管理器（Windows用chocolatey，Mac用homebrew，Linux用apt/yum）；3) 使用Anaconda发行版（适合数据科学）。安装后可通过命令行输入'python --version'验证。",
                "Python的基本语法": "Python使用缩进来表示代码块（通常4个空格），不需要大括号。变量不需要声明类型，是动态类型语言。使用#表示单行注释，'''或\"\"\"表示多行注释。语句以换行符结束，不需要分号。",
                "Python的数据类型": "Python的主要数据类型包括：数字（int, float, complex）、字符串（str）、列表（list）、元组（tuple）、字典（dict）、集合（set）、布尔值（bool）。所有类型都是对象，支持动态类型转换。",
                "Python的函数": "Python使用def关键字定义函数。函数可以接受位置参数、关键字参数、默认参数和可变参数（*args, **kwargs）。函数是一等对象，可以作为参数传递。支持lambda表达式创建匿名函数。",
                "Python的面向对象": "Python支持面向对象编程，使用class关键字定义类。支持继承、多态、封装。所有类都继承自object。可以使用@property装饰器创建属性。支持特殊方法（如__init__, __str__）来实现运算符重载。",
                "Python的模块和包": "模块是包含Python代码的文件，使用import导入。包是包含__init__.py的目录，用于组织模块。标准库提供了丰富的模块（如os, sys, json, datetime等）。可以使用pip安装第三方包。",
                "Python的异常处理": "Python使用try-except-finally处理异常。可以捕获特定异常类型，使用raise抛出异常。异常是对象，可以包含错误信息。建议使用具体的异常类型而不是通用的Exception。",
                "Python的文件操作": "使用open()函数打开文件，支持多种模式（r, w, a, rb, wb等）。使用with语句可以自动关闭文件。可以使用read(), readline(), readlines()读取，write(), writelines()写入。",
                "Python的列表推导式": "列表推导式是Python的语法糖，用于快速创建列表。格式为[表达式 for 变量 in 可迭代对象 if 条件]。比传统循环更简洁高效。",
                "如何学习Python": "学习Python建议：1) 从基础语法开始；2) 多做练习和项目；3) 阅读优秀代码；4) 参与开源项目；5) 使用IDE（如PyCharm, VSCode）提高效率；6) 参考官方文档和教程。",
                "Python的虚拟环境": "虚拟环境用于隔离项目依赖，使用venv或virtualenv创建。激活后安装的包只在该环境中。使用'python -m venv env'创建，'source env/bin/activate'（Linux/Mac）或'env\\Scripts\\activate'（Windows）激活。",
                "Python的装饰器": "装饰器是修改函数或类行为的函数，使用@符号应用。本质是函数包装器，可以添加功能（如日志、缓存、权限检查）而不修改原函数。可以嵌套使用多个装饰器。",
                "Python的生成器": "生成器是使用yield关键字的函数，返回迭代器。可以按需生成值，节省内存。适合处理大量数据或无限序列。可以使用生成器表达式创建。",
            },
            "机器学习": {
                "什么是机器学习": "机器学习是人工智能的一个分支，通过算法让计算机从数据中自动学习规律和模式，无需显式编程。主要分为监督学习、无监督学习和强化学习三大类。",
                "监督学习": "监督学习使用标记的训练数据（输入-输出对）来训练模型。目标是学习从输入到输出的映射函数。常见算法包括线性回归、逻辑回归、决策树、随机森林、支持向量机、神经网络等。",
                "无监督学习": "无监督学习从未标记的数据中发现隐藏的模式和结构。主要包括聚类（如K-means）、降维（如PCA）、关联规则挖掘等。常用于数据探索和预处理。",
                "深度学习": "深度学习是使用多层神经网络进行学习的机器学习方法。包括卷积神经网络（CNN）用于图像处理，循环神经网络（RNN/LSTM）用于序列数据，Transformer用于自然语言处理。",
                "如何开始学习机器学习": "学习机器学习建议：1) 掌握数学基础（线性代数、概率论、微积分）；2) 学习Python和数据处理库（NumPy, Pandas）；3) 学习机器学习库（Scikit-learn）；4) 学习深度学习框架（TensorFlow, PyTorch）；5) 完成实际项目。",
                "机器学习的评估指标": "分类问题常用准确率、精确率、召回率、F1分数、ROC曲线。回归问题常用均方误差（MSE）、平均绝对误差（MAE）、R²分数。需要根据问题类型选择合适的指标。",
                "过拟合和欠拟合": "过拟合是模型在训练集上表现好但测试集上表现差，通常因为模型过于复杂。欠拟合是模型无法捕捉数据规律，通常因为模型过于简单。可以通过交叉验证、正则化、增加数据等方法解决。",
                "特征工程": "特征工程是选择和转换特征以提高模型性能的过程。包括特征选择、特征变换、特征创建、缺失值处理、异常值处理等。好的特征工程往往比算法选择更重要。",
            },
            "智能体": {
                "什么是智能体（Agent）": "智能体（Agent）是能够感知环境、自主决策并采取行动以实现目标的系统。它可以是软件（如自动客服、自动驾驶决策模块），也可以是具身实体（如机器人）。与普通程序最大的区别是：智能体需要根据环境反馈调整行动，而不只是执行固定流程。",
                "如何学习智能体": "学习智能体建议分四步：1) 理解核心概念：感知、决策、行动、反馈；2) 了解常见范式：规则式、反应式、规划式、多智能体协作；3) 动手实践：用现成框架做一个能调用工具的小智能体；4) 进阶：学习记忆、规划、工具调用与评估方法。",
            },
            "java": {
                "什么是Java": "Java是一种面向对象的编程语言，由Sun Microsystems（现Oracle）在1995年发布。Java具有'一次编写，到处运行'的特点，通过JVM实现跨平台。广泛应用于企业级应用、Android开发、Web后端等领域。",
                "Java的基本语法": "Java是强类型语言，变量需要声明类型。使用大括号{}表示代码块。语句以分号结束。使用//表示单行注释，/* */表示多行注释。所有代码必须在类中。",
                "Java的面向对象": "Java是纯面向对象语言，所有代码都在类中。支持类、继承、接口、抽象类、多态等特性。使用public, private, protected控制访问权限。",
                "Java的集合框架": "Java集合框架包括List（ArrayList, LinkedList）、Set（HashSet, TreeSet）、Map（HashMap, TreeMap）等接口和实现类。提供了丰富的集合操作功能。",
                "Java的异常处理": "Java使用try-catch-finally处理异常。异常分为检查异常（Checked）和运行时异常（Runtime）。可以使用throws声明可能抛出的异常，throw抛出异常。",
                "Java的多线程": "Java支持多线程编程，可以通过继承Thread类或实现Runnable接口创建线程。使用synchronized实现同步，使用wait/notify实现线程通信。Java 5+提供了并发包（java.util.concurrent）。",
            },
            "操作系统": {
                "什么是进程": "进程（Process）是操作系统进行资源分配与调度的基本单位。你可以把它理解为：一个“正在运行的程序”的实例，它拥有独立的地址空间（代码、数据、堆、栈等）以及运行所需的系统资源（打开的文件、网络连接等）。进程之间彼此隔离，操作系统通过进程来管理程序的执行。",
                "进程和线程的区别": "进程是资源分配的基本单位，线程（Thread）是CPU调度的基本单位。一个进程可以包含多个线程；同一进程内的线程共享进程的内存空间和资源（如堆、全局变量、打开的文件），但每个线程有自己的栈和寄存器上下文。进程切换开销通常更大，线程切换更轻量，但线程共享资源也更容易出现并发问题。",
                "什么是线程": "线程是进程内的执行单元，是CPU调度的基本单位。线程共享进程的大部分资源（地址空间、打开文件等），但拥有独立的执行上下文（栈、寄存器等）。多线程可以让一个程序并发执行多个任务，提高响应性或吞吐量。",
                "什么是PCB": "PCB（Process Control Block，进程控制块）是操作系统用来描述和管理进程的数据结构，通常包含进程状态、程序计数器、寄存器、内存管理信息、调度信息、打开文件列表等。操作系统依靠PCB在进程切换时保存/恢复现场，并进行调度管理。"
            },
            "通用": {
                "如何开始学习": "开始学习建议：1) 明确学习目标和动机；2) 制定详细的学习计划；3) 选择合适的学习资源（书籍、教程、课程）；4) 动手实践，多做项目；5) 加入学习社区，寻求帮助；6) 定期复习和总结。",
                "如何保持学习动力": "保持学习动力方法：1) 设定小目标和里程碑；2) 记录学习进度和成就；3) 加入学习社区，寻找学习伙伴；4) 将学习与兴趣结合；5) 定期休息，避免 burnout；6) 庆祝小成就。",
                "如何提高学习效率": "提高学习效率：1) 使用番茄工作法；2) 主动学习而非被动阅读；3) 费曼学习法（教别人）；4) 间隔重复记忆；5) 做笔记和总结；6) 实践和项目驱动；7) 找到最佳学习时间。",
                "如何选择学习资源": "选择学习资源考虑：1) 资源质量（作者、评价、更新频率）；2) 适合自己水平（入门/进阶）；3) 学习方式（视频/书籍/实践）；4) 社区支持；5) 价格和可获取性；6) 结合多种资源学习。",
                "遇到问题怎么办": "遇到问题解决步骤：1) 仔细阅读错误信息；2) 查阅官方文档；3) 搜索相关问题（Stack Overflow, GitHub Issues）；4) 调试和日志；5) 向社区求助；6) 分解问题，逐步解决。",
                "如何做学习笔记": "做笔记建议：1) 使用结构化方法（如康奈尔笔记法）；2) 记录关键概念和例子；3) 用自己的话总结；4) 添加代码示例和图表；5) 定期复习和更新；6) 使用数字工具（如Markdown, Notion）。",
                "如何准备面试": "准备面试：1) 复习基础知识；2) 练习编程题（LeetCode等）；3) 准备项目介绍；4) 了解公司和职位；5) 准备常见问题；6) 模拟面试；7) 保持自信和积极态度。",
            }
        }
    
    def process(self, message: Message) -> Message:
        """处理问答请求"""
        try:
            content = message.content
            if "action" not in content:
                content["action"] = "generate"
            action = content["action"]
            
            # 验证消息，get_history 和 delete_history 操作不需要 question 字段
            required_fields = ["action"]
            if action not in ["get_history", "delete_history"]:
                required_fields.append("question")
            
            valid, error_msg = self.validate_message(message, required_fields)
            if not valid:
                return self.send_message(
                    receiver=message.sender,
                    msg_type=MessageType.ERROR,
                    content={"status": "error", "message": error_msg}
                )
            
            # 处理 get_history 操作
            if action == "get_history":
                user_id = content.get("user_id", "default")
                limit = content.get("limit", 50)
                history = self.storage.get_qa_history(user_id, limit)
                return self.send_message(
                    receiver=message.sender,
                    msg_type=MessageType.RESPONSE,
                    content={"status": "success", "data": history}
                )
            # 处理 delete_history 操作
            elif action == "delete_history":
                user_id = content.get("user_id", "default")
                timestamp = content.get("timestamp")
                question = content.get("question")
                
                # 验证必要字段
                if not timestamp:
                    return self.send_message(
                        receiver=message.sender,
                        msg_type=MessageType.ERROR,
                        content={"status": "error", "message": "缺少必需字段: timestamp"}
                    )
                
                # 执行删除操作
                success = self.storage.delete_qa_history(user_id, timestamp, question)
                if success:
                    return self.send_message(
                        receiver=message.sender,
                        msg_type=MessageType.RESPONSE,
                        content={"status": "success", "message": "删除成功"}
                    )
                else:
                    return self.send_message(
                        receiver=message.sender,
                        msg_type=MessageType.ERROR,
                        content={"status": "error", "message": "删除失败，未找到匹配的记录"}
                    )
            
            question = content.get("question", "").strip()
            
            # 验证问题不为空
            if not question:
                return self.send_message(
                    receiver=message.sender,
                    msg_type=MessageType.ERROR,
                    content={"status": "error", "message": "问题不能为空"}
                )
            
            subject = content.get("subject", "通用")
            user_id = content.get("user_id", "default")
            qa_id = uuid.uuid4().hex[:16]
            track_event("qa_asked", user_id, {
                "qa_id": qa_id,
                "subject": subject,
                "action": action,
                "question_length": len(question),
            })

            # 2. 每轮对话：只使用短期记忆（最近N轮）
            chat_history = self._conversation_history(user_id, content.get("conversation_id"))
            
            # 省略式追问先补全主题（"怎么学" → "如何学习智能体"），再结合对话历史
            contextual_question = self._expand_question_with_subject(question, subject)
            enhanced_question = self._enhance_question_with_context(contextual_question, chat_history, subject)
            
            result = self.answer_question(enhanced_question, subject, chat_history)
            answer = result.get("answer", "")
            # 学习路径类回答给出下一步：主动询问是否要转成学习计划
            answer = self._with_plan_offer(question, answer, subject)
            confidence = result.get("confidence", 0.2)
            source = result.get("source", "default")

            if answer:
                self.storage.save_qa_history(
                    user_id,
                    question,
                    answer,
                    subject,
                    metadata={
                        "confidence": confidence,
                        "source": source,
                        "source_label": SOURCE_LABELS.get(source, "通用建议")
                    }
                )
                conversation_memory.append(user_id, question, answer)
                
                # ========== 多智能体协作：主动通知其他 Agent ==========
                # 通知 ServiceAgent 更新学习进度（检测到用户在学习新知识）
                if confidence >= 0.6:
                    self._notify_progress_update(question, subject, user_id)
                
                # 通知 ServiceAgent 创建复习计划（高质量回答值得复习）
                if confidence >= 0.7:
                    self._notify_create_review(question, answer, subject, user_id)
                # ========== 协作通知结束 ==========

            # 评估答案质量
            quality_assessment = self._evaluate_answer_quality(answer, question, confidence) if answer else {
                "quality_score": 0.0,
                "issues": ["无答案"],
                "suggestions": ["请尝试重新提问或检查知识库"]
            }
            track_event("qa_answered", user_id, {
                "qa_id": qa_id,
                "subject": subject,
                "source": source,
                "confidence": confidence,
                "has_answer": bool(answer),
                "quality_score": quality_assessment.get("quality_score"),
            })

            response_content = {
                "status": "success",
                "question": question,
                "answer": answer,
                "confidence": confidence,
                "subject": subject,
                "source": source,
                "source_label": SOURCE_LABELS.get(source, "通用建议"),
                "trace": result.get("trace", []),
                "context_preview": result.get("context_preview", ""),
                "context_snippets": result.get("context_snippets", []),
                "quality": quality_assessment
            }

            return self.send_message(
                receiver=message.sender,
                msg_type=MessageType.RESPONSE,
                content=response_content
            )
            
        except Exception as e:
            logger.error(f"QA Agent 处理请求失败: {str(e)}", exc_info=True)
            return self.send_message(
                receiver=message.sender,
                msg_type=MessageType.ERROR,
                content={
                    "status": "error", 
                    "message": f"抱歉，处理您的问题时遇到系统错误: {str(e)}",
                    "answer": f"系统暂时无法回答您的问题，请稍后重试。（错误代码: {str(e)}）"
                }
            )

    def answer_question(self, question: str, subject: str, chat_history: str) -> Dict[str, Any]:
        sub_questions = self._split_into_subquestions(question)
        if len(sub_questions) > 1:
            trace = []
            confidences = []
            sources = []
            answers = []
            
            for i, sub_q in enumerate(sub_questions, 1):
                sub_result = self._single_turn_answer(sub_q, subject, chat_history)
                trace.append({"question": sub_q, **sub_result})
                confidences.append(sub_result.get("confidence", 0.2))
                sources.append(sub_result.get("source", "default"))
                answers.append(f"**问题{i}：{sub_q}**\n{sub_result.get('answer', '')}")
            
            overall_confidence = min(confidences) if confidences else 0.2
            combined_answer = "\n\n---\n\n".join(answers)
            
            return {
                "answer": combined_answer,
                "confidence": overall_confidence,
                "source": "multi_hop",
                "trace": trace
            }
        else:
            return self._single_turn_answer(question, subject, chat_history)

    def _single_turn_answer(self, question: str, subject: str, chat_history: str) -> Dict[str, Any]:
        question_lower = question.lower().strip()
        keywords = self._extract_keywords(question_lower)
         # 1. 匹配当前主题知识库
        answer, score, matched = self._match_question(question_lower, keywords, self.knowledge_base.get(subject, {}))
        if answer:
            confidence = self._confidence_from_score(score, base=0.75)
            # 格式化答案为引导式回答
            formatted_answer = self._format_guided_answer(question, answer, subject, matched)
            return {"answer": formatted_answer, "confidence": confidence, "source": "knowledge_base", "matched": matched}

        best_cross = None
        best_score = 0.0
        best_subject = None
        best_matched = None
        for subj, kb in self.knowledge_base.items():
            if subj == subject:
                continue
            match, sc, matched_question = self._match_question(question_lower, keywords, kb)
            if sc > best_score:
                best_score = sc
                best_cross = match
                best_subject = subj
                best_matched = matched_question
        # 2. 匹配跨主题知识库
        if best_cross and best_score >= 0.5:
            confidence = self._confidence_from_score(best_score, base=0.6)
            # 格式化跨主题答案为引导式回答
            formatted_answer = self._format_guided_answer(question, best_cross, best_subject, best_matched)
            formatted_answer += f"\n\n注意：这个回答主要来自{best_subject}的相关知识，可以帮助你理解当前问题。"
            return {
                "answer": formatted_answer,
                "confidence": confidence,
                "source": "knowledge_base_cross",
                "matched": best_matched,
                "related_subject": best_subject
            }
         # 3. 匹配通用知识库
        generic_answer, generic_score, matched_question = self._match_question(
            question_lower, keywords, self.knowledge_base.get("通用", {})
        )
        if generic_answer:
            confidence = self._confidence_from_score(generic_score, base=0.55)
            # 格式化通用答案为引导式回答
            formatted_answer = self._format_guided_answer(question, generic_answer, "通用学习策略", matched_question)
            return {
                "answer": formatted_answer,
                "confidence": confidence,
                "source": "knowledge_base_cross",
                "matched": matched_question
            }

        # 4. 使用RAG生成回答
        try:
            rag = get_rag_helper()
            rag_details = rag.ask_with_details(question, subject=subject, chat_history=chat_history)

            if rag_details and rag_details.get("answer") and len(rag_details.get("answer", "").strip()) > 0:
                mode = rag_details.get("mode", "rag_llm")
                confidence = 0.65 if mode == "rag_llm" else 0.55
                if rag_details.get("fallback"):
                    confidence = 0.5
                
                rag_answer = rag_details.get("answer", "")
                context_preview = rag_details.get("context_preview", "")
                context_snippets = rag_details.get("context_snippets") or self._extract_context_snippets(context_preview, question)
                # 增强RAG答案，使其更具引导性
                enhanced_answer = self._enhance_rag_answer(question, rag_answer, subject)
                
                return {
                    "answer": enhanced_answer,
                    "confidence": confidence,
                    "source": mode,
                    "context_preview": context_preview,
                    "context_snippets": context_snippets
                }
            else:
                # RAG返回空或无效，记录日志并回退到默认
                logger.warning(f"RAG returned empty answer for question: {question}")
                raise ValueError("RAG returned empty answer")
                
        except Exception as e:
            logger.error(f"RAG调用失败或无效: {str(e)}")
            # 失败后继续执行，使用默认回答
            
        # 提供更友好的默认引导式回答
        default_answer = self._get_default_guided_answer(question, subject)
        return {"answer": default_answer, "confidence": 0.2, "source": "default"}
        
    def _enhance_rag_answer(self, question: str, rag_answer: str, subject: str) -> str:
        """
        增强RAG生成的答案，保持简洁清晰
        """
        # 直接返回RAG生成的答案，去除冗余的引导性内容
        return rag_answer.strip()
    
    def _extract_context_snippets(self, context_preview: str, question: str) -> List[Dict[str, str]]:
        if not context_preview:
            return []
        snippets: List[Dict[str, str]] = []
        keywords = self._extract_keywords(question.lower())
        pattern = re.compile(r'\[片段(\d+)\]\s*(.*?)(?=\[片段\d+\]|$)', re.S)
        matches = list(pattern.finditer(context_preview))
        if not matches:
            # 尝试解析“=== 第n页 ===”结构
            fallback_parts = re.split(r'===\s*第(\d+)页\s*===', context_preview)
            if len(fallback_parts) > 1:
                for i in range(1, len(fallback_parts), 2):
                    page = fallback_parts[i]
                    text = fallback_parts[i + 1] if i + 1 < len(fallback_parts) else ""
                    filtered_text = self._filter_snippet_content(text, keywords)
                    snippets.append({
                        "title": f"第{page}页",
                        "content": filtered_text
                    })
                return snippets
        
        for match in matches:
            idx = match.group(1)
            snippet_text = match.group(2)
            filtered_text = self._filter_snippet_content(snippet_text, keywords)
            snippets.append({
                "title": f"片段{idx}",
                "content": filtered_text
            })
        return snippets

    def _filter_snippet_content(self, text: str, keywords: List[str]) -> str:
        cleaned = text.strip().replace('\n', ' ')
        if not cleaned:
            return ''
        sentences = [s.strip() for s in re.split(r'[。！？]', cleaned) if s.strip()]
        if not sentences:
            return cleaned[:200]
        if keywords:
            matched = [s for s in sentences if any(kw in s.lower() for kw in keywords)]
        else:
            matched = []
        target = matched[:2] if matched else sentences[:2]
        joined = '。'.join(target)
        return (joined[:200] + '…') if len(joined) > 200 else joined
        
    def _get_default_guided_answer(self, question: str, subject: str) -> str:
        """
        生成默认的引导式回答，即使没有找到直接答案
        """
        if "什么是" in question or "什么叫" in question:
            term = question.replace("什么是", "").replace("什么叫", "").replace("？", "").replace("?", "").strip()
            if term:
                return f"**{term}** 是{subject}领域的一个重要概念。建议从以下几方面了解：\n\n1. 核心定义与作用\n2. 与相关概念的区别\n3. 实际应用场景\n\n你可以问我更具体的问题。"
        
        return f"抱歉，暂未找到关于'{question}'的信息。建议：\n\n1. 查阅官方文档或教程\n2. 尝试更具体的问题描述\n3. 检查相关主题的知识点"

    def _extract_keywords(self, question: str) -> list:
        stop_words = {"的", "是", "什么", "如何", "怎么", "为什么", "？", "?", "吗", "呢", "了", "在", "有", "和", "与", "或"}
        words = re.findall(r'\w+', question)
        return [w for w in words if w not in stop_words and len(w) > 1]

    def _format_guided_answer(self, question: str, answer: str, subject: str, matched_question: str) -> str:
        """
        将标准答案格式化为简洁清晰的回答
        """
        # 直接返回核心答案，去除冗余的开场白和结尾
        formatted = answer.strip()
        
        # 如果答案本身已经很完整，直接返回
        if len(formatted) > 50:
            return formatted
        
        # 如果答案太短，添加简洁的补充说明
        if matched_question and matched_question != question:
            formatted = f"{formatted}\n\n（这是关于{matched_question}的回答）"
        
        return formatted
        
    def _match_question(self, question: str, keywords: list, knowledge_base: Dict[str, str]) -> Tuple[str, float, str]:
        best_match = None
        best_score = 0.0
        matched_question = None

        for key, value in knowledge_base.items():
            key_lower = key.lower()
            question_lower = question.lower()
            score = 0.0

            # 完全匹配或包含匹配
            if question_lower in key_lower or key_lower in question_lower:
                score = 1.0
            # 关键词匹配 - 更宽松的匹配逻辑
            elif keywords:
                # 计算匹配的关键词比例
                matched_keywords = sum(1 for kw in keywords if kw in key_lower)
                score = matched_keywords / len(keywords) if keywords else 0
                
                # 提高关键词匹配的权重，特别是对于Python相关问题
                if 'python' in question_lower and 'python' in key_lower:
                    score += 0.2
                
                # 对于常见问法词（什么是/如何/特点等），只在“问法相同”时加分，避免误匹配到无关条目
                if matched_keywords > 0:
                    if "什么是" in question_lower and "什么是" in key_lower:
                        score += 0.3
                    if "如何" in question_lower and "如何" in key_lower:
                        score += 0.2
                    if any(w in question_lower for w in ["特点", "特性", "优势"]) and any(w in key_lower for w in ["特点", "特性", "优势"]):
                        score += 0.2
                    if any(w in question_lower for w in ["介绍", "概述"]) and any(w in key_lower for w in ["介绍", "概述"]):
                        score += 0.15

            # 确保分数不超过1.0
            score = min(score, 1.0)

            if score > best_score:
                best_score = score
                best_match = value
                matched_question = key

        return best_match, min(best_score, 1.0), matched_question

    def _split_into_subquestions(self, question: str) -> List[str]:
        cleaned = question.strip()
        if len(cleaned) < 15:
            return [cleaned]
        parts = re.split(r'[？?。；]|并且|以及|同时|并', cleaned)
        parts = [p.strip() for p in parts if p and len(p.strip()) > 5]
        return parts if len(parts) > 1 else [cleaned]

    def _confidence_from_score(self, score: float, base: float = 0.6) -> float:
        score = max(0.0, min(score, 1.0))
        return min(base + score * 0.4, 0.98)
    
    def _knowledge_base_answer(self, question: str, subject: str):
        """在流式路径中复用内置知识库（当前主题 → 通用主题），命中返回(答案, 来源)。"""
        text = (question or "").lower().strip()
        if not text:
            return None, None
        keywords = self._extract_keywords(text)
        answer, score, matched = self._match_question(text, keywords, self.knowledge_base.get(subject, {}))
        source = "knowledge_base"
        if not answer or score < 0.5:
            answer, score, matched = self._match_question(text, keywords, self.knowledge_base.get("通用", {}))
            source = "knowledge_base_cross"
        if answer and score >= 0.5:
            return self._format_guided_answer(question, answer, subject, matched), source
        return None, None

    def _with_plan_offer(self, question: str, answer: str, subject: str) -> str:
        """学习路径类回答结尾询问是否转成学习计划，供"需要"这类确认承接。"""
        if (answer and subject and subject != "通用"
                and re.search(r"(怎么学|如何学|怎么入门|如何入门)", question)
                and "计划" not in answer):
            return answer.rstrip() + f"\n\n需要我为你制定一份{subject}的学习计划吗？"
        return answer

    def _conversation_history(self, user_id: str, conversation_id: str | None) -> str:
        """优先读取当前会话记录；无记录时回退短期记忆。"""
        if conversation_id:
            try:
                conversation = self.storage.get_conversation(conversation_id) or {}
                if conversation.get("user_id") != user_id:
                    return conversation_memory.get_history_as_text(user_id)
                lines = []
                for item in conversation.get("messages", [])[-12:]:
                    role = item.get("role")
                    text = str(item.get("content") or "").strip()
                    if role in {"user", "assistant"} and text:
                        lines.append(("用户" if role == "user" else "助手") + "：" + text[:700])
                if lines:
                    return "\n".join(lines[-8:])
            except Exception as exc:
                logger.warning("加载当前会话上下文失败: %s", type(exc).__name__)
        return conversation_memory.get_history_as_text(user_id)

    def _expand_question_with_subject(self, question: str, subject: str) -> str:
        """把省略式追问补全为完整问题，避免"怎么学"被答成通用学习方法。"""
        text = (question or "").strip()
        if not text or not subject or subject == "通用":
            return text
        if subject.lower() in text.lower():
            return text
        if re.fullmatch(r"(?:怎么学|如何学|怎么学习|如何学习|怎么开始|如何开始|怎么入门|如何入门|怎么上手|如何上手)[。.!！?？]*", text):
            return f"如何学习{subject}"
        if re.fullmatch(r"(?:是什么|是什么意思|怎么理解|如何理解)[。.!！?？]*", text):
            return f"什么是{subject}"
        if re.fullmatch(r"(?:详细点|详细一点|具体点|具体一点|展开|展开说说|再说说|多说点)[。.!！?？]*", text):
            return f"请更详细地解释{subject}"
        if len(text) <= 8 and re.match(r"^(?:它|这个|那个|这|那|还有|继续|然后|原理|细节|为什么)", text):
            return f"关于{subject}：{text}"
        return text

    def _enhance_question_with_context(self, question: str, chat_history: str, subject: str) -> str:
        """从对话历史中提取上下文，增强问题理解"""
        if not chat_history or len(chat_history.strip()) < 10:
            return question
        
        # 提取最近3轮对话的关键信息
        history_lines = chat_history.split('\n')
        recent_context = []
        
        # 反向查找最近的问答对
        for line in reversed(history_lines[-20:]):  # 只看最近20行
            line = line.strip()
            if not line:
                continue
            
            # 识别问题
            if line.startswith('Q:') or line.startswith('问题:'):
                recent_context.insert(0, f"之前的问题：{line[2:].strip()}")
        
        # 如果有相关上下文，添加到问题中
        if recent_context:
            context_summary = ' '.join(recent_context[-2:])  # 只保留最近2条
            # 检查问题是否与上下文相关
            if any(keyword in question.lower() for keyword in ['它', '这', '那个', '上述', '刚才']):
                enhanced = f"上下文：{context_summary}\n当前问题：{question}"
                return enhanced
        
        return question
    
    def _evaluate_answer_quality(self, answer: str, question: str, confidence: float) -> Dict[str, Any]:
        """评估答案质量"""
        if not answer or len(answer.strip()) < 10:
            return {
                "quality_score": 0.2,
                "issues": ["答案过短"],
                "suggestions": ["尝试提供更详细的回答"]
            }
        
        quality_metrics = {
            "length_score": 0,
            "structure_score": 0,
            "relevance_score": 0,
            "completeness_score": 0
        }
        
        # 1. 长度评分
        answer_len = len(answer)
        if 50 <= answer_len <= 1000:
            quality_metrics["length_score"] = 1.0
        elif answer_len < 50:
            quality_metrics["length_score"] = 0.5
        else:
            quality_metrics["length_score"] = 0.8
        
        # 2. 结构评分
        structure_indicators = ['：', ':', '\n\n', '1.', '2.', '（', '(', '•', '-']
        structure_count = sum(1 for ind in structure_indicators if ind in answer)
        quality_metrics["structure_score"] = min(structure_count / 3, 1.0)
        
        # 3. 相关性评分（检查是否包含问题中的关键词）
        question_keywords = set(re.findall(r'[\u4e00-\u9fff]{2,}', question))
        answer_lower = answer.lower()
        matched_keywords = sum(1 for kw in question_keywords if kw in answer_lower)
        quality_metrics["relevance_score"] = matched_keywords / max(len(question_keywords), 1)
        
        # 4. 完整性评分（检查是否有完整的句子）
        sentence_endings = ['。', '.', '！', '!', '？', '?']
        sentence_count = sum(1 for end in sentence_endings if end in answer)
        quality_metrics["completeness_score"] = min(sentence_count / 3, 1.0)
        
        # 综合质量得分
        quality_score = (
            quality_metrics["length_score"] * 0.25 +
            quality_metrics["structure_score"] * 0.25 +
            quality_metrics["relevance_score"] * 0.30 +
            quality_metrics["completeness_score"] * 0.20
        )
        
        # 结合置信度
        final_quality = (quality_score * 0.7 + confidence * 0.3)
        
        issues = []
        suggestions = []
        
        if quality_metrics["length_score"] < 0.6:
            issues.append("答案可能过短")
            suggestions.append("提供更多细节")
        
        if quality_metrics["structure_score"] < 0.5:
            issues.append("答案结构不够清晰")
            suggestions.append("使用列表或分段组织答案")
        
        if quality_metrics["relevance_score"] < 0.5:
            issues.append("答案相关性可能不足")
            suggestions.append("更直接地回答问题的核心")
        
        return {
            "quality_score": round(final_quality, 2),
            "metrics": quality_metrics,
            "issues": issues,
            "suggestions": suggestions
        }
    
    # ========== 多智能体协作方法 ==========
    
    def _notify_progress_update(self, question: str, subject: str, user_id: str):
        """
        通知 ServiceAgent 更新学习进度
        当用户学习新知识时自动触发
        """
        self.send_to_agent(
            receiver="service",
            action=MessageAction.NOTIFY,
            content={
                "type": "learning_progress",
                "question": question,
                "subject": subject,
                "user_id": user_id,
                "auto_detected": True  # 标记为自动检测
            },
            priority=MessagePriority.LOW
        )
    
    def _notify_create_review(self, question: str, answer: str, subject: str, user_id: str):
        """
        通知 ServiceAgent 创建复习计划
        当 QAAgent 生成高质量回答时自动触发
        """
        # 提取关键概念用于复习
        key_concepts = self._extract_key_concepts(question, answer)
        
        self.send_to_agent(
            receiver="service",
            action=MessageAction.REQUEST,
            content={
                "type": "create_review",
                "question": question,
                "answer": answer,
                "subject": subject,
                "user_id": user_id,
                "key_concepts": key_concepts,
                "auto_generated": True  # 标记为自动生成
            },
            priority=MessagePriority.NORMAL,
            requires_response=False
        )
    
    def _extract_key_concepts(self, question: str, answer: str) -> List[str]:
        """从问答中提取关键概念"""
        # 简单实现：提取问句中的核心词
        concepts = []
        
        # 提取"什么是X"中的X
        import re
        what_patterns = re.findall(r'什么是(.+?)[？?]?', question)
        concepts.extend(what_patterns)
        
        # 提取"X是什么"中的X
        is_patterns = re.findall(r'(.+?)是什么', question)
        concepts.extend(is_patterns)
        
        # 去重并返回
        return list(set([c.strip() for c in concepts if c.strip()]))
    
    def _handle_response(self, msg: AgentMessage):
        """处理来自 ServiceAgent 的响应"""
        content = msg.content
        if content.get("type") == "review_created":
            logger.info(f"复习计划已创建: {content.get('subject', 'unknown')}")
    
    def process_stream(self, message: Message):
        """流式处理问答请求，返回生成器"""
        try:
            content = message.content
            if "action" not in content:
                content["action"] = "generate"
            action = content["action"]
            
            # 非生成类操作，直接调用 process 并 yield 结果
            if action in ["get_history", "delete_history"]:
                response = self.process(message)
                yield {"type": "complete", "data": response.content}
                return

            question = content.get("question", "").strip()
            if not question:
                yield {"type": "error", "message": "问题不能为空"}
                return

            subject = content.get("subject", "通用")
            user_id = content.get("user_id", "default")
            image_data = content.get("image", None)

            # 2. 每轮对话：只使用短期记忆
            chat_history = self._conversation_history(user_id, content.get("conversation_id"))
            contextual_question = self._expand_question_with_subject(question, subject)
            enhanced_question = self._enhance_question_with_context(contextual_question, chat_history, subject)

            full_answer = ""
            real_source = "rag_llm"  # 兜底；实际来源以 meta 中 RAG 返回的为准
            is_general = False

            # 先查内置知识库（与 process 路径一致）：命中就不再把概念问题交给无关资料
            kb_answer, kb_source = self._knowledge_base_answer(contextual_question, subject)
            if kb_answer:
                kb_answer = self._with_plan_offer(question, kb_answer, subject)
                real_source = kb_source or "knowledge_base"
                is_general = False
                yield {"type": "meta", "data": {"mode": real_source, "source": real_source, "is_general": False}}
                # 打字机式逐段吐出，与 RAG 路径保持一致
                rag_service = get_rag_helper()
                for piece in rag_service.stream_text_pieces(kb_answer):
                    full_answer += piece
                    yield {"type": "token", "data": piece}
            else:
                # 调用 RAG 流式接口
                rag = get_rag_helper()
                for chunk in rag.ask_with_details_stream(enhanced_question, subject=subject, chat_history=chat_history, image_data=image_data):
                    chunk_type = chunk.get("type")
                    data = chunk.get("data")

                    if chunk_type == "meta":
                        if isinstance(data, dict):
                            real_source = data.get("source") or data.get("mode") or real_source
                            is_general = bool(data.get("is_general"))
                        yield {"type": "meta", "data": data}
                    elif chunk_type == "token":
                        full_answer += data
                        yield {"type": "token", "data": data}
                    elif chunk_type == "error":
                        yield {"type": "error", "message": data}

            # 生成完成后，保存历史：来源用 RAG 实际返回的 mode，不再写死 rag_llm
            if full_answer:
                confidence = 0.5 if is_general else 0.8
                source = real_source

                self.storage.save_qa_history(
                    user_id,
                    question,
                    full_answer,
                    subject,
                    metadata={
                        "confidence": confidence,
                        "source": source,
                        "source_label": SOURCE_LABELS.get(source, "通用建议")
                    }
                )
                conversation_memory.append(user_id, question, full_answer)
                
        except Exception as e:
            logger.error("QA Stream Error: %s", type(e).__name__)
            yield {"type": "error", "message": "回答暂时无法生成，请稍后重试。"}
    
    # 保持函数签名返回三个值：最佳匹配答案、匹配分数、匹配的问题

