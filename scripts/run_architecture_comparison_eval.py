import argparse
import json
import os
import re
import sys
import time
import traceback
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Set


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)


COMPLEX_DEFAULT = os.path.join(ROOT_DIR, "data", "complex_task_testcases.jsonl")
SINGLE_DEFAULT = os.path.join(ROOT_DIR, "data", "intent_testcases.jsonl")
OUTPUT_DIR = os.path.join(ROOT_DIR, "data", "eval_results")
SUPPORTED_LABELS = {"plan", "qa", "resource", "progress", "quiz", "review"}


def load_jsonl(path: str, limit: int = 0) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = (line or "").strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if limit > 0 and len(rows) >= limit:
                break
    return rows


def normalize_expected_subtasks(values: Sequence[str]) -> List[str]:
    normalized: List[str] = []
    for value in values:
        item = str(value or "").strip().lower()
        if item in SUPPORTED_LABELS and item not in normalized:
            normalized.append(item)
    return normalized


def infer_subtasks_from_response(response: Dict[str, Any]) -> List[str]:
    labels: Set[str] = set()
    if not isinstance(response, dict):
        return []

    plan = response.get("plan")
    if isinstance(plan, dict) and plan:
        labels.add("plan")

    resources = response.get("resources")
    if isinstance(resources, list) and resources:
        labels.add("resource")

    progress = response.get("progress")
    if isinstance(progress, dict) and progress:
        labels.add("progress")

    if response.get("type") == "progress_update":
        labels.add("progress")

    questions = response.get("questions")
    if response.get("quiz_id") or (isinstance(questions, list) and questions):
        labels.add("quiz")

    if response.get("review") is not None or response.get("review_id") or response.get("next_review_at"):
        labels.add("review")

    answer = str(response.get("answer") or "").strip()
    source = str(response.get("source") or "").strip()
    context_snippets = response.get("context_snippets")
    if answer or source or context_snippets:
        labels.add("qa")

    intent = str(response.get("intent") or "").strip().lower()
    if intent.startswith("plan"):
        labels.add("plan")

    return sorted(labels)


def infer_primary_label(response: Dict[str, Any], actual_labels: Sequence[str]) -> str:
    if not actual_labels:
        return "unknown"

    intent = str(response.get("intent") or "").strip().lower()
    if intent.startswith("plan"):
        return "plan"

    priority = ["plan", "resource", "progress", "quiz", "review", "qa"]
    for label in priority:
        if label in actual_labels:
            return label
    return actual_labels[0]


def compact_response(response: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "status": response.get("status"),
        "type": response.get("type"),
        "intent": response.get("intent"),
        "message": str(response.get("message") or "")[:160],
        "answer_preview": str(response.get("answer") or "")[:160],
        "has_plan": isinstance(response.get("plan"), dict) and bool(response.get("plan")),
        "resource_count": len(response.get("resources") or []) if isinstance(response.get("resources"), list) else 0,
        "has_progress": isinstance(response.get("progress"), dict) and bool(response.get("progress")),
        "question_count": len(response.get("questions") or []) if isinstance(response.get("questions"), list) else 0,
        "has_review": response.get("review") is not None or bool(response.get("review_id")),
    }


class SingleAgentBaseline:
    """
    Monolithic baseline:
    - one controller
    - one primary intent per request
    - no compound-task decomposition
    """

    def __init__(self) -> None:
        from agents.planner_service import PlannerService
        from agents.progress_service import ProgressService
        from agents.qa_agent import QAAgent
        from agents.resource_service import ResourceService
        from core.message import Message, MessageType
        from nlp.intent_detection import detect_intent, extract_duration, extract_subject

        self.Message = Message
        self.MessageType = MessageType
        self.detect_intent = detect_intent
        self.extract_duration = extract_duration
        self.extract_subject = extract_subject

        self.qa_agent = QAAgent()
        self.planner_agent = PlannerService()
        self.resource_agent = ResourceService()
        self.progress_agent = ProgressService()

        try:
            from agents.quiz_service import QuizService
            self.quiz_agent = QuizService()
        except Exception:
            self.quiz_agent = None

        try:
            from agents.review_service import ReviewService
            self.review_agent = ReviewService()
        except Exception:
            self.review_agent = None

    def _matches_any(self, query: str, patterns: Sequence[str]) -> bool:
        return any(re.search(pattern, query, re.IGNORECASE) for pattern in patterns)

    def _detect_compound_intents(self, query: str) -> List[str]:
        intent_patterns = {
            "plan": [
                r"学习计划",
                r"复习计划",
                r"制定.*计划",
                r"规划.*(天|周|学习|路线|安排)",
                r"安排.*(天|周|学习|路线|计划)",
                r"学习路线",
                r"学习路径",
                r"学习安排",
                r"(在|用).*(天|周|个月)内.*(学完|补完|复习|掌握)",
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
                r"练习题",
                r"测试题",
                r"测验",
                r"考我",
                r"自测",
                r"小测",
                r"检验",
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

    def _merge_serial_results(
        self,
        query: str,
        subject: str,
        intents: Sequence[str],
        qa_result: Optional[Dict[str, Any]] = None,
        plan_result: Optional[Dict[str, Any]] = None,
        progress_result: Optional[Dict[str, Any]] = None,
        resource_result: Optional[Dict[str, Any]] = None,
        quiz_result: Optional[Dict[str, Any]] = None,
        review_result: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "status": "success",
            "type": "complex",
            "subject": subject,
            "intent_chain": list(intents),
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
            result["intent"] = plan_result.get("intent")
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
                "message": "未能完成串行任务处理",
                "type": "complex",
                "subject": subject,
                "intent_chain": list(intents),
            }

        result["message"] = "已完成" + "、".join(completed_parts)
        return result

    def _handle_compound(self, query: str, user_id: str, intents: Sequence[str]) -> Dict[str, Any]:
        subject = self.extract_subject(query)
        qa_result = self._handle_qa(query, user_id) if "qa" in intents else None
        plan_result = self._handle_plan(query, user_id) if "plan" in intents else None
        progress_result = self._handle_progress(query, user_id) if "progress" in intents else None
        resource_result = self._handle_resource(query, user_id) if "resource" in intents else None
        quiz_result = self._handle_quiz(query, user_id) if "quiz" in intents else None
        review_result = None
        if "review" in intents:
            review_content = (qa_result or {}).get("answer") or query
            review_result = self._handle_review(query, user_id, subject=subject, content_text=review_content)

        return self._merge_serial_results(
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

    def _process_with(self, agent: Any, receiver: str, content: Dict[str, Any]) -> Dict[str, Any]:
        if agent is None:
            return {"status": "error", "message": f"{receiver} unavailable"}
        message = self.Message(
            sender="single_agent_baseline",
            receiver=receiver,
            msg_type=self.MessageType.REQUEST,
            content=content,
        )
        response = agent.process(message)
        return response.content if hasattr(response, "content") else response

    def _handle_plan(self, query: str, user_id: str) -> Dict[str, Any]:
        subject = self.extract_subject(query)
        duration = self.extract_duration(query)
        return self._process_with(
            self.planner_agent,
            "planner",
            {
                "action": "create",
                "user_id": user_id,
                "subject": subject,
                "duration": duration,
            },
        )

    def _handle_qa(self, query: str, user_id: str) -> Dict[str, Any]:
        subject = self.extract_subject(query)
        result = self._process_with(
            self.qa_agent,
            "qa",
            {
                "action": "generate",
                "question": query,
                "subject": subject,
                "user_id": user_id,
            },
        )
        if "subject" not in result:
            result["subject"] = subject
        return result

    def _handle_resource(self, query: str, user_id: str) -> Dict[str, Any]:
        subject = self.extract_subject(query)
        return self._process_with(
            self.resource_agent,
            "resource",
            {
                "action": "recommend",
                "subject": subject,
                "user_id": user_id,
                "count": 5,
            },
        )

    def _handle_progress(self, query: str, user_id: str) -> Dict[str, Any]:
        subject = self.extract_subject(query)
        update_patterns = ["完成", "学完", "学习了", "学了", "更新", "记录", "已完成"]
        action = "update" if any(token in query for token in update_patterns) else "get"
        content: Dict[str, Any] = {
            "action": action,
            "user_id": user_id,
            "subject": subject,
        }
        if action == "update":
            content["progress_data"] = {
                "completed_tasks": 1,
                "total_tasks": 10,
                "study_time": 30,
                "note": query,
                "date": datetime.now().strftime("%Y-%m-%d"),
            }
        else:
            content["detail"] = True
        return self._process_with(self.progress_agent, "progress", content)

    def _handle_quiz(self, query: str, user_id: str) -> Dict[str, Any]:
        subject = self.extract_subject(query)
        num_match = re.search(r"(\d+)\s*道", query)
        difficulty_match = re.search(r"(简单|easy|初级|中等|medium|中级|困难|hard|高级)", query, re.IGNORECASE)
        return self._process_with(
            self.quiz_agent,
            "quiz",
            {
                "action": "generate",
                "query": query,
                "subject": subject,
                "difficulty": difficulty_match.group(1) if difficulty_match else "medium",
                "num_questions": int(num_match.group(1)) if num_match else 5,
                "user_id": user_id,
            },
        )

    def _handle_review(
        self,
        query: str,
        user_id: str,
        subject: Optional[str] = None,
        content_text: str = "",
    ) -> Dict[str, Any]:
        query_lower = query.lower()
        subject = subject or self.extract_subject(query)

        if self._matches_any(query, [r"安排", r"计划", r"清单", r"重点", r"提纲", r"复习", r"回顾", r"巩固"]):
            action = "schedule"
        elif "完成" in query or "complete" in query_lower:
            action = "complete"
        else:
            action = "next"

        return self._process_with(
            self.review_agent,
            "review",
            {
                "action": action,
                "query": query,
                "user_id": user_id,
                "subject": subject,
                "content": content_text or query,
                "importance": "medium",
            },
        )

    def process_request(self, query: str, user_id: str = "default") -> Dict[str, Any]:
        intents = self._detect_compound_intents(query)
        if len(intents) >= 2:
            return self._handle_compound(query, user_id, intents)

        intent = self.detect_intent(query)
        if intent == "plan":
            return self._handle_plan(query, user_id)
        if intent == "resource":
            return self._handle_resource(query, user_id)
        if intent == "progress":
            return self._handle_progress(query, user_id)
        if intent == "quiz":
            return self._handle_quiz(query, user_id)
        if intent == "review":
            return self._handle_review(query, user_id)
        return self._handle_qa(query, user_id)


def evaluate_complex_cases(
    runner: Any,
    dataset_path: str,
    limit: int,
    user_prefix: str,
) -> Dict[str, Any]:
    rows = load_jsonl(dataset_path, limit=limit)
    details: List[Dict[str, Any]] = []
    completed_cases = 0
    expected_subtask_total = 0
    matched_subtask_total = 0
    elapsed_values: List[float] = []
    per_scenario: Dict[str, Dict[str, Any]] = defaultdict(lambda: {"total": 0, "completed": 0})
    per_subtask: Dict[str, Dict[str, Any]] = defaultdict(lambda: {"expected": 0, "matched": 0})

    for index, row in enumerate(rows, 1):
        query = str(row.get("query") or "").strip()
        expected = normalize_expected_subtasks(row.get("expected_subtasks") or [])
        scenario = str((row.get("meta") or {}).get("scenario") or "unknown")
        user_id = f"{user_prefix}_complex_{index}"

        started = time.time()
        error = ""
        response: Dict[str, Any]
        try:
            response = runner.process_request(query, user_id=user_id)
        except Exception as exc:
            response = {"status": "error", "message": str(exc)}
            error = traceback.format_exc(limit=3)
        elapsed_ms = round((time.time() - started) * 1000, 2)
        elapsed_values.append(elapsed_ms)

        actual = infer_subtasks_from_response(response)
        matched = [label for label in expected if label in actual]
        complete = set(expected).issubset(set(actual))

        if complete:
            completed_cases += 1
        expected_subtask_total += len(expected)
        matched_subtask_total += len(matched)

        per_scenario[scenario]["total"] += 1
        if complete:
            per_scenario[scenario]["completed"] += 1

        for label in expected:
            per_subtask[label]["expected"] += 1
            if label in actual:
                per_subtask[label]["matched"] += 1

        details.append(
            {
                "index": index,
                "query": query,
                "scenario": scenario,
                "expected_subtasks": expected,
                "actual_subtasks": actual,
                "matched_subtasks": matched,
                "completed": complete,
                "partial_score": round(len(matched) / len(expected), 4) if expected else 0.0,
                "elapsed_ms": elapsed_ms,
                "response": compact_response(response),
                "error": error,
            }
        )

    scenario_summary = {}
    for scenario, stats in sorted(per_scenario.items()):
        total = stats["total"]
        completed = stats["completed"]
        scenario_summary[scenario] = {
            "total": total,
            "completed": completed,
            "completion_rate": round(completed / total, 4) if total else 0.0,
        }

    subtask_summary = {}
    for label, stats in sorted(per_subtask.items()):
        expected = stats["expected"]
        matched = stats["matched"]
        subtask_summary[label] = {
            "expected": expected,
            "matched": matched,
            "hit_rate": round(matched / expected, 4) if expected else 0.0,
        }

    total_cases = len(rows)
    return {
        "dataset_path": dataset_path,
        "total_cases": total_cases,
        "completed_cases": completed_cases,
        "completion_rate": round(completed_cases / total_cases, 4) if total_cases else 0.0,
        "matched_subtasks": matched_subtask_total,
        "expected_subtasks": expected_subtask_total,
        "subtask_accuracy": round(matched_subtask_total / expected_subtask_total, 4)
        if expected_subtask_total
        else 0.0,
        "avg_elapsed_ms": round(sum(elapsed_values) / len(elapsed_values), 2) if elapsed_values else 0.0,
        "per_scenario": scenario_summary,
        "per_subtask": subtask_summary,
        "details": details,
    }


def evaluate_single_cases(
    runner: Any,
    dataset_path: str,
    limit: int,
    user_prefix: str,
) -> Dict[str, Any]:
    rows = load_jsonl(dataset_path, limit=limit)
    details: List[Dict[str, Any]] = []
    correct_cases = 0
    elapsed_values: List[float] = []
    per_label: Dict[str, Dict[str, Any]] = defaultdict(lambda: {"total": 0, "correct": 0})

    for index, row in enumerate(rows, 1):
        query = str(row.get("query") or row.get("question") or "").strip()
        expected = str(row.get("ground_truth") or "").strip().lower()
        user_id = f"{user_prefix}_single_{index}"

        started = time.time()
        error = ""
        try:
            response = runner.process_request(query, user_id=user_id)
        except Exception as exc:
            response = {"status": "error", "message": str(exc)}
            error = traceback.format_exc(limit=3)
        elapsed_ms = round((time.time() - started) * 1000, 2)
        elapsed_values.append(elapsed_ms)

        actual_labels = infer_subtasks_from_response(response)
        primary_label = infer_primary_label(response, actual_labels)
        correct = expected in actual_labels
        if correct:
            correct_cases += 1

        per_label[expected]["total"] += 1
        if correct:
            per_label[expected]["correct"] += 1

        details.append(
            {
                "index": index,
                "query": query,
                "expected_label": expected,
                "actual_labels": actual_labels,
                "primary_label": primary_label,
                "correct": correct,
                "elapsed_ms": elapsed_ms,
                "response": compact_response(response),
                "error": error,
            }
        )

    label_summary = {}
    for label, stats in sorted(per_label.items()):
        total = stats["total"]
        correct = stats["correct"]
        label_summary[label] = {
            "total": total,
            "correct": correct,
            "accuracy": round(correct / total, 4) if total else 0.0,
        }

    total_cases = len(rows)
    return {
        "dataset_path": dataset_path,
        "total_cases": total_cases,
        "correct_cases": correct_cases,
        "accuracy": round(correct_cases / total_cases, 4) if total_cases else 0.0,
        "avg_elapsed_ms": round(sum(elapsed_values) / len(elapsed_values), 2) if elapsed_values else 0.0,
        "per_label": label_summary,
        "details": details,
    }


def build_comparison_report(single_report: Dict[str, Any], multi_report: Dict[str, Any]) -> Dict[str, Any]:
    single_complex = single_report.get("complex", {})
    multi_complex = multi_report.get("complex", {})
    single_single = single_report.get("single", {})
    multi_single = multi_report.get("single", {})

    return {
        "complex_completion_rate_delta": round(
            multi_complex.get("completion_rate", 0.0) - single_complex.get("completion_rate", 0.0), 4
        ),
        "complex_subtask_accuracy_delta": round(
            multi_complex.get("subtask_accuracy", 0.0) - single_complex.get("subtask_accuracy", 0.0), 4
        ),
        "complex_avg_elapsed_ms_delta": round(
            multi_complex.get("avg_elapsed_ms", 0.0) - single_complex.get("avg_elapsed_ms", 0.0), 2
        ),
        "single_task_accuracy_delta": round(
            multi_single.get("accuracy", 0.0) - single_single.get("accuracy", 0.0), 4
        ),
        "single_task_avg_elapsed_ms_delta": round(
            multi_single.get("avg_elapsed_ms", 0.0) - single_single.get("avg_elapsed_ms", 0.0), 2
        ),
    }


def write_report(report: Dict[str, Any], output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)


def print_summary(report: Dict[str, Any]) -> None:
    single_report = report.get("single_agent", {})
    multi_report = report.get("multi_agent", {})
    compare = report.get("comparison", {})

    print("=== Architecture Comparison Summary ===")
    print(f"Generated At: {report.get('generated_at')}")
    print("")
    print("[Single-Agent Baseline]")
    print(f"Complex Completion: {single_report.get('complex', {}).get('completion_rate', 0.0):.2%}")
    print(f"Complex Subtask Accuracy: {single_report.get('complex', {}).get('subtask_accuracy', 0.0):.2%}")
    print(f"Single-Task Accuracy: {single_report.get('single', {}).get('accuracy', 0.0):.2%}")
    print("")
    print("[Multi-Agent]")
    print(f"Complex Completion: {multi_report.get('complex', {}).get('completion_rate', 0.0):.2%}")
    print(f"Complex Subtask Accuracy: {multi_report.get('complex', {}).get('subtask_accuracy', 0.0):.2%}")
    print(f"Single-Task Accuracy: {multi_report.get('single', {}).get('accuracy', 0.0):.2%}")
    print("")
    print("[Delta: Multi - Single]")
    print(f"Complex Completion Delta: {compare.get('complex_completion_rate_delta', 0.0):+.2%}")
    print(f"Complex Subtask Accuracy Delta: {compare.get('complex_subtask_accuracy_delta', 0.0):+.2%}")
    print(f"Single-Task Accuracy Delta: {compare.get('single_task_accuracy_delta', 0.0):+.2%}")
    print("")
    print(f"Report: {report.get('output_path')}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare single-agent baseline vs multi-agent architecture.")
    parser.add_argument("--complex-dataset", default=COMPLEX_DEFAULT)
    parser.add_argument("--single-dataset", default=SINGLE_DEFAULT)
    parser.add_argument("--complex-limit", type=int, default=0)
    parser.add_argument("--single-limit", type=int, default=0)
    parser.add_argument("--user-prefix", default="compare")
    parser.add_argument("--enable-rag", action="store_true")
    parser.add_argument("--output", default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.enable_rag:
        os.environ["RAG_ENABLED"] = "false"

    from core.coordinator import AgentCoordinator  # noqa: E402

    if not args.enable_rag:
        import agents.planner_service as planner_service  # noqa: E402
        import agents.qa_agent as qa_agent  # noqa: E402
        import agents.quiz_service as quiz_service  # noqa: E402
        import agents.resource_service as resource_service  # noqa: E402
        import nlp.intent_detection as intent_detection  # noqa: E402

        def _fast_single_turn(self, question, subject, chat_history):
            return {
                "answer": self._get_default_guided_answer(question, subject),
                "confidence": 0.2,
                "source": "default",
            }

        def _fast_generate_quiz(self, subject, difficulty="medium", num_questions=5, user_id="default"):
            difficulty_normalized = self.difficulty_mapping.get(str(difficulty).lower(), "medium")
            questions = self._generate_questions_from_template(
                subject=subject,
                difficulty=difficulty_normalized,
                num_questions=num_questions,
            )
            return {
                "status": "success",
                "quiz_id": f"eval_{user_id}_{subject}_{num_questions}",
                "subject": subject,
                "difficulty": difficulty_normalized,
                "questions": questions[:num_questions],
                "num_questions": min(len(questions), num_questions),
            }

        planner_service.PlannerService._enhance_plan_with_rag = lambda self, weekly_plan, subject, duration, profile=None: None
        qa_agent.QAAgent._single_turn_answer = _fast_single_turn
        qa_agent.get_rag_helper = lambda: None
        quiz_service.get_rag_helper = lambda: None
        quiz_service.QuizService.generate_quiz = _fast_generate_quiz
        resource_service.ResourceService._fetch_external_resources = lambda self, subject, count: []
        resource_service.ResourceService._search_online_resources = lambda self, subject, limit: []
        intent_detection.detect_intent_llm = lambda query, timeout=5.0: None

    single_runner = SingleAgentBaseline()
    multi_runner = AgentCoordinator()

    generated_at = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")

    single_report = {
        "complex": evaluate_complex_cases(
            runner=single_runner,
            dataset_path=args.complex_dataset,
            limit=args.complex_limit,
            user_prefix=f"{args.user_prefix}_single_agent",
        ),
        "single": evaluate_single_cases(
            runner=single_runner,
            dataset_path=args.single_dataset,
            limit=args.single_limit,
            user_prefix=f"{args.user_prefix}_single_agent",
        ),
    }

    multi_report = {
        "complex": evaluate_complex_cases(
            runner=type(
                "MultiAgentRunner",
                (),
                {"process_request": lambda self, query, user_id="default": multi_runner.handle_complex_request(query, user_id)},
            )(),
            dataset_path=args.complex_dataset,
            limit=args.complex_limit,
            user_prefix=f"{args.user_prefix}_multi_agent",
        ),
        "single": evaluate_single_cases(
            runner=multi_runner,
            dataset_path=args.single_dataset,
            limit=args.single_limit,
            user_prefix=f"{args.user_prefix}_multi_agent",
        ),
    }

    report: Dict[str, Any] = {
        "generated_at": generated_at,
        "single_agent": single_report,
        "multi_agent": multi_report,
        "comparison": build_comparison_report(single_report, multi_report),
    }

    output_path = args.output
    if not output_path:
        output_path = os.path.join(
            OUTPUT_DIR,
            f"architecture_comparison_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        )
    report["output_path"] = output_path
    write_report(report, output_path)
    print_summary(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
