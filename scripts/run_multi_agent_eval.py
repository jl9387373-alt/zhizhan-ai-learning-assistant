import argparse
import json
import os
import sys
import time
import traceback
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, Iterable, List, Sequence, Set


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

    message = str(response.get("message") or "").strip()
    if message:
        if "学习计划" in message:
            labels.add("plan")
        if "复习计划已安排" in message:
            labels.add("review")
        if "进度已更新" in message:
            labels.add("progress")

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
    data: Dict[str, Any] = {
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
    return data


def invoke_complex_case(
    coordinator: Any,
    query: str,
    user_id: str,
    entrypoint: str,
) -> Dict[str, Any]:
    if entrypoint == "process_request":
        return coordinator.process_request(query, user_id=user_id)
    return coordinator.handle_complex_request(query, user_id=user_id)


def evaluate_complex_cases(
    coordinator: Any,
    dataset_path: str,
    entrypoint: str,
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
            response = invoke_complex_case(coordinator, query, user_id, entrypoint)
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
        "entrypoint": entrypoint,
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
    coordinator: Any,
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
            response = coordinator.process_request(query, user_id=user_id)
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


def write_report(report: Dict[str, Any], output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)


def print_summary(report: Dict[str, Any]) -> None:
    complex_report = report.get("complex", {})
    single_report = report.get("single", {})

    print("=== Multi-Agent Evaluation Summary ===")
    print(f"Generated At: {report.get('generated_at')}")
    print("")
    print("[Complex Tasks]")
    print(f"Cases: {complex_report.get('total_cases', 0)}")
    print(f"Completion Rate: {complex_report.get('completion_rate', 0.0):.2%}")
    print(f"Subtask Accuracy: {complex_report.get('subtask_accuracy', 0.0):.2%}")
    print(f"Avg Elapsed: {complex_report.get('avg_elapsed_ms', 0.0)} ms")
    print("")
    print("[Single Tasks]")
    print(f"Cases: {single_report.get('total_cases', 0)}")
    print(f"Accuracy: {single_report.get('accuracy', 0.0):.2%}")
    print(f"Avg Elapsed: {single_report.get('avg_elapsed_ms', 0.0)} ms")
    print("")
    print(f"Report: {report.get('output_path')}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate multi-agent collaboration and single-task routing.")
    parser.add_argument("--complex-dataset", default=COMPLEX_DEFAULT)
    parser.add_argument("--single-dataset", default=SINGLE_DEFAULT)
    parser.add_argument("--complex-entrypoint", choices=["handle_complex_request", "process_request"], default="handle_complex_request")
    parser.add_argument("--complex-limit", type=int, default=0)
    parser.add_argument("--single-limit", type=int, default=0)
    parser.add_argument("--user-prefix", default="eval")
    parser.add_argument("--enable-rag", action="store_true")
    parser.add_argument("--skip-single", action="store_true")
    parser.add_argument("--skip-complex", action="store_true")
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

    coordinator = AgentCoordinator()

    generated_at = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    report: Dict[str, Any] = {
        "generated_at": generated_at,
        "complex": {},
        "single": {},
    }

    if not args.skip_complex:
        report["complex"] = evaluate_complex_cases(
            coordinator=coordinator,
            dataset_path=args.complex_dataset,
            entrypoint=args.complex_entrypoint,
            limit=args.complex_limit,
            user_prefix=args.user_prefix,
        )

    if not args.skip_single:
        report["single"] = evaluate_single_cases(
            coordinator=coordinator,
            dataset_path=args.single_dataset,
            limit=args.single_limit,
            user_prefix=args.user_prefix,
        )

    output_path = args.output
    if not output_path:
        output_path = os.path.join(OUTPUT_DIR, f"multi_agent_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    report["output_path"] = output_path

    write_report(report, output_path)
    print_summary(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
