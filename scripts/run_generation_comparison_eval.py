import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Sequence


ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


from rag.evaluation import load_jsonl_dataset
from rag.metrics import calculate_answer_quality
from rag.prompting import build_qa_prompt
from rag.service import GENERAL_SUBJECT, RAGService


DEFAULT_DATASET = ROOT_DIR / "data" / "rag_testcases_manual_curated_100.jsonl"
OUTPUT_DIR = ROOT_DIR / "data" / "eval_results"
GENERATION_METRICS = ("faithfulness", "answer_relevancy", "answer_correctness")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare pure-LLM generation against the system generation logic under the same retrieved context."
    )
    parser.add_argument("--dataset", type=str, default=str(DEFAULT_DATASET))
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--subject", type=str, default="")
    return parser.parse_args()


def extract_expected_keywords(ground_truth: str, limit: int = 10) -> List[str]:
    stop_words = {
        "这个",
        "那个",
        "一种",
        "可以",
        "用于",
        "包括",
        "主要",
        "通过",
        "进行",
        "以及",
        "其中",
        "系统",
        "问题",
    }
    import re

    tokens = re.findall(r"[\u4e00-\u9fff]{2,}|[A-Za-z0-9_]{3,}", str(ground_truth or "").lower())
    result: List[str] = []
    for token in tokens:
        if token in stop_words or token in result:
            continue
        result.append(token)
        if len(result) >= limit:
            break
    return result


def normalize_sources(case: Dict[str, Any]) -> List[str]:
    meta = case.get("meta") if isinstance(case.get("meta"), dict) else {}
    acceptable_raw = case.get("acceptable_sources") or meta.get("acceptable_sources") or []
    primary = case.get("primary_source") or meta.get("primary_source") or meta.get("source_file") or ""

    sources: List[str] = []
    primary_name = Path(str(primary or "")).name.strip()
    if primary_name:
        sources.append(primary_name)

    if isinstance(acceptable_raw, list):
        for item in acceptable_raw:
            name = Path(str(item or "")).name.strip()
            if name and name not in sources:
                sources.append(name)
    return sources


def invoke_llm(service: RAGService, question: str, subject: str, context: str) -> str:
    api_key = service.cfg["api_key"]
    llm = service._build_llm(api_key)
    if llm is None or service.PromptTemplate is None:
        return ""
    prompt = build_qa_prompt({"PromptTemplate": service.PromptTemplate})
    response = (prompt | llm).invoke(
        {
            "question": question,
            "subject": subject,
            "context": context,
            "history": service._format_chat_history(""),
        }
    )
    return service._sanitize_answer(getattr(response, "content", None) or str(response))


def generate_system_answer(
    service: RAGService,
    question: str,
    subject: str,
    docs: Sequence[Any],
    focus_terms: Sequence[str],
    context: str,
) -> tuple[str, str]:
    return service.generate_answer_with_postcheck(
        question=question,
        subject=subject,
        docs=list(docs),
        focus_terms=list(focus_terms),
        context=context,
        chat_history="",
        allow_llm=True,
    )


def build_case_result(
    service: RAGService,
    case: Dict[str, Any],
) -> Dict[str, Any]:
    question = str(case.get("question") or case.get("query") or "").strip()
    meta = case.get("meta") if isinstance(case.get("meta"), dict) else {}
    subject = str(case.get("subject") or meta.get("subject") or GENERAL_SUBJECT).strip() or GENERAL_SUBJECT
    ground_truth = str(case.get("ground_truth") or "").strip()
    acceptable_sources = normalize_sources(case)

    started = time.time()
    docs, focus_terms = service._retrieve(question, subject, "")
    retrieval_time = time.time() - started
    snippets = service._docs_to_snippets(list(docs))
    contexts = [str(getattr(doc, "page_content", "") or "") for doc in docs if str(getattr(doc, "page_content", "") or "").strip()]
    context_text = service._build_context(list(docs))

    system_started = time.time()
    system_answer, system_mode = generate_system_answer(service, question, subject, docs, focus_terms, context_text)
    system_elapsed = time.time() - system_started

    llm_started = time.time()
    pure_llm_answer = invoke_llm(service, question, subject, context_text) if context_text else ""
    pure_llm_elapsed = time.time() - llm_started

    expected_keywords = extract_expected_keywords(ground_truth) if ground_truth else []
    system_metrics = calculate_answer_quality(
        answer=system_answer,
        context=contexts,
        question=question,
        expected_keywords=expected_keywords,
        expected_aspects=[],
        ground_truth=ground_truth,
        context_snippets=snippets,
        acceptable_sources=acceptable_sources,
    )
    pure_llm_metrics = calculate_answer_quality(
        answer=pure_llm_answer,
        context=contexts,
        question=question,
        expected_keywords=expected_keywords,
        expected_aspects=[],
        ground_truth=ground_truth,
        context_snippets=snippets,
        acceptable_sources=acceptable_sources,
    )

    return {
        "question": question,
        "subject": subject,
        "ground_truth": ground_truth,
        "retrieval": {
            "doc_count": len(docs),
            "focus_terms": list(focus_terms),
            "retrieval_time_s": round(retrieval_time, 4),
            "context_snippets": snippets[:5],
            "acceptable_sources": acceptable_sources,
        },
        "pure_llm": {
            "answer": pure_llm_answer,
            "response_time_s": round(pure_llm_elapsed, 4),
            "metrics": {name: round(float(pure_llm_metrics.get(name, 0.0)), 4) for name in GENERATION_METRICS},
        },
        "system": {
            "answer": system_answer,
            "mode": system_mode,
            "response_time_s": round(system_elapsed, 4),
            "metrics": {name: round(float(system_metrics.get(name, 0.0)), 4) for name in GENERATION_METRICS},
        },
    }


def summarize_cases(cases: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "pure_llm": {name: 0.0 for name in GENERATION_METRICS},
        "system": {name: 0.0 for name in GENERATION_METRICS},
        "wins": {
            "pure_llm": {name: 0 for name in GENERATION_METRICS},
            "system": {name: 0 for name in GENERATION_METRICS},
            "tie": {name: 0 for name in GENERATION_METRICS},
        },
        "avg_response_time_s": {"pure_llm": 0.0, "system": 0.0, "retrieval": 0.0},
    }
    if not cases:
        return summary

    for side in ("pure_llm", "system"):
        for name in GENERATION_METRICS:
            values = [float(case.get(side, {}).get("metrics", {}).get(name, 0.0)) for case in cases]
            summary[side][name] = round(mean(values), 4) if values else 0.0

    summary["avg_response_time_s"]["pure_llm"] = round(
        mean(float(case.get("pure_llm", {}).get("response_time_s", 0.0)) for case in cases), 4
    )
    summary["avg_response_time_s"]["system"] = round(
        mean(float(case.get("system", {}).get("response_time_s", 0.0)) for case in cases), 4
    )
    summary["avg_response_time_s"]["retrieval"] = round(
        mean(float(case.get("retrieval", {}).get("retrieval_time_s", 0.0)) for case in cases), 4
    )

    for name in GENERATION_METRICS:
        for case in cases:
            left = float(case.get("pure_llm", {}).get("metrics", {}).get(name, 0.0))
            right = float(case.get("system", {}).get("metrics", {}).get(name, 0.0))
            if right > left:
                summary["wins"]["system"][name] += 1
            elif left > right:
                summary["wins"]["pure_llm"][name] += 1
            else:
                summary["wins"]["tie"][name] += 1

    return summary


def save_report(report: Dict[str, Any]) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / f"generation_comparison_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def main() -> None:
    args = parse_args()
    rows = load_jsonl_dataset(args.dataset, limit=args.limit, subject=args.subject)
    if not rows:
        raise RuntimeError("No evaluation rows loaded.")

    service = RAGService()
    if not service.ensure_ready():
        raise RuntimeError("RAG service is not ready.")
    if not service.cfg.get("api_key"):
        raise RuntimeError("API key is required for pure LLM generation comparison.")

    cases: List[Dict[str, Any]] = []
    for index, case in enumerate(rows, start=1):
        result = build_case_result(service, case)
        result["index"] = index
        cases.append(result)
        print(
            f"[{index}/{len(rows)}] "
            f"rel(system/pure)={result['system']['metrics']['answer_relevancy']:.4f}/{result['pure_llm']['metrics']['answer_relevancy']:.4f} "
            f"mode={result['system']['mode']}"
        )

    summary = summarize_cases(cases)
    report = {
        "generated_at": datetime.now().isoformat(),
        "dataset": str(Path(args.dataset).resolve()),
        "limit": args.limit,
        "subject": args.subject,
        "comparison_mode": "same_retrieved_context",
        "cases": cases,
        "summary": summary,
    }
    output_path = save_report(report)

    print("=" * 72)
    print("Generation comparison completed")
    print("=" * 72)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"report: {output_path}")


if __name__ == "__main__":
    main()
