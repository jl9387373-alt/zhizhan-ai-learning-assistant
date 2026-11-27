#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Enhance completed RAG evaluation jobs with judge-based and semantic metrics."""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Optional, Sequence


ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


from core.config import config
from rag.metrics import calculate_keyword_coverage, calculate_rouge_l_f1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Post-process a completed RAG eval job with LLM judge metrics and human-review samples."
    )
    parser.add_argument("--input", required=True, help="Path to a completed evaluation job JSON.")
    parser.add_argument(
        "--output",
        default="",
        help="Optional output JSON path. Defaults to data/rag_evals/<stem>_enhanced_<timestamp>.json",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Maximum number of cases to enhance. Use 0 for all cases in the input job.",
    )
    parser.add_argument(
        "--judge-limit",
        type=int,
        default=0,
        help="Maximum number of cases that call the LLM judge. Use 0 for all processed cases.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=24,
        help="Number of cases to include in the human review sample table.",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=20,
        help="Persist intermediate JSON progress every N processed cases.",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.2,
        help="Sleep seconds between LLM judge calls to reduce rate-limit pressure.",
    )
    return parser.parse_args()


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip().lower()


def _tokenize(text: str) -> List[str]:
    tokens: List[str] = []
    for part in re.findall(r"[\u4e00-\u9fff]+|[a-zA-Z0-9_]{2,}", _normalize_text(text)):
        if re.fullmatch(r"[\u4e00-\u9fff]+", part):
            if len(part) == 2:
                tokens.append(part)
            elif len(part) > 2:
                tokens.extend(part[index:index + 2] for index in range(len(part) - 1))
        else:
            tokens.append(part)
    return tokens


def _safe_div(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def _clamp(score: float) -> float:
    return max(0.0, min(1.0, float(score)))


def _strip_follow_up(answer: str) -> str:
    text = str(answer or "")
    text = re.sub(r"\[FOLLOW_UP\].*?\[/FOLLOW_UP\]", "", text, flags=re.DOTALL | re.IGNORECASE)
    return re.sub(r"\s+\Z", "", text).strip()


def _truncate(text: str, limit: int) -> str:
    text = re.sub(r"\s+", " ", str(text or "")).strip()
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)] + "…"


def _extract_numbers(text: str) -> List[str]:
    return re.findall(r"\d+(?:\.\d+)?", str(text or ""))


def _extract_expected_keywords(ground_truth: str, limit: int = 12) -> List[str]:
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
        "因为",
        "如果",
        "其中",
        "以及",
        "系统",
        "学习",
        "知识",
        "问题",
        "用户",
        "模型",
    }
    result: List[str] = []
    for token in _tokenize(ground_truth):
        if token in stop_words:
            continue
        if token not in result:
            result.append(token)
        if len(result) >= limit:
            break
    return result


_SIMILARITY_EMBEDDER: Any = None
_SIMILARITY_EMBEDDER_FAILED = False


def _get_similarity_embedder() -> Any:
    global _SIMILARITY_EMBEDDER
    global _SIMILARITY_EMBEDDER_FAILED

    if _SIMILARITY_EMBEDDER is not None:
        return _SIMILARITY_EMBEDDER
    if _SIMILARITY_EMBEDDER_FAILED:
        return None

    model_name = os.getenv("RAG_EVAL_SIMILARITY_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    try:
        try:
            from langchain_huggingface import HuggingFaceEmbeddings
        except Exception:
            from langchain_community.embeddings import HuggingFaceEmbeddings  # type: ignore

        _SIMILARITY_EMBEDDER = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"local_files_only": True},
        )
        return _SIMILARITY_EMBEDDER
    except Exception:
        _SIMILARITY_EMBEDDER_FAILED = True
        return None


def _cosine_similarity(vector_a: Sequence[float], vector_b: Sequence[float]) -> float:
    if not vector_a or not vector_b or len(vector_a) != len(vector_b):
        return 0.0
    dot = sum(a * b for a, b in zip(vector_a, vector_b))
    norm_a = sum(a * a for a in vector_a) ** 0.5
    norm_b = sum(b * b for b in vector_b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return _clamp(dot / (norm_a * norm_b))


def semantic_similarity(text_a: str, text_b: str) -> float:
    embedder = _get_similarity_embedder()
    if embedder is not None:
        try:
            vector_a = embedder.embed_query(_normalize_text(text_a)[:2000])
            vector_b = embedder.embed_query(_normalize_text(text_b)[:2000])
            return _cosine_similarity(vector_a, vector_b)
        except Exception:
            pass

    tokens_a = set(_tokenize(text_a))
    tokens_b = set(_tokenize(text_b))
    if not tokens_a and not tokens_b:
        return 1.0
    if not tokens_a or not tokens_b:
        return 0.0
    return _clamp(len(tokens_a & tokens_b) / len(tokens_a | tokens_b))


def semantic_correctness(answer: str, ground_truth: str, expected_keywords: Sequence[str]) -> Dict[str, float]:
    answer_clean = _strip_follow_up(answer)
    truth_clean = _strip_follow_up(ground_truth)
    semantic = semantic_similarity(answer_clean, truth_clean)
    rouge_l = calculate_rouge_l_f1(answer_clean, truth_clean)
    keyword_score = calculate_keyword_coverage(answer_clean, list(expected_keywords))

    answer_numbers = set(_extract_numbers(answer_clean))
    truth_numbers = set(_extract_numbers(truth_clean))
    if truth_numbers:
        number_consistency = _safe_div(len(answer_numbers & truth_numbers), len(truth_numbers))
    else:
        number_consistency = 1.0

    combined = _clamp(0.7 * semantic + 0.15 * keyword_score + 0.1 * rouge_l + 0.05 * number_consistency)
    return {
        "semantic_similarity": round(semantic, 4),
        "keyword_coverage_gt": round(keyword_score, 4),
        "number_consistency": round(number_consistency, 4),
        "answer_correctness_semantic": round(combined, 4),
    }


class JudgeClient:
    def __init__(self) -> None:
        config.reload()
        self.provider = str(config.get("rag.model_provider", "openai"))
        self.model_name = str(config.get("rag.model_name", "qwen3-max"))
        self.base_url = str(config.get("rag.base_url", "https://api.openai.com/v1"))
        self.api_key = str(config.get("rag.api_key", "")) or os.getenv("OPENAI_API_KEY") or os.getenv("DASHSCOPE_API_KEY")
        self.timeout = int(config.get("rag.llm_timeout", 20))
        self._llm = None

    def available(self) -> bool:
        return bool(self.api_key)

    def _build_llm(self) -> Any:
        if self._llm is not None:
            return self._llm

        if self.provider == "openai":
            from langchain_openai import ChatOpenAI

            self._llm = ChatOpenAI(
                model=self.model_name,
                openai_api_key=self.api_key,
                openai_api_base=self.base_url,
                temperature=0.0,
                max_tokens=800,
                timeout=self.timeout,
            )
            return self._llm

        if self.provider == "dashscope":
            try:
                from langchain_community.chat_models import ChatDashScope as DashLLM
            except ImportError:
                from langchain_community.chat_models.tongyi import ChatTongyi as DashLLM

            self._llm = DashLLM(
                model=self.model_name,
                dashscope_api_key=self.api_key,
                temperature=0.0,
                max_tokens=800,
                timeout=self.timeout,
            )
            return self._llm

        raise ValueError(f"Unsupported judge provider: {self.provider}")

    def evaluate(self, question: str, answer: str, ground_truth: str, contexts: Sequence[str]) -> Dict[str, Any]:
        llm = self._build_llm()
        context_text = "\n\n".join(
            f"[Context {index}]\n{_truncate(context, 900)}"
            for index, context in enumerate(contexts[:4], start=1)
        )
        prompt = (
            "你是一个严格但公平的 RAG 评测裁判。"
            "请只输出 JSON，不要输出解释性前后缀。\n"
            "评分规则：\n"
            "1. faithfulness_score：答案中的信息有多少能被给定 context 支撑，0 到 1。\n"
            "2. answer_relevancy_score：答案是否切题、是否真正回答了问题，0 到 1。\n"
            "3. unsupported_claims：列出最多 3 条没有在 context 中找到依据的重要陈述。\n"
            "4. supported_claims：列出最多 3 条被 context 支撑的重要陈述。\n"
            "5. rationale：一句简短中文结论。\n\n"
            f"Question:\n{_truncate(question, 1200)}\n\n"
            f"Reference Answer:\n{_truncate(ground_truth, 1600)}\n\n"
            f"Model Answer:\n{_truncate(_strip_follow_up(answer), 2200)}\n\n"
            f"Retrieved Context:\n{context_text or '(empty)'}\n\n"
            "请返回如下 JSON 结构："
            '{"faithfulness_score":0.0,"answer_relevancy_score":0.0,'
            '"supported_claims":[""],"unsupported_claims":[""],"rationale":""}'
        )

        response = llm.invoke(prompt)
        content = getattr(response, "content", response)
        text = content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)
        data = _parse_json_object(text)
        return {
            "faithfulness_llm": round(_clamp(float(data.get("faithfulness_score", 0.0))), 4),
            "answer_relevancy_llm": round(_clamp(float(data.get("answer_relevancy_score", 0.0))), 4),
            "supported_claims": _ensure_str_list(data.get("supported_claims")),
            "unsupported_claims": _ensure_str_list(data.get("unsupported_claims")),
            "rationale": str(data.get("rationale", "")).strip(),
        }


def _ensure_str_list(value: Any) -> List[str]:
    if not isinstance(value, list):
        return []
    result: List[str] = []
    for item in value:
        text = str(item or "").strip()
        if text:
            result.append(text)
    return result[:3]


def _parse_json_object(text: str) -> Dict[str, Any]:
    cleaned = str(text or "").strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        value = json.loads(cleaned)
        return value if isinstance(value, dict) else {}
    except Exception:
        pass

    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if not match:
        return {}
    try:
        value = json.loads(match.group(0))
        return value if isinstance(value, dict) else {}
    except Exception:
        return {}


def load_job(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Invalid job JSON: {path}")
    return data


def compute_summary(cases: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    metric_values: Dict[str, List[float]] = defaultdict(list)
    response_times: List[float] = []

    for case in cases:
        metrics = case.get("metrics") if isinstance(case.get("metrics"), dict) else {}
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                metric_values[key].append(float(value))
        response_time = case.get("response_time")
        if isinstance(response_time, (int, float)) and float(response_time) > 0:
            response_times.append(float(response_time))

    avg_metrics = {
        key: round(mean(values), 4)
        for key, values in metric_values.items()
        if values
    }
    return {
        "total_cases": len(cases),
        "avg_response_time": round(mean(response_times), 2) if response_times else 0.0,
        "avg_metrics": avg_metrics,
    }


def _case_subject(case: Dict[str, Any]) -> str:
    subject = str(case.get("subject") or "").strip()
    if subject:
        return subject
    meta = case.get("meta") if isinstance(case.get("meta"), dict) else {}
    return str(meta.get("subject") or "unknown").strip() or "unknown"


def _score_for_sampling(case: Dict[str, Any], metric_name: str) -> float:
    metrics = case.get("metrics") if isinstance(case.get("metrics"), dict) else {}
    value = metrics.get(metric_name, 0.0)
    return float(value) if isinstance(value, (int, float)) else 0.0


def _unique_extend(target: List[Dict[str, Any]], source: Iterable[Dict[str, Any]], limit: int) -> None:
    seen = {id(item) for item in target}
    for item in source:
        marker = id(item)
        if marker in seen:
            continue
        target.append(item)
        seen.add(marker)
        if len(target) >= limit:
            break


def build_human_review_sample(cases: Sequence[Dict[str, Any]], sample_size: int) -> List[Dict[str, Any]]:
    if not cases:
        return []

    ordered_low_faith = sorted(cases, key=lambda case: _score_for_sampling(case, "faithfulness_llm"))
    ordered_low_correct = sorted(cases, key=lambda case: _score_for_sampling(case, "answer_correctness_semantic"))
    ordered_disagreement = sorted(
        cases,
        key=lambda case: abs(
            _score_for_sampling(case, "faithfulness_llm") - _score_for_sampling(case, "faithfulness")
        )
        + abs(
            _score_for_sampling(case, "answer_correctness_semantic") - _score_for_sampling(case, "answer_correctness")
        ),
        reverse=True,
    )

    chosen: List[Dict[str, Any]] = []
    bucket = max(1, sample_size // 3)
    _unique_extend(chosen, ordered_low_faith, bucket)
    _unique_extend(chosen, ordered_low_correct, bucket * 2)
    _unique_extend(chosen, ordered_disagreement, bucket * 3)

    by_subject: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for case in cases:
        by_subject[_case_subject(case)].append(case)
    subjects = sorted(by_subject.keys(), key=lambda key: len(by_subject[key]), reverse=True)

    while len(chosen) < sample_size:
        progressed = False
        for subject in subjects:
            candidates = sorted(
                by_subject[subject],
                key=lambda case: (
                    _score_for_sampling(case, "faithfulness_llm")
                    + _score_for_sampling(case, "answer_correctness_semantic")
                ),
            )
            before = len(chosen)
            _unique_extend(chosen, candidates, len(chosen) + 1)
            if len(chosen) > before:
                progressed = True
            if len(chosen) >= sample_size:
                break
        if not progressed:
            break

    return chosen[:sample_size]


def write_human_review_files(cases: Sequence[Dict[str, Any]], markdown_path: Path, csv_path: Path) -> None:
    headers = [
        "id",
        "subject",
        "question",
        "answer_excerpt",
        "ground_truth_excerpt",
        "matched_files",
        "faithfulness_llm",
        "answer_correctness_semantic",
        "human_faithfulness",
        "human_correctness",
        "notes",
    ]

    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for index, case in enumerate(cases, start=1):
            metrics = case.get("metrics") if isinstance(case.get("metrics"), dict) else {}
            writer.writerow(
                {
                    "id": index,
                    "subject": _case_subject(case),
                    "question": _truncate(case.get("question", ""), 120),
                    "answer_excerpt": _truncate(_strip_follow_up(case.get("answer", "")), 200),
                    "ground_truth_excerpt": _truncate(case.get("ground_truth", ""), 160),
                    "matched_files": ", ".join(case.get("matched_files") or []),
                    "faithfulness_llm": metrics.get("faithfulness_llm", ""),
                    "answer_correctness_semantic": metrics.get("answer_correctness_semantic", ""),
                    "human_faithfulness": "",
                    "human_correctness": "",
                    "notes": "",
                }
            )

    lines = [
        "# RAG Human Review Sample",
        "",
        "| ID | Subject | Question | Faithfulness-LLM | Correctness-Semantic | Human Faithfulness | Human Correctness | Notes |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for index, case in enumerate(cases, start=1):
        metrics = case.get("metrics") if isinstance(case.get("metrics"), dict) else {}
        lines.append(
            "| {id} | {subject} | {question} | {faith} | {correct} |  |  |  |".format(
                id=index,
                subject=_truncate(_case_subject(case), 20).replace("|", "/"),
                question=_truncate(case.get("question", ""), 60).replace("|", "/"),
                faith=metrics.get("faithfulness_llm", ""),
                correct=metrics.get("answer_correctness_semantic", ""),
            )
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- `answer_excerpt`、`ground_truth_excerpt` 和 `matched_files` 已同步写入 CSV，便于人工逐条核查。",
            "- 建议人工标签使用：`1=好`、`0.5=部分成立`、`0=不成立`。",
        ]
    )
    markdown_path.write_text("\n".join(lines), encoding="utf-8")


def default_output_paths(input_path: Path) -> tuple[Path, Path, Path]:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = ROOT_DIR / "data" / "rag_evals"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{input_path.stem}_enhanced_{timestamp}.json"
    md_path = out_dir / f"human_review_sample_{timestamp}.md"
    csv_path = out_dir / f"human_review_sample_{timestamp}.csv"
    return json_path, md_path, csv_path


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    input_path = Path(args.input).resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input job not found: {input_path}")

    output_path, markdown_path, csv_path = default_output_paths(input_path)
    if args.output:
        output_path = Path(args.output).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)

    job = load_job(input_path)
    original_cases = job.get("cases")
    if not isinstance(original_cases, list):
        raise ValueError("Input job does not contain a valid cases list.")

    limit = int(args.limit)
    cases = original_cases[:limit] if limit > 0 else original_cases
    judge_limit = int(args.judge_limit) if int(args.judge_limit) > 0 else len(cases)

    judge = JudgeClient()
    judge_enabled = judge.available()

    enhanced_cases: List[Dict[str, Any]] = []
    started_at = time.time()

    print("=" * 72)
    print("Enhanced RAG evaluation started")
    print("=" * 72)
    print(f"input: {input_path}")
    print(f"output: {output_path}")
    print(f"judge_enabled: {judge_enabled}")
    print(f"judge_provider: {judge.provider}")
    print(f"judge_model: {judge.model_name}")
    print(f"cases_to_process: {len(cases)}")
    print(f"judge_limit: {judge_limit}")

    for index, raw_case in enumerate(cases, start=1):
        case = json.loads(json.dumps(raw_case, ensure_ascii=False))
        metrics = case.get("metrics") if isinstance(case.get("metrics"), dict) else {}
        case["metrics"] = metrics

        question = str(case.get("question") or "")
        answer = str(case.get("answer") or "")
        ground_truth = str(case.get("ground_truth") or "")
        contexts = [
            str(item.get("content") or "")
            for item in (case.get("context_snippets") or [])
            if isinstance(item, dict) and item.get("content")
        ]
        expected_keywords = _extract_expected_keywords(ground_truth)

        semantic_scores = semantic_correctness(
            answer=answer,
            ground_truth=ground_truth,
            expected_keywords=expected_keywords,
        )
        metrics.update(semantic_scores)

        judge_result: Dict[str, Any] = {}
        if judge_enabled and index <= judge_limit:
            try:
                judge_result = judge.evaluate(
                    question=question,
                    answer=answer,
                    ground_truth=ground_truth,
                    contexts=contexts,
                )
                metrics["faithfulness_llm"] = judge_result["faithfulness_llm"]
                metrics["answer_relevancy_llm"] = judge_result["answer_relevancy_llm"]
            except Exception as exc:
                judge_result = {"error": str(exc)}
                metrics.setdefault("faithfulness_llm", metrics.get("faithfulness", 0.0))
                metrics.setdefault("answer_relevancy_llm", metrics.get("answer_relevancy", 0.0))
            time.sleep(max(0.0, float(args.sleep)))
        else:
            metrics.setdefault("faithfulness_llm", metrics.get("faithfulness", 0.0))
            metrics.setdefault("answer_relevancy_llm", metrics.get("answer_relevancy", 0.0))

        case["judge"] = judge_result
        enhanced_cases.append(case)

        if index % max(1, int(args.save_every)) == 0 or index == len(cases):
            payload = {
                **job,
                "source_job_path": str(input_path),
                "status": "done" if index == len(cases) else "running",
                "updated_at": datetime.now().isoformat(),
                "progress": {"done": index, "total": len(cases)},
                "enhancement": {
                    "judge_enabled": judge_enabled,
                    "judge_provider": judge.provider,
                    "judge_model": judge.model_name,
                    "judge_limit": judge_limit,
                    "elapsed_seconds": round(time.time() - started_at, 2),
                },
                "summary": compute_summary(enhanced_cases),
                "cases": enhanced_cases,
            }
            save_json(output_path, payload)
            print(f"[{index}/{len(cases)}] saved -> {output_path.name}")

    final_payload = load_job(output_path)
    sample_cases = build_human_review_sample(final_payload.get("cases") or [], sample_size=max(1, int(args.sample_size)))
    write_human_review_files(sample_cases, markdown_path=markdown_path, csv_path=csv_path)

    final_payload["human_review_sample"] = {
        "markdown_path": str(markdown_path),
        "csv_path": str(csv_path),
        "sample_size": len(sample_cases),
    }
    final_payload["updated_at"] = datetime.now().isoformat()
    final_payload["summary"] = compute_summary(final_payload.get("cases") or [])
    save_json(output_path, final_payload)

    print("=" * 72)
    print("Enhanced RAG evaluation completed")
    print("=" * 72)
    print(json.dumps(final_payload.get("summary") or {}, ensure_ascii=False, indent=2))
    print(f"human_review_markdown: {markdown_path}")
    print(f"human_review_csv: {csv_path}")


if __name__ == "__main__":
    main()
