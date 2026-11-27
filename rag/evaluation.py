"""Synchronous RAG evaluation helpers built on the project's unified metrics."""

from __future__ import annotations

import csv
import json
import os
import re
from datetime import datetime
from statistics import mean
from typing import Any, Dict, List, Optional, Sequence

from core.logger import logger
from rag.metrics import calculate_answer_quality
from rag.rag_chain import get_rag_helper


METRIC_NAMES = [
    "context_precision",
    "context_recall",
    "hit_rate",
    "mrr",
    "faithfulness",
    "answer_relevancy",
    "answer_correctness",
]
RETRIEVAL_METRIC_NAMES = METRIC_NAMES[:4]
GENERATION_METRIC_NAMES = METRIC_NAMES[4:]


def load_jsonl_dataset(path: str, limit: int = 0, subject: str = "") -> List[Dict[str, Any]]:
    """Load a JSONL RAG dataset and optionally filter by subject."""
    cases: List[Dict[str, Any]] = []
    abs_path = os.path.abspath(path)
    if not os.path.exists(abs_path):
        return cases

    with open(abs_path, "r", encoding="utf-8-sig") as file:
        for line in file:
            line = str(line or "").strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(item, dict):
                continue
            meta = item.get("meta") if isinstance(item.get("meta"), dict) else {}
            item_subject = str(item.get("subject") or meta.get("subject") or "").strip()
            if subject and item_subject != subject:
                continue
            cases.append(item)
            if limit > 0 and len(cases) >= limit:
                break
    return cases


class RAGEvaluator:
    def __init__(self, rag_helper: Optional[Any] = None):
        self.rag = rag_helper or get_rag_helper()

    def prepare_test_dataset(self, test_cases: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Run the RAG pipeline and capture rich evaluation inputs for each case."""
        dataset: List[Dict[str, Any]] = []
        logger.info("Preparing evaluation dataset with %s cases", len(test_cases))

        for case in test_cases:
            question = str(case.get("question") or case.get("query") or "").strip()
            if not question:
                continue

            meta = case.get("meta") if isinstance(case.get("meta"), dict) else {}
            subject = str(case.get("subject") or meta.get("subject") or "通用").strip() or "通用"
            ground_truth = str(case.get("ground_truth") or "").strip()
            acceptable_sources = self._normalize_sources(case)

            details = self.rag.ask_with_details(question, subject=subject, chat_history="") or {}
            snippets = details.get("context_snippets") or []
            if not isinstance(snippets, list):
                snippets = []
            contexts = [
                str(item.get("content") or "")
                for item in snippets
                if isinstance(item, dict) and item.get("content")
            ]
            matched_files = self._match_source_files(snippets, acceptable_sources)

            dataset.append(
                {
                    "question": question,
                    "subject": subject,
                    "meta": meta,
                    "answer": str(details.get("answer") or ""),
                    "ground_truth": ground_truth,
                    "contexts": contexts,
                    "context_snippets": snippets,
                    "acceptable_sources": acceptable_sources,
                    "matched_files": matched_files,
                    "response_time": self._to_float(details.get("response_time")),
                    "confidence": self._to_float(details.get("confidence")),
                    "mode": str(details.get("mode") or ""),
                    "fallback": bool(details.get("fallback")),
                }
            )

        return dataset

    def evaluate(
        self,
        dataset: Sequence[Dict[str, Any]],
        output_path: str = "data/evaluation_results.csv",
        summary_path: str = "",
    ) -> Dict[str, Any]:
        """Evaluate a prepared dataset and export both row-level and summary artifacts."""
        rows: List[Dict[str, Any]] = []

        for item in dataset:
            question = str(item.get("question") or "").strip()
            if not question:
                continue

            subject = str(item.get("subject") or "通用").strip() or "通用"
            answer = str(item.get("answer") or "")
            ground_truth = str(item.get("ground_truth") or "")
            contexts = [str(context or "") for context in (item.get("contexts") or []) if str(context or "").strip()]
            snippets = list(item.get("context_snippets") or [])
            acceptable_sources = list(item.get("acceptable_sources") or [])
            expected_keywords = self._extract_expected_keywords(ground_truth) if ground_truth else []
            metrics = calculate_answer_quality(
                answer=answer,
                context=contexts,
                question=question,
                expected_keywords=expected_keywords,
                expected_aspects=[],
                ground_truth=ground_truth,
                context_snippets=snippets,
                acceptable_sources=acceptable_sources,
            )

            matched_files = item.get("matched_files")
            if not isinstance(matched_files, list):
                matched_files = self._match_source_files(snippets, acceptable_sources)

            row = {
                "question": question,
                "subject": subject,
                "mode": str(item.get("mode") or ""),
                "fallback": bool(item.get("fallback")),
                "response_time": round(self._to_float(item.get("response_time")), 4),
                "confidence": round(self._to_float(item.get("confidence")), 4),
                "context_count": len(contexts),
                "matched_files": "|".join(str(name or "") for name in matched_files if str(name or "").strip()),
                "acceptable_sources": "|".join(
                    str(name or "") for name in acceptable_sources if str(name or "").strip()
                ),
                "answer": answer,
                "ground_truth": ground_truth,
                **{name: round(self._to_float(metrics.get(name)), 4) for name in METRIC_NAMES},
            }
            rows.append(row)

        summary = self._summarize(rows)
        resolved_output_path = os.path.abspath(output_path)
        resolved_summary_path = (
            os.path.abspath(summary_path) if summary_path else self._summary_path_for(resolved_output_path)
        )
        self._write_csv(rows, resolved_output_path)
        self._write_summary_json(summary, resolved_summary_path, resolved_output_path)
        logger.info(
            "Evaluation finished. CSV saved to %s, summary saved to %s",
            resolved_output_path,
            resolved_summary_path,
        )
        return {
            "rows": rows,
            "summary": summary,
            "output_path": resolved_output_path,
            "summary_path": resolved_summary_path,
        }

    def evaluate_test_cases(
        self,
        test_cases: Sequence[Dict[str, Any]],
        output_path: str = "data/evaluation_results.csv",
        summary_path: str = "",
    ) -> Dict[str, Any]:
        """Prepare test cases and evaluate them in one step."""
        dataset = self.prepare_test_dataset(test_cases)
        return self.evaluate(dataset, output_path=output_path, summary_path=summary_path)

    def _extract_expected_keywords(self, ground_truth: str, limit: int = 10) -> List[str]:
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
        tokens = re.findall(r"[\u4e00-\u9fff]{2,}|[a-zA-Z0-9_]{3,}", str(ground_truth or "").lower())
        result: List[str] = []
        for token in tokens:
            if token in stop_words or token in result:
                continue
            result.append(token)
            if len(result) >= limit:
                break
        return result

    def _normalize_sources(self, case: Dict[str, Any]) -> List[str]:
        meta = case.get("meta") if isinstance(case.get("meta"), dict) else {}
        acceptable_raw = case.get("acceptable_sources") or meta.get("acceptable_sources") or []
        primary = case.get("primary_source") or meta.get("primary_source") or meta.get("source_file") or ""

        sources: List[str] = []
        primary_name = os.path.basename(str(primary or "")).strip()
        if primary_name:
            sources.append(primary_name)

        if isinstance(acceptable_raw, list):
            for item in acceptable_raw:
                name = os.path.basename(str(item or "")).strip()
                if name and name not in sources:
                    sources.append(name)
        return sources

    def _match_source_files(
        self,
        snippets: Sequence[Dict[str, Any]],
        acceptable_sources: Sequence[str],
    ) -> List[str]:
        acceptable = {
            os.path.basename(str(item or "")).strip()
            for item in acceptable_sources
            if str(item or "").strip()
        }
        matched: List[str] = []
        for snippet in snippets:
            metadata = snippet.get("metadata") if isinstance(snippet, dict) else {}
            if not isinstance(metadata, dict):
                continue
            source = os.path.basename(
                str(metadata.get("file_name") or metadata.get("file_path") or metadata.get("source") or "")
            ).strip()
            if source and source in acceptable and source not in matched:
                matched.append(source)
        return matched

    def _summarize(self, rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        if not rows:
            return {
                "generated_at": datetime.now().isoformat(),
                "total_cases": 0,
                "avg_response_time": 0.0,
                "avg_confidence": 0.0,
                "fallback_rate": 0.0,
                "source_match_rate": 0.0,
                "retrieval_metrics": {name: 0.0 for name in RETRIEVAL_METRIC_NAMES},
                "generation_metrics": {name: 0.0 for name in GENERATION_METRIC_NAMES},
                "avg_metrics": {name: 0.0 for name in METRIC_NAMES},
                "subjects": {},
            }

        avg_metrics = {
            name: round(sum(self._to_float(row.get(name)) for row in rows) / len(rows), 4)
            for name in METRIC_NAMES
        }
        response_times = [
            self._to_float(row.get("response_time"))
            for row in rows
            if self._to_float(row.get("response_time")) > 0
        ]
        confidence_scores = [
            self._to_float(row.get("confidence"))
            for row in rows
            if self._to_float(row.get("confidence")) > 0
        ]

        subject_groups: Dict[str, List[Dict[str, Any]]] = {}
        for row in rows:
            subject = str(row.get("subject") or "unknown").strip() or "unknown"
            subject_groups.setdefault(subject, []).append(row)

        subject_summary: Dict[str, Dict[str, Any]] = {}
        for subject, items in sorted(subject_groups.items()):
            subject_response_times = [
                self._to_float(item.get("response_time"))
                for item in items
                if self._to_float(item.get("response_time")) > 0
            ]
            subject_confidence_scores = [
                self._to_float(item.get("confidence"))
                for item in items
                if self._to_float(item.get("confidence")) > 0
            ]
            subject_summary[subject] = {
                "total_cases": len(items),
                "avg_response_time": round(mean(subject_response_times), 4) if subject_response_times else 0.0,
                "avg_confidence": round(mean(subject_confidence_scores), 4) if subject_confidence_scores else 0.0,
                "avg_metrics": {
                    metric_name: round(
                        sum(self._to_float(item.get(metric_name)) for item in items) / len(items),
                        4,
                    )
                    for metric_name in METRIC_NAMES
                },
            }

        return {
            "generated_at": datetime.now().isoformat(),
            "total_cases": len(rows),
            "avg_response_time": round(mean(response_times), 4) if response_times else 0.0,
            "avg_confidence": round(mean(confidence_scores), 4) if confidence_scores else 0.0,
            "fallback_rate": round(sum(1 for row in rows if bool(row.get("fallback"))) / len(rows), 4),
            "source_match_rate": round(
                sum(1 for row in rows if str(row.get("matched_files") or "").strip()) / len(rows),
                4,
            ),
            "retrieval_metrics": {name: avg_metrics[name] for name in RETRIEVAL_METRIC_NAMES},
            "generation_metrics": {name: avg_metrics[name] for name in GENERATION_METRIC_NAMES},
            "avg_metrics": avg_metrics,
            "subjects": subject_summary,
        }

    def _write_csv(self, rows: Sequence[Dict[str, Any]], output_path: str) -> None:
        directory = os.path.dirname(output_path)
        if directory:
            os.makedirs(directory, exist_ok=True)

        fieldnames = [
            "question",
            "subject",
            "mode",
            "fallback",
            "response_time",
            "confidence",
            "context_count",
            "matched_files",
            "acceptable_sources",
            "answer",
            "ground_truth",
            *METRIC_NAMES,
        ]
        with open(output_path, "w", encoding="utf-8-sig", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow({name: row.get(name, "") for name in fieldnames})

    def _write_summary_json(self, summary: Dict[str, Any], summary_path: str, csv_path: str) -> None:
        directory = os.path.dirname(summary_path)
        if directory:
            os.makedirs(directory, exist_ok=True)

        payload = {
            "generated_at": summary.get("generated_at"),
            "output_csv_path": csv_path,
            "metric_names": METRIC_NAMES,
            "summary": summary,
        }
        with open(summary_path, "w", encoding="utf-8-sig") as file:
            json.dump(payload, file, ensure_ascii=False, indent=2)

    def _summary_path_for(self, output_path: str) -> str:
        base, _ = os.path.splitext(output_path)
        return f"{base}.summary.json"

    def _to_float(self, value: Any) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0


def run_sample_evaluation() -> Dict[str, Any]:
    evaluator = RAGEvaluator()
    test_cases = [
        {
            "question": "什么是软件开发生命周期？",
            "ground_truth": "软件开发生命周期是软件从需求、设计、开发、测试到交付与维护的完整过程。",
            "meta": {"subject": "通用"},
        },
        {
            "question": "瀑布模型和敏捷开发有什么区别？",
            "ground_truth": "瀑布模型强调线性阶段推进，敏捷开发强调迭代和快速响应变化。",
            "meta": {"subject": "通用"},
        },
    ]
    return evaluator.evaluate_test_cases(test_cases)


if __name__ == "__main__":
    run_sample_evaluation()
