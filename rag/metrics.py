#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Metrics used by the local RAG evaluation pipeline."""

from __future__ import annotations

import os
import re
import time
from functools import lru_cache
from math import sqrt
from statistics import mean
from typing import Any, Dict, List, Optional, Sequence

from core.logger import logger


def _safe_div(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def _clamp(score: float) -> float:
    return max(0.0, min(1.0, float(score)))


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


def _token_set(text: str) -> set[str]:
    return set(_tokenize(text))


def _split_sentences(text: str) -> List[str]:
    parts = re.split(r"[。！？!?；;\n]+", str(text or ""))
    return [part.strip() for part in parts if part.strip()]


def _extract_numbers(text: str) -> List[str]:
    return re.findall(r"\d+(?:\.\d+)?", str(text or ""))


def _strip_follow_up_block(text: str) -> str:
    cleaned = re.sub(r"\[FOLLOW_UP\].*?\[/FOLLOW_UP\]", "", str(text or ""), flags=re.DOTALL | re.IGNORECASE)
    return cleaned.strip()


def _dedupe_texts(items: Sequence[str], limit: Optional[int] = None) -> List[str]:
    unique: List[str] = []
    seen = set()
    for item in items:
        text = str(item or "").strip()
        if not text:
            continue
        normalized = _normalize_text(text)
        if normalized in seen:
            continue
        seen.add(normalized)
        unique.append(text)
        if limit is not None and len(unique) >= limit:
            break
    return unique


def _lcs_length(seq1: Sequence[str], seq2: Sequence[str]) -> int:
    if not seq1 or not seq2:
        return 0
    previous = [0] * (len(seq2) + 1)
    for token1 in seq1:
        current = [0]
        for index, token2 in enumerate(seq2, start=1):
            if token1 == token2:
                current.append(previous[index - 1] + 1)
            else:
                current.append(max(previous[index], current[-1]))
        previous = current
    return previous[-1]


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
    hf_endpoint = os.getenv("HF_ENDPOINT")
    if hf_endpoint:
        os.environ["HF_ENDPOINT"] = hf_endpoint

    try:
        try:
            from langchain_huggingface import HuggingFaceEmbeddings
        except Exception:
            from langchain_community.embeddings import HuggingFaceEmbeddings  # type: ignore

        _SIMILARITY_EMBEDDER = HuggingFaceEmbeddings(model_name=model_name, model_kwargs={"local_files_only": True})
        return _SIMILARITY_EMBEDDER
    except Exception as exc:
        _SIMILARITY_EMBEDDER_FAILED = True
        logger.warning(f"Failed to initialize local similarity embedder, fallback to lexical similarity: {exc}")
        return None


@lru_cache(maxsize=4096)
def _embed_text(text: str) -> tuple[float, ...]:
    embedder = _get_similarity_embedder()
    if embedder is None:
        return tuple()
    try:
        vector = embedder.embed_query(_normalize_text(text)[:2000])
        return tuple(float(value) for value in vector)
    except Exception as exc:
        logger.warning(f"Failed to embed evaluation text, fallback to lexical similarity: {exc}")
        return tuple()


def _cosine_similarity(vector_a: Sequence[float], vector_b: Sequence[float]) -> float:
    if not vector_a or not vector_b or len(vector_a) != len(vector_b):
        return 0.0
    dot = sum(a * b for a, b in zip(vector_a, vector_b))
    norm_a = sqrt(sum(a * a for a in vector_a))
    norm_b = sqrt(sum(b * b for b in vector_b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return _clamp(dot / (norm_a * norm_b))


def _semantic_similarity(text_a: str, text_b: str) -> float:
    vector_a = _embed_text(text_a)
    vector_b = _embed_text(text_b)
    if vector_a and vector_b:
        return _cosine_similarity(vector_a, vector_b)
    tokens_a = _token_set(text_a)
    tokens_b = _token_set(text_b)
    if not tokens_a and not tokens_b:
        return 1.0
    if not tokens_a or not tokens_b:
        return 0.0
    return _clamp(len(tokens_a & tokens_b) / len(tokens_a | tokens_b))



def _prepare_context_segments(context: Sequence[str]) -> List[str]:
    segments = [str(item or "").strip() for item in context if str(item or "").strip()]
    joined = " ".join(segments).strip()
    if joined:
        segments.append(joined)
    return _dedupe_texts(segments)



def _support_score(candidate: str, context: Sequence[str]) -> float:
    candidate_text = str(candidate or "").strip()
    if not candidate_text:
        return 0.0

    candidate_normalized = _normalize_text(candidate_text)
    candidate_tokens = _token_set(candidate_text)
    candidate_numbers = _extract_numbers(candidate_text)
    best_score = 0.0

    for segment in _prepare_context_segments(context):
        segment_normalized = _normalize_text(segment)
        if not segment_normalized:
            continue

        if candidate_normalized and candidate_normalized in segment_normalized:
            return 1.0

        segment_tokens = _token_set(segment)
        lexical = _safe_div(len(candidate_tokens & segment_tokens), len(candidate_tokens)) if candidate_tokens else 0.0
        semantic = _semantic_similarity(candidate_text, segment)
        score = max(lexical, 0.6 * semantic + 0.4 * lexical)

        if candidate_numbers:
            supported_numbers = sum(1 for number in candidate_numbers if number in segment_normalized)
            number_score = _safe_div(supported_numbers, len(candidate_numbers))
            score *= 0.7 + 0.3 * number_score

        best_score = max(best_score, _clamp(score))

    return best_score



def _extract_reference_units(
    ground_truth: str,
    expected_keywords: Optional[Sequence[str]] = None,
    limit: int = 12,
) -> List[str]:
    text = str(ground_truth or "").strip()
    units: List[str] = []

    if text:
        raw_parts = re.split(r"(?:[\r\n]+|[。！？!?；;])", text)
        for part in raw_parts:
            cleaned = re.sub(r"^[\s#>*\-\d\.\)\(]+", "", part).strip()
            if len(cleaned) >= 4:
                units.append(cleaned)

    if not units and text:
        units.append(text)

    if expected_keywords:
        units.extend(str(item or "").strip() for item in expected_keywords if str(item or "").strip())

    return _dedupe_texts(units, limit=limit)


def _extract_source_name(snippet: Dict[str, Any]) -> str:
    metadata = snippet.get("metadata") if isinstance(snippet, dict) else {}
    if not isinstance(metadata, dict):
        return ""
    file_name = str(metadata.get("file_name") or "")
    file_path = str(metadata.get("file_path") or "")
    source = file_name or file_path
    return os.path.basename(source).strip()


def _snippet_relevance_score(
    snippet: Dict[str, Any],
    question: str,
    expected_keywords: Sequence[str],
    acceptable_sources: Optional[Sequence[str]] = None,
) -> float:
    content = str(snippet.get("content") or "")
    content_tokens = _token_set(content)
    if not content_tokens:
        return 0.0

    question_tokens = _token_set(question)
    keyword_tokens = {kw.lower() for kw in expected_keywords if str(kw).strip()}
    source_name = _extract_source_name(snippet)
    acceptable = {os.path.basename(str(item or "")).strip() for item in (acceptable_sources or []) if str(item).strip()}

    if source_name and source_name in acceptable:
        return 1.0

    keyword_overlap = _safe_div(len(content_tokens & keyword_tokens), len(keyword_tokens)) if keyword_tokens else 0.0
    question_overlap = _safe_div(len(content_tokens & question_tokens), len(question_tokens)) if question_tokens else 0.0
    return _clamp(0.65 * keyword_overlap + 0.35 * question_overlap)


def _relevant_ranks(
    context_snippets: Sequence[Dict[str, Any]],
    question: str,
    expected_keywords: Sequence[str],
    acceptable_sources: Optional[Sequence[str]] = None,
    threshold: float = 0.25,
) -> List[int]:
    ranks: List[int] = []
    for index, snippet in enumerate(context_snippets, start=1):
        score = _snippet_relevance_score(
            snippet=snippet,
            question=question,
            expected_keywords=expected_keywords,
            acceptable_sources=acceptable_sources,
        )
        if score >= threshold:
            ranks.append(index)
    return ranks


def calculate_success_rate(success_count: int, total_count: int) -> float:
    return _safe_div(success_count, total_count)


def calculate_average_response_time(response_times: List[float]) -> float:
    if not response_times:
        return 0.0
    return sum(response_times) / len(response_times)


def calculate_keyword_coverage(answer: str, expected_keywords: List[str]) -> float:
    if not expected_keywords:
        return 1.0
    answer_lower = _normalize_text(answer)
    found = [keyword for keyword in expected_keywords if _normalize_text(keyword) in answer_lower]
    return _clamp(_safe_div(len(found), len(expected_keywords)))


def calculate_structure_score(answer: str) -> float:
    if not answer:
        return 0.0
    indicators = [
        ("\n\n", 2.0),
        ("1.", 1.0),
        ("2.", 1.0),
        ("3.", 1.0),
        ("- ", 1.0),
        ("* ", 1.0),
        ("：", 0.5),
        (":", 0.5),
        ("\t", 0.5),
        ("    ", 0.5),
    ]
    total_weight = sum(weight for _, weight in indicators)
    score = sum(weight for marker, weight in indicators if marker in answer)
    return _clamp(_safe_div(score, total_weight))


def calculate_relevance_score(answer: str, expected_aspects: List[str]) -> float:
    if not expected_aspects:
        return 1.0
    answer_lower = _normalize_text(answer)
    matched = sum(1 for aspect in expected_aspects if _normalize_text(aspect) in answer_lower)
    return _clamp(_safe_div(matched, len(expected_aspects)))


def calculate_length_score(answer: str) -> float:
    length = len(str(answer or ""))
    if 50 <= length <= 2000:
        return 1.0
    if length < 50:
        return _clamp(_safe_div(length, 50))
    if length <= 3000:
        return 0.8
    return _clamp(0.8 * _safe_div(3000, length))


def calculate_confidence_score(
    keyword_coverage: float,
    context_relevance: float,
    length_score: float,
    structure_score: float,
) -> float:
    confidence = (
        keyword_coverage * 0.35
        + context_relevance * 0.30
        + length_score * 0.20
        + structure_score * 0.15
    )
    return _clamp(confidence)


def calculate_context_relevance(answer: str, context: List[str]) -> float:
    if not answer or not context:
        return 0.0
    context_tokens = _token_set(" ".join(context))
    answer_tokens = _token_set(answer)
    if not context_tokens:
        return 0.0
    return _clamp(_safe_div(len(answer_tokens & context_tokens), len(context_tokens)))


def calculate_context_precision(
    context_snippets: Sequence[Dict[str, Any]],
    question: str,
    expected_keywords: Sequence[str],
    acceptable_sources: Optional[Sequence[str]] = None,
) -> float:
    if not context_snippets:
        return 0.0
    relevant_count = len(
        _relevant_ranks(
            context_snippets=context_snippets,
            question=question,
            expected_keywords=expected_keywords,
            acceptable_sources=acceptable_sources,
        )
    )
    return _clamp(_safe_div(relevant_count, len(context_snippets)))


def calculate_context_recall(
    context: Sequence[str],
    expected_keywords: Sequence[str],
    ground_truth: str = "",
) -> float:
    reference_units = _extract_reference_units(
        ground_truth=ground_truth,
        expected_keywords=expected_keywords,
    )
    if not reference_units:
        return 1.0

    covered_count = sum(1 for unit in reference_units if _support_score(unit, context) >= 0.55)
    return _clamp(_safe_div(covered_count, len(reference_units)))


def calculate_hit_rate(
    context_snippets: Sequence[Dict[str, Any]],
    question: str,
    expected_keywords: Sequence[str],
    acceptable_sources: Optional[Sequence[str]] = None,
    top_k: Optional[int] = None,
) -> float:
    if not context_snippets:
        return 0.0
    snippets = list(context_snippets[:top_k]) if top_k else list(context_snippets)
    ranks = _relevant_ranks(
        context_snippets=snippets,
        question=question,
        expected_keywords=expected_keywords,
        acceptable_sources=acceptable_sources,
    )
    return 1.0 if ranks else 0.0


def calculate_mrr(
    context_snippets: Sequence[Dict[str, Any]],
    question: str,
    expected_keywords: Sequence[str],
    acceptable_sources: Optional[Sequence[str]] = None,
) -> float:
    ranks = _relevant_ranks(
        context_snippets=context_snippets,
        question=question,
        expected_keywords=expected_keywords,
        acceptable_sources=acceptable_sources,
    )
    if not ranks:
        return 0.0
    return _clamp(1.0 / min(ranks))


def calculate_faithfulness(answer: str, context: Sequence[str]) -> float:
    claims = _split_sentences(answer)
    if not claims:
        return 0.0

    support_scores = [_support_score(claim, context) for claim in claims if _token_set(claim)]
    if not support_scores:
        return 0.0
    return _clamp(mean(support_scores))


def calculate_answer_relevancy(question: str, answer: str) -> float:
    if not question or not answer:
        return 0.0
    semantic = _semantic_similarity(question, answer)
    question_tokens = _token_set(question)
    answer_tokens = _token_set(answer)
    lexical_focus = _safe_div(len(question_tokens & answer_tokens), len(question_tokens)) if question_tokens else 1.0
    return _clamp(0.7 * semantic + 0.3 * lexical_focus)


def calculate_rouge_l_f1(answer: str, ground_truth: str) -> float:
    answer_tokens = _tokenize(answer)
    truth_tokens = _tokenize(ground_truth)
    if not answer_tokens and not truth_tokens:
        return 1.0
    if not answer_tokens or not truth_tokens:
        return 0.0
    lcs = _lcs_length(answer_tokens, truth_tokens)
    precision = _safe_div(lcs, len(answer_tokens))
    recall = _safe_div(lcs, len(truth_tokens))
    if precision + recall == 0:
        return 0.0
    return _clamp(2 * precision * recall / (precision + recall))


def calculate_answer_correctness(
    answer: str,
    ground_truth: str,
    expected_keywords: Optional[Sequence[str]] = None,
) -> float:
    if not ground_truth:
        return 0.0

    answer_tokens = _token_set(answer)
    truth_tokens = _token_set(ground_truth)
    jaccard = _safe_div(len(answer_tokens & truth_tokens), len(answer_tokens | truth_tokens)) if (answer_tokens or truth_tokens) else 1.0
    rouge_l = calculate_rouge_l_f1(answer, ground_truth)
    semantic_similarity = _semantic_similarity(answer, ground_truth)
    return _clamp(0.45 * semantic_similarity + 0.35 * rouge_l + 0.20 * jaccard)


def calculate_answer_quality(
    answer: str,
    context: List[str],
    question: str,
    expected_keywords: Optional[List[str]] = None,
    expected_aspects: Optional[List[str]] = None,
    ground_truth: str = "",
    context_snippets: Optional[Sequence[Dict[str, Any]]] = None,
    acceptable_sources: Optional[Sequence[str]] = None,
) -> Dict[str, float]:
    answer = _strip_follow_up_block(answer)
    expected_keywords = list(expected_keywords or [])
    snippets = list(context_snippets or [])

    context_precision = calculate_context_precision(
        context_snippets=snippets,
        question=question,
        expected_keywords=expected_keywords,
        acceptable_sources=acceptable_sources,
    )
    context_recall = calculate_context_recall(
        context=context,
        expected_keywords=expected_keywords,
        ground_truth=ground_truth,
    )
    hit_rate = calculate_hit_rate(
        context_snippets=snippets,
        question=question,
        expected_keywords=expected_keywords,
        acceptable_sources=acceptable_sources,
    )
    mrr = calculate_mrr(
        context_snippets=snippets,
        question=question,
        expected_keywords=expected_keywords,
        acceptable_sources=acceptable_sources,
    )
    faithfulness = calculate_faithfulness(answer=answer, context=context)
    answer_relevancy = calculate_answer_relevancy(question=question, answer=answer)
    answer_correctness = calculate_answer_correctness(
        answer=answer,
        ground_truth=ground_truth,
        expected_keywords=expected_keywords,
    )

    return {
        "context_precision": context_precision,
        "context_recall": context_recall,
        "hit_rate": hit_rate,
        "mrr": mrr,
        "faithfulness": faithfulness,
        "answer_relevancy": answer_relevancy,
        "answer_correctness": answer_correctness,
    }


def calculate_recall(true_positives: int, false_negatives: int) -> float:
    return _safe_div(true_positives, true_positives + false_negatives)


def calculate_precision(true_positives: int, false_positives: int) -> float:
    return _safe_div(true_positives, true_positives + false_positives)


def calculate_f1_score(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def calculate_retrieval_quality_metrics(results: List[Dict[str, Any]]) -> Dict[str, float]:
    if not results:
        return {
            "avg_response_time": 0.0,
            "avg_confidence": 0.0,
            "avg_docs_retrieved": 0.0,
            "success_rate": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1_score": 0.0,
        }

    response_times = [float(item.get("response_time", 0.0)) for item in results]
    confidences = [float(item.get("confidence", 0.0)) for item in results]
    docs_counts = [float(item.get("docs_count", 0.0)) for item in results]
    successes = [bool(item.get("success", False)) for item in results]

    true_positives = sum(int(item.get("true_positives", 0)) for item in results)
    false_positives = sum(int(item.get("false_positives", 0)) for item in results)
    false_negatives = sum(int(item.get("false_negatives", 0)) for item in results)

    precision = calculate_precision(true_positives, false_positives)
    recall = calculate_recall(true_positives, false_negatives)
    f1_score = calculate_f1_score(precision, recall)

    return {
        "avg_response_time": calculate_average_response_time(response_times),
        "avg_confidence": calculate_average_response_time(confidences),
        "avg_docs_retrieved": calculate_average_response_time(docs_counts),
        "success_rate": calculate_success_rate(sum(successes), len(successes)),
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
    }


def calculate_accuracy_metrics(
    correct: int,
    total: int,
    true_positives: int = 0,
    false_positives: int = 0,
    false_negatives: int = 0,
) -> Dict[str, float]:
    accuracy = calculate_success_rate(correct, total)
    precision = calculate_precision(true_positives, false_positives)
    recall = calculate_recall(true_positives, false_negatives)
    f1_score = calculate_f1_score(precision, recall)
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "correct": correct,
        "total": total,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
    }


def calculate_overall_score(
    functionality: Dict[str, bool],
    intent_accuracy: float,
    retrieval_score: float,
    answer_score: float,
    performance_score: float,
) -> float:
    weights = {
        "functionality": 0.30,
        "intent_accuracy": 0.20,
        "retrieval_score": 0.25,
        "answer_score": 0.15,
        "performance_score": 0.10,
    }
    func_score = _safe_div(sum(bool(value) for value in functionality.values()), len(functionality)) if functionality else 0.0
    overall = (
        func_score * weights["functionality"]
        + intent_accuracy * weights["intent_accuracy"]
        + retrieval_score * weights["retrieval_score"]
        + answer_score * weights["answer_score"]
        + performance_score * weights["performance_score"]
    )
    return _clamp(overall)


def calculate_performance_metrics(response_times: List[float], cache_hits: int, total_queries: int) -> Dict[str, float]:
    if not response_times:
        return {
            "avg_response_time": 0.0,
            "min_response_time": 0.0,
            "max_response_time": 0.0,
            "cache_hit_rate": 0.0,
        }

    cache_hit_rate = 0.0
    if total_queries > 1:
        cache_hit_rate = cache_hits / (total_queries - 1)

    return {
        "avg_response_time": calculate_average_response_time(response_times),
        "min_response_time": min(response_times),
        "max_response_time": max(response_times),
        "cache_hit_rate": _clamp(cache_hit_rate),
    }


def generate_improvement_recommendations(results: Dict[str, Any]) -> List[str]:
    recommendations: List[str] = []

    if not results.get("llm_available", False):
        recommendations.append("LLM 未配置，建议补充模型 API 配置。")

    if not results.get("vector_store_ready", False):
        recommendations.append("向量库未就绪，建议检查索引文件和构建流程。")

    summary = results.get("summary") if isinstance(results.get("summary"), dict) else {}
    retrieval_metrics = results.get("retrieval_metrics") if isinstance(results.get("retrieval_metrics"), dict) else {}
    generation_metrics = results.get("generation_metrics") if isinstance(results.get("generation_metrics"), dict) else {}

    if not retrieval_metrics:
        retrieval_metrics = summary.get("retrieval_metrics") if isinstance(summary.get("retrieval_metrics"), dict) else {}
    if not generation_metrics:
        generation_metrics = summary.get("generation_metrics") if isinstance(summary.get("generation_metrics"), dict) else {}

    retrieval_quality = results.get("retrieval_quality") if isinstance(results.get("retrieval_quality"), dict) else {}
    answer_quality = results.get("answer_quality") if isinstance(results.get("answer_quality"), dict) else {}

    context_recall = retrieval_metrics.get("context_recall", retrieval_quality.get("recall", 0.0))
    context_precision = retrieval_metrics.get("context_precision", retrieval_quality.get("precision", 0.0))
    hit_rate = retrieval_metrics.get("hit_rate", retrieval_quality.get("success_rate", 0.0))
    mrr = retrieval_metrics.get("mrr", 0.0)

    if context_recall < 0.7:
        recommendations.append("上下文召回率偏低，建议补充知识库覆盖或优化召回策略。")
    if context_precision < 0.7:
        recommendations.append("上下文精确率偏低，建议减少噪声片段并优化重排策略。")
    if hit_rate < 0.85:
        recommendations.append("Hit Rate 偏低，建议先确保每个问题至少能命中 1 个相关证据。")
    if mrr < 0.6:
        recommendations.append("MRR 偏低，建议优化混合检索后的排序和重排逻辑。")

    faithfulness = generation_metrics.get("faithfulness", answer_quality.get("avg_faithfulness", 0.0))
    answer_relevancy = generation_metrics.get("answer_relevancy", answer_quality.get("avg_relevance_score", 0.0))
    answer_correctness = generation_metrics.get("answer_correctness", answer_quality.get("avg_answer_correctness", 0.0))

    if faithfulness < 0.8:
        recommendations.append("答案忠实性偏低，建议加强基于证据生成的约束，降低幻觉。")
    if answer_relevancy < 0.75:
        recommendations.append("答案相关性偏低，建议优化问题理解与回答聚焦能力。")
    if answer_correctness < 0.7:
        recommendations.append("答案正确性偏低，建议结合标准答案进一步优化生成质量。")

    performance = results.get("performance", {})
    if performance.get("avg_response_time", 0.0) > 1.0:
        recommendations.append("系统平均响应时间偏高，建议优化代码路径或增加缓存。")
    if performance.get("cache_hit_rate", 0.0) < 0.5 and performance.get("total_queries", 0) > 5:
        recommendations.append("缓存命中率偏低，建议调整缓存粒度与容量。")

    return recommendations


def get_score_description(score: float, metric_name: str) -> str:
    if score >= 0.9:
        return f"{metric_name}优秀（{score:.2%}）"
    if score >= 0.8:
        return f"{metric_name}良好（{score:.2%}）"
    if score >= 0.7:
        return f"{metric_name}一般（{score:.2%}）"
    if score >= 0.6:
        return f"{metric_name}及格（{score:.2%}）"
    return f"{metric_name}需要改进（{score:.2%}）"


def format_evaluation_summary(results: Dict[str, Any]) -> str:
    summary: List[str] = []
    summary.append("===== RAG 系统评测摘要 =====")
    summary.append(f"评测时间: {results.get('evaluation_time', '')}")
    summary.append("")

    summary_data = results.get("summary") if isinstance(results.get("summary"), dict) else {}
    overall_score = summary_data.get("overall_score", 0.0)
    summary.append(f"总体评分: {get_score_description(overall_score, '总体评分')}")
    summary.append("")

    summary.append("【功能完整性】")
    functionality = results.get("functionality", {})
    for key, value in functionality.items():
        summary.append(f"  {key}: {'正常' if value else '异常'}")
    summary.append("")

    retrieval_metrics = results.get("retrieval_metrics") if isinstance(results.get("retrieval_metrics"), dict) else {}
    generation_metrics = results.get("generation_metrics") if isinstance(results.get("generation_metrics"), dict) else {}
    if not retrieval_metrics:
        retrieval_metrics = summary_data.get("retrieval_metrics") if isinstance(summary_data.get("retrieval_metrics"), dict) else {}
    if not generation_metrics:
        generation_metrics = summary_data.get("generation_metrics") if isinstance(summary_data.get("generation_metrics"), dict) else {}

    retrieval_quality = results.get("retrieval_quality") if isinstance(results.get("retrieval_quality"), dict) else {}
    answer_quality = results.get("answer_quality") if isinstance(results.get("answer_quality"), dict) else {}

    summary.append("【检索质量】")
    summary.append(f"  Context Precision: {retrieval_metrics.get('context_precision', retrieval_quality.get('precision', 0.0)):.4f}")
    summary.append(f"  Context Recall: {retrieval_metrics.get('context_recall', retrieval_quality.get('recall', 0.0)):.4f}")
    summary.append(f"  Hit Rate: {retrieval_metrics.get('hit_rate', retrieval_quality.get('success_rate', 0.0)):.4f}")
    summary.append(f"  MRR: {retrieval_metrics.get('mrr', 0.0):.4f}")
    summary.append("")

    summary.append("【生成质量】")
    summary.append(f"  Faithfulness: {generation_metrics.get('faithfulness', answer_quality.get('avg_faithfulness', 0.0)):.4f}")
    summary.append(f"  Answer Relevancy: {generation_metrics.get('answer_relevancy', answer_quality.get('avg_relevance_score', 0.0)):.4f}")
    summary.append(f"  Answer Correctness: {generation_metrics.get('answer_correctness', answer_quality.get('avg_answer_correctness', 0.0)):.4f}")
    summary.append("")

    summary.append("【性能指标】")
    performance = results.get("performance", {})
    summary.append(f"  平均响应时间: {performance.get('avg_response_time', 0.0):.4f} 秒")
    summary.append(f"  最小响应时间: {performance.get('min_response_time', 0.0):.4f} 秒")
    summary.append(f"  最大响应时间: {performance.get('max_response_time', 0.0):.4f} 秒")
    summary.append(f"  缓存命中率: {performance.get('cache_hit_rate', 0.0):.2%}")
    summary.append(f"  总查询数: {performance.get('total_queries', 0)}")
    summary.append("")

    recommendations = summary_data.get("recommendations", [])
    if recommendations:
        summary.append("【改进建议】")
        for index, item in enumerate(recommendations, start=1):
            summary.append(f"  {index}. {item}")

    summary.append("==========================")
    return "\n".join(summary)


def benchmark_retrieval_performance(rag_service: Any, queries: List[str], iterations: int = 3) -> Dict[str, float]:
    response_times: List[float] = []

    for _ in range(2):
        for query in queries[:2]:
            try:
                rag_service.ask(query)
            except Exception:
                pass

    for _ in range(iterations):
        for query in queries:
            try:
                started = time.time()
                rag_service.ask(query)
                response_times.append(time.time() - started)
            except Exception:
                pass

    if not response_times:
        return {}

    total_cost = sum(response_times)
    return {
        "avg_time": total_cost / len(response_times),
        "min_time": min(response_times),
        "max_time": max(response_times),
        "total_queries": len(response_times),
        "throughput": _safe_div(len(response_times), total_cost),
    }




