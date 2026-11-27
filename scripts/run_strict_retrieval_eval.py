import argparse
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


from core.logger import logger
from rag.evaluation import load_jsonl_dataset
from rag.metrics import (
    calculate_context_precision,
    calculate_context_recall,
    calculate_hit_rate,
    calculate_mrr,
)
from rag.vector_store import VectorStoreService


DEFAULT_DATASET = ROOT_DIR / "data" / "rag_testcases_all_565.jsonl"
OUTPUT_DIR = ROOT_DIR / "data" / "eval_results"
METHOD_LABELS = {
    "vector": "单一向量检索",
    "keyword": "单一关键词检索",
    "hybrid_rrf": "混合检索+RRF",
}
METRIC_KEYS = ("context_precision", "context_recall", "hit_rate", "mrr")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run strict retrieval evaluation with Context Precision / Context Recall / Hit Rate / MRR."
    )
    parser.add_argument("--dataset", type=str, default=str(DEFAULT_DATASET))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--subject", type=str, default="")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument(
        "--scope",
        type=str,
        default="both",
        choices=("single", "multi", "both"),
        help="single: only primary source is accepted; multi: all acceptable sources are accepted.",
    )
    parser.add_argument("--ci-samples", type=int, default=3000, help="Bootstrap samples for 95% CI.")
    parser.add_argument(
        "--sig-samples",
        type=int,
        default=6000,
        help="Permutation samples for paired significance test.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible reliability stats.")
    parser.add_argument("--rebuild", action="store_true")
    return parser.parse_args()


def basename(path_text: str) -> str:
    return os.path.basename(str(path_text or "")).strip()


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
    tokens = re.findall(r"[\u4e00-\u9fff]{2,}|[A-Za-z0-9_]{3,}", str(ground_truth or "").lower())
    result: List[str] = []
    for token in tokens:
        if token in stop_words or token in result:
            continue
        result.append(token)
        if len(result) >= limit:
            break
    return result


def normalize_sources(meta: Dict[str, Any], scope: str) -> List[str]:
    primary = basename(meta.get("primary_source") or meta.get("source_file") or "")
    if scope == "single":
        return [primary] if primary else []

    acceptable: List[str] = []
    raw_sources = meta.get("acceptable_sources")
    if primary:
        acceptable.append(primary)
    if isinstance(raw_sources, list):
        for item in raw_sources:
            name = basename(item)
            if name and name not in acceptable:
                acceptable.append(name)
    return acceptable


def prepare_vector_service(reuse_existing: bool = True) -> VectorStoreService:
    service = VectorStoreService()
    service.reload_config()
    service._ensure_embeddings()
    if service.embeddings is None:
        raise RuntimeError("Embedding model is unavailable, strict retrieval evaluation cannot continue.")

    persist_dir, persist_dir_rel = service._get_persist_dirs()
    if reuse_existing and os.path.isdir(persist_dir):
        load_path = persist_dir_rel if os.path.isdir(persist_dir_rel) else persist_dir
        logger.info("Loading existing vector store from %s", load_path)
        service.vs = service.imports["FAISS"].load_local(  # type: ignore[index]
            load_path,
            service.embeddings,
            allow_dangerous_deserialization=True,
        )
        service._init_document_index()
        return service

    if not service.load_or_build():
        raise RuntimeError("Failed to load or build vector store.")
    return service


def retrieve_docs(
    service: VectorStoreService,
    method: str,
    question: str,
    subject: str,
    top_k: int,
) -> List[Any]:
    if method == "vector":
        if service.vs is None:
            return []
        return [doc for doc, _ in service.vs.similarity_search_with_score(question, k=top_k)]
    if method == "keyword":
        return [item["doc"] for item in service.keyword_search(question, top_k=top_k)]
    if method == "hybrid_rrf":
        return list(service.hybrid_search(question, top_k=top_k, subject=subject))
    raise ValueError(f"Unsupported method: {method}")


def docs_to_snippets(docs: Sequence[Any]) -> List[Dict[str, Any]]:
    snippets: List[Dict[str, Any]] = []
    for index, doc in enumerate(docs, start=1):
        metadata = getattr(doc, "metadata", {}) or {}
        snippets.append(
            {
                "index": index,
                "title": str(
                    metadata.get("chunk_title")
                    or metadata.get("title")
                    or metadata.get("file_name")
                    or f"doc-{index}"
                ),
                "content": str(getattr(doc, "page_content", "") or ""),
                "metadata": metadata,
            }
        )
    return snippets


def _create_result_bucket(method: str, scope: str, top_k: int) -> Dict[str, Any]:
    return {
        "method": method,
        "method_label": METHOD_LABELS.get(method, method),
        "scope": scope,
        "top_k": top_k,
        "_precision_sum": 0.0,
        "_recall_sum": 0.0,
        "_hit_rate_sum": 0.0,
        "_mrr_sum": 0.0,
        "_elapsed_sum": 0.0,
        "_metric_values": {name: [] for name in METRIC_KEYS},
        "details": [],
    }


def evaluate_method(
    service: VectorStoreService,
    rows: Sequence[Dict[str, Any]],
    method: str,
    top_k: int,
    scopes: Sequence[str],
) -> List[Dict[str, Any]]:
    buckets = {scope: _create_result_bucket(method, scope, top_k) for scope in scopes}

    for index, row in enumerate(rows, start=1):
        question = str(row.get("question") or row.get("query") or "").strip()
        if not question:
            continue

        meta = row.get("meta") if isinstance(row.get("meta"), dict) else {}
        subject = str(row.get("subject") or meta.get("subject") or "通用").strip() or "通用"
        ground_truth = str(row.get("ground_truth") or "").strip()
        expected_keywords = extract_expected_keywords(ground_truth) if ground_truth else []

        started = time.time()
        docs = retrieve_docs(
            service=service,
            method=method,
            question=question,
            subject=subject,
            top_k=top_k,
        )
        elapsed_ms = round((time.time() - started) * 1000, 2)
        snippets = docs_to_snippets(docs)
        contexts = [
            str(getattr(doc, "page_content", "") or "")
            for doc in docs
            if str(getattr(doc, "page_content", "") or "").strip()
        ]
        retrieved_sources = [
            basename(
                (getattr(doc, "metadata", {}) or {}).get("file_name")
                or (getattr(doc, "metadata", {}) or {}).get("file_path")
                or (getattr(doc, "metadata", {}) or {}).get("source")
            )
            for doc in docs
        ]

        for scope in scopes:
            acceptable_sources = normalize_sources(meta, scope)
            precision = calculate_context_precision(
                context_snippets=snippets,
                question=question,
                expected_keywords=expected_keywords,
                acceptable_sources=acceptable_sources,
            )
            recall = calculate_context_recall(
                context=contexts,
                expected_keywords=expected_keywords,
                ground_truth=ground_truth,
            )
            hit_rate = calculate_hit_rate(
                context_snippets=snippets,
                question=question,
                expected_keywords=expected_keywords,
                acceptable_sources=acceptable_sources,
                top_k=top_k,
            )
            reciprocal_rank = calculate_mrr(
                context_snippets=snippets,
                question=question,
                expected_keywords=expected_keywords,
                acceptable_sources=acceptable_sources,
            )

            bucket = buckets[scope]
            bucket["_precision_sum"] += precision
            bucket["_recall_sum"] += recall
            bucket["_hit_rate_sum"] += hit_rate
            bucket["_mrr_sum"] += reciprocal_rank
            bucket["_elapsed_sum"] += elapsed_ms
            bucket["_metric_values"]["context_precision"].append(float(precision))
            bucket["_metric_values"]["context_recall"].append(float(recall))
            bucket["_metric_values"]["hit_rate"].append(float(hit_rate))
            bucket["_metric_values"]["mrr"].append(float(reciprocal_rank))
            bucket["details"].append(
                {
                    "index": index,
                    "question": question,
                    "subject": subject,
                    "scope": scope,
                    "ground_truth": ground_truth,
                    "acceptable_sources": acceptable_sources,
                    "expected_keywords": expected_keywords,
                    "retrieved_sources": retrieved_sources,
                    "context_precision": round(precision, 4),
                    "context_recall": round(recall, 4),
                    "hit_rate": round(hit_rate, 4),
                    "mrr": round(reciprocal_rank, 4),
                    "elapsed_ms": elapsed_ms,
                }
            )

    results: List[Dict[str, Any]] = []
    for scope, bucket in buckets.items():
        total = len(bucket["details"])
        results.append(
            {
                "method": bucket["method"],
                "method_label": bucket["method_label"],
                "scope": scope,
                "top_k": bucket["top_k"],
                "total_cases": total,
                "context_precision": round(bucket["_precision_sum"] / total, 4) if total else 0.0,
                "context_recall": round(bucket["_recall_sum"] / total, 4) if total else 0.0,
                "hit_rate": round(bucket["_hit_rate_sum"] / total, 4) if total else 0.0,
                "mrr": round(bucket["_mrr_sum"] / total, 4) if total else 0.0,
                "avg_elapsed_ms": round(bucket["_elapsed_sum"] / total, 2) if total else 0.0,
                "metric_values": bucket["_metric_values"],
                "details": bucket["details"],
            }
        )
    return results


def _bootstrap_ci(values: Sequence[float], samples: int, rng: np.random.Generator) -> Dict[str, Any]:
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return {"mean": 0.0, "std": 0.0, "ci95": [0.0, 0.0], "samples": 0}

    mean_value = float(arr.mean())
    std_value = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
    if arr.size == 1 or samples <= 0:
        return {
            "mean": round(mean_value, 4),
            "std": round(std_value, 4),
            "ci95": [round(mean_value, 4), round(mean_value, 4)],
            "samples": int(arr.size),
        }

    idx = rng.integers(0, arr.size, size=(samples, arr.size))
    means = arr[idx].mean(axis=1)
    lower, upper = np.percentile(means, [2.5, 97.5])
    return {
        "mean": round(mean_value, 4),
        "std": round(std_value, 4),
        "ci95": [round(float(lower), 4), round(float(upper), 4)],
        "samples": int(arr.size),
    }


def _paired_permutation_test(
    a_values: Sequence[float],
    b_values: Sequence[float],
    samples: int,
    rng: np.random.Generator,
) -> Dict[str, Any]:
    a = np.asarray(list(a_values), dtype=float)
    b = np.asarray(list(b_values), dtype=float)
    size = min(a.size, b.size)
    if size == 0:
        return {
            "mean_diff": 0.0,
            "ci95_diff": [0.0, 0.0],
            "p_value": 1.0,
            "significant_0_05": False,
            "samples": 0,
        }

    diffs = a[:size] - b[:size]
    observed = float(diffs.mean())
    diff_std = float(diffs.std(ddof=1)) if size > 1 else 0.0
    effect_size = (observed / diff_std) if diff_std > 1e-12 else 0.0

    if size == 1 or samples <= 0:
        return {
            "mean_diff": round(observed, 4),
            "ci95_diff": [round(observed, 4), round(observed, 4)],
            "effect_size_dz": round(effect_size, 4),
            "p_value": 1.0,
            "significant_0_05": False,
            "samples": int(size),
        }

    idx = rng.integers(0, size, size=(samples, size))
    boot_means = diffs[idx].mean(axis=1)
    ci_low, ci_high = np.percentile(boot_means, [2.5, 97.5])

    signs = rng.choice(np.array([-1.0, 1.0]), size=(samples, size))
    perm_means = (diffs * signs).mean(axis=1)
    p_val = float((np.sum(np.abs(perm_means) >= abs(observed)) + 1) / (samples + 1))

    return {
        "mean_diff": round(observed, 4),
        "ci95_diff": [round(float(ci_low), 4), round(float(ci_high), 4)],
        "effect_size_dz": round(effect_size, 4),
        "p_value": round(p_val, 6),
        "significant_0_05": bool(p_val < 0.05),
        "samples": int(size),
    }


def attach_reliability_stats(
    results: List[Dict[str, Any]],
    ci_samples: int,
    sig_samples: int,
    seed: int,
) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)

    for item in results:
        values = item.get("metric_values") or {}
        ci_bundle: Dict[str, Any] = {}
        for metric in METRIC_KEYS:
            ci_bundle[metric] = _bootstrap_ci(values.get(metric, []), ci_samples, rng)
        item["confidence_intervals"] = ci_bundle

    by_scope: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for item in results:
        scope = str(item.get("scope") or "multi")
        method = str(item.get("method") or "")
        by_scope.setdefault(scope, {})[method] = item

    significance: Dict[str, Any] = {}
    for scope, method_map in by_scope.items():
        hybrid = method_map.get("hybrid_rrf")
        if not hybrid:
            continue
        comparisons: Dict[str, Any] = {}
        hybrid_values = hybrid.get("metric_values") or {}
        for baseline_method in ("vector", "keyword"):
            baseline = method_map.get(baseline_method)
            if not baseline:
                continue
            baseline_values = baseline.get("metric_values") or {}
            label = f"hybrid_rrf_vs_{baseline_method}"
            comparisons[label] = {}
            for metric in METRIC_KEYS:
                comparisons[label][metric] = _paired_permutation_test(
                    a_values=hybrid_values.get(metric, []),
                    b_values=baseline_values.get(metric, []),
                    samples=sig_samples,
                    rng=rng,
                )
        significance[scope] = comparisons

    for item in results:
        item.pop("metric_values", None)

    return {
        "ci_method": "bootstrap_mean_ci",
        "ci_samples": ci_samples,
        "significance_method": "paired_sign_flip_permutation_test",
        "significance_samples": sig_samples,
        "seed": seed,
        "significance_tests": significance,
    }


def build_summary(results: Sequence[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    summary: Dict[str, List[Dict[str, Any]]] = {}
    for item in results:
        scope = str(item.get("scope") or "multi")
        summary.setdefault(scope, []).append(
            {
                "method": item["method_label"],
                "context_precision": item["context_precision"],
                "context_recall": item["context_recall"],
                "hit_rate": item["hit_rate"],
                "mrr": item["mrr"],
                "context_precision_ci95": (item.get("confidence_intervals", {}).get("context_precision", {}).get("ci95") or [0.0, 0.0]),
                "context_recall_ci95": (item.get("confidence_intervals", {}).get("context_recall", {}).get("ci95") or [0.0, 0.0]),
                "hit_rate_ci95": (item.get("confidence_intervals", {}).get("hit_rate", {}).get("ci95") or [0.0, 0.0]),
                "mrr_ci95": (item.get("confidence_intervals", {}).get("mrr", {}).get("ci95") or [0.0, 0.0]),
                "avg_elapsed_ms": item["avg_elapsed_ms"],
                "total_cases": item["total_cases"],
            }
        )
    return summary


def save_report(report: Dict[str, Any]) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = OUTPUT_DIR / f"rag_retrieval_strict_eval_{timestamp}.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8-sig")
    return path


def main() -> None:
    args = parse_args()
    rows = load_jsonl_dataset(args.dataset, limit=args.limit, subject=args.subject)
    if not rows:
        raise RuntimeError("No evaluation rows were loaded from the dataset.")

    service = prepare_vector_service(reuse_existing=not args.rebuild)
    methods = ["vector", "keyword", "hybrid_rrf"]
    scopes = ["single", "multi"] if args.scope == "both" else [args.scope]

    results: List[Dict[str, Any]] = []
    for method in methods:
        results.extend(
            evaluate_method(
                service=service,
                rows=rows,
                method=method,
                top_k=args.top_k,
                scopes=scopes,
            )
        )

    reliability = attach_reliability_stats(
        results=results,
        ci_samples=max(0, args.ci_samples),
        sig_samples=max(0, args.sig_samples),
        seed=args.seed,
    )

    report = {
        "generated_at": datetime.now().isoformat(),
        "dataset": os.path.abspath(args.dataset),
        "limit": args.limit,
        "subject": args.subject,
        "top_k": args.top_k,
        "reliability": reliability,
        "definition": {
            "context_precision": "relevant retrieved chunks / total retrieved chunks",
            "context_recall": "retrieved supporting information units / all required information units",
            "hit_rate": "whether top-k contains at least one relevant chunk",
            "mrr": "reciprocal rank of the first relevant chunk",
            "scope_note": {
                "single": "Only the primary source file is treated as acceptable evidence source.",
                "multi": "All acceptable source files are treated as acceptable evidence sources.",
            },
        },
        "summary": build_summary(results),
        "results": results,
    }
    output_path = save_report(report)

    print("=" * 72)
    print("Strict retrieval evaluation finished")
    print("=" * 72)
    for scope, items in report["summary"].items():
        print(f"[{scope}]")
        print(f"{'方法':<16} {'CtxPrec':<10} {'CtxRecall':<10} {'HitRate':<10} {'MRR':<10} {'样本数':<8}")
        for item in items:
            print(
                f"{item['method']:<16} "
                f"{item['context_precision']:<10.4f} "
                f"{item['context_recall']:<10.4f} "
                f"{item['hit_rate']:<10.4f} "
                f"{item['mrr']:<10.4f} "
                f"{item['total_cases']:<8}"
            )
        print("-" * 72)
        print("95% CI:")
        for item in items:
            cp_ci = item.get("context_precision_ci95", [0.0, 0.0])
            cr_ci = item.get("context_recall_ci95", [0.0, 0.0])
            hr_ci = item.get("hit_rate_ci95", [0.0, 0.0])
            mrr_ci = item.get("mrr_ci95", [0.0, 0.0])
            print(
                f"{item['method']:<16} "
                f"CP[{cp_ci[0]:.4f},{cp_ci[1]:.4f}] "
                f"CR[{cr_ci[0]:.4f},{cr_ci[1]:.4f}] "
                f"HR[{hr_ci[0]:.4f},{hr_ci[1]:.4f}] "
                f"MRR[{mrr_ci[0]:.4f},{mrr_ci[1]:.4f}]"
            )
        scope_tests = reliability.get("significance_tests", {}).get(scope, {})
        if scope_tests:
            print("Significance (paired permutation, hybrid vs baseline):")
            for cmp_name, metric_tests in scope_tests.items():
                print(f"  {cmp_name}:")
                for metric in METRIC_KEYS:
                    stat = metric_tests.get(metric, {})
                    print(
                        "    "
                        f"{metric}: diff={float(stat.get('mean_diff', 0.0)):+.4f}, "
                        f"95%CI=[{float((stat.get('ci95_diff') or [0.0, 0.0])[0]):.4f},"
                        f"{float((stat.get('ci95_diff') or [0.0, 0.0])[1]):.4f}], "
                        f"p={float(stat.get('p_value', 1.0)):.6f}"
                    )
        print("-" * 72)
    print(f"Saved report: {output_path}")


if __name__ == "__main__":
    main()
