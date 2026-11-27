import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Sequence


ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


from core.logger import logger
from rag.vector_store import VectorStoreService


DEFAULT_DATASETS = [
    ROOT_DIR / "data" / "rag_testcases.jsonl",
    ROOT_DIR / "data" / "rag_testcases_additional.jsonl",
]
OUTPUT_DIR = ROOT_DIR / "data" / "eval_results"


def load_jsonl(paths: Sequence[Path], limit: int = 0, subject: str = "") -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for path in paths:
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8-sig") as f:
            for line in f:
                line = (line or "").strip()
                if not line:
                    continue
                obj = json.loads(line)
                meta = obj.get("meta") if isinstance(obj.get("meta"), dict) else {}
                if subject and str(meta.get("subject") or "") != subject:
                    continue
                rows.append(obj)
                if limit > 0 and len(rows) >= limit:
                    return rows
    return rows


def basename(path_text: str) -> str:
    return os.path.basename(str(path_text or "")).strip()


def doc_basename(doc: Any) -> str:
    metadata = getattr(doc, "metadata", {}) or {}
    return basename(metadata.get("file_name") or metadata.get("file_path") or metadata.get("source"))


def normalize_target_sources(meta: Dict[str, Any]) -> tuple[str, List[str]]:
    primary_source = basename(meta.get("primary_source") or meta.get("source_file") or "")

    acceptable_sources: List[str] = []
    acceptable_raw = meta.get("acceptable_sources")
    if isinstance(acceptable_raw, list):
        for item in acceptable_raw:
            source_name = basename(item)
            if source_name and source_name not in acceptable_sources:
                acceptable_sources.append(source_name)

    if primary_source and primary_source not in acceptable_sources:
        acceptable_sources.insert(0, primary_source)

    return primary_source, acceptable_sources


def dedupe_docs_by_source(docs: Sequence[Any], top_k: int) -> List[Any]:
    unique_docs: List[Any] = []
    seen_sources = set()
    for doc in docs:
        source_name = doc_basename(doc)
        source_key = source_name or f"__empty__:{len(unique_docs)}"
        if source_key in seen_sources:
            continue
        seen_sources.add(source_key)
        unique_docs.append(doc)
        if len(unique_docs) >= top_k:
            break
    return unique_docs


def precision_at_k(relevant_ranks: Sequence[int], retrieved_count: int) -> float:
    if retrieved_count <= 0:
        return 0.0
    return len(relevant_ranks) / retrieved_count


def recall_at_k(relevant_ranks: Sequence[int]) -> float:
    return 1.0 if relevant_ranks else 0.0


def f1_score(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def mrr(relevant_ranks: Sequence[int]) -> float:
    if not relevant_ranks:
        return 0.0
    return 1.0 / min(relevant_ranks)


def hit_at_k(relevant_ranks: Sequence[int], k: int) -> float:
    return 1.0 if any(rank <= k for rank in relevant_ranks) else 0.0


def prepare_vector_service(reuse_existing: bool = True) -> VectorStoreService:
    service = VectorStoreService()
    service.reload_config()
    service._ensure_embeddings()
    if service.embeddings is None:
        raise RuntimeError("向量嵌入模型未就绪，无法执行向量/混合检索评测")

    persist_dir, persist_dir_rel = service._get_persist_dirs()
    if reuse_existing and os.path.isdir(persist_dir):
        load_path = persist_dir_rel if os.path.isdir(persist_dir_rel) else persist_dir
        logger.info(f"直接加载现有向量库: {load_path}")
        service.vs = service.imports["FAISS"].load_local(  # type: ignore[index]
            load_path,
            service.embeddings,
            allow_dangerous_deserialization=True,
        )
        service._init_document_index()
        return service

    if not service.load_or_build():
        raise RuntimeError("向量库加载或构建失败")
    return service


def retrieve_docs(
    service: VectorStoreService,
    method: str,
    question: str,
    subject: str,
    top_k: int,
    unique_source: bool,
    candidate_multiplier: int,
) -> List[Any]:
    raw_k = max(top_k, top_k * max(candidate_multiplier, 1)) if unique_source else top_k

    if method == "vector":
        if service.vs is None:
            return []
        results = service.vs.similarity_search_with_score(question, k=raw_k)
        docs = [doc for doc, _ in results]
        return dedupe_docs_by_source(docs, top_k) if unique_source else docs[:top_k]
    if method == "keyword":
        docs = [item["doc"] for item in service.keyword_search(question, top_k=raw_k)]
        return dedupe_docs_by_source(docs, top_k) if unique_source else docs[:top_k]
    if method == "hybrid_rrf":
        docs = service.hybrid_search(question, top_k=raw_k, subject=subject)
        return dedupe_docs_by_source(docs, top_k) if unique_source else docs[:top_k]
    raise ValueError(f"不支持的评测方法: {method}")


def evaluate_method(
    service: VectorStoreService,
    rows: Sequence[Dict[str, Any]],
    method: str,
    top_k: int,
    unique_source: bool,
    candidate_multiplier: int,
) -> Dict[str, Any]:
    details: List[Dict[str, Any]] = []
    precision_sum = 0.0
    recall_sum = 0.0
    f1_sum = 0.0
    mrr_sum = 0.0
    hit_at_1_sum = 0.0
    hit_at_3_sum = 0.0
    hit_at_5_sum = 0.0
    hit_at_10_sum = 0.0
    rank1_hits = 0
    total_retrieved = 0
    total_elapsed = 0.0

    for index, row in enumerate(rows, 1):
        question = str(row.get("question") or "").strip()
        meta = row.get("meta") if isinstance(row.get("meta"), dict) else {}
        subject = str(meta.get("subject") or "通用")
        primary_source, acceptable_sources = normalize_target_sources(meta)
        acceptable_set = set(acceptable_sources)

        started = time.time()
        docs = retrieve_docs(
            service=service,
            method=method,
            question=question,
            subject=subject,
            top_k=top_k,
            unique_source=unique_source,
            candidate_multiplier=candidate_multiplier,
        )
        elapsed_ms = round((time.time() - started) * 1000, 2)
        total_elapsed += elapsed_ms

        retrieved_files = [doc_basename(doc) for doc in docs]
        relevant_ranks = [
            rank
            for rank, name in enumerate(retrieved_files, start=1)
            if acceptable_set and name in acceptable_set
        ]
        matched_sources = [
            name for name in retrieved_files if acceptable_set and name in acceptable_set
        ]

        precision = precision_at_k(relevant_ranks, len(docs))
        recall = recall_at_k(relevant_ranks)
        f1 = f1_score(precision, recall)
        reciprocal_rank = mrr(relevant_ranks)
        hit1 = hit_at_k(relevant_ranks, 1)
        hit3 = hit_at_k(relevant_ranks, 3)
        hit5 = hit_at_k(relevant_ranks, 5)
        hit10 = hit_at_k(relevant_ranks, 10)

        precision_sum += precision
        recall_sum += recall
        f1_sum += f1
        mrr_sum += reciprocal_rank
        hit_at_1_sum += hit1
        hit_at_3_sum += hit3
        hit_at_5_sum += hit5
        hit_at_10_sum += hit10
        rank1_hits += int(hit1)
        total_retrieved += len(docs)

        details.append(
            {
                "index": index,
                "question": question,
                "subject": subject,
                "expected_source_file": primary_source,
                "primary_source_file": primary_source,
                "acceptable_source_files": acceptable_sources,
                "retrieved_source_files": retrieved_files,
                "evaluation_mode": "unique_source" if unique_source else "raw_chunks",
                "relevant_ranks": relevant_ranks,
                "matched_source_files": matched_sources,
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1": round(f1, 4),
                "mrr": round(reciprocal_rank, 4),
                "hit_at_1": round(hit1, 4),
                "hit_at_3": round(hit3, 4),
                "hit_at_5": round(hit5, 4),
                "hit_at_10": round(hit10, 4),
                "elapsed_ms": elapsed_ms,
            }
        )

    total = len(rows)
    return {
        "method": method,
        "top_k": top_k,
        "evaluation_mode": "unique_source" if unique_source else "raw_chunks",
        "candidate_multiplier": candidate_multiplier if unique_source else 1,
        "total_cases": total,
        "precision": round(precision_sum / total, 4) if total else 0.0,
        "recall": round(recall_sum / total, 4) if total else 0.0,
        "f1_score": round(f1_sum / total, 4) if total else 0.0,
        "mrr": round(mrr_sum / total, 4) if total else 0.0,
        "hit_at_1": round(hit_at_1_sum / total, 4) if total else 0.0,
        "hit_at_3": round(hit_at_3_sum / total, 4) if total else 0.0,
        "hit_at_5": round(hit_at_5_sum / total, 4) if total else 0.0,
        "hit_at_10": round(hit_at_10_sum / total, 4) if total else 0.0,
        "rank1_hits": rank1_hits,
        "avg_retrieved_sources": round(total_retrieved / total, 2) if total else 0.0,
        "avg_elapsed_ms": round(total_elapsed / total, 2) if total else 0.0,
        "details": details,
    }


def build_summary(results: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    labels = {
        "vector": "单一向量检索",
        "keyword": "单一关键词检索",
        "hybrid_rrf": "混合检索+RRF",
    }
    for result in results:
        items.append(
            {
                "method": labels.get(result["method"], result["method"]),
                "precision": result["precision"],
                "recall": result["recall"],
                "f1_score": result["f1_score"],
                "mrr": result["mrr"],
                "hit_at_1": result["hit_at_1"],
                "hit_at_3": result["hit_at_3"],
                "hit_at_5": result["hit_at_5"],
                "hit_at_10": result["hit_at_10"],
                "rank1_hits": result["rank1_hits"],
                "avg_retrieved_sources": result["avg_retrieved_sources"],
                "avg_elapsed_ms": result["avg_elapsed_ms"],
                "total_cases": result["total_cases"],
            }
        )
    return items


def save_report(report: Dict[str, Any]) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = OUTPUT_DIR / f"rag_retrieval_eval_{timestamp}.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8-sig")
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="评测向量、关键词、混合+RRF 三种 RAG 检索策略")
    parser.add_argument("--top-k", type=int, default=10, help="每个问题取前 K 个检索结果")
    parser.add_argument("--limit", type=int, default=50, help="评测样本数，0 表示全部")
    parser.add_argument("--subject", type=str, default="", help="只评测指定主题")
    parser.add_argument(
        "--dataset",
        action="append",
        default=[],
        help="指定 JSONL 测试集路径，可重复传入；未传时使用默认两份 RAG 测试集",
    )
    parser.add_argument("--rebuild", action="store_true", help="忽略现有向量库，强制触发重建")
    parser.add_argument(
        "--raw-chunks",
        action="store_true",
        help="使用原始 chunk 级评测口径，不按 source_file 去重",
    )
    parser.add_argument(
        "--candidate-multiplier",
        type=int,
        default=5,
        help="按来源去重时，先召回 top_k * multiplier 个候选，再截取前 top_k 个唯一来源",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_paths = [Path(item) for item in args.dataset] if args.dataset else list(DEFAULT_DATASETS)
    rows = load_jsonl(dataset_paths, limit=args.limit, subject=args.subject)
    if not rows:
        raise RuntimeError("未加载到任何 RAG 测试样本")

    service = prepare_vector_service(reuse_existing=not args.rebuild)

    methods = ["vector", "keyword", "hybrid_rrf"]
    unique_source = not args.raw_chunks
    results = [
        evaluate_method(
            service,
            rows,
            method,
            args.top_k,
            unique_source=unique_source,
            candidate_multiplier=args.candidate_multiplier,
        )
        for method in methods
    ]
    summary = build_summary(results)
    report = {
        "generated_at": datetime.now().isoformat(),
        "dataset_paths": [str(path) for path in dataset_paths],
        "limit": args.limit,
        "subject": args.subject,
        "top_k": args.top_k,
        "evaluation_mode": "unique_source" if unique_source else "raw_chunks",
        "candidate_multiplier": args.candidate_multiplier if unique_source else 1,
        "summary": summary,
        "results": results,
    }
    output_path = save_report(report)

    print("=" * 72)
    print("RAG 检索质量评测完成")
    print("=" * 72)
    print(f"{'方法':<20} {'Precision':<10} {'Recall':<10} {'F1':<10} {'MRR':<10} {'样本数':<8}")
    for item in summary:
        print(
            f"{item['method']:<20} "
            f"{item['precision']:<10.4f} "
            f"{item['recall']:<10.4f} "
            f"{item['f1_score']:<10.4f} "
            f"{item['mrr']:<10.4f} "
            f"{item['total_cases']:<8}"
        )
    print("-" * 72)
    print(f"{'方法':<20} {'Hit@1':<10} {'Hit@3':<10} {'Hit@5':<10} {'Hit@10':<10} {'Rank1':<8}")
    for item in summary:
        print(
            f"{item['method']:<20} "
            f"{item['hit_at_1']:<10.4f} "
            f"{item['hit_at_3']:<10.4f} "
            f"{item['hit_at_5']:<10.4f} "
            f"{item['hit_at_10']:<10.4f} "
            f"{item['rank1_hits']:<8}"
        )
    print(f"结果文件: {output_path}")


if __name__ == "__main__":
    main()
