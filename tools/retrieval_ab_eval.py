"""纯向量 vs 混合检索 小样本对比评估（辅助人工判定命中）

用法：
    python tools/retrieval_ab_eval.py [--queries queries.json] [--top-k 3]

说明：
    针对术语密集类 query，对比「纯向量检索」与「关键词+语义混合检索」的 top-K 结果，
    输出两种模式的来源文件名 + 摘要，便于人工判定哪个更准。
    若 query 提供 acceptable_sources（期望命中的文件名），会自动统计命中率。

注意：
    - 需在能访问 embedding 网关的机器运行（北邮 API）；首次会自动 load_or_build 向量库。
    - 内置 10 个示例 query 是通用 CS 术语，**请按需替换为你知识库里真实存在的术语**，
      否则两种检索都可能命中为空，对比失去意义。
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag.vector_store import VectorStoreService


# 内置 10 个术语密集类 query 示例（★ 请替换为知识库里的真实术语！）
DEFAULT_QUERIES = [
    {"question": "什么是虚拟内存？", "acceptable_sources": []},
    {"question": "进程调度算法有哪些？", "acceptable_sources": []},
    {"question": "B+树索引的工作原理", "acceptable_sources": []},
    {"question": "死锁产生的四个必要条件", "acceptable_sources": []},
    {"question": "TCP 三次握手过程", "acceptable_sources": []},
    {"question": "事务的 ACID 特性", "acceptable_sources": []},
    {"question": "哈希冲突如何解决", "acceptable_sources": []},
    {"question": "卷积神经网络的核心结构", "acceptable_sources": []},
    {"question": "Transformer 的注意力机制", "acceptable_sources": []},
    {"question": "PageRank 算法思想", "acceptable_sources": []},
]


def basename_of(meta):
    # 真实文件名存在 file_name 字段；source 是类型标识(markdown_file/pdf_file)，仅兜底
    src = (meta or {}).get("file_name") or (meta or {}).get("file_path") or (meta or {}).get("source") or ""
    return os.path.basename(str(src))


def vector_topk(vs, query, top_k):
    results = vs.vs.similarity_search_with_score(query, k=top_k)
    out = []
    for doc, score in results:
        meta = getattr(doc, "metadata", {}) or {}
        out.append({
            "source": basename_of(meta),
            "score": round(float(score), 4),
            "content": (getattr(doc, "page_content", "") or "")[:120],
        })
    return out


_svc = None
def get_service():
    global _svc
    if _svc is None:
        from rag.service import get_rag_service
        _svc = get_rag_service()
        _svc.ensure_ready()
    return _svc

def hybrid_topk(svc, query, top_k):
    # 走产品真实检索管线（hybrid + 启发式 rerank + focus 优先级 + 同义词扩展）
    svc.cfg["top_k"] = top_k
    docs, _ = svc._retrieve(query, "通用", "")
    items = []
    for d in docs[:top_k]:
        meta = getattr(d, "metadata", {}) or {}
        content = str(getattr(d, "page_content", "") or "")[:120].replace("\n", " ")
        items.append({"source": basename_of(meta), "content": content})
    return items


def hit(items, acceptable):
    if not acceptable:
        return None
    acc = {os.path.basename(str(a)) for a in acceptable}
    return any(it["source"] in acc for it in items)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", default="")
    ap.add_argument("--top-k", type=int, default=3)
    args = ap.parse_args()

    queries = DEFAULT_QUERIES
    if args.queries and os.path.exists(args.queries):
        with open(args.queries, encoding="utf-8") as f:
            queries = json.load(f)

    vs = VectorStoreService()
    print("加载/构建向量库（首次需调用 embedding 网关，请耐心等待）...")
    vs.load_or_build()

    print(f"\n{'=' * 70}\n对比评估 {len(queries)} 个 query，top_k={args.top_k}\n{'=' * 70}")
    v_hit = h_hit = 0
    records = []
    for i, q in enumerate(queries, 1):
        q_text = q.get("question") or q.get("query") or ""
        acc = q.get("acceptable_sources") or []
        v_items = vector_topk(vs, q_text, args.top_k)
        h_items = hybrid_topk(get_service(), q_text, args.top_k)
        v_ok = hit(v_items, acc)
        h_ok = hit(h_items, acc)
        if v_ok:
            v_hit += 1
        if h_ok:
            h_hit += 1

        print(f"\n[{i}] {q_text}")
        print("  纯向量 top-%d:" % args.top_k)
        for it in v_items:
            print(f"    - {it['source']}  {it['content']}")
        print("  混合检索 top-%d:" % args.top_k)
        for it in h_items:
            print(f"    - {it['source']}  {it['content']}")
        if acc:
            print(f"  期望来源: {acc}  | 纯向量命中:{v_ok} 混合命中:{h_ok}")

        records.append({
            "question": q_text,
            "acceptable_sources": acc,
            "vector_topk": [r["source"] for r in v_items],
            "hybrid_topk": [r["source"] for r in h_items],
            "vector_hit": bool(v_ok),
            "hybrid_hit": bool(h_ok),
        })

    summary = {
        "total": len(queries),
        "vector_hit": v_hit,
        "hybrid_hit": h_hit,
        "vector_rate": round(v_hit / len(queries), 2) if queries else 0,
        "hybrid_rate": round(h_hit / len(queries), 2) if queries else 0,
        "details": records,
    }
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "retrieval_ab_result.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n结构化结果已写入: {out_path}")

    if any(q.get("acceptable_sources") for q in queries):
        print(f"命中率: 纯向量 {v_hit}/{len(queries)} = {v_hit / len(queries):.0%}"
              f"  | 混合 {h_hit}/{len(queries)} = {h_hit / len(queries):.0%}")
    else:
        print("\n提示: 未提供 acceptable_sources，请人工对照上方输出判定哪种检索更准，并自行统计命中数。")


if __name__ == "__main__":
    main()
