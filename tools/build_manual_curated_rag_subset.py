import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set


ROOT_DIR = Path(__file__).resolve().parent.parent
INPUT_PATHS = [
    ROOT_DIR / "data" / "rag_testcases.jsonl",
    ROOT_DIR / "data" / "rag_testcases_additional.jsonl",
]
OUTPUT_PATH = ROOT_DIR / "data" / "rag_testcases_manual_curated_100.jsonl"
TARGET_PER_SOURCE = 10


def add_all(target: Set[str], *items: str) -> None:
    for item in items:
        if item:
            target.add(item)


def curated_sources(primary: str, question: str) -> List[str]:
    q = question
    acceptable: Set[str] = {primary}

    if primary == "如何高效利用AI进行学习.md":
        if any(k in q for k in ["如何高效利用 AI 进行学习", "如何高效利用 AI 进行学习", "学习建议"]):
            add_all(acceptable, "计算机专业核心课程学习指南.md")

    elif primary == "计算机专业核心课程学习指南.md":
        if any(k in q for k in ["软件工程与实践", "软件工程"]):
            add_all(acceptable, "软件工程与常见设计模式.md")
        elif any(k in q for k in ["硬件与系统架构", "计算机系统"]):
            add_all(acceptable, "计算机组成原理重点笔记.md", "操作系统.pdf")
        elif any(k in q for k in ["数学基础"]):
            add_all(acceptable, "离散数学核心概念总结.md", "线性代数与概率统计学习要点.md")
        elif any(k in q for k in ["核心软件基础"]):
            add_all(
                acceptable,
                "操作系统.pdf",
                "数据库系统与SQL核心知识.md",
                "计算机网络核心知识总结.md",
            )
        elif any(k in q for k in ["学习建议"]):
            add_all(acceptable, "如何高效利用AI进行学习.md")

    elif primary == "离散数学核心概念总结.md":
        pass

    elif primary == "操作系统.pdf":
        if any(k in q for k in ["分页", "段式存储", "文件系统", "缓冲区"]):
            add_all(acceptable, "操作系统内存管理与文件系统专题.md")
        if any(k in q for k in ["共享内存", "管道", "调度", "临界区", "进程", "线程", "死锁"]):
            add_all(acceptable, "操作系统进程线程与调度专题.md")

    elif primary == "README.md":
        pass

    elif primary == "计算机组成原理重点笔记.md":
        if any(k in q for k in ["总线", "I/O", "CPU", "指令系统"]):
            add_all(acceptable, "计算机组成与指令执行基础.md")
        if any(k in q for k in ["计算机系统概述", "数据的表示与运算", "存储器层次结构"]):
            add_all(acceptable, "计算机基础综合知识导论.md")

    elif primary == "程序员求职面试与简历指南.md":
        pass

    elif primary == "RAG技术原理简述.md":
        pass

    elif primary == "软件工程与常见设计模式.md":
        if any(k in q for k in ["软件测试"]):
            add_all(acceptable, "软件测试与调试方法入门.md")

    elif primary == "计算机专业课核心考点摘要.md":
        if any(k in q for k in ["数据库", "Database"]):
            add_all(
                acceptable,
                "数据库系统与SQL核心知识.md",
                "数据库索引事务与优化专题.md",
                "数据库范式与ER建模基础.md",
            )
        if any(k in q for k in ["数据结构与算法", "DSA"]):
            add_all(acceptable, "数据结构与算法基础专题.md")
        if any(k in q for k in ["计算机网络", "Computer Networks"]):
            add_all(acceptable, "计算机网络核心知识总结.md", "计算机网络传输层与应用层详解.md")
        if any(k in q for k in ["操作系统", "OS"]):
            add_all(
                acceptable,
                "操作系统.pdf",
                "操作系统进程线程与调度专题.md",
                "操作系统内存管理与文件系统专题.md",
            )

    ordered = [primary]
    for item in sorted(acceptable):
        if item not in ordered:
            ordered.append(item)
    return ordered


def load_rows() -> List[Dict]:
    rows: List[Dict] = []
    global_index = 0
    for path in INPUT_PATHS:
        with path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                global_index += 1
                obj["_global_index"] = global_index
                obj["_source_path"] = path.name
                obj["_source_line"] = line_no
                rows.append(obj)
    return rows


def build_subset(rows: List[Dict]) -> List[Dict]:
    grouped: Dict[str, List[Dict]] = defaultdict(list)
    for row in rows:
        meta = row.get("meta", {}) if isinstance(row.get("meta"), dict) else {}
        primary = str(meta.get("primary_source") or meta.get("source_file") or "").strip()
        grouped[primary].append(row)

    subset: List[Dict] = []
    for primary in sorted(grouped):
        chosen = grouped[primary][:TARGET_PER_SOURCE]
        for row in chosen:
            meta = row["meta"]
            primary_source = str(meta.get("primary_source") or meta.get("source_file") or "").strip()
            row["meta"]["primary_source"] = primary_source
            row["meta"]["acceptable_sources"] = curated_sources(primary_source, str(row.get("question") or ""))
            row["meta"]["acceptable_source_mode"] = "ai_assisted_manual_curated_v1"
            row["meta"]["curated_subset"] = "manual100"
            row["meta"]["original_global_index"] = row["_global_index"]
            row["meta"]["original_dataset"] = row["_source_path"]
            subset.append(row)
    return subset


def save_subset(rows: List[Dict]) -> None:
    lines: List[str] = []
    for row in rows:
        row = dict(row)
        row.pop("_global_index", None)
        row.pop("_source_path", None)
        row.pop("_source_line", None)
        lines.append(json.dumps(row, ensure_ascii=False))
    OUTPUT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    rows = load_rows()
    subset = build_subset(rows)
    save_subset(subset)
    print(f"saved: {OUTPUT_PATH}")
    print(f"rows: {len(subset)}")


if __name__ == "__main__":
    main()
