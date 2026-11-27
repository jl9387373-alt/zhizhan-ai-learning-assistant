import json
from pathlib import Path
from typing import Dict, Iterable, List, Set


ROOT_DIR = Path(__file__).resolve().parent.parent
DATASET_PATHS = [
    ROOT_DIR / "data" / "rag_testcases.jsonl",
    ROOT_DIR / "data" / "rag_testcases_additional.jsonl",
]


LEARNING_DOCS = {
    "如何高效利用AI进行学习.md",
    "计算机专业核心课程学习指南.md",
}
OS_DOCS = {
    "操作系统.pdf",
    "操作系统进程线程与调度专题.md",
    "操作系统内存管理与文件系统专题.md",
}
ARCH_DOCS = {
    "计算机组成原理重点笔记.md",
    "计算机组成与指令执行基础.md",
    "计算机基础综合知识导论.md",
}
SE_DOCS = {
    "软件工程与常见设计模式.md",
    "软件测试与调试方法入门.md",
}
RAG_DOCS = {"RAG技术原理简述.md"}
JOB_DOCS = {"程序员求职面试与简历指南.md"}
DISCRETE_DOCS = {"离散数学核心概念总结.md"}
README_DOCS = {"README.md"}
CORE_SUMMARY_DOCS = {
    "计算机专业课核心考点摘要.md",
    "计算机专业核心课程学习指南.md",
}
DB_DOCS = {
    "数据库系统与SQL核心知识.md",
    "数据库索引事务与优化专题.md",
    "数据库范式与ER建模基础.md",
    "计算机专业课核心考点摘要.md",
}
NETWORK_DOCS = {
    "计算机网络核心知识总结.md",
    "计算机网络传输层与应用层详解.md",
    "计算机专业课核心考点摘要.md",
}


def add_all(target: Set[str], items: Iterable[str]) -> None:
    for item in items:
        if item:
            target.add(item)


def contains_any(text: str, keywords: Iterable[str]) -> bool:
    return any(keyword in text for keyword in keywords)


def infer_acceptable_sources(question: str, meta: Dict[str, object]) -> List[str]:
    primary_source = str(meta.get("primary_source") or meta.get("source_file") or "").strip()
    subject = str(meta.get("subject") or "").strip()
    text = f"{question} {subject}"
    acceptable: Set[str] = set()

    if primary_source:
        acceptable.add(primary_source)

    if primary_source in LEARNING_DOCS:
        add_all(acceptable, LEARNING_DOCS)
    elif primary_source == "操作系统.pdf":
        add_all(acceptable, OS_DOCS)
    elif primary_source == "计算机组成原理重点笔记.md":
        add_all(acceptable, ARCH_DOCS)
    elif primary_source == "软件工程与常见设计模式.md":
        add_all(acceptable, SE_DOCS)
    elif primary_source == "RAG技术原理简述.md":
        add_all(acceptable, RAG_DOCS)
    elif primary_source == "程序员求职面试与简历指南.md":
        add_all(acceptable, JOB_DOCS)
    elif primary_source == "离散数学核心概念总结.md":
        add_all(acceptable, DISCRETE_DOCS)
    elif primary_source == "README.md":
        add_all(acceptable, README_DOCS)
    elif primary_source == "计算机专业课核心考点摘要.md":
        add_all(acceptable, CORE_SUMMARY_DOCS)

    if contains_any(text, ["操作系统", "分页", "分段", "调度", "进程", "线程", "管道", "共享内存", "死锁", "文件系统", "内存管理"]):
        add_all(acceptable, OS_DOCS)

    if contains_any(text, ["CPU", "I/O", "总线", "指令系统", "硬件", "系统架构", "数据的表示", "运算", "计算机系统概述"]):
        add_all(acceptable, ARCH_DOCS)

    if contains_any(text, ["软件工程", "设计模式", "SOLID", "软件测试", "需求", "建模", "结构型模式", "行为型模式", "创建型模式"]):
        add_all(acceptable, SE_DOCS)

    if contains_any(text, ["数据库", "SQL", "事务", "索引", "范式", "ER"]):
        add_all(acceptable, DB_DOCS)

    if contains_any(text, ["计算机网络", "TCP", "UDP", "HTTP", "HTTPS", "DNS", "传输层", "应用层", "Socket"]):
        add_all(acceptable, NETWORK_DOCS)

    if contains_any(text, ["学习", "主动思考", "间隔重复", "学习建议", "AI 进行学习", "AI进行学习"]):
        add_all(acceptable, LEARNING_DOCS)

    if contains_any(text, ["RAG", "检索增强", "混合检索", "向量检索", "关键词检索", "RRF", "核心流程"]):
        add_all(acceptable, RAG_DOCS)

    if contains_any(text, ["面试", "简历", "软技能", "求职", "推荐练习资源"]):
        add_all(acceptable, JOB_DOCS)

    if contains_any(text, ["离散数学", "集合", "函数", "图论", "数理逻辑", "命题逻辑"]):
        add_all(acceptable, DISCRETE_DOCS)

    if contains_any(text, ["知识库文档存放指南", "导入网页", "支持的文件格式", "提示"]):
        add_all(acceptable, README_DOCS)

    ordered: List[str] = []
    if primary_source:
        ordered.append(primary_source)
    for item in sorted(acceptable):
        if item not in ordered:
            ordered.append(item)
    return ordered


def upgrade_dataset(path: Path) -> int:
    updated_rows: List[str] = []
    count = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            meta = obj.get("meta") if isinstance(obj.get("meta"), dict) else {}
            primary_source = str(meta.get("primary_source") or meta.get("source_file") or "").strip()
            acceptable_sources = infer_acceptable_sources(str(obj.get("question") or ""), meta)
            meta["primary_source"] = primary_source
            meta["acceptable_sources"] = acceptable_sources
            meta["acceptable_source_mode"] = "rule_enriched_v1"
            obj["meta"] = meta
            updated_rows.append(json.dumps(obj, ensure_ascii=False))
            count += 1

    path.write_text("\n".join(updated_rows) + "\n", encoding="utf-8")
    return count


def main() -> None:
    total = 0
    for path in DATASET_PATHS:
        total += upgrade_dataset(path)
        print(f"updated: {path} -> {path.name}")
    print(f"total_rows: {total}")


if __name__ == "__main__":
    main()
