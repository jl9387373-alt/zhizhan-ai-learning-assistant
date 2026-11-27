# -*- coding: utf-8 -*-
"""主题-文档对应关系：查询知识库是否覆盖某个主题。

用法：
    python tools/topic_coverage.py               # 打印全部主题清单
    python tools/topic_coverage.py RAG           # 查询某主题
    python tools/topic_coverage.py --json        # 以 JSON 输出
    python tools/topic_coverage.py --md          # 输出 markdown 表格

目的：把"库内是否有该主题"从"靠猜"变成"可查询"。
"""
from __future__ import annotations

import json
import os
import re
import sys
from typing import Dict, List, Tuple

# 主题识别规则：自上而下匹配，命中即止。顺序重要（越具体越靠前）。
TOPIC_RULES: List[Tuple[str, str]] = [
    (r"^agent|智能体", "AI Agent"),
    (r"^rag|检索增强", "RAG"),
    (r"操作系统|内存管理|文件系统", "操作系统"),
    (r"计算机网络|网络传输|^网络", "计算机网络"),
    (r"计算机组成|指令执行", "计算机组成原理"),
    (r"数据结构|算法基础", "数据结构与算法"),
    (r"数据库|sql", "数据库"),
    (r"离散数学", "离散数学"),
    (r"线性代数|概率统计", "数学基础"),
    (r"机器学习", "机器学习"),
    (r"深度学习|神经网络", "深度学习"),
    (r"软件工程|设计模式", "软件工程"),
    (r"软件测试|调试", "软件测试"),
    (r"安全|攻击", "计算机安全"),
    (r"python", "Python"),
    (r"c\+\+|cpp", "C++"),
    (r"java(?!script)", "Java"),
    (r"javascript|前端", "JavaScript/前端"),
    (r"求职|面试|简历", "求职与面试"),
    (r"高效利用ai|学习方法", "学习方法"),
    (r"产品经理|竞品|需求分析|^prd|产品需求", "产品经理"),
    (r"职场|沟通|汇报|向上管理", "职场沟通"),
    (r"计算机专业|核心课程|考点", "计算机专业课总览"),
    (r"计算机基础", "计算机基础"),
]

# 没有专门文档、但会被用户高频问到的主题：显式声明为缺口
KNOWN_GAPS = ["操作系统原理细节", "编译原理", "分布式系统", "Kubernetes", "产品数据分析"]


def classify(filename: str) -> str:
    stem = os.path.splitext(filename)[0]
    low = stem.lower()
    for pattern, topic in TOPIC_RULES:
        if re.search(pattern, low):
            return topic
    return "未归类"


def scan(root: str) -> Dict[str, List[str]]:
    doc_dir = os.path.join(root, "data", "documents")
    mapping: Dict[str, List[str]] = {}
    if not os.path.isdir(doc_dir):
        return mapping
    for name in sorted(os.listdir(doc_dir)):
        if not name.lower().endswith((".md", ".pdf", ".txt")):
            continue
        mapping.setdefault(classify(name), []).append(name)
    return mapping


def query(mapping: Dict[str, List[str]], keyword: str) -> Dict[str, object]:
    key = keyword.strip().lower()
    if not key:
        return {"topic": keyword, "covered": False, "docs": [], "note": "空主题"}
    hits: List[str] = []
    for topic, docs in mapping.items():
        if key in topic.lower():
            hits.extend(docs)
        else:
            hits.extend(d for d in docs if key in d.lower())
    return {
        "topic": keyword,
        "covered": bool(hits),
        "docs": sorted(set(hits)),
        "note": "" if hits else "知识库未覆盖该主题，应明确告知用户，而不是用相近片段作答",
    }


def main() -> int:
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    args = [a for a in sys.argv[1:]]
    as_json = "--json" in args
    as_md = "--md" in args
    args = [a for a in args if not a.startswith("--")]

    mapping = scan(root)
    if not mapping:
        print(f"未找到知识库目录: {os.path.join(root, 'data', 'documents')}", file=sys.stderr)
        return 1

    if args:
        result = query(mapping, args[0])
        if as_json:
            print(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            status = "已覆盖" if result["covered"] else "未覆盖"
            print(f"主题：{result['topic']}\n状态：{status}")
            for d in result["docs"]:
                print(f"  - {d}")
            if result["note"]:
                print(f"提示：{result['note']}")
        return 0

    total = sum(len(v) for v in mapping.values())
    if as_json:
        print(json.dumps({"total": total, "mapping": mapping}, ensure_ascii=False, indent=2))
        return 0
    if as_md:
        print("| 主题 | 文档数 | 文档 |")
        print("|---|---|---|")
        for topic in sorted(mapping):
            docs = mapping[topic]
            print(f"| {topic} | {len(docs)} | {'、'.join(docs)} |")
        return 0

    print(f"知识库文档总数：{total}")
    print(f"主题数：{len(mapping)}\n")
    for topic in sorted(mapping):
        print(f"[{topic}] {len(mapping[topic])} 篇")
        for d in mapping[topic]:
            print(f"    - {d}")
    print("\n已知缺口（用户会问、库里没有）：" + "、".join(KNOWN_GAPS))
    return 0


if __name__ == "__main__":
    sys.exit(main())
