# -*- coding: utf-8 -*-
"""逐条重放 Bad Case，用真实模型验证修复是否在端到端生效。

判定口径（v2）：不只看「有没有答错主题」，还要看「有没有答对」。

    BC-01  必须真正使用知识库（docs_count > 0）且有学习顺序
    BC-02  不出现本项目术语
    BC-04  正文无来源声明，且开头不出现标题回显
    BC-05  首句是定义，且不答成软技能
"""
from __future__ import annotations

import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from rag.service import RAGService  # noqa: E402

BANNED_SOURCE = "基于模型通用知识"
PROJECT_FEATURE_WORDS = ("混合检索", "动态切分")
INTERVIEW_WORDS = ("抗压能力", "沟通能力", "学习能力")
ECHO_PATTERN = re.compile(r"^[A-Za-z\u4e00-\u9fff]{1,12}是[：:]")


def ask(svc, question, subject="通用"):
    details = svc.ask_with_details(question, subject=subject, chat_history="") or {}
    return {
        "answer": str(details.get("answer") or ""),
        "mode": details.get("mode") or "",
        "docs": details.get("docs_count", 0),
    }


def check_bc01(r):
    """RAG 怎么学：必须用上知识库，且给出入门顺序。"""
    if r["docs"] <= 0:
        return False, "未使用知识库（资料 0），答案来自通用知识"
    a = r["answer"]
    ok = any(s in a for s in ("先", "第一步", "阶段", "学习路径", "步骤"))
    return ok, "已用知识库且有学习顺序" if ok else "已用知识库但未给出学习顺序"


def check_bc02(r):
    a = r["answer"]
    bad = [w for w in PROJECT_FEATURE_WORDS if w in a]
    return (not bad), ("出现本项目术语：" + "、".join(bad)) if bad else "未出现项目术语"


def check_bc04(r):
    a = r["answer"].strip()
    if BANNED_SOURCE in a:
        return False, "正文含来源声明"
    if ECHO_PATTERN.match(a):
        return False, "开头为标题回显（如「agent是：…」）"
    return True, "无重复声明、无标题回显"


def check_bc05(r):
    a = r["answer"]
    bad = [w for w in INTERVIEW_WORDS if w in a]
    if bad:
        return False, "答成软技能：" + "、".join(bad)
    first = re.split(r"[。\n]", a.strip())[0]
    if not any(k in first for k in ("指", "是一种", "是", "定义", "属于")):
        return False, "首句不是定义：" + first[:30]
    return True, "首句为定义且未答成软技能"


CASES = [
    ("BC-01 学习路径答非所问", "RAG怎么学", "RAG", check_bc01),
    ("BC-02 学习目标答成项目介绍", "我想学习agent", "agent", check_bc02),
    ("BC-04 来源提示重复", "什么是agent", "agent", check_bc04),
    ("BC-05 问 agent 答成候选人评价", "解释agent", "agent", check_bc05),
]


def main() -> int:
    svc = RAGService()
    if not svc.ensure_ready():
        print("知识库未就绪：请先运行 python tools/rebuild_knowledge_base.py")
        return 1

    print("=" * 78)
    print("Bad Case 逐条重放（真实模型，判定口径 v2）")
    print("=" * 78)

    passed = 0
    for title, question, subject, checker in CASES:
        r = ask(svc, question, subject)
        ok, reason = checker(r)
        passed += 1 if ok else 0
        print(f"\n【{title}】")
        print(f"  输入    : {question}（主题 {subject}）")
        print(f"  模式    : {r['mode']}  资料数: {r['docs']}")
        print(f"  判定    : {'通过' if ok else '未通过'} —— {reason}")
        print(f"  答案前80字: {r['answer'][:80].replace(chr(10), ' ')}")

    print("\n" + "-" * 78)
    print(f"自动判定：{passed}/{len(CASES)} 通过")
    print("-" * 78)
    print(
        "\n【BC-03 连续对话丢失任务】需手动重放，三轮依次输入：\n"
        "  1) 解释一下 agent\n"
        "  2) 制定一个学习计划      ← 期望：直接创建 agent 计划，或仅追问周期\n"
        "  3) 上面不是说了 agent    ← 期望：继续创建计划，不再解释概念\n"
        "  判定：第三轮结束是否生成了主题为 agent 的学习计划。"
    )
    return 0 if passed == len(CASES) else 2


if __name__ == "__main__":
    sys.exit(main())
