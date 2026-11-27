import json
import itertools
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List


ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"

INTENT_PATH = DATA_DIR / "intent_testcases.jsonl"
COMPLEX_PATH = DATA_DIR / "complex_task_testcases.jsonl"
RAG_PATH = DATA_DIR / "rag_testcases.jsonl"
RAG_ADDITIONAL_PATH = DATA_DIR / "rag_testcases_additional.jsonl"

INTENT_TARGETS = {
    "plan": 120,
    "progress": 110,
    "qa": 150,
    "quiz": 110,
    "resource": 120,
    "review": 110,
}

COMPLEX_SCENARIO_TARGET = 25
RAG_SUBJECT_TARGET = 70


def load_jsonl(path: Path) -> List[dict]:
    rows: List[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def append_unique_rows(path: Path, rows: Iterable[dict], text_key: str) -> int:
    existing = load_jsonl(path)
    seen = {str(row.get(text_key) or "").strip() for row in existing}
    additions: List[dict] = []
    for row in rows:
        text = str(row.get(text_key) or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        additions.append(row)

    if not additions:
        return 0

    with path.open("a", encoding="utf-8", newline="\n") as f:
        if path.stat().st_size > 0:
            f.write("\n")
        for index, row in enumerate(additions):
            f.write(json.dumps(row, ensure_ascii=False))
            if index != len(additions) - 1:
                f.write("\n")
    return len(additions)


def build_intent_additions() -> List[dict]:
    existing = load_jsonl(INTENT_PATH)
    counts = Counter(row["ground_truth"] for row in existing)
    additions: List[dict] = []

    plan_subjects = [
        "Python 基础", "Java 面向对象", "MySQL 入门", "操作系统", "计算机网络", "数据结构",
        "前端基础", "Linux 命令行", "软件测试", "机器学习入门", "Spring Boot", "Golang",
        "Redis", "Docker", "C 语言", "算法刷题", "计算机组成原理", "离散数学", "数据库系统", "设计模式",
    ]
    progress_subjects = [
        "Python 函数和模块", "Java 集合", "MySQL 多表查询", "操作系统进程管理", "计算机网络 TCP/IP",
        "数据结构树和图", "前端 HTML/CSS", "Linux 文件系统", "软件测试用例设计", "机器学习线性回归",
        "Spring Boot 基础", "Golang 并发", "Redis 缓存", "Docker 镜像和容器", "C 语言指针",
        "算法动态规划", "组成原理流水线", "离散数学命题逻辑", "数据库索引", "设计模式单例与工厂",
    ]
    quiz_subjects = [
        "Python", "Java", "MySQL", "操作系统", "计算机网络", "数据结构", "前端开发", "Linux",
        "软件工程", "机器学习", "Spring Boot", "Golang", "Redis", "Docker", "C 语言", "算法",
        "计算机组成原理", "离散数学", "数据库", "设计模式",
    ]
    review_subjects = [
        "Python 基础", "Java 基础", "数据库索引", "操作系统内存管理", "计算机网络分层模型",
        "数据结构排序算法", "前端 JavaScript", "Linux 常用命令", "软件工程生命周期", "机器学习评估指标",
        "Spring Boot 核心概念", "Golang 协程", "Redis 数据结构", "Docker 常用命令", "C 语言数组与指针",
        "动态规划", "组成原理存储系统", "离散数学集合论", "SQL 查询优化", "设计模式",
    ]

    plan_templates = [
        "请帮我制定一个{duration}天的{subject}学习计划。",
        "我想在{duration}天内系统学习{subject}，请帮我安排每天的任务。",
        "麻烦给我做一个{subject}的{duration}天复习计划。",
        "为了期末复习，请帮我规划一份{duration}天的{subject}学习路线。",
    ]
    progress_templates = [
        "我今天学完了{subject}，帮我记录一下进度。",
        "更新一下我的学习进度：{subject}这部分已经完成了。",
        "帮我登记一下，{subject}我已经看完了。",
        "我刚复习完{subject}，请更新我的学习进度。",
    ]
    quiz_templates = [
        "给我出5道关于{subject}的练习题。",
        "来一套{subject}基础测验题，5道就行。",
        "请围绕{subject}出几道选择题让我练习。",
        "帮我生成一组{subject}自测题。",
    ]
    review_templates = [
        "帮我整理一下{subject}的复习提纲。",
        "我想巩固{subject}，请给我列出复习重点。",
        "请帮我回顾一下{subject}这部分的核心知识点。",
        "把{subject}帮我梳理成一份考前复习清单。",
    ]
    durations = [7, 10, 14, 21, 30]

    generators = {
        "plan": (
            counts["plan"],
            INTENT_TARGETS["plan"],
            (
                {"query": template.format(duration=duration, subject=subject), "ground_truth": "plan"}
                for subject, duration, template in itertools.product(plan_subjects, durations, plan_templates)
            ),
        ),
        "progress": (
            counts["progress"],
            INTENT_TARGETS["progress"],
            (
                {"query": template.format(subject=subject), "ground_truth": "progress"}
                for subject, template in itertools.product(progress_subjects, progress_templates)
            ),
        ),
        "quiz": (
            counts["quiz"],
            INTENT_TARGETS["quiz"],
            (
                {"query": template.format(subject=subject), "ground_truth": "quiz"}
                for subject, template in itertools.product(quiz_subjects, quiz_templates)
            ),
        ),
        "review": (
            counts["review"],
            INTENT_TARGETS["review"],
            (
                {"query": template.format(subject=subject), "ground_truth": "review"}
                for subject, template in itertools.product(review_subjects, review_templates)
            ),
        ),
    }

    existing_queries = {row["query"] for row in existing}
    for label, (current, target, source_rows) in generators.items():
        needed = max(0, target - current)
        if needed == 0:
            continue
        collected = 0
        for row in source_rows:
            query = row["query"]
            if query in existing_queries:
                continue
            existing_queries.add(query)
            additions.append(row)
            collected += 1
            if collected >= needed:
                break
        if collected < needed:
            raise RuntimeError(f"{label} 样本不足，目标缺口 {needed}，仅生成 {collected}")

    return additions


def build_complex_additions() -> List[dict]:
    existing = load_jsonl(COMPLEX_PATH)
    counts = Counter(row["meta"]["scenario"] for row in existing)
    existing_queries = {row["query"] for row in existing}
    additions: List[dict] = []

    subject_rows = [
        ("Python", "列表推导式和函数封装", "beginner", 10),
        ("Java", "集合框架和泛型", "intermediate", 14),
        ("MySQL", "索引和多表查询", "beginner", 12),
        ("操作系统", "进程线程与内存管理", "intermediate", 14),
        ("计算机网络", "TCP/IP 与拥塞控制", "intermediate", 10),
        ("数据结构", "树、堆和图", "beginner", 15),
        ("前端开发", "HTML、CSS 和 JavaScript", "beginner", 14),
        ("Linux", "命令行与文件权限", "beginner", 10),
        ("软件工程", "需求、设计与测试", "intermediate", 12),
        ("机器学习", "监督学习基础", "intermediate", 15),
        ("Spring Boot", "IOC、AOP 与 Web 开发", "intermediate", 14),
        ("Golang", "协程、channel 和工程实践", "intermediate", 12),
        ("Redis", "数据结构和缓存场景", "intermediate", 10),
        ("Docker", "镜像、容器与部署", "beginner", 9),
        ("计算机组成原理", "指令执行和存储层次", "intermediate", 13),
        ("离散数学", "命题逻辑和图论", "beginner", 11),
        ("设计模式", "单例、工厂与策略", "intermediate", 10),
        ("算法", "贪心、二分和动态规划", "intermediate", 15),
    ]

    scenario_specs = {
        "plan_resource": {
            "expected_subtasks": ["plan", "resource"],
            "expected_agents": ["Coordinator Agent", "Service Agent"],
            "templates": [
                "帮我制定一个{days}天的{subject}学习计划，并推荐适合复习{focus}的资料。",
                "我想在{days}天内补一下{subject}，请先做计划，再给我推荐关于{focus}的学习资源。",
                "请围绕{subject}帮我排一份{days}天路线，同时推荐几份讲{focus}的资料。",
            ],
        },
        "qa_quiz": {
            "expected_subtasks": ["qa", "quiz"],
            "expected_agents": ["Coordinator Agent", "QA Agent", "Service Agent"],
            "templates": [
                "先解释一下{subject}里的{focus}，然后出3道题检验我是否理解。",
                "请先讲清楚{focus}在{subject}中的作用，再给我出几道练习题。",
                "我总分不清{subject}的{focus}，先解释，再来一组小测题。",
            ],
        },
        "progress_review": {
            "expected_subtasks": ["progress", "review"],
            "expected_agents": ["Coordinator Agent", "Service Agent"],
            "templates": [
                "我已经学完了{subject}里的{focus}，帮我更新进度，并安排下一轮复习重点。",
                "请记录一下，我这几天把{subject}的{focus}看完了，再给我整理复习清单。",
                "我刚复习完{subject}的{focus}，先更新学习进度，再告诉我后续怎么复习。",
            ],
        },
        "plan_quiz": {
            "expected_subtasks": ["plan", "quiz"],
            "expected_agents": ["Coordinator Agent", "Service Agent"],
            "templates": [
                "帮我做一个{days}天的{subject}复习计划，再顺便出几道关于{focus}的题让我热身。",
                "我想在{days}天内补完{subject}，请先排计划，然后给我来一组{focus}练习题。",
                "请规划一个{days}天的{subject}学习安排，最后再出题测我对{focus}的掌握情况。",
            ],
        },
        "qa_resource": {
            "expected_subtasks": ["qa", "resource"],
            "expected_agents": ["Coordinator Agent", "QA Agent", "Service Agent"],
            "templates": [
                "解释一下{subject}中的{focus}，再推荐两份适合继续深入的资料。",
                "先给我讲清楚{focus}是什么，然后推荐一些学习{subject}的资源。",
                "我想弄懂{subject}的{focus}，请先解释，再给我配几份资料。",
            ],
        },
        "plan_resource_quiz": {
            "expected_subtasks": ["plan", "resource", "quiz"],
            "expected_agents": ["Coordinator Agent", "Service Agent"],
            "templates": [
                "请帮我制定一个{days}天的{subject}学习计划，推荐资料，并出3道关于{focus}的练习题。",
                "我想系统复习{subject}，请先排{days}天计划，再推荐资源，最后出题检测{focus}。",
                "围绕{subject}给我做一份{days}天安排，顺带推荐讲{focus}的资料，并给我小测一下。",
            ],
        },
        "qa_review": {
            "expected_subtasks": ["qa", "review"],
            "expected_agents": ["Coordinator Agent", "QA Agent", "Service Agent"],
            "templates": [
                "请解释一下{subject}里的{focus}，然后帮我整理成复习提纲。",
                "先讲清楚{focus}，再把{subject}这部分的复习重点列给我。",
                "我想复盘{subject}的{focus}，先解释，再给我一份考前回顾清单。",
            ],
        },
        "progress_resource": {
            "expected_subtasks": ["progress", "resource"],
            "expected_agents": ["Coordinator Agent", "Service Agent"],
            "templates": [
                "我已经学完了{subject}的{focus}，帮我看看当前进度，并推荐下一步资料。",
                "请记录一下{subject}里{focus}这部分我已经完成了，再推荐后续学习资源。",
                "我刚把{subject}的{focus}过完，先更新进度，再告诉我接下来该看什么资料。",
            ],
        },
    }

    for scenario, spec in scenario_specs.items():
        needed = max(0, COMPLEX_SCENARIO_TARGET - counts[scenario])
        if needed == 0:
            continue
        collected = 0
        for subject, focus, difficulty, days in subject_rows:
            for template in spec["templates"]:
                query = template.format(subject=subject, focus=focus, days=days)
                if query in existing_queries:
                    continue
                existing_queries.add(query)
                additions.append(
                    {
                        "query": query,
                        "expected_subtasks": spec["expected_subtasks"],
                        "expected_agents": spec["expected_agents"],
                        "meta": {
                            "subject": subject,
                            "scenario": scenario,
                            "difficulty": difficulty,
                        },
                    }
                )
                collected += 1
                if collected >= needed:
                    break
            if collected >= needed:
                break
        if collected < needed:
            raise RuntimeError(f"{scenario} 样本不足，目标缺口 {needed}，仅生成 {collected}")

    return additions


def strip_question_suffix(question: str) -> str:
    return question.rstrip("？?。.!！ ")


def build_rag_additions() -> List[dict]:
    base_rows = load_jsonl(RAG_PATH)
    additional_rows = load_jsonl(RAG_ADDITIONAL_PATH)
    all_rows = base_rows + additional_rows
    counts = Counter(row["meta"]["subject"] for row in all_rows)
    existing_questions = {row["question"] for row in all_rows}
    additions: List[dict] = []

    by_subject: Dict[str, List[dict]] = {}
    for row in all_rows:
        by_subject.setdefault(row["meta"]["subject"], []).append(row)

    question_wrappers = [
        "请简要说明：{stem}。",
        "如果让我快速复习，{stem}？",
        "能不能用更精炼的方式回答：{stem}？",
        "请用学习笔记口吻概括一下：{stem}。",
        "面向初学者，{stem}？",
        "请把这个问题讲得更清楚一些：{stem}。",
        "我在复习时看到这个问题：{stem}，请概括回答。",
        "请围绕这个知识点做个简明解释：{stem}。",
        "把它当成考前速记题来回答：{stem}。",
        "请从核心概念角度回答：{stem}。",
        "如果只保留重点，{stem}？",
        "请给出一版更适合入门复习的回答：{stem}。",
    ]

    type_cycle = ["definition", "summary", "compare", "steps", "usage", "principle", "benefit", "review"]

    for subject in sorted(by_subject):
        needed = max(0, RAG_SUBJECT_TARGET - counts[subject])
        if needed == 0:
            continue
        collected = 0
        for row in by_subject[subject]:
            stem = strip_question_suffix(row["question"])
            for index, wrapper in enumerate(question_wrappers):
                question = wrapper.format(stem=stem)
                if question in existing_questions:
                    continue
                existing_questions.add(question)
                additions.append(
                    {
                        "question": question,
                        "ground_truth": row["ground_truth"],
                        "meta": {
                            "subject": subject,
                            "source_file": row["meta"].get("source_file", "综合知识库.md"),
                            "type": type_cycle[index % len(type_cycle)],
                        },
                    }
                )
                collected += 1
                if collected >= needed:
                    break
            if collected >= needed:
                break
        if collected < needed:
            raise RuntimeError(f"{subject} RAG 样本不足，目标缺口 {needed}，仅生成 {collected}")

    return additions


def summarize_counts() -> None:
    intent_counts = Counter(row["ground_truth"] for row in load_jsonl(INTENT_PATH))
    complex_counts = Counter(row["meta"]["scenario"] for row in load_jsonl(COMPLEX_PATH))
    rag_counts = Counter(
        row["meta"]["subject"] for row in (load_jsonl(RAG_PATH) + load_jsonl(RAG_ADDITIONAL_PATH))
    )

    print("intent_testcases.jsonl")
    for label, count in sorted(intent_counts.items()):
        print(f"  {label}: {count}")
    print(f"  total: {sum(intent_counts.values())}")

    print("\ncomplex_task_testcases.jsonl")
    for label, count in sorted(complex_counts.items()):
        print(f"  {label}: {count}")
    print(f"  total: {sum(complex_counts.values())}")

    print("\nrag_testcases + rag_testcases_additional")
    for label, count in sorted(rag_counts.items()):
        print(f"  {label}: {count}")
    print(f"  total: {sum(rag_counts.values())}")


def main() -> None:
    intent_added = append_unique_rows(INTENT_PATH, build_intent_additions(), "query")
    complex_added = append_unique_rows(COMPLEX_PATH, build_complex_additions(), "query")
    rag_added = append_unique_rows(RAG_ADDITIONAL_PATH, build_rag_additions(), "question")

    print(f"新增 intent 样本: {intent_added}")
    print(f"新增 complex 样本: {complex_added}")
    print(f"新增 rag 样本: {rag_added}")
    print()
    summarize_counts()


if __name__ == "__main__":
    main()
