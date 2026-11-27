from __future__ import annotations

import os
import re
import sys
import threading
from typing import Dict, List, Optional, Sequence, Tuple


SUBJECT_KEYWORDS: Dict[str, List[str]] = {
    "python": ["python", "py", "廖雪峰python", "装饰器", "列表推导式", "生成器"],
    "java": ["java", "jvm", "jre", "jdk", "集合框架", "泛型"],
    "javascript": ["javascript", "js", "前端", "es6", "html", "css", "react", "vue"],
    "机器学习": ["机器学习", "ml", "machine learning", "梯度下降", "监督学习", "回归"],
    "深度学习": ["深度学习", "dl", "卷积神经网络", "cnn", "rnn", "transformer"],
    "人工智能": ["人工智能", "ai"],
    "agent": ["agent", "智能体", "ai agent", "autonomous agent"],
    "数据库": ["数据库", "sql", "mysql", "postgres", "oracle", "索引", "事务", "多表查询"],
    "数据结构": ["数据结构", "算法", "链表", "栈", "队列", "树", "图", "bfs", "dfs"],
    "动态规划": ["动态规划", "状态转移", "最优子结构"],
    "操作系统": ["操作系统", "os", "进程", "线程", "内存管理", "虚拟内存", "死锁", "linux 内核"],
    "网络": ["计算机网络", "网络", "tcp", "ip", "http", "https", "dns", "tcp/ip", "拥塞控制"],
    "Linux": ["linux", "shell", "命令行", "权限管理", "管道"],
    "软件测试": ["软件测试", "测试用例", "黑盒", "白盒", "边界值", "等价类", "自动化测试"],
    "产品经理": ["产品经理", "产品岗", "产品实习", "需求分析", "竞品分析", "用户调研", "prd", "产品原型"],
}

# 明确主题后缀/前缀：用于从"为我制定RAG学习计划""学完经济学"等句子中抽取自由主题
_TOPIC_STOPWORDS = {
    "学习计划", "复习计划", "制定", "创建", "为我", "帮我", "给我", "一个", "一份",
    "学完", "完成", "已经", "的", "了", "部分", "章节", "检索", "基础", "学习", "复习",
    "记录", "更新", "进度", "计划", "安排", "内容", "东西", "知识点",
}

# 计划请求里的修饰词/时间词：它们描述"计划本身"，不是学习主题
# （截图 bug："为我制定一个详细计划"被当成主题"详细"；"帮我制定本周计划"被当成"本周"）
_TOPIC_MODIFIERS = {
    "详细", "简单", "完整", "系统", "具体", "全面", "精简", "快速", "高效", "深入",
    "本周", "这周", "下周", "下个月", "本月", "今天", "明天", "近期", "最近", "一份", "一个",
}

# 自由主题候选里的泛化词：抽出来也不是一个可学习的主题
_TOPIC_GENERIC_CANDIDATES = {
    "学过的内容", "学过的", "学过", "最近学过", "最近学的", "上面说的", "刚才说的",
}

# 面试主题：从"我想面试产品经理实习生"里抽取"岗位 + 面试"，交给面试准备计划生成器
_INTERVIEW_ROLE_STOP_PREFIXES = (
    "怎么", "如何", "什么", "哪些", "需要", "应该", "注意", "常见", "准备", "打算",
    "流程", "技巧", "经验", "问题", "题目", "官", "穿", "自我", "中", "时", "后",
    "前", "的", "过程", "环节", "现场", "电话", "视频", "轮", "面", "我", "你", "想", "要", "去",
)
_INTERVIEW_ROLE_STOP_SUFFIXES = ("面试", "应聘", "求职", "经验", "技巧", "准备", "流程", "问题", "机会")

# 进度溯源追问：这类问题必须读真实业务数据，不能交给知识库RAG
PROGRESS_INQUIRY_PATTERNS: Sequence[str] = (
    r"根据什么.{0,6}(更新|算|记录)",
    r"依据什么.{0,6}(更新|算|记录)",
    r"怎么算出来的",
    r"如何计算.{0,4}进度",
    r"进度.{0,6}(怎么算|怎么来的|依据|根据)",
    r"剩[下余].{0,6}(什么|哪些|多少|任务|%|％|进度)",
    r"还[有剩].{0,6}(什么|哪些|多少|任务|没学|没完成)",
    r"哪[些个].{0,4}(任务|没学|未完成|没完成)",
    r"剩下.{0,4}\d+\s*[%％]",
    r"\d+\s*[%％].{0,6}(是什么|怎么来|哪些)",
    r"(为什么|为啥).{0,6}(\d+\s*[%％]|这个进度)",
    r"进度.{0,4}依据",
)


def is_progress_inquiry(query: str) -> bool:
    """判断是否为进度溯源追问（根据什么更新/剩下的是什么等）。"""
    text = (query or "").strip()
    if not text:
        return False
    return any(re.search(p, text, re.IGNORECASE) for p in PROGRESS_INQUIRY_PATTERNS)


def _clean_interview_role(raw: str) -> str:
    role = (raw or "").strip(" 　,，。.!！?？的")
    role = re.sub(
        r"^(?:请|麻烦)?\s*(?:帮我|给我|为我|我想|我要|想|要|准备|打算|计划)?\s*"
        r"(?:制定|做|创建|生成|安排)?\s*(?:一份|一个)?\s*(?:关于)?\s*",
        "", role,
    ).strip(" 　,，。.!！?？的")
    for suffix in _INTERVIEW_ROLE_STOP_SUFFIXES:
        if role.endswith(suffix) and len(role) > len(suffix):
            role = role[: -len(suffix)]
    return role.strip(" 　的")


def extract_interview_subject(query: str) -> Optional[str]:
    """从"我想面试产品经理实习生""产品经理面试怎么准备""帮我制定产品经理面试计划"中
    抽取面试主题，返回"岗位 + 面试/求职"；不是面试意图时返回 None。"""
    text = (query or "").strip()
    if not text or not re.search(r"面试|应聘|求职", text):
        return None

    candidates = []
    # "帮我制定产品经理实习生面试的4周学习计划"
    match = re.search(
        r"([\u4e00-\u9fa5A-Za-z]{2,16}?)\s*(?:面试|应聘|求职)\s*(?:的)?\s*"
        r"(?:\d+\s*(?:天|周|个月|月))?\s*(?:学习|复习)?\s*(?:准备)?\s*(?:计划|路线|规划)",
        text,
    )
    if match:
        candidates.append(match.group(1))
    # "我想面试产品经理实习生"
    match = re.search(r"(?:面试|应聘|求职)\s*([\u4e00-\u9fa5A-Za-z]{2,14})", text)
    if match:
        candidates.append(match.group(1))
    # "产品经理面试怎么准备"
    match = re.search(
        r"([\u4e00-\u9fa5A-Za-z]{2,10})\s*(?:面试|应聘|求职)\s*(?:$|[，,。；;！!？?]|怎么|如何|准备|流程|经验|技巧|问题|题|该|要|需要)",
        text,
    )
    if match:
        candidates.append(match.group(1))

    for raw in candidates:
        role = _clean_interview_role(re.split(r"[，,。；;！!？?\s]", raw)[0])
        if len(role) < 2 or len(role) > 12:
            continue
        if role.startswith(_INTERVIEW_ROLE_STOP_PREFIXES):
            continue
        if any(tok in role for tok in ("什么", "怎么", "如何")):
            continue
        suffix = "求职" if ("求职" in text and "面试" not in text) else "面试"
        return f"{role}{suffix}"
    return None


def _clean_free_subject(candidate: str) -> Optional[str]:
    """清洗自由主题候选：剥掉"本周/详细/一份"等修饰，过滤泛化词。"""
    candidate = candidate.strip(" 　,，。.!！?？的")
    # 反复剥离前缀修饰词："本周的学习计划"→"学习计划"→""
    changed = True
    while changed and candidate:
        changed = False
        for modifier in _TOPIC_MODIFIERS:
            if candidate.startswith(modifier) and len(candidate) > len(modifier):
                candidate = candidate[len(modifier):].strip(" 　,，。.!！?？的")
                changed = True
                break
    candidate = re.sub(r"(?:的)?\s*(?:学习|复习)(?:计划|路线|规划)\s*$", "", candidate).strip()
    candidate = re.sub(r"(?:的)?\s*(?:计划|路线)\s*$", "", candidate).strip()
    candidate = re.sub(r"\d+\s*(天|周|个月|月)\s*$", "", candidate).strip()
    candidate = candidate.strip(" 　,，。.!！?？的")
    if not candidate or len(candidate) < 2 or len(candidate) > 24:
        return None
    if candidate in _TOPIC_STOPWORDS or candidate in _TOPIC_MODIFIERS:
        return None
    if candidate in _TOPIC_GENERIC_CANDIDATES:
        return None
    # 候选里残留“制定/创建/生成/安排”说明抽到的是指令，不是主题
    if any(verb in candidate for verb in ("制定", "创建", "生成", "安排")):
        return None
    return candidate


def extract_free_subject(query: str) -> Optional[str]:
    """从明确措辞中抽取固定关键词表之外的主题，如 RAG、经济学。
    只认"制定/学完/学习 + 主题"这类明确结构，避免把普通问句误判为主题。"""
    text = (query or "").strip()
    if not text:
        return None

    interview_subject = extract_interview_subject(text)
    if interview_subject:
        return interview_subject

    patterns = (
        # "我想学习X，为我制定一个学习路线/计划" —— 主题在指令之前，必须优先匹配
        r"(?:我想|我要|我想学|我要学)\s*(?:学习|学一下|了解一下)?\s*(.+?)\s*[,，。;；]\s*(?:为|帮|给)?\s*(?:我)?\s*(?:制定|做|创建|生成|安排)\s*(?:一份|一个)?\s*(?:\d+\s*(?:天|周|个月|月))?\s*(?:的)?\s*(?:学习|复习)?\s*(?:计划|路线|规划)",
        r"(?:为|帮|给)\s*(?:我)?\s*(?:制定|做|创建|生成|安排)\s*(?:一份|一个)?\s*(?:\d+\s*(?:天|周|个月|月))?\s*(.+?)\s*(?:的)?\s*(?:学习|复习)?\s*(?:计划|路线|规划)",
        r"制定\s*(?:一份|一个)?\s*(?:\d+\s*(?:天|周|个月|月))?\s*(.+?)\s*(?:的)?\s*(?:学习|复习)?\s*(?:计划|路线|规划)",
        # 查询式："有没有RAG的学习计划""是否有经济学的学习路线"
        r"(?:有没有|是否有|有么|\w+有没有)\s*(?:一份|一个)?\s*(?:\d+\s*(?:天|周|个月|月))?\s*(.+?)\s*(?:的)?\s*(?:学习|复习)?\s*(?:计划|路线|规划)",
        # "agent 的学习计划"：主题 + 的 + 计划，直接取主题
        r"^(?:我想|我要|帮我|给我|为我)?\s*(.+?)\s*的\s*(?:学习|复习)?\s*(?:计划|路线|规划)\s*[。.!！]?\s*$",
        # "我想学X/复习X" —— 不再匹配裸"我想…"；"'我想学习'"必须排在"'我想学'"前面
        r"(?:我想学习|我要学习|我想学|我要学|想学习|想学|学习|复习|学完|学完了|已完成|完成)\s*(?:了)?\s*(.+?)\s*(?:的)?\s*(?:学习|复习|部分|章节|内容|检索)?\s*$",
        r"(.+?)\s*的\s*(?:检索|基础|章节|部分)(?:内容|部分)?\s*(?:我)?\s*(?:已经)?\s*(?:学完|完成)",
        # "主题 + 疑问"结构："rag怎么学""经济学是什么" —— 连续对话里最常出现
        r"(.+?)\s*(?:怎么学|如何学|怎么理解|是什么|是什么意思|有什么用|怎么用)\s*[?？]?\s*$",
        # 主题在问句开头、后面还有补充要求时，先保留明确主题。
        # 例："RAG怎么学？请给我一个具体的入门顺序。"
        r"^\s*(.+?)\s*(?:怎么学|如何学|怎么理解|是什么|是什么意思|有什么用|怎么用)\s*[?？。.!！]\s*(?:请|能否|可以|帮我|再|另外).+",
        # "解释/介绍 + 主题"："解释一下agent""解释 Agent""讲讲RAG"
        r"^(?:请)?\s*(?:解释一下|解释|介绍一下|介绍|说明一下|说明|讲讲|讲一下|说说|聊一下|聊聊)\s*(.+?)\s*[?？。.]?\s*$",
        # "什么是 + 主题"（补足上面模式未覆盖的写法）
        r"^什么是\s*(.+?)\s*[?？。.]?\s*$",
        # 回答式主题回指："上面不是说了agent""刚提到的agent""你说的agent"
        # 必须含明确的回指动词，避免把"制定一个学习计划"之类误抽成主题
        r"^(?:上面|刚才|刚刚|前面|之前)?\s*(?:不是)?\s*(?:说的|说了|提到|讲的是|指的是|指)\s*(?:的是)?\s*(.+?)\s*(?:吧|吗|呢)?\s*[?？。.]?\s*$",
    )
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if not match:
            continue
        candidate = match.group(1).strip(" 　,，。.!！?？的")
        candidate = re.sub(
            r"^(我想|我要|我想学|我要学|帮我|为我|给我|学习|学|复习|学完|完成|已经|解释一下|解释|介绍|讲讲|说说|说的|说了|提到|指的是|指的是)\s*",
            "", candidate,
        ).strip()
        # 关键清洗：候选里若混入"为我制定/帮我制定/制定...计划"等指令成分，截断到指令之前，
        # 避免把"智能体，为我制定一个学习路线"整句当成主题（截图 bug）
        candidate = re.split(r"[,，;；]?\s*(?:为|帮|给)\s*(?:我)?\s*(?:制定|做|创建|生成|安排)|[,，;；]\s*制定", candidate)[0]
        candidate = _clean_free_subject(candidate)
        if not candidate:
            continue
        # 若候选只是通用问句成分，视为无明确主题
        if any(tok in candidate for tok in ("什么", "怎么", "为什么", "多少", "进度", "吗")):
            continue
        return candidate
    return None


VALID_INTENTS = {"plan", "qa", "resource", "progress", "quiz", "review"}

_INTENT_SYSTEM_PROMPT = """你是一个学习助手系统的意图分类器。
用户会发送一条中文消息，你需要判断它属于以下哪个意图，只输出标签本身：
plan, qa, resource, progress, quiz, review

规则：
1. 只能输出一个标签，不要解释。
2. 如果无法确定，输出 qa。
"""

_llm_client = None


def _get_llm_client():
    global _llm_client
    if _llm_client is not None:
        return _llm_client

    try:
        root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if root_dir not in sys.path:
            sys.path.insert(0, root_dir)

        from core.config import config

        api_key = str(config.get("rag.api_key", "")) or os.getenv("OPENAI_API_KEY", "")
        base_url = str(config.get("rag.base_url", "https://api.openai.com/v1"))
        model = str(config.get("rag.model_name", "gpt-3.5-turbo"))
        if not api_key:
            return None

        try:
            from openai import OpenAI

            client = OpenAI(api_key=api_key, base_url=base_url)
            client._intent_model = model
            _llm_client = client
            return _llm_client
        except ImportError:
            pass

        try:
            from langchain_openai import ChatOpenAI

            client = ChatOpenAI(
                model=model,
                api_key=api_key,
                base_url=base_url,
                temperature=0,
                max_tokens=10,
            )
            client._intent_model = model
            client._is_langchain = True
            _llm_client = client
            return _llm_client
        except ImportError:
            pass
    except Exception:
        return None

    return None


def detect_intent_llm(query: str, timeout: float = 5.0) -> Optional[str]:
    if not query or not query.strip():
        return None

    client = _get_llm_client()
    if client is None:
        return None

    model = getattr(client, "_intent_model", "gpt-3.5-turbo")
    is_langchain = getattr(client, "_is_langchain", False)
    result_box = [None]
    error_box = [None]

    def _call() -> None:
        try:
            if is_langchain:
                from langchain_core.messages import HumanMessage, SystemMessage

                resp = client.invoke(
                    [
                        SystemMessage(content=_INTENT_SYSTEM_PROMPT),
                        HumanMessage(content=query.strip()),
                    ]
                )
                result_box[0] = str(resp.content).strip().lower()
            else:
                resp = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": _INTENT_SYSTEM_PROMPT},
                        {"role": "user", "content": query.strip()},
                    ],
                    max_tokens=10,
                    temperature=0,
                )
                result_box[0] = str(resp.choices[0].message.content).strip().lower()
        except Exception as exc:
            error_box[0] = exc

    thread = threading.Thread(target=_call, daemon=True)
    thread.start()
    thread.join(timeout=timeout)

    if thread.is_alive() or error_box[0] is not None:
        return None

    raw = (result_box[0] or "").strip().lower()
    token = raw.split()[0] if raw.split() else raw
    if token in VALID_INTENTS:
        return token
    if token in {"question", "问答"}:
        return "qa"
    return None


def _score_patterns(query: str, patterns: Sequence[Tuple[str, float]]) -> float:
    score = 0.0
    for pattern, base_score in patterns:
        if re.search(pattern, query, re.IGNORECASE):
            hits = len(re.findall(pattern, query, re.IGNORECASE)) or 1
            score = max(score, min(0.99, base_score + 0.02 * (hits - 1)))
    return score


def _detect_intent_rules(query: str) -> str:
    query = (query or "").strip()
    if not query:
        return "qa"

    query_lower = query.lower()

    qa_keywords = ["解释一下", "什么是", "什么叫做", "什么意思", "定义", "为什么", "如何", "怎样", "怎么", "？", "?"]
    if any(keyword in query_lower for keyword in qa_keywords):
        if not any(token in query_lower for token in ["计划", "规划", "安排", "推荐", "资料", "测验", "复习", "进度"]):
            return "qa"

    intent_patterns: Dict[str, Sequence[Tuple[str, float]]] = {
        "plan": [
            (r"制定.*计划|创建.*计划|学习.*规划|学习计划|复习计划", 0.97),
            (r"做一个.*计划|列.*计划|列一份.*计划", 0.94),
            (r"规划.*(天|周|个月)|安排.*(天|周|个月)", 0.93),
            (r"排.*(计划|安排|路线|每天学什么)|先.*(做|排|列).*(计划|安排)", 0.92),
            (r"\d+\s*(天|周|个月).*(计划|安排|路线)", 0.9),
            (r"(在|用).*(天|周|个月)内.*(学完|补完|复习|掌握)", 0.9),
            (r"做安排|过一遍|学习路径|学习路线|学习安排", 0.86),
        ],
        "resource": [
            (r"推荐.*(资源|资料|教程|视频|网站|课程|书|文章)|配几份资料", 0.95),
            (r"学习资料|学习资源|推荐|资料|教程|课程|网站|文章|书", 0.82),
        ],
        "progress": [
            (r"查看.*进度|当前.*进度|我的.*进度|学习.*进度", 0.95),
            (r"更新.*进度|记录.*进度|同步.*进度|帮我记录", 0.92),
            (r"学完了|学完|完成了|已经完成|已经学完|学习完成|完成学习", 0.9),
        ],
        "quiz": [
            (r"出\s*\d*\s*道.*题|出几道.*题|出题|来几道题|来一组.*题", 0.96),
            (r"练习题|测试题|测验|考我|自测|小测|判断题|小题", 0.92),
            (r"检测|检测题|掌握情况|热身|检验", 0.88),
        ],
        "review": [
            (r"安排复习|复习重点|复习提纲|复习清单|复盘|回顾|巩固", 0.95),
            (r"总结.*重点|记忆曲线|下次复习|复习", 0.86),
        ],
        "qa": [
            (r"什么是|什么叫做|什么意思|定义|解释一下|说明一下", 0.95),
            (r"为什么|如何|怎样|怎么|原理|区别|关系|作用", 0.88),
            (r"\?|？", 0.7),
        ],
    }

    scores = {intent: _score_patterns(query, patterns) for intent, patterns in intent_patterns.items()}

    if scores["plan"] >= 0.9 and scores["resource"] >= 0.9:
        return "plan"
    if scores["plan"] >= 0.9 and scores["quiz"] >= 0.88:
        return "plan"
    if scores["progress"] >= 0.9 and scores["review"] >= 0.86:
        return "progress"

    best_intent = max(scores.items(), key=lambda item: item[1])[0]
    if scores[best_intent] <= 0:
        return "qa"
    return best_intent


_LEARN_INQUIRY_MARKERS = (
    "我想学", "我要学", "想学习", "想学", "我想了解", "怎么学", "如何学", "从哪学", "从哪开始",
    "怎么入手", "如何入门", "怎么入门", "学习路线", "学习路径", "零基础", "从零开始", "入门",
)
_PLAN_REQUEST_MARKERS = ("计划", "规划", "安排", "制定", "创建", "生成", "列一份", "排一下")


def is_learning_inquiry(query: str) -> bool:
    """是否为"我想学习X / X怎么学"这类学习路径询问。
    这类请求属于问答（先给入门顺序），不是"制定计划"，只有出现明确计划措辞才交给 plan。"""
    text = (query or "").strip().lower()
    if not text:
        return False
    if any(m in text for m in _PLAN_REQUEST_MARKERS):
        return False
    return any(m in text for m in _LEARN_INQUIRY_MARKERS)


def detect_intent(query: str) -> str:
    # 学习路径询问必须先于 LLM 分类兜底，避免被误判为 plan 而直接建计划
    if is_learning_inquiry(query):
        return "qa"
    llm_result = detect_intent_llm(query)
    if llm_result:
        return llm_result
    return _detect_intent_rules(query)


def _cn_number_to_int(token: str) -> Optional[int]:
    if not token:
        return None
    if token.isdigit():
        return int(token)

    mapping = {"一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9, "十": 10}
    if token == "十":
        return 10
    if len(token) == 2 and token[0] == "十" and token[1] in mapping:
        return 10 + mapping[token[1]]
    if len(token) == 2 and token[1] == "十" and token[0] in mapping:
        return mapping[token[0]] * 10
    if len(token) == 3 and token[1] == "十" and token[0] in mapping and token[2] in mapping:
        return mapping[token[0]] * 10 + mapping[token[2]]
    if token in mapping:
        return mapping[token]
    return None


def extract_duration(query: str) -> str:
    query = (query or "").strip()
    match = re.search(r"(\d+|[一二两三四五六七八九十]+)\s*(天|周|个月|月)", query)
    if not match:
        return "4周"

    value = _cn_number_to_int(match.group(1)) or 4
    unit = match.group(2)

    if unit == "天":
        weeks = max(1, (value + 6) // 7)
        return f"{weeks}周"
    if unit in {"个月", "月"}:
        return f"{max(1, value * 4)}周"
    return f"{value}周"


_ELLIPTICAL_PATTERNS = (
    r"^(?:它|这个|那个|这|那|上述|刚才|上面|继续|还有|然后|接着)(?:呢|怎么|如何)?[？?。!！]*$",
    r"^(?:怎么|如何|为什么|为啥)(?:学|学习|理解|开始|入门|上手|用|做|准备|复习|安排)?[？?。!！]*$",
    r"^(?:详细点|详细一点|具体点|具体一点|展开|展开说说|再说说|多说点|举个例子|继续讲|接着讲|继续)[。.!！?？]*$",
)


def is_elliptical_query(query: str) -> bool:
    """判断是否为依赖上文的省略式追问（"怎么学""详细点"），用于继承会话主题。"""
    text = (query or "").strip()
    if not text or len(text) > 20:
        return False
    return any(re.match(pattern, text) for pattern in _ELLIPTICAL_PATTERNS)


def extract_subject(query: str) -> str:
    query_lower = (query or "").lower()

    # 面试/求职主题优先：岗位本身是已知词时，也要保留"面试"这层语境
    interview_subject = extract_interview_subject(query)
    if interview_subject:
        return interview_subject

    best_subject = "通用"
    best_score = 0

    for subject, keywords in SUBJECT_KEYWORDS.items():
        score = 0
        if subject.lower() in query_lower:
            score += max(2, len(subject))
        for keyword in keywords:
            keyword_lower = keyword.lower()
            if keyword_lower in query_lower:
                score = max(score, len(keyword_lower) + (2 if keyword_lower == subject.lower() else 0))
        if score > best_score:
            best_subject = subject
            best_score = score

    # 固定关键词表未命中时，尝试抽取明确提及的自由主题（RAG / 经济学等）
    free_subject = extract_free_subject(query)
    if free_subject:
        free_lower = free_subject.lower()
        # 常见别名/同义词归一到规范主题，避免计划/知识库分成两个主题
        if free_lower in {"agent", "ai agent", "autonomous agent", "智能体", "人工智能体"}:
            return "agent"
        # 用户明确点名了关键词表中的某个"子主题词"（如 Vue 属于 javascript、
        # 装饰器 属于 python），且该词不同于归类名 → 用用户原话，避免被上位类覆盖。
        # 仅当用户原话就是这个词本身时生效；"python的装饰器"这类子主题仍归 python。
        if best_subject != "通用":
            keyword_to_subject = {
                kw.lower(): subject for subject, kws in SUBJECT_KEYWORDS.items() for kw in kws
            }
            if keyword_to_subject.get(free_lower) == best_subject and free_lower != best_subject.lower():
                return free_subject
        # 关键词表完全未命中 → 用自由主题（RAG / 经济学 / 量子计算等）
        if best_subject == "通用":
            return free_subject

    return best_subject


