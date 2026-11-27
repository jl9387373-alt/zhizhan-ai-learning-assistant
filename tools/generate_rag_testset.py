import argparse
import json
import os
import re
import random
import sys
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rag.content_loader import load_pdf


def _extract_subject_from_filename(name: str) -> str:
    n = name.lower()
    if "操作系统" in name or "os" in n:
        return "操作系统"
    if "离散" in name:
        return "离散数学"
    if "组成原理" in name:
        return "计算机组成原理"
    if "软件工程" in name or "设计模式" in name:
        return "软件工程"
    if "面试" in name or "简历" in name:
        return "求职面试"
    if "RAG" in name or "rag" in n:
        return "RAG"
    if "学习" in name:
        return "学习方法"
    return "通用"


def _pick_def_terms(text: str, limit: int) -> List[str]:
    terms: List[str] = []
    patterns = [
        r"([一-龥A-Za-z0-9_]{2,20})\s*[（(].{0,20}[)）]\s*是",
        r"什么是\s*([一-龥A-Za-z0-9_]{2,20})",
        r"([一-龥A-Za-z0-9_]{2,20})\s*是指",
        r"([一-龥A-Za-z0-9_]{2,20})\s*指的是",
    ]
    for pat in patterns:
        for m in re.finditer(pat, text):
            t = (m.group(1) or "").strip()
            if 2 <= len(t) <= 20 and t not in terms:
                terms.append(t)
            if len(terms) >= limit:
                return terms
    return terms


def _pick_compare_pairs(text: str, limit: int) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    patterns = [
        ("diff", r"([一-龥A-Za-z0-9_]{2,20})\s*(与|和)\s*([一-龥A-Za-z0-9_]{2,20})\s*的区别"),
        ("vs", r"([一-龥A-Za-z0-9_]{2,20})\s*vs\.?\s*([一-龥A-Za-z0-9_]{2,20})"),
    ]
    for kind, pat in patterns:
        for m in re.finditer(pat, text):
            a = (m.group(1) or "").strip()
            b = (m.group(3) or "").strip() if kind == "diff" else (m.group(2) or "").strip()
            if a and b and a != b:
                if (a, b) not in pairs and (b, a) not in pairs:
                    pairs.append((a, b))
            if len(pairs) >= limit:
                return pairs
    return pairs


def _read_text_file(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except UnicodeDecodeError:
        with open(path, "r", encoding="gbk", errors="ignore") as f:
            return f.read()

def _compact_text(text: str) -> str:
    text = re.sub(r"===\s*第\d+页\s*===", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _extract_ground_truth(text: str, must_terms: List[str], window_before: int = 120, window_after: int = 380) -> str:
    if not text:
        return ""
    compact = _compact_text(text)
    idx = -1
    for t in must_terms:
        p = compact.find(t)
        if p < 0:
            return ""
        idx = p if idx < 0 else min(idx, p)
    start = max(0, idx - window_before)
    end = min(len(compact), idx + window_after)
    snippet = compact[start:end].strip()
    snippet = re.sub(r"^[，,。.\-\s]+", "", snippet)
    snippet = re.sub(r"[，,。.\-\s]+$", "", snippet)
    return snippet


def _load_documents(doc_dir: str) -> List[Dict[str, str]]:
    docs: List[Dict[str, str]] = []
    for name in os.listdir(doc_dir):
        full = os.path.join(doc_dir, name)
        if not os.path.isfile(full):
            continue
        ext = os.path.splitext(name)[1].lower()
        if ext not in [".md", ".markdown", ".txt", ".pdf"]:
            continue
        if ext == ".pdf":
            content = load_pdf(full) or ""
        else:
            content = _read_text_file(full)
        if len(content.strip()) < 200:
            continue
        docs.append(
            {
                "file_name": name,
                "file_path": full,
                "subject": _extract_subject_from_filename(name),
                "content": content,
                "ext": ext,
            }
        )
    return docs


def _extract_md_sections(md_text: str) -> List[Tuple[str, str]]:
    lines = md_text.splitlines()
    heading_re = re.compile(r"^(#{1,4})\s+(.+?)\s*$")
    sections: List[Tuple[str, str]] = []
    title = None
    buf: List[str] = []

    def flush():
        nonlocal title, buf
        if title is None:
            buf = []
            return
        body = "\n".join(buf).strip()
        if body:
            sections.append((title, body))
        buf = []

    for line in lines:
        m = heading_re.match(line.strip())
        if m:
            flush()
            title = m.group(2).strip()
            continue
        buf.append(line)

    flush()
    return sections


def _normalize_title(title: str) -> str:
    t = title.strip()
    t = re.sub(r"^\d+\.\s*", "", t)
    t = t.strip().strip("：:").strip()
    return t


def _make_question_for_title(title: str, rng: random.Random) -> str:
    t = _normalize_title(title)
    if not t:
        return "请概括这部分内容的核心要点。"

    if t.startswith("什么是"):
        return t if t.endswith(("？", "?")) else (t + "？")
    if t.startswith("如何"):
        return t if t.endswith(("？", "?")) else (t + "？")
    if "区别" in t or " vs " in t.lower() or "vs" in t.lower():
        base = re.sub(r"\s*vs\.?\s*", "和", t, flags=re.IGNORECASE)
        base = base.replace("的区别", "")
        templates = [
            "{x}有什么区别？",
            "对比一下{x}的差异。",
            "{x}分别适合什么场景？",
        ]
        return rng.choice(templates).format(x=base)

    templates = [
        "用通俗的话解释一下“{t}”。",
        "请概括“{t}”的核心要点。",
        "“{t}”主要讲了什么？",
        "总结一下“{t}”这一部分的重点。",
        "“{t}”在实际中有什么用？",
    ]
    return rng.choice(templates).format(t=t)


def _make_question_for_term(term: str, rng: random.Random) -> str:
    t = term.strip().strip("：:").strip()
    templates = [
        "什么是{t}？",
        "用一句话解释{t}。",
        "请用通俗的例子说明{t}。",
        "{t}主要解决什么问题？",
        "{t}有什么特点？",
    ]
    return rng.choice(templates).format(t=t)


def _make_question_for_pair(a: str, b: str, rng: random.Random) -> str:
    templates = [
        "{a}和{b}有什么区别？",
        "对比一下{a}与{b}的核心差异。",
        "{a}和{b}各自适合哪些场景？",
    ]
    return rng.choice(templates).format(a=a, b=b)


def generate_testcases(documents: List[Dict[str, str]], per_doc: int) -> List[Dict[str, str]]:
    testcases: List[Dict[str, str]] = []
    os_texts: List[str] = []
    rng = random.Random(20260309)
    for doc in documents:
        text = doc["content"]
        subject = doc["subject"]
        file_name = doc["file_name"]
        ext = doc.get("ext", "")
        if subject == "操作系统":
            os_texts.append(text)

        if ext in [".md", ".markdown"]:
            sections = _extract_md_sections(text)
            used = 0
            for title, body in sections:
                if used >= per_doc:
                    break
                t = _normalize_title(title)
                if len(t) < 2 or len(t) > 60:
                    continue
                if "目录" in t or "参考" in t:
                    continue
                q = _make_question_for_title(title, rng)
                gt = _extract_ground_truth(body, [t], window_before=0, window_after=520) or _compact_text(body)[:520]
                testcases.append(
                    {
                        "question": q,
                        "ground_truth": gt,
                        "meta": {"subject": subject, "source_file": file_name, "type": "section"},
                    }
                )
                used += 1

        terms = _pick_def_terms(text, limit=max(2, per_doc))
        for t in terms[:per_doc]:
            gt = _extract_ground_truth(text, [t])
            testcases.append(
                {
                    "question": _make_question_for_term(t, rng),
                    "ground_truth": gt,
                    "meta": {"subject": subject, "source_file": file_name, "type": "definition"},
                }
            )

        pairs = _pick_compare_pairs(text, limit=max(1, per_doc // 2))
        for a, b in pairs:
            gt = _extract_ground_truth(text, [a, b])
            testcases.append(
                {
                    "question": _make_question_for_pair(a, b, rng),
                    "ground_truth": gt,
                    "meta": {"subject": subject, "source_file": file_name, "type": "comparison"},
                }
            )

    if os_texts:
        os_blob = "\n".join(os_texts)
        seeds = [
            "进程",
            "线程",
            "PCB",
            "死锁",
            "虚拟内存",
            "分页",
            "页面置换",
            "调度",
        ]
        for term in seeds:
            gt = _extract_ground_truth(os_blob, [term])
            testcases.append(
                {
                    "question": _make_question_for_term(term, rng),
                    "ground_truth": gt,
                    "meta": {"subject": "操作系统", "source_file": "操作系统.pdf", "type": "seed"},
                }
            )

    return testcases


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--doc-dir", default=os.path.join("data", "documents"))
    parser.add_argument("--per-doc", type=int, default=6)
    parser.add_argument("--out", default=os.path.join("data", "testsets", "rag_testcases.jsonl"))
    args = parser.parse_args()

    documents = _load_documents(args.doc_dir)
    testcases = generate_testcases(documents, per_doc=args.per_doc)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for case in testcases:
            f.write(json.dumps(case, ensure_ascii=False) + "\n")

    print(f"Generated {len(testcases)} test cases -> {args.out}")


if __name__ == "__main__":
    main()
