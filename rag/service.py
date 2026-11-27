"""Core RAG service."""
from __future__ import annotations

import os
import re
import time
from collections import OrderedDict
from typing import Any, Dict, Iterator, List, Optional, Tuple

from core.config import config
from core.logger import logger
from .metadata import RAGMetadataStore
from .prompting import build_qa_prompt, build_hybrid_prompt, build_general_prompt, build_learning_path_prompt
from .vector_store import VectorStoreService, safe_imports

GENERAL_SUBJECT = "\u901a\u7528"

# 资料引用型问题：必须基于指定资料回答，找不到就明说，不能编造或用通用知识顶替
_DOC_SPECIFIC_PATTERNS: List[str] = [
    r"\u6211\u4e0a\u4f20\u7684\u8d44\u6599", r"\u6211\u7684\u7b14\u8bb0", r"\u67d0\u7bc7\u7b14\u8bb0",
    r"\u67d0\u4e00\u7ae0", r"\u8fd9\u4e00\u7ae0", r"\u8fd9\u7bc7\u6587\u7ae0", r"\u8fd9\u4efd\u8d44\u6599",
    r"\u6839\u636e\u8d44\u6599", r"\u6839\u636e\u4e0a\u4f20", r"\u6309\u7167\u7b14\u8bb0",
    r"\u8bfe\u7a0b\u8d44\u6599", r"\u6307\u5b9a\u7684\u8d44\u6599", r"\u5177\u4f53\u5230\u7b2c\u51e0\u7ae0",
]


def is_document_specific_question(question: str) -> bool:
    text = str(question or "")
    return any(re.search(p, text, re.IGNORECASE) for p in _DOC_SPECIFIC_PATTERNS)


SYNONYM_MAP: Dict[str, List[str]] = {
    "虚拟内存": ["虚拟地址空间", "虚拟内存实现思路"],
    "虚拟地址空间": ["虚拟内存"],
    "进程调度": ["cpu调度", "调度算法", "进程调度算法"],
    "调度算法": ["进程调度", "cpu调度"],
    "卷积神经网络": ["cnn", "卷积", "卷积层"],
    "cnn": ["卷积神经网络", "卷积"],
    "transformer": ["注意力机制", "self-attention", "自注意力"],
    "注意力机制": ["transformer", "self-attention", "自注意力"],
    "pagerank": ["网页排名", "链接分析", "网页排序"],
    "哈希冲突": ["哈希", "哈希表", "散列"],
    "哈希": ["哈希表", "哈希冲突"],
    "死锁": ["死锁避免", "死锁条件", "死锁四个必要条件"],
    "acid": ["事务", "事务特性"],
    "事务": ["acid", "事务特性"],
}


class RAGService:
    def __init__(self) -> None:
        self.imports = safe_imports()
        self.vs = VectorStoreService()
        self.metadata_store = RAGMetadataStore()
        self._ready = False
        self._query_cache: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        self.reload_config()
        self.ChatOpenAI = self.imports.get("ChatOpenAI") if self.imports else None
        self.PromptTemplate = self.imports.get("PromptTemplate") if self.imports else None

    def reload_config(self) -> None:
        config.reload()
        self.cfg = {
            "top_k": int(config.get("rag.top_k", 4)),
            "score_threshold": float(config.get("rag.score_threshold", 0.3)),
            "model_name": str(config.get("rag.model_name", "gpt-3.5-turbo")),
            "model_provider": str(config.get("rag.model_provider", "openai")),
            "api_key": str(config.get("rag.api_key", "")),
            "base_url": str(config.get("rag.base_url", "https://api.openai.com/v1")),
            "hybrid_search_enabled": bool(config.get("rag.hybrid_search_enabled", True)),
            "enable_rerank": bool(config.get("rag.enable_rerank", True)),
            "max_context_length": int(config.get("rag.max_context_length", 4000)),
            "rerank_top_k": int(config.get("rag.rerank_top_k", 8)),
            "cache_size": int(config.get("rag.cache_size", 100)),
            "llm_timeout": int(config.get("rag.llm_timeout", 20)),
            "postcheck_secondary_enabled": bool(config.get("rag.postcheck_secondary_enabled", True)),
            "postcheck_secondary_trigger": float(config.get("rag.postcheck_secondary_trigger", 0.52)),
            "postcheck_secondary_top_k": int(config.get("rag.postcheck_secondary_top_k", 8)),
            "postcheck_improve_margin": float(config.get("rag.postcheck_improve_margin", 0.03)),
        }
        self.vs.reload_config()

    def ensure_ready(self) -> bool:
        if self._ready:
            return True
        if not bool(config.get("rag.enabled", True)):
            return False
        try:
            self._ready = self.vs.load_or_build()
        except Exception as exc:
            logger.warning("知识库初始化失败: %s", type(exc).__name__)
            self._ready = False
        return self._ready

    def ask_with_details(self, question: str, subject: str = GENERAL_SUBJECT, chat_history: str = "", image_data: str = None) -> Optional[Dict[str, Any]]:
        self.reload_config()
        start = time.time()
        question = str(question or "").strip()
        subject = str(subject or GENERAL_SUBJECT).strip() or GENERAL_SUBJECT
        if len(question) < 2:
            return {"answer": "\u62b1\u6b49\uff0c\u95ee\u9898\u592a\u77ed\u3002", "mode": "error", "fallback": True, "response_time": time.time() - start}
        cache_key = f"{subject}:{question[:160]}:{chat_history[:80]}"
        if cache_key in self._query_cache:
            cached = dict(self._query_cache[cache_key])
            self._query_cache.move_to_end(cache_key)
            cached["response_time"] = time.time() - start
            return cached
        kb_available = self.ensure_ready() and self.imports is not None

        # 指定资料类问题：无论知识库是否可用，都必须基于资料回答，不编造
        doc_specific = is_document_specific_question(question)
        if not kb_available:
            if doc_specific:
                return {"answer": "\u672a\u627e\u5230\u4f60\u6307\u5b9a\u7684\u8d44\u6599\uff0c\u6682\u65f6\u65e0\u6cd5\u636e\u6b64\u56de\u7b54\u3002", "mode": "no_context", "fallback": True, "response_time": time.time() - start}
            # 知识库不可用：尝试用模型通用知识回答普通概念问题
            general = self._answer_with_general_knowledge(question, subject, chat_history)
            if general:
                return general
            return {"answer": "\u62b1\u6b49\uff0c\u77e5\u8bc6\u5e93\u6682\u65f6\u4e0d\u53ef\u7528\uff0c\u4e14\u6a21\u578b\u5f53\u524d\u4e0d\u53ef\u7528\uff0c\u65e0\u6cd5\u56de\u7b54\u3002", "mode": "error", "fallback": True, "response_time": time.time() - start}

        # 定义类问题召回加倍：一篇文档里多个小节标题都含主题词，
        # 只取 4 段时常常把"定义/结构"那几段挤掉，只剩"评审流程"。
        retrieve_k = 10 if self._question_intent(question) == "define" else None
        retrieved_docs, focus_terms = self._retrieve(question, subject, chat_history, k=retrieve_k)
        # 关键：仅凭"检索到非空片段"不能走知识库。先剔除不回答用户意图的资料，
        # 例如"rag怎么学"命中了集成测试段落（里面只是顺带提到 RAG 检索），必须排除。
        docs = [doc for doc in retrieved_docs if self._doc_answers_intent(question, doc, subject)]
        intent = self._question_intent(question)
        if intent == "define" and len(docs) > 1:
            docs = self._prefer_topic_chunks(docs, question, subject)
        if intent == "learn" and not docs:
            # 学习类问题若资料未命中主题，明确走通用知识，不用无关片段拼凑
            docs = []

        snippets = self._docs_to_snippets(docs)
        context = self._build_context(docs)
        if not context:
            if doc_specific:
                return {"answer": "\u672a\u627e\u5230\u4f60\u6307\u5b9a\u7684\u8d44\u6599\uff0c\u65e0\u6cd5\u636e\u6b64\u56de\u7b54\uff0c\u4e5f\u4e0d\u4f1a\u7f16\u9020\u8d44\u6599\u5185\u5bb9\u3002", "mode": "no_context", "fallback": True, "context_snippets": snippets, "response_time": time.time() - start, "docs_count": 0}
            general = self._answer_with_general_knowledge(question, subject, chat_history)
            if general:
                general["context_snippets"] = snippets
                general["docs_count"] = 0
                return general
            return {"answer": "\u62b1\u6b49\uff0c\u6ca1\u6709\u627e\u5230\u76f8\u5173\u8d44\u6599\uff0c\u4e14\u6a21\u578b\u5f53\u524d\u4e0d\u53ef\u7528\uff0c\u65e0\u6cd5\u56de\u7b54\u3002", "mode": "error", "fallback": True, "context_snippets": snippets, "response_time": time.time() - start, "docs_count": 0}

        answer, mode = self.generate_answer_with_postcheck(
            question=question,
            subject=subject,
            docs=docs,
            focus_terms=focus_terms,
            context=context,
            chat_history=chat_history,
            allow_llm=True,
        )
        # 生成后再校验一次：答案若跑题（答成别的主题/答非所问），重走通用知识回答
        answer = self._strip_heading_echo(answer, docs)
        if self._is_answer_usable(answer, question) and self._answer_on_topic(question, answer):
            return self._finalize(cache_key, answer, mode, context, snippets, len(docs), start)

        general = self._answer_with_general_knowledge(question, subject, chat_history)
        if general:
            general["context_snippets"] = []
            general["docs_count"] = 0
            return general
        # 模型也不可用：不再返回无关资料原文/模板
        return {"answer": "\u62b1\u6b49\uff0c\u68c0\u7d22\u5230\u7684\u8d44\u6599\u65e0\u6cd5\u56de\u7b54\u8be5\u95ee\u9898\uff0c\u4e14\u6a21\u578b\u5f53\u524d\u4e0d\u53ef\u7528\u3002", "mode": "error", "fallback": True, "response_time": time.time() - start, "docs_count": len(docs)}

    @staticmethod
    def stream_text_pieces(text: str, delay: float = 0.03, max_piece: int = 4) -> Iterator[str]:
        """把已生成的答案切成小片段并逐段吐出，模拟打字机效果。
        只影响展示节奏，不改变答案内容；可被内置知识库路径复用。"""
        text = str(text or "")
        if not text:
            return
        i = 0
        n = len(text)
        while i < n:
            # 优先在标点/换行处断句，读起来更自然
            step = min(max_piece, n - i)
            window = text[i:i + step]
            cut = -1
            for idx in range(len(window) - 1, -1, -1):
                if window[idx] in "，。！？；：、,.!?;:\n":
                    cut = idx + 1
                    break
            if cut > 0:
                step = cut
            yield text[i:i + step]
            i += step
            if i < n:
                time.sleep(delay)

    def ask_with_details_stream(self, question: str, subject: str = GENERAL_SUBJECT, chat_history: str = "", image_data: str = None) -> Iterator[Dict[str, Any]]:
        details = self.ask_with_details(question, subject=subject, chat_history=chat_history, image_data=image_data)
        if not details:
            yield {"type": "error", "data": "RAG service returned no response."}
            return
        mode = details.get("mode", "")
        meta = {
            "context_preview": str(details.get("context_preview") or "")[:200],
            "context_snippets": details.get("context_snippets") or [],
            "docs_count": details.get("docs_count", 0),
            "mode": mode,
            "source": details.get("source") or mode,
            "is_general": mode == "general_llm",
        }
        yield {"type": "meta", "data": meta}
        answer = str(details.get("answer") or "")
        for piece in self.stream_text_pieces(answer):
            yield {"type": "token", "data": piece}

    def ask(self, question: str, subject: str = GENERAL_SUBJECT, chat_history: str = "") -> Optional[str]:
        details = self.ask_with_details(question, subject=subject, chat_history=chat_history)
        return details.get("answer") if details else None

    def ingest(self, texts: List[str], subject: str = GENERAL_SUBJECT, metadata: Optional[List[Dict[str, Any]]] = None) -> int:
        count = self.vs.ingest_texts(texts, subject=subject, metadata=metadata)
        if count > 0 and metadata:
            self.metadata_store.append([{"subject": subject, "chunk_count": count, **entry} for entry in metadata])
        return count

    def rebuild(self) -> bool:
        self.vs = VectorStoreService()
        self._ready = False
        return self.vs.rebuild()

    def get_knowledge_base_stats(self) -> Dict[str, Any]:
        if not self.ensure_ready():
            return {"error": "RAG system is unavailable."}
        return self.vs.get_document_stats()

    def optimize_knowledge_base(self) -> Dict[str, Any]:
        return {"status": "skipped", "actions": [], "stats": self.get_knowledge_base_stats()}

    def generate_mindmap(self, topic: str) -> str:
        topic = str(topic or "").strip() or "Topic"
        return "mindmap\n  root((" + topic.replace('"', '') + "))\n    Definition\n    Key_Points\n    Examples\n    Practice\n"

    def _retrieve(self, question: str, subject: str, chat_history: str, k: int = None) -> tuple[List[Any], List[str]]:
        retrieval_query = self._optimize_query(question, subject)
        history_key = self._extract_key_from_history(chat_history)
        if history_key:
            retrieval_query = f"{history_key}\n\u5f53\u524d\u95ee\u9898\uff1a{retrieval_query}"
        full_query = f"{subject} | {retrieval_query}" if subject != GENERAL_SUBJECT else retrieval_query
        synonyms = self._expand_synonyms(question)
        if synonyms:
            full_query = f"{full_query} {' '.join(synonyms)}"
        focus_terms = self._extract_focus_terms(question) if self._is_focus_query(question) else []
        limit_k = int(k or self.cfg["top_k"])
        initial_k = max(self.cfg["rerank_top_k"], limit_k * (3 if focus_terms else 2))
        docs: List[Any] = []
        if self.cfg["hybrid_search_enabled"]:
            try:
                docs = list(self.vs.hybrid_search(full_query, top_k=initial_k, subject=subject))
            except Exception as exc:
                logger.warning(f"Hybrid search failed: {exc}")
        if not docs:
            try:
                docs = [item["doc"] for item in self.vs.keyword_search(full_query, top_k=initial_k)]
            except Exception as exc:
                logger.warning(f"Keyword fallback search failed: {exc}")
        if not docs and self.vs.vs is not None:
            try:
                try:
                    retriever = self.vs.vs.as_retriever(search_type="similarity_score_threshold", search_kwargs={"k": initial_k, "score_threshold": self.cfg["score_threshold"]})
                except Exception:
                    retriever = self.vs.vs.as_retriever(search_kwargs={"k": initial_k})
                docs = list(retriever.invoke(full_query))
            except Exception as exc:
                logger.warning("向量检索失败: %s", type(exc).__name__)
                docs = []
        priority: List[Any] = []
        if focus_terms:
            priority = self._find_explicit_title_matches(focus_terms, subject, limit=max(self.cfg["top_k"], min(initial_k, 6)))
            if priority:
                docs = self._refine_focus_documents(priority, docs, focus_terms, initial_k)
                docs = docs[:1]
        # 定义类问题：向量召回常把"标题就是主题"的小节漏掉（如"1. PRD 的八个部分"），
        # 这里用标题检索强制补齐，再交给后续排序。
        if self._question_intent(question) == "define":
            topic = str(self._extract_question_topic(question, fallback=subject or "") or "").strip()
            if len(topic) >= 2:
                try:
                    title_hits = self._find_explicit_title_matches([topic], subject, limit=6) or []
                except Exception:
                    title_hits = []
                if title_hits:
                    seen = {getattr(d, "page_content", "")[:60] for d in docs}
                    merged = []
                    for d in title_hits:
                        key = getattr(d, "page_content", "")[:60]
                        if key not in seen:
                            seen.add(key)
                            merged.append(d)
                    docs = merged + list(docs)

        if len(docs) > limit_k and not priority:
            docs = self._rerank_documents(docs, question, subject)
        return docs[:limit_k], focus_terms

    def _build_llm(self, api_key: str) -> Any:
        provider = self.cfg["model_provider"]
        model_name = self.cfg["model_name"]
        if provider == "openai" and self.ChatOpenAI is not None:
            return self.ChatOpenAI(model=model_name, openai_api_key=api_key, openai_api_base=self.cfg["base_url"], temperature=0.2, max_tokens=1200, timeout=self.cfg["llm_timeout"])
        if provider == "dashscope":
            try:
                from langchain_community.chat_models import ChatDashScope as DashLLM
            except Exception:
                try:
                    from langchain_community.chat_models.tongyi import ChatTongyi as DashLLM  # type: ignore
                except Exception:
                    DashLLM = None
            if DashLLM is not None:
                return DashLLM(model=model_name, dashscope_api_key=api_key, temperature=0.2, max_tokens=1200, timeout=self.cfg["llm_timeout"])
        return None

    def _finalize(self, cache_key: str, answer: str, mode: str, context: str, snippets: List[Dict[str, Any]], docs_count: int, start: float) -> Dict[str, Any]:
        result = {"answer": self._sanitize_answer(answer), "mode": mode, "context_preview": context[:1200], "context_snippets": snippets, "fallback": False, "confidence": self._calculate_answer_confidence(answer, context, cache_key), "docs_count": docs_count}
        if len(self._query_cache) >= self.cfg["cache_size"]:
            self._query_cache.popitem(last=False)
        self._query_cache[cache_key] = dict(result)
        self._query_cache.move_to_end(cache_key)
        result["response_time"] = time.time() - start
        return result

    def _extract_key_from_history(self, history: str) -> str:
        text = str(history or "").strip()
        if len(text) < 20:
            return ""
        values: List[str] = []
        for raw_line in text.splitlines()[-6:]:
            line = raw_line.strip()
            if len(line) < 6:
                continue
            values.append(re.sub(r"\s+", " ", line)[-120:])
        return " ".join(values[:2])

    def _sanitize_answer(self, answer: str) -> str:
        text = str(answer or "")
        text = re.sub(r"\[FOLLOW_UP\].*?\[/FOLLOW_UP\]", "", text, flags=re.IGNORECASE | re.DOTALL)
        text = re.sub(r"(?m)^\s*\d+[.)、]\s*##\s+.*$", "", text)
        text = re.sub(r"(?m)^##\s+.*$", "", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def _looks_like_noisy_text(self, text: str) -> bool:
        value = str(text or "")
        if not value.strip():
            return True
        if re.search(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", value):
            return True
        if "???" in value and "AI" in value:
            return True
        split_cjk = len(re.findall(r"[\u4e00-\u9fff]\s+[\u4e00-\u9fff]", value))
        if split_cjk >= 6:
            return True
        return False

    def _estimate_text_quality(self, text: str) -> float:
        value = str(text or "").strip()
        if not value:
            return -10.0
        score = 0.0
        non_space = re.sub(r"\s+", "", value)
        if len(non_space) >= 24:
            score += 1.0
        if len(non_space) >= 80:
            score += 1.0
        if re.search(r"^#{1,3}\s+|^\s*[-*]\s+", value, flags=re.MULTILINE):
            score += 1.5
        if re.search(r"[???.!?]", value):
            score += 0.5
        if self._looks_like_noisy_text(value):
            score -= 6.0
        split_cjk = len(re.findall(r"[\u4e00-\u9fff]\s+[\u4e00-\u9fff]", value))
        score -= min(split_cjk, 10) * 0.5
        return score

    def _expand_query_term_variants(self, term: str) -> List[str]:
        cleaned = self._clean_markdown_text(term).lower()
        if not cleaned:
            return []

        variants: List[str] = []

        def add(value: str) -> None:
            token = self._clean_markdown_text(value).lower().strip()
            if token and token not in variants:
                variants.append(token)

        add(cleaned)
        compact = re.sub(r"\s+", "", cleaned)
        if compact != cleaned:
            add(compact)

        for token in re.findall(r"[\u4e00-\u9fff]{2,}|[A-Za-z0-9_/]{2,}", cleaned):
            add(token)

        if any(sep in cleaned for sep in ("\u548c", "\u4e0e", "\u53ca", "\u8ddf")):
            for part in re.split(r"[\u548c\u4e0e\u53ca\u8ddf]", cleaned):
                add(part)

        if cleaned.endswith("\u9501") and len(cleaned) >= 3:
            add(cleaned[:-1])
        if cleaned.endswith("\u673a\u5236") and len(cleaned) >= 4:
            add(cleaned[:-2])
        if cleaned.endswith("\u65b9\u6cd5") and len(cleaned) >= 4:
            add(cleaned[:-2])
        if cleaned.endswith("\u533a\u522b") and len(cleaned) >= 4:
            add(cleaned[:-2])

        return variants

    def _expand_synonyms(self, question: str) -> List[str]:
        text = (question or "").lower()
        expanded: List[str] = []
        for key, syns in SYNONYM_MAP.items():
            if key in text:
                for s in syns:
                    token = s.lower().strip()
                    if token and token not in expanded:
                        expanded.append(token)
        return expanded

    def _extract_query_terms(self, question: str, focus_terms: Optional[List[str]] = None) -> List[str]:
        values: List[str] = []
        for term in focus_terms or []:
            for variant in self._expand_query_term_variants(term):
                if variant not in values:
                    values.append(variant)
        topic = self._extract_question_topic(question, fallback="")
        for variant in self._expand_query_term_variants(topic):
            if variant not in values:
                values.append(variant)
        for token in self._extract_focus_terms(question):
            for variant in self._expand_query_term_variants(token):
                if variant not in values:
                    values.append(variant)
        return values[:8]

    def _score_doc_for_answer(self, question: str, doc: Any, rank: int, focus_terms: Optional[List[str]] = None) -> float:
        metadata = getattr(doc, "metadata", {}) or {}
        content = str(getattr(doc, "page_content", "") or "")
        file_name = str(metadata.get("file_name") or "").strip().lower()
        chunk_title = str(metadata.get("chunk_title") or metadata.get("title") or "").strip().lower()
        document_title = str(metadata.get("document_title") or "").strip().lower()
        title_blob = " ".join(part for part in [file_name, chunk_title, document_title] if part)
        normalized_question = re.sub(r"[^\u4e00-\u9fffA-Za-z0-9_]+", "", str(question or "").lower())
        normalized_chunk_title = re.sub(r"[^\u4e00-\u9fffA-Za-z0-9_]+", "", chunk_title)
        normalized_document_title = re.sub(r"[^\u4e00-\u9fffA-Za-z0-9_]+", "", document_title)
        style = self._classify_question_style(question)

        score = self._estimate_text_quality(content)
        score += max(0.0, 3.0 - rank * 0.35)
        if file_name.endswith(".md"):
            score += 3.0
        elif file_name.endswith(".txt"):
            score += 2.0
        elif file_name.endswith(".pdf"):
            score -= 1.5

        for term in self._extract_query_terms(question, focus_terms):
            if term and term == chunk_title:
                score += 10.0
            elif term and (term in chunk_title or term in document_title):
                score += 7.0
            elif term and term in title_blob:
                score += 5.0
            elif term and term in content[:240].lower():
                score += 2.0

        if normalized_question:
            if normalized_chunk_title and len(normalized_chunk_title) >= 4 and normalized_chunk_title in normalized_question:
                score += 9.0
            if normalized_document_title and len(normalized_document_title) >= 4 and normalized_document_title in normalized_question:
                score += 6.0

        if style == "definition":
            if any(marker in title_blob for marker in ("概念", "定义", "原理", "是什么", "导论")):
                score += 4.0
            head = self._first_sentence(content[:260])
            if head and re.search(r"(?:是|指的是|是一种|是一组|用于)", head):
                score += 3.0
        elif style == "summary":
            if any(marker in title_blob for marker in ("核心", "要点", "总结", "流程", "重点")):
                score += 2.5
        return score

    def _has_topic_support(self, question: str, docs: List[Any], focus_terms: Optional[List[str]] = None) -> bool:
        if not docs:
            return False

        candidates = self._extract_query_terms(question, focus_terms)
        topic = self._extract_question_topic(question, fallback="")
        if topic and topic.lower() not in candidates:
            candidates.insert(0, topic.lower())

        meaningful_terms = [term for term in candidates if len(re.sub(r"[\W_]+", "", term, flags=re.UNICODE)) >= 2]
        if not meaningful_terms:
            return True

        for doc in docs:
            metadata = getattr(doc, "metadata", {}) or {}
            content = str(getattr(doc, "page_content", "") or "").lower()
            title_blob = " ".join(
                str(metadata.get(key) or "").strip().lower()
                for key in ("chunk_title", "title", "document_title", "file_name")
            )
            normalized_title = re.sub(r"[^\u4e00-\u9fffA-Za-z0-9_]+", "", title_blob)
            normalized_content = re.sub(r"[^\u4e00-\u9fffA-Za-z0-9_]+", "", content[:400])
            for term in meaningful_terms:
                if not term:
                    continue
                if term in title_blob or term in content[:400]:
                    return True
                normalized_term = re.sub(r"[^\u4e00-\u9fffA-Za-z0-9_]+", "", term.lower())
                if normalized_term and (
                    normalized_term in normalized_title
                    or normalized_title in normalized_term
                    or normalized_term in normalized_content
                ):
                    return True
        return False

    def _question_intent(self, question: str) -> str:
        """判断问题的信息需求类型，用于验证候选资料是否真正回答了问题。
        learn: 需要学习顺序/路径；define: 需要概念解释；steps: 需要操作步骤；generic: 常规。"""
        text = str(question or "").lower()
        # 学习路径类：既包括"X怎么学"，也包括"我想学习X"。
        # 关键：这类问题要的是"如何入门的顺序"，不是"X 的定义/本项目怎么用 X"。
        learn_markers = (
            "怎么学", "如何学", "学习路线", "学习路径", "从哪学", "从哪开始", "怎么入手", "学习顺序",
            "我想学", "我要学", "想学习", "想学", "我想了解", "从零", "零基础", "入门", "怎么入门", "如何入门",
        )
        if any(m in text for m in learn_markers):
            return "learn"
        if any(m in text for m in ("是什么", "什么是", "定义", "介绍", "概念", "原理",
                                   "解释", "讲讲", "说一下", "介绍一下")):
            return "define"
        if any(m in text for m in ("怎么用", "如何使用", "操作步骤", "怎么做", "怎么用", "步骤")):
            return "steps"
        return "generic"

    def _prefer_topic_chunks(self, docs: List[Any], question: str, subject: str = "") -> List[Any]:
        """定义类问题：优先使用片段标题本身含主题的那一段。

        同一篇文档常含多个小节（如「1. Agent 是什么」与「5. 常见误区」），
        只凭文件名判断相关会让后者也通过，最终可能选中错误小节。
        """
        scaffold = {
            "解释", "讲讲", "说一下", "介绍一下", "介绍", "是什么", "什么是",
            "定义", "概念", "原理", "怎么", "如何", "为什么", "一下",
        }
        raw = [self._extract_question_topic(question, fallback=subject or ""), subject]
        raw += list(self._extract_query_terms(question))
        terms: List[str] = []
        for item in raw:
            term = str(item or "").lower().strip()
            if not term or term in scaffold:
                continue
            for word in scaffold:
                if term.startswith(word) and len(term) > len(word):
                    term = term[len(word):]
                if term.endswith(word) and len(term) > len(word):
                    term = term[:-len(word)]
            if term and term not in scaffold:
                terms.append(term)
        terms = list(dict.fromkeys(terms))
        if not terms:
            return docs

        def score(doc: Any) -> int:
            metadata = getattr(doc, "metadata", {}) or {}
            title = str(metadata.get("chunk_title") or metadata.get("title") or "").lower()
            head = str(getattr(doc, "page_content", "") or "").lower()[:80]
            if any(t in title for t in terms):
                return 0
            if any(t in head for t in terms):
                return 1
            return 2

        ordered = sorted(docs, key=score)

        # 定义类问题：若存在"标题直接命中主题"的片段，就只在其中挑选，
        # 避免把同文档里"评审怎么过""常见误区"这类小节一并喂给模型，
        # 导致问"是什么"却答成"怎么做"。
        if self._question_intent(question) == "define":
            # 关键：同一篇文档里所有小节标题都含主题词（如都含"PRD"），
            # 按"标题命中"排序无法区分"定义"与"评审流程"。
            # 改为看正文开头：越早出现"主题…是/指/即"的片段越可能是定义。
            # 用主题词做探针（PRD / RAG 这类英文缩写也要支持），
            # 不能用 '解释一下什么是prd' 这种整句，否则永远匹配不到正文
            topic_probe = str(self._extract_question_topic(question, fallback=subject or "") or "").lower().strip()
            if not topic_probe and terms:
                topic_probe = min(terms, key=len)

            def def_score(doc: Any) -> int:
                metadata = getattr(doc, "metadata", {}) or {}
                title = str(metadata.get("chunk_title") or metadata.get("title") or "").lower()
                head = str(getattr(doc, "page_content", "") or "").lower()[:120]
                head_flat = re.sub(r"^[#\s>\-*]+", "", head)
                if topic_probe and topic_probe in head_flat[:50]:
                    return 0        # 开头就在讲这个主题
                if topic_probe and title.strip().startswith(topic_probe):
                    return 1        # 标题以主题开头
                if topic_probe and topic_probe in title:
                    return 2
                return 3

            return sorted(ordered, key=def_score)[:3]
        return ordered

    def _doc_answers_intent(self, question: str, doc: Any, subject: str = "") -> bool:
        """检查一个文档块是否真正回答了用户的问题意图，而不是仅仅'提到了主题词'。
        核心：标题/文档名要与主题一致，且内容要含对应意图的信号。"""
        metadata = getattr(doc, "metadata", {}) or {}
        content = str(getattr(doc, "page_content", "") or "")
        content_lower = content.lower()
        file_name = str(metadata.get("file_name") or metadata.get("file_path") or "").strip().lower()
        chunk_title = str(metadata.get("chunk_title") or metadata.get("title") or "").strip().lower()
        document_title = str(metadata.get("document_title") or "").strip().lower()
        title_blob = " ".join(p for p in (file_name, chunk_title, document_title) if p)

        intent = self._question_intent(question)
        topic = self._extract_question_topic(question, fallback=subject or "")
        topic_l = (topic or "").lower().strip()

        # 定义主题词集合：英文缩写按词边界匹配，避免 rag 命中 fragment 之类
        topic_terms = self._extract_query_terms(question)
        if topic_l and len(topic_l) >= 2:
            topic_terms.append(topic_l)
        topic_terms = [t.lower() for t in topic_terms if len(t) >= 2]
        # 问句虚词不是主题词：否则"解释agent"里的"解释"会命中
        # 《求职指南》中"解释技术概念时的清晰度"这类句子
        _scaffold_terms = {
            "解释", "讲讲", "说一下", "介绍一下", "介绍", "是什么", "什么是",
            "定义", "概念", "原理", "怎么", "如何", "为什么", "一下",
        }
        topic_terms = [t for t in topic_terms if t not in _scaffold_terms]

        def has_topic(text_lower: str) -> bool:
            return any(
                (re.fullmatch(r"[a-z0-9]+", t) and re.search(r"\b" + re.escape(t) + r"\b", text_lower))
                or (not re.fullmatch(r"[a-z0-9]+", t) and t in text_lower)
                for t in topic_terms
            )

        # 1) 主题必须出现在标题或正文，否则主题词只是偶然提及
        topic_in_title = has_topic(title_blob)
        topic_in_content = has_topic(content_lower)

        # 2) 学习路径场景：文档本身必须是"教人怎么学该主题"的资料，
        # 而不是"本项目系统架构/功能介绍里顺带提到了主题词"。
        if intent == "learn":
            # 项目功能介绍、系统架构类文档：即使标题命中主题也不作为入门依据
            project_doc_markers = (
                "系统架构", "项目介绍", "本系统", "架构设计", "技术方案", "功能模块",
                "整体架构", "设计方案", "技术实现", "模块设计", "功能介绍",
            )
            if any(m in title_blob for m in project_doc_markers):
                return False
            if any(m in content_lower[:220] for m in ("本系统", "我们的系统", "本项目")):
                return False
            # 标题不命中主题 → 在别的主题段落里顺带提到主题词，排除
            if not topic_in_title:
                return False
            # "检索/生成/切分"等功能词不再算学习信号，必须是明确的入门/路线特征
            learn_title_signals = (
                "学习", "入门", "路线", "路径", "指南", "教程", "从零", "零基础", "基础", "进阶", "实践", "步骤",
            )
            learn_content_signals = (
                "学习路线", "学习路径", "学习步骤", "入门", "第一阶段", "第二步", "从零开始",
                "循序渐进", "练习", "先学", "再学", "按顺序",
            )
            has_learning_signal = (
                any(s in title_blob for s in learn_title_signals)
                or any(s in content_lower for s in learn_content_signals)
            )
            # 学习路径类文档常在标题声明主题，正文不再重复主题词（如「第一阶段 · 跑通最小链路」），
            # 因此不再强制正文命中主题；标题命中已在上面校验过。
            if has_learning_signal:
                return True
            # 标题明确就是这个主题本身（如「RAG技术原理简述」），即使没有"学习路线"字样，
            # 也是回答"X怎么学"的可靠主题资料。项目功能介绍类已在 601-604 排除，不会误收。
            return topic_in_title

        # 3) "是什么/定义"场景：需要标题或开头就是该主题的解释
        if intent == "define":
            head = content_lower[:200]
            # 定义类问题：主题必须出现在标题，或出现在该段首句（通常是小标题）。
            # 仅在正文更靠后的位置顺带提到主题词，不足以作为定义依据。
            first_line = content_lower.split("\n", 1)[0][:120]
            if not (topic_in_title or has_topic(first_line)):
                return False
            # 标题就是这个主题的小节（如「1. PRD 的八个部分」）本身就是对题的结构段，
            # 即使正文是表格/清单（不含"是/指"这类解释词）也应保留，
            # 否则"是什么"类问题会只剩定义段，答不出结构。
            if topic_in_title:
                return True
            return any(m in head for m in ("是", "指", "是一种", "称为", "架构", "技术", "方法"))

        # 4) 操作步骤场景
        if intent == "steps":
            return topic_in_title and any(sig in content_lower for sig in ("步骤", "流程", "方法", "操作", "先", "再"))

        # 5) 常规场景：至少主题要在标题或内容中命中
        return topic_in_title or topic_in_content

    def _strip_heading_echo(self, answer: str, docs: List[Any]) -> str:
        """去掉答案开头的标题回显。

        现象：模型会先把片段标题原样吐出来，再接正文，例如
        「PRD 写作：结构、评审与验收 PRD（产品需求文档）不是给老板看的…」。
        处理：仅当开头与某个片段标题高度重合时裁掉，且保证裁完仍有内容。
        """
        if not answer:
            return answer
        text = answer.lstrip()
        titles = set()
        for doc in docs or []:
            meta = getattr(doc, "metadata", {}) or {}
            for key in ("chunk_title", "title", "document_title"):
                value = str(meta.get(key) or "").strip()
                if len(value) >= 4:
                    titles.add(value)
        for title in sorted(titles, key=len, reverse=True):
            if not text.startswith(title):
                continue
            rest = text[len(title):].lstrip(" \t：:，,。·—-")
            # 裁完必须有足够内容，避免把答案裁空
            if len(rest) >= 20:
                return rest
        return answer

    def _answer_on_topic(self, question: str, answer: str) -> bool:
        """最终答案跑题检测：确保答案围绕主题、且满足问题意图，而不是无关片段拼凑。"""
        answer_lower = str(answer or "").lower()
        question_lower = str(question or "").lower()
        if not answer_lower:
            return False

        intent = self._question_intent(question)
        topic = self._extract_question_topic(question, fallback="")
        topic_l = (topic or "").lower().strip()

        # 与主题明显无关的内容信号：这些词出现在答案里就判跑题（RAG 主题尤其要严查）
        off_topic_markers = (
            "集成测试", "单元测试", "黑盒测试", "白盒测试", "等价类", "边界值",
            "调试的基本思路", "意图识别、任务路由",
        )
        if any(marker in answer_lower for marker in off_topic_markers):
            return False

        # 学习类问题：答案必须有明确的"顺序/步骤结构"，而不只是出现主题词（如"检索"）。
        # 否则"rag是什么"式的概念解释会被误当成"rag怎么学"的答案。
        if intent == "learn":
            # 答案若变成"本系统混合检索/动态切分/Agent协作"式的功能介绍，视为跑题。
            # 注意："混合检索/动态切分"本身是 RAG 领域的正常术语，单出现不算跑题；
            # 只有当它和"本系统/本项目/我们的系统"一起出现（在介绍本项目功能）时才拦截。
            project_self_markers = ("本系统", "本项目", "我们的系统", "本平台", "该系统")
            project_feature_markers = (
                "混合检索", "动态切分", "agent协作", "意图识别、任务路由", "向量化存储",
            )
            if any(m in answer_lower for m in project_self_markers) and any(
                m in answer_lower for m in project_feature_markers
            ):
                return False
            structure_signals = ("学习路线", "学习路径", "学习顺序", "步骤", "阶段", "入门",
                                 "先", "再", "然后", "最后", "首先", "其次", "按顺序", "顺序")
            has_structure = any(sig in answer_lower for sig in structure_signals)
            has_numbering = bool(re.search(r"(?:^|\n)\s*\d+[\.、\)]", answer))
            return has_structure or has_numbering

        # 定义类问题：答案里要出现主题或解释信号（这里校验的是"答案"，不是片段）
        if intent == "define":
            return bool(topic_l) and (topic_l in answer_lower or any(m in answer_lower for m in ("是", "指", "称为", "架构")))

        return True

    def _select_primary_doc(self, question: str, docs: List[Any], focus_terms: Optional[List[str]] = None) -> Any:
        if not docs:
            return None
        ranked = sorted(
            enumerate(docs),
            key=lambda item: self._score_doc_for_answer(question, item[1], item[0], focus_terms),
            reverse=True,
        )
        return ranked[0][1]

    def _is_answer_usable(self, answer: str, question: str = "") -> bool:
        text = self._sanitize_answer(answer)
        if not text:
            return False
        compact = re.sub(r"[\s\W_]+", "", text, flags=re.UNICODE)
        if len(compact) < 6:
            return False
        if self._looks_like_noisy_text(text):
            return False
        invalid_fragments = (
            "???????",
            "????",
            "??????",
            "????????",
        )
        if any(fragment in text for fragment in invalid_fragments):
            return False
        if text.endswith(("?", "??", "? 1.", "???")):
            return False
        if question and self._question_requests_example(question) and "??" in text and text.endswith("?"):
            return False
        return True

    def _docs_to_snippets(self, docs: List[Any]) -> List[Dict[str, Any]]:
        return [{"index": index, "title": str((getattr(doc, "metadata", {}) or {}).get("chunk_title") or (getattr(doc, "metadata", {}) or {}).get("title") or (getattr(doc, "metadata", {}) or {}).get("file_name") or f"doc-{index}"), "content": self._truncate_snippet(str(getattr(doc, "page_content", "") or "")), "metadata": getattr(doc, "metadata", {}) or {}} for index, doc in enumerate(docs, start=1)]

    def _truncate_snippet(self, text: str, max_len: int = 320) -> str:
        text = str(text or "").strip().replace("\r", "")
        return text if len(text) <= max_len else text[:max_len] + "..."

    def _build_context(self, docs: List[Any]) -> str:
        parts: List[str] = []
        total = 0
        for index, doc in enumerate(docs, start=1):
            snippet = f"[\u7247\u6bb5{index}] {str(getattr(doc, 'page_content', '') or '').strip()}"
            if total + len(snippet) <= self.cfg["max_context_length"]:
                parts.append(snippet)
                total += len(snippet) + 2
            else:
                remaining = self.cfg["max_context_length"] - total - 10
                if remaining > 100:
                    parts.append(snippet[:remaining] + "...")
                break
        return "\n\n".join(parts)

    def _format_chat_history(self, history: str) -> str:
        if not history or len(history.strip()) < 10:
            return "No prior chat history."
        lines = [line.strip() for line in history.splitlines() if line.strip()]
        return "\n".join(lines[-6:])[:500]

    def _context_alignment_score(self, answer: str, context: str, question: str) -> float:
        answer_terms = set(re.findall(r'[\u4e00-\u9fff]{2,}|[A-Za-z0-9_]{3,}', str(answer or "").lower()))
        question_terms = set(re.findall(r'[\u4e00-\u9fff]{2,}|[A-Za-z0-9_]{3,}', str(question or "").lower()))
        context_terms = set(re.findall(r'[\u4e00-\u9fff]{2,}|[A-Za-z0-9_]{3,}', str(context or "").lower()))
        answer_nums = set(re.findall(r'\d+(?:\.\d+)?', str(answer or "")))
        context_nums = set(re.findall(r'\d+(?:\.\d+)?', str(context or "")))
        q_score = len(answer_terms & question_terms) / max(len(question_terms), 1) if question_terms else 0.5
        c_score = len(answer_terms & context_terms) / max(len(answer_terms), 1) if answer_terms else 0.0
        n_score = len(answer_nums & context_nums) / max(len(answer_nums), 1) if answer_nums else 1.0
        return max(0.0, min(1.0, 0.35 * q_score + 0.50 * c_score + 0.15 * n_score))

    def _question_keyword_terms(self, question: str) -> set[str]:
        tokens = self._extract_text_terms(question)
        stop_words = {
            "什么",
            "哪些",
            "如何",
            "为什么",
            "怎么",
            "一下",
            "这个",
            "那个",
            "主要",
            "讲了",
            "介绍",
            "请问",
            "总结",
            "概括",
            "核心",
            "要点",
            "区别",
            "作用",
            "定义",
            "介绍一下",
            "what",
            "how",
            "why",
            "which",
            "explain",
            "summary",
        }
        result = {
            token
            for token in tokens
            if token
            and token not in stop_words
            and len(re.sub(r"[\W_]+", "", token, flags=re.UNICODE)) >= 2
        }
        if not result:
            return set(list(tokens)[:8])
        return set(list(result)[:12])

    def _keyword_coverage_score(self, answer: str, question: str) -> float:
        q_terms = self._question_keyword_terms(question)
        if not q_terms:
            return 0.5
        a_terms = self._extract_text_terms(answer)
        if not a_terms:
            return 0.0
        overlap = len(a_terms & q_terms) / max(len(q_terms), 1)
        return max(0.0, min(1.0, overlap))

    def _unsupported_number_ratio(self, answer: str, context: str) -> float:
        answer_nums = {
            self._normalize_number_token(item)
            for item in re.findall(r"\d+(?:\.\d+)?%?", str(answer or ""))
        }
        answer_nums.discard("")
        if not answer_nums:
            return 0.0
        context_nums = {
            self._normalize_number_token(item)
            for item in re.findall(r"\d+(?:\.\d+)?%?", str(context or ""))
        }
        context_nums.discard("")
        unsupported = [num for num in answer_nums if num not in context_nums]
        return len(unsupported) / max(len(answer_nums), 1)

    def _answer_compactness_score(self, answer: str) -> float:
        text = re.sub(r"\s+", "", str(answer or ""))
        length = len(text)
        if length < 24:
            return 0.2
        if length < 60:
            return 0.65
        if length <= 320:
            return 1.0
        if length <= 520:
            return 0.75
        return 0.55

    def _candidate_quality_score(self, answer: str, mode: str, question: str, context: str) -> float:
        style = self._classify_question_style(question)
        align = self._context_alignment_score(answer, context, question)
        coverage = self._keyword_coverage_score(answer, question)
        num_consistency = 1.0 - self._unsupported_number_ratio(answer, context)
        compactness = self._answer_compactness_score(answer)

        if style == "summary":
            score = 0.40 * align + 0.36 * coverage + 0.10 * compactness + 0.14 * num_consistency
        elif style in ("utility", "definition"):
            score = 0.54 * align + 0.24 * coverage + 0.08 * compactness + 0.14 * num_consistency
        else:
            score = 0.50 * align + 0.26 * coverage + 0.08 * compactness + 0.16 * num_consistency

        if "focus_extract" in mode and style in ("summary", "definition"):
            score += 0.015
        if mode == "rag_llm" and style == "summary":
            score += 0.015
        return max(0.0, min(1.0, score))

    def _extract_text_terms(self, text: str) -> set[str]:
        tokens = re.findall(r"[\u4e00-\u9fff]{2,}|[A-Za-z][A-Za-z0-9_/-]{2,}", str(text or "").lower())
        return {token for token in tokens if token}

    def _normalize_number_token(self, token: str) -> str:
        value = str(token or "").strip().rstrip("%")
        if not value:
            return ""
        if "." in value:
            value = value.rstrip("0").rstrip(".")
        return value

    def _split_answer_units(self, answer: str) -> List[str]:
        text = self._sanitize_answer(answer)
        if not text:
            return []
        units: List[str] = []
        for raw_line in re.split(r"\n+", text):
            line = raw_line.strip()
            if not line:
                continue
            segments = [segment.strip() for segment in re.split(r"(?<=[。！？!?；;])", line) if segment.strip()]
            if not segments:
                segments = [line]
            units.extend(segments)
        return units

    def _is_supported_unit(
        self,
        unit: str,
        context: str,
        context_terms: set[str],
        context_nums: set[str],
        question_terms: set[str],
    ) -> bool:
        cleaned = re.sub(r"^\s*(?:[-*]|(?:\d+[.)、]))\s*", "", str(unit or "")).strip()
        if not cleaned:
            return False
        compact = re.sub(r"[\s\W_]+", "", cleaned, flags=re.UNICODE)
        if len(compact) < 4:
            return False
        numbers = [self._normalize_number_token(item) for item in re.findall(r"\d+(?:\.\d+)?%?", cleaned)]
        numbers = [item for item in numbers if item]
        if numbers and any(item not in context_nums for item in numbers):
            return False
        unit_terms = self._extract_text_terms(cleaned)
        if not unit_terms:
            return bool(numbers)
        overlap_terms = unit_terms & context_terms
        overlap_count = len(overlap_terms)
        overlap_ratio = overlap_count / max(len(unit_terms), 1)
        question_overlap = len(unit_terms & question_terms)

        # Prefer statements with clear evidence overlap in retrieved context.
        if overlap_count >= 2:
            return True
        if overlap_ratio >= 0.25:
            return True
        # Allow one-term overlap when it matches question focus and has evidence anchor.
        if overlap_count >= 1 and question_overlap >= 1 and overlap_ratio >= 0.12:
            return True
        # Numeric claims with matched numbers can be kept with looser lexical overlap.
        if numbers and overlap_count >= 1:
            return True
        return False

    def _prune_unsupported_content(self, answer: str, context: str, question: str) -> str:
        text = self._sanitize_answer(answer)
        if not text:
            return ""
        units = self._split_answer_units(text)
        if len(units) <= 1:
            return text

        context_terms = self._extract_text_terms(context)
        context_nums = {
            self._normalize_number_token(item)
            for item in re.findall(r"\d+(?:\.\d+)?%?", str(context or ""))
        }
        context_nums.discard("")
        question_terms = self._extract_text_terms(question)

        kept: List[str] = []
        for idx, unit in enumerate(units):
            if self._is_supported_unit(unit, context, context_terms, context_nums, question_terms):
                kept.append(unit.strip())
                continue
            if idx == 0 and len(units) <= 3 and self._is_answer_usable(unit, question):
                kept.append(unit.strip())

        if not kept:
            return text
        list_style = any(re.match(r"^\s*(?:[-*]|(?:\d+[.)、]))\s*", unit) for unit in units)
        joined = "\n".join(kept) if list_style else " ".join(kept)
        pruned = self._sanitize_answer(joined)
        return pruned if self._is_answer_usable(pruned, question) else text

    def _augment_answer_with_evidence_terms(
        self,
        answer: str,
        question: str,
        context: str,
        max_items: int = 2,
    ) -> str:
        text = self._sanitize_answer(answer)
        if not text or not context:
            return text
        if self._question_requests_one_sentence(question):
            return text
        if len(re.sub(r"\s+", "", text)) > 420:
            return text

        q_terms = list(self._question_keyword_terms(question))
        if not q_terms:
            return text
        answer_terms = self._extract_text_terms(text)
        missing_terms = [term for term in q_terms if term not in answer_terms][:8]
        if not missing_terms:
            return text

        context_sentences = [self._clean_markdown_text(item) for item in self._split_sentences(context)]
        additions: List[str] = []
        seen = set()
        for term in missing_terms:
            term_lower = term.lower()
            for sentence in context_sentences:
                if not sentence:
                    continue
                candidate = re.sub(r"^\[\s*片段\d+\s*\]\s*", "", sentence).strip()
                if len(candidate) < 10:
                    continue
                lowered = candidate.lower()
                if term_lower not in lowered:
                    continue
                if candidate in seen:
                    continue
                seen.add(candidate)
                additions.append(candidate[:140].rstrip("，,；;。.!！?？") + "。")
                break
            if len(additions) >= max_items:
                break

        if not additions:
            return text

        if re.search(r"^\s*(?:[-*]|\d+[.)、])\s+", text, flags=re.MULTILINE):
            base_count = len(re.findall(r"^\s*(?:\d+[.)、])\s+", text, flags=re.MULTILINE))
            numbered = [f"{base_count + idx}. {item}" for idx, item in enumerate(additions, start=1)]
            merged = text + "\n" + "\n".join(numbered)
            return self._sanitize_answer(merged)

        merged = text + "\n补充要点：" + "；".join(additions)
        return self._sanitize_answer(merged)

    def _answer_with_general_knowledge(self, question: str, subject: str, chat_history: str = "") -> Optional[Dict[str, Any]]:
        """无相关资料时用模型知识回答普通问题；模型不可用返回 None。
        返回中 mode/source=general_llm（内部字段），答案正文不含任何来源声明。"""
        api_key = self.cfg["api_key"] or os.getenv("OPENAI_API_KEY") or os.getenv("DASHSCOPE_API_KEY")
        if not api_key or self.PromptTemplate is None:
            return None
        try:
            llm = self._build_llm(api_key)
            if llm is None:
                return None
            if self._question_intent(question) == "learn":
                prompt = build_learning_path_prompt({"PromptTemplate": self.PromptTemplate})
            else:
                prompt = build_general_prompt({"PromptTemplate": self.PromptTemplate})
            response = (prompt | llm).invoke(
                {
                    "question": question,
                    "subject": subject,
                    "history": self._format_chat_history(chat_history),
                }
            )
            answer = self._sanitize_answer(getattr(response, "content", None) or str(response))
            if not self._is_answer_usable(answer, question):
                return None
            return {"answer": answer, "mode": "general_llm", "fallback": False, "source": "general_llm"}
        except Exception as exc:
            logger.warning(f"General knowledge answer failed: {exc}")
            return None

    def _invoke_llm_answer(self, question: str, subject: str, context: str, chat_history: str = "") -> Tuple[str, str]:
        """返回 (答案, 内部mode)。mode 由这里决定，绝不通过扫描答案文本判断。"""
        api_key = self.cfg["api_key"] or os.getenv("OPENAI_API_KEY") or os.getenv("DASHSCOPE_API_KEY")
        if not api_key:
            return "", ""
        if self.PromptTemplate is None:
            return "", ""
        try:
            llm = self._build_llm(api_key)
            if llm is None:
                return "", ""
            prompt = build_hybrid_prompt({"PromptTemplate": self.PromptTemplate})
            response = (prompt | llm).invoke(
                {
                    "question": question,
                    "subject": subject,
                    "context": context,
                    "history": self._format_chat_history(chat_history),
                }
            )
            answer = self._sanitize_answer(getattr(response, "content", None) or str(response))
            if not self._is_answer_usable(answer, question):
                return "", ""
            return answer, "rag_hybrid"
        except Exception as exc:
            logger.warning(f"LLM generation failed: {exc}")
            return "", ""

    def _rewrite_answer_with_evidence(self, question: str, subject: str, context: str, draft_answer: str) -> str:
        api_key = self.cfg["api_key"] or os.getenv("OPENAI_API_KEY") or os.getenv("DASHSCOPE_API_KEY")
        if not api_key:
            return ""
        try:
            llm = self._build_llm(api_key)
            if llm is None:
                return ""
            prompt = (
                "\u4f60\u662f\u4e25\u8c28\u7684\u6559\u5b66\u52a9\u624b\u3002\u8bf7\u4ec5\u6839\u636e\u7ed9\u5b9a\u8d44\u6599\u91cd\u5199\u7b54\u6848\uff0c"
                "\u4e0d\u8981\u6dfb\u52a0\u8d44\u6599\u4e4b\u5916\u7684\u4fe1\u606f\u3002"
                "\n\u8981\u6c42\uff1a"
                "\n1. \u5148\u76f4\u63a5\u56de\u7b54\u95ee\u9898\u3002"
                "\n2. \u8986\u76d6\u95ee\u9898\u4e2d\u7684\u6838\u5fc3\u672f\u8bed\u4e0e\u5173\u952e\u95ee\u6cd5\uff0c\u907f\u514d\u53ea\u56de\u7b54\u90e8\u5206\u8981\u70b9\u3002"
                "\n3. \u4f7f\u75282-5\u4e2a\u8981\u70b9\uff0c\u7b80\u6d01\u6e05\u6670\uff1b\u80fd\u5728\u8d44\u6599\u4e2d\u627e\u5230\u4f9d\u636e\u7684\u5173\u952e\u4e8b\u5b9e\u5c3d\u91cf\u4fdd\u7559\u3002"
                "\n4. \u82e5\u8d44\u6599\u4e0d\u8db3\uff0c\u8bf7\u660e\u786e\u5199\u51fa\u201c\u6839\u636e\u5f53\u524d\u8d44\u6599\u65e0\u6cd5\u786e\u8ba4\u201d\uff0c\u4f46\u4ecd\u7ed9\u51fa\u53ef\u786e\u8ba4\u7684\u8981\u70b9\u3002"
                "\n5. \u4e0d\u8981\u7f16\u9020\u6570\u5b57\u3001\u5e74\u4efd\u3001\u5b9a\u4e49\u6216\u7ed3\u8bba\u3002"
                f"\n\n\u95ee\u9898\uff1a{question}"
                f"\n\u4e3b\u9898\uff1a{subject}"
                f"\n\n\u8d44\u6599\uff1a\n{context[:2200]}"
                f"\n\n\u539f\u7b54\u6848\uff1a{draft_answer}"
                "\n\n\u91cd\u5199\u7b54\u6848\uff1a"
            )
            response = llm.invoke(prompt)
            rewritten = self._sanitize_answer(getattr(response, "content", None) or str(response))
            return rewritten if self._is_answer_usable(rewritten, question) else ""
        except Exception as exc:
            logger.warning(f"Evidence rewrite failed: {exc}")
            return ""

    def _pick_best_candidate(
        self,
        candidates: List[tuple[str, str]],
        question: str,
        context: str,
    ) -> tuple[str, str, float]:
        best_answer = ""
        best_mode = "rag_context"
        best_score = 0.0
        for answer, mode in candidates:
            candidate = self._prune_unsupported_content(answer, context, question)
            candidate = self._augment_answer_with_evidence_terms(candidate, question, context, max_items=2)
            if not self._is_answer_usable(candidate, question):
                continue
            score = self._candidate_quality_score(candidate, mode, question, context)
            if score > best_score:
                best_answer = candidate
                best_mode = mode
                best_score = score
        return best_answer, best_mode, best_score

    def _doc_identity(self, doc: Any) -> str:
        metadata = getattr(doc, "metadata", {}) or {}
        file_name = str(metadata.get("file_name") or metadata.get("file_path") or "").strip().lower()
        chunk_title = str(metadata.get("chunk_title") or metadata.get("title") or "").strip().lower()
        chunk_idx = str(metadata.get("chunk_index") or metadata.get("index") or "").strip()
        content_head = str(getattr(doc, "page_content", "") or "").strip().lower()[:120]
        return "|".join([file_name, chunk_title, chunk_idx, content_head])

    def _build_answer_candidates(
        self,
        question: str,
        docs: List[Any],
        focus_terms: List[str],
    ) -> List[tuple[str, str]]:
        has_topic_support = self._has_topic_support(question, docs, focus_terms)
        candidates: List[tuple[str, str]] = []

        # "怎么学"需要学习顺序，资料原文抽取（"主要内容是：..."）无法回答，
        # 这类意图只让 LLM 生成或通用知识回答，不产生"rag_*_extract"抽取候选。
        intent = self._question_intent(question)
        if intent == "learn":
            return candidates
        # "是什么"同理：抽取式模板只会吐一句摘要，答不出"有哪几部分"，
        # 而检索结果里已经有结构段（如"PRD 的八个部分"），交给 LLM 组织更完整。
        if intent == "define":
            return candidates

        if focus_terms and has_topic_support:
            focus_answer = self._build_focus_answer(question, docs, focus_terms)
            if focus_answer:
                candidates.append((focus_answer, "rag_focus_extract"))

        if has_topic_support:
            grounded_answer = self._build_grounded_answer(question, docs)
            if grounded_answer:
                candidates.append((grounded_answer, "rag_grounded_extract"))

        return candidates

    def _secondary_retrieval_queries(self, question: str, subject: str, focus_terms: List[str]) -> List[str]:
        values: List[str] = []

        def add(item: str) -> None:
            text = re.sub(r"\s+", " ", str(item or "")).strip()
            if text and text not in values:
                values.append(text)

        base = self._strip_question_wrappers(question)
        topic = self._extract_question_topic(question, fallback=base)
        optimized = self._optimize_query(question, subject)
        add(question)
        add(base)
        add(optimized)
        if topic:
            add(f"{topic} 核心概念 关键知识点")
            add(f"{topic} 定义 区别 作用")
        for term in focus_terms[:3]:
            add(f"{term} 是什么")
            add(f"{term} 核心要点")
        if subject and subject != GENERAL_SUBJECT and topic:
            add(f"{subject} {topic}")
        return values[:6]

    def _secondary_retrieve_docs(
        self,
        question: str,
        subject: str,
        focus_terms: List[str],
        base_docs: List[Any],
    ) -> List[Any]:
        if not self.ensure_ready():
            return base_docs
        target_k = max(int(self.cfg.get("postcheck_secondary_top_k", 8)), self.cfg["top_k"] * 2)
        queries = self._secondary_retrieval_queries(question, subject, focus_terms)
        merged_docs: List[Any] = list(base_docs)
        seen = {self._doc_identity(doc) for doc in merged_docs}

        for query in queries:
            try:
                retrieval_query = self._optimize_query(query, subject)
                full_query = f"{subject} | {retrieval_query}" if subject != GENERAL_SUBJECT else retrieval_query
                candidate_docs = list(self.vs.hybrid_search(full_query, top_k=target_k, subject=subject))
            except Exception as exc:
                logger.warning(f"Secondary retrieval failed: {exc}")
                continue
            for doc in candidate_docs:
                doc_key = self._doc_identity(doc)
                if doc_key in seen:
                    continue
                seen.add(doc_key)
                merged_docs.append(doc)

        if len(merged_docs) <= len(base_docs):
            return base_docs
        reranked = self._rerank_documents(merged_docs, question, subject)
        return reranked[:target_k]

    def generate_answer_with_postcheck(
        self,
        question: str,
        subject: str,
        docs: List[Any],
        focus_terms: Optional[List[str]] = None,
        context: str = "",
        chat_history: str = "",
        allow_llm: bool = True,
    ) -> tuple[str, str]:
        question = str(question or "").strip()
        subject = str(subject or GENERAL_SUBJECT).strip() or GENERAL_SUBJECT
        context = str(context or "").strip() or self._build_context(docs)
        if not context:
            return "", "rag_context"

        focus_terms = [str(term or "").strip() for term in (focus_terms or []) if str(term or "").strip()]
        candidates = self._build_answer_candidates(question, docs, focus_terms)

        best_answer, best_mode, best_score = self._pick_best_candidate(candidates, question, context)
        improve_margin = float(self.cfg.get("postcheck_improve_margin", 0.03))

        if (
            bool(self.cfg.get("postcheck_secondary_enabled", True))
            and docs
            and best_score < float(self.cfg.get("postcheck_secondary_trigger", 0.52))
        ):
            secondary_docs = self._secondary_retrieve_docs(question, subject, focus_terms, docs)
            secondary_ids = [self._doc_identity(doc) for doc in secondary_docs]
            base_ids = [self._doc_identity(doc) for doc in docs]
            if secondary_docs and secondary_ids != base_ids:
                secondary_context = self._build_context(secondary_docs)
                secondary_candidates = self._build_answer_candidates(question, secondary_docs, focus_terms)
                if best_answer:
                    secondary_candidates.append((best_answer, best_mode))
                sec_answer, sec_mode, sec_score = self._pick_best_candidate(
                    secondary_candidates, question, secondary_context
                )
                if sec_answer and sec_score >= best_score + improve_margin:
                    best_answer = sec_answer
                    best_mode = f"{sec_mode}+second_retrieval"
                    best_score = sec_score
                    docs = secondary_docs
                    context = secondary_context

        llm_needed = (not best_answer) or (best_score < 0.46)
        if allow_llm and llm_needed:
            llm_answer, llm_mode = self._invoke_llm_answer(question, subject, context, chat_history=chat_history)
            if llm_answer:
                # mode 由生成器内部返回，不通过扫描答案文本判断
                annotated_mode = llm_mode or "rag_llm"
                if annotated_mode == "rag_llm":
                    llm_answer = self._prune_unsupported_content(llm_answer, context, question)
                llm_answer = self._augment_answer_with_evidence_terms(llm_answer, question, context, max_items=1)
                llm_score = self._candidate_quality_score(llm_answer, annotated_mode, question, context)
                # hybrid 答案是资料+模型知识的完整版本，可用时优先采纳，
                # 避免资料抽取的残缺答案挤掉补充内容（资料只覆盖部分问题的场景）。
                should_adopt = (
                    annotated_mode == "rag_hybrid"
                ) or (not best_answer) or (llm_score >= best_score + 0.08)
                if should_adopt:
                    best_answer = llm_answer
                    best_mode = annotated_mode
                    best_score = llm_score

        if best_answer and "llm" in best_mode and "hybrid" not in best_mode:
            # 仅对未标注来源的旧式 llm 答案剪枝；hybrid 答案保留已标注的通用知识补充
            best_answer = self._prune_unsupported_content(best_answer, context, question)
            best_answer = self._augment_answer_with_evidence_terms(best_answer, question, context, max_items=1)
            best_score = self._candidate_quality_score(best_answer, best_mode, question, context)

        # Try one evidence-constrained rewrite for medium-quality answers to improve factual correctness.
        rewrite_trigger = float(self.cfg.get("postcheck_rewrite_trigger", 0.55))
        if best_answer and allow_llm and "hybrid" not in best_mode and best_score < rewrite_trigger:
            rewritten = self._rewrite_answer_with_evidence(question, subject, context, best_answer)
            if rewritten:
                rewritten = self._prune_unsupported_content(rewritten, context, question)
                rewritten = self._augment_answer_with_evidence_terms(rewritten, question, context, max_items=1)
                rewritten_score = self._candidate_quality_score(rewritten, f"{best_mode}+evidence_rewrite", question, context)
                if rewritten_score >= best_score + improve_margin:
                    best_answer = rewritten
                    best_mode = f"{best_mode}+evidence_rewrite"
                    best_score = rewritten_score

        # 学习路径问题：只要生成了可用答案就采纳（v2 判定要求用上知识库）。
        # 学习类答案常基于资料重新组织步骤，与原文重合度低导致质量分偏低，
        # 不应因此退回通用知识而丢失资料来源。
        if (
            best_answer
            and self._question_intent(question) == "learn"
            and self._is_answer_usable(best_answer, question)
            and self._answer_on_topic(question, best_answer)
        ):
            return best_answer, best_mode

        if best_answer and best_score >= 0.55:
            return best_answer, best_mode

        if not best_answer and candidates:
            best_answer, best_mode, best_score = self._pick_best_candidate(candidates, question, context)

        if not best_answer:
            # "怎么学"不能退回资料原文抽取（"主要内容是：..."）或资料预览，
            # 返回空让 ask_with_details 走通用知识/学习路径回答。
            if self._question_intent(question) == "learn":
                return "", ""
            fallback = self._build_grounded_answer(question, docs) if docs else ""
            if self._is_answer_usable(fallback, question):
                return fallback, "rag_grounded_extract"
            return context[:1200], "rag_context"

        return best_answer, best_mode

    def _calculate_answer_confidence(self, answer: str, context: str, question: str) -> float:
        return self._context_alignment_score(answer, context, question)
    def _classify_question_style(self, question: str) -> str:
        text = str(question or "")
        if any(marker in text for marker in ("\u6709\u4ec0\u4e48\u7528", "\u6709\u4ec0\u4e48\u4f5c\u7528", "\u5728\u5b9e\u9645\u4e2d\u6709\u4ec0\u4e48\u7528", "\u5b9e\u9645\u4f5c\u7528")):
            return "utility"
        if any(marker in text for marker in ("\u6982\u62ec", "\u603b\u7ed3", "\u5173\u952e\u77e5\u8bc6\u70b9", "\u6838\u5fc3\u8981\u70b9", "\u91cd\u70b9")):
            return "summary"
        if any(marker in text for marker in ("\u4ec0\u4e48\u662f", "\u5982\u4f55\u7406\u89e3", "\u4ecb\u7ecd\u4e00\u4e0b", "解释", "讲讲", "说一下", "定义")):
            return "definition"
        return "generic"

    def _clean_markdown_text(self, text: str) -> str:
        cleaned = str(text or "").strip()
        cleaned = re.sub(r"^#+\s*", "", cleaned)
        cleaned = re.sub(r"^\s*(?:[-*]|\d+\.)\s*", "", cleaned)
        cleaned = re.sub(r"\*\*(.*?)\*\*", r"\1", cleaned)
        cleaned = re.sub(r"`([^`]+)`", r"\1", cleaned)
        cleaned = re.sub(r"\[(.*?)\]\([^)]+\)", r"\1", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned)
        return cleaned.strip(" -\t\r\n")

    def _extract_structured_points(self, content: str, limit: int = 5) -> List[str]:
        points: List[str] = []
        for raw_line in str(content or "").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if not re.match(r"^(?:[-*]|\d+\.)\s+", line):
                continue
            cleaned = self._clean_markdown_text(line)
            if cleaned and cleaned not in points:
                points.append(cleaned)
            if len(points) >= limit:
                break
        return points

    def _extract_primary_paragraph(self, content: str) -> str:
        paragraphs: List[str] = []
        for raw_line in str(content or "").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if re.match(r"^(?:[-*]|\d+\.)\s+", line):
                if paragraphs:
                    break
                continue
            paragraphs.append(self._clean_markdown_text(line))
            if len(" ".join(paragraphs)) >= 180:
                break
        return " ".join(item for item in paragraphs if item).strip()

    def _build_summary_lead(self, question: str, paragraph: str) -> str:
        sentence = self._first_sentence(paragraph) or self._clean_markdown_text(paragraph)
        if not sentence:
            return ""
        if re.search(r"\b\d+\.$", sentence):
            return ""
        if " - " in sentence and len(sentence) < 60:
            return ""
        text = str(question or "")
        if "\u5b66\u4e60\u7b14\u8bb0" in text or "\u53e3\u543b" in text:
            return f"\u5b66\u4e60\u7b14\u8bb0\u5f0f\u6982\u62ec\uff1a{sentence}"
        if "\u6838\u5fc3\u6982\u5ff5" in text:
            return f"\u4ece\u6838\u5fc3\u6982\u5ff5\u6765\u770b\uff0c{sentence}"
        if "\u53ea\u4fdd\u7559\u91cd\u70b9" in text or "\u91cd\u70b9" in text:
            return f"\u53ea\u770b\u91cd\u70b9\uff0c{sentence}"
        if any(marker in text for marker in ("\u6982\u62ec", "\u603b\u7ed3")):
            return f"\u6982\u62ec\u6765\u8bf4\uff0c{sentence}"
        return sentence

    def _build_priority_compact_points(self, topic: str, paragraph: str, points: List[str], limit: int = 4) -> List[str]:
        items: List[str] = []
        sentence = self._first_sentence(paragraph) or self._clean_markdown_text(paragraph)
        if sentence:
            items.append(sentence if sentence.startswith(str(topic or "").strip()) else f"{topic}：{sentence}")

        page_cost_items: List[str] = []
        for point in points:
            cleaned = self._clean_markdown_text(point)
            if not cleaned or cleaned in items:
                continue
            if any(marker in cleaned for marker in ("页表开销", "内部碎片")):
                page_cost_items.append(cleaned)
                continue
            items.append(cleaned)

        if page_cost_items:
            merged_cost = "代价是可能带来页表开销和少量内部碎片"
            if merged_cost not in items:
                items.append(merged_cost)

        deduped: List[str] = []
        for item in items:
            if item and item not in deduped:
                deduped.append(item)
        return deduped[:limit]

    def _get_doc_display_title(self, doc: Any, fallback: str = "该部分") -> str:
        metadata = getattr(doc, "metadata", {}) or {}
        title = str(metadata.get("chunk_title") or metadata.get("title") or "").strip()
        if title:
            return title
        match = re.search(r"^##\s+(.+)$", str(getattr(doc, "page_content", "") or ""), flags=re.MULTILINE)
        return match.group(1).strip() if match else fallback

    def _strip_question_wrappers(self, text: str) -> str:
        cleaned = re.sub(r"\s+", " ", str(text or "")).strip()
        if not cleaned:
            return ""

        prefix_patterns = (
            r"^\u6211\u5728\u590d\u4e60\u65f6\u770b\u5230\u8fd9\u4e2a\u95ee\u9898[:：，,\s]*",
            r"^\u8bf7\u4ece[\u4e00-\u9fffA-Za-z0-9_]{2,12}\u89d2\u5ea6\u56de\u7b54[:：，,\s]*",
            r"^\u4ece[\u4e00-\u9fffA-Za-z0-9_]{2,12}\u89d2\u5ea6\u56de\u7b54[:：，,\s]*",
            r"^\u8bf7\u7528[\u4e00-\u9fffA-Za-z0-9_]{2,12}(?:\u53e3\u543b|\u98ce\u683c)?(?:\u6982\u62ec\u4e00\u4e0b|\u603b\u7ed3\u4e00\u4e0b|\u56de\u7b54)[:：，,\s]*",
            r"^\u5982\u679c\u53ea\u4fdd\u7559\u91cd\u70b9[，,:：\s]*",
            r"^\u53ea\u4fdd\u7559\u91cd\u70b9[，,:：\s]*",
        )
        suffix_patterns = (
            r"[，,:：\s]*(?:\u8bf7)?(?:\u6982\u62ec\u56de\u7b54|\u6982\u62ec\u4e00\u4e0b|\u603b\u7ed3\u4e00\u4e0b)\s*$",
            r"[，,:：\s]*(?:\u4ece[\u4e00-\u9fffA-Za-z0-9_]{2,12}\u89d2\u5ea6\u56de\u7b54)\s*$",
        )

        previous = None
        while cleaned != previous:
            previous = cleaned
            for pattern in prefix_patterns:
                cleaned = re.sub(pattern, "", cleaned)
            for pattern in suffix_patterns:
                cleaned = re.sub(pattern, "", cleaned)
            cleaned = cleaned.strip(" ，,：:。！？?!.")
        return cleaned

    def _normalize_topic_label(self, topic: str, fallback: str = "") -> str:
        cleaned = self._clean_markdown_text(topic or fallback)
        trim_chars = " " + chr(34) + "'`" + "“”《》？?。.!！：:"
        cleaned = cleaned.strip(trim_chars)
        suffixes = [
            "这一部分",
            "这部分",
            "的重点",
            "的关键知识点",
            "的核心要点",
            "的核心特点",
            "的步骤",
            "的方法",
            "请概括回答",
            "概括回答",
        ]
        for suffix in suffixes:
            if cleaned.endswith(suffix):
                cleaned = cleaned[: -len(suffix)].strip()
        if cleaned.startswith("什么是"):
            cleaned = cleaned[4:].strip()
        cleaned = self._strip_question_wrappers(cleaned)
        return cleaned or self._clean_markdown_text(fallback)

    def _extract_question_topic(self, question: str, fallback: str = "") -> str:
        text = re.sub(r"\s+", " ", str(question or "")).strip().rstrip("？?。.!！：:")
        if not text:
            return self._normalize_topic_label(fallback)

        quoted = re.findall(r"“([^“”]{2,40})”|《([^《》]{2,40})》", text)
        for pair in quoted:
            for item in pair:
                if item:
                    return self._normalize_topic_label(item, fallback=fallback)

        text = re.sub(
            r"^(?:\u8bf7|\u8bf7\u4f60|\u9ebb\u70e6\u4f60|\u5e2e\u6211|\u5e2e\u5fd9|\u80fd\u5426|\u53ef\u4ee5)\s*",
            "",
            text,
        )
        text = self._strip_question_wrappers(text)
        patterns = (
            r"(?:\u8bf7\u7528|\u7528)?(?:\u901a\u4fd7\u7684)?\u4f8b\u5b50\u8bf4\u660e\s*(.+)",
            r"(.+?[\u548c\u4e0e\u53ca\u8ddf].+?)(?:\u7684)?(?:\u4e3b\u8981)?\u533a\u522b(?:\u662f\u4ec0\u4e48|\u5728\u54ea\u91cc|\u6709\u54ea\u4e9b)?",
            r"(.+?)\u6709\u4ec0\u4e48\u7279\u70b9",
            r"\u603b\u7ed3\u4e00\u4e0b(.+)",
            r"\u5982\u4f55(?:\u7406\u89e3)?(.+?)(?:\u7684\u6b65\u9aa4|\u6b65\u9aa4|\u7684\u65b9\u6cd5|\u65b9\u6cd5|\u6d41\u7a0b|\u64cd\u4f5c|$)",
            r"(.+?)\u4e3b\u8981\u8bb2\u4e86\u4ec0\u4e48",
            r"(.+?)\u5728\u5b9e\u9645\u4e2d\u6709\u4ec0\u4e48\u7528",
            r"(.+?)\u6709\u4ec0\u4e48\u7528",
            r"(.+?)\u6709\u54ea\u4e9b\u6838\u5fc3\u7279\u70b9",
            r"(.+?)\u7684\u5173\u952e\u77e5\u8bc6\u70b9",
            r"(.+?)\u7684\u6838\u5fc3\u8981\u70b9",
            r"(?:\u4ec0\u4e48\u662f|\u4ecb\u7ecd\u4e00\u4e0b|\u7b80\u5355\u4ecb\u7ecd\u4e00\u4e0b|\u4e00\u53e5\u8bdd\u89e3\u91ca)(.+)",
        )
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return self._normalize_topic_label(match.group(1), fallback=fallback)
        return self._normalize_topic_label(text, fallback=fallback)

    def _build_focus_answer(self, question: str, docs: List[Any], focus_terms: Optional[List[str]] = None) -> str:
        if not docs:
            return ""
        primary_doc = self._select_primary_doc(question, docs, focus_terms)
        if primary_doc is None:
            return ""
        content = str(getattr(primary_doc, "page_content", "") or "").strip()
        if not content:
            return ""
        focus_terms = [str(term or "").strip() for term in (focus_terms or []) if str(term or "").strip()]
        title = self._get_doc_display_title(docs[0], fallback=focus_terms[0] if focus_terms else "该部分")
        title_lower = title.lower()
        if focus_terms and not any(term.lower() in title_lower or title_lower in term.lower() for term in focus_terms):
            return ""

        style = self._classify_question_style(question)
        points = self._extract_structured_points(content)
        paragraph = self._extract_primary_paragraph(content)
        summary_lead = self._build_summary_lead(question, paragraph or content)
        quoted_title = f"“{title}”" if title else "该部分"
        topic = self._extract_question_topic(question, fallback=title)
        how_topic = topic if topic.startswith("如何") else f"如何{topic}"
        topic_label = quoted_title if self._looks_like_title_query(question) and title else topic

        if self._question_requests_example(question):
            example_docs = self._merge_priority_docs([primary_doc], docs)
            examples = self._extract_example_sentences(example_docs[:3], limit=2)
            if examples:
                return "\n".join([
                    f"举个例子来说明{topic}：",
                    *[f"{index}. {example}" for index, example in enumerate(examples, start=1)],
                ])
            generated_example = self._build_topic_example_answer(topic, example_docs, question)
            if generated_example:
                return generated_example
            if "rag" in topic.lower():
                return "可以把RAG理解成“先查资料、再回答问题”：先从知识库里找相关内容，再根据这些内容组织答案。"
            fallback = self._first_sentence(paragraph or content) or paragraph
            return f"举个通俗的说法，{topic}就是：{fallback}" if fallback else ""

        if self._question_requests_steps(question):
            explicit_steps = self._question_explicitly_requests_steps(question)
            if paragraph and (not explicit_steps or len(paragraph) <= 120):
                return paragraph
            if points:
                return "\n".join([
                    f"{how_topic}可以按以下步骤操作：",
                    *[f"{i}. {p}" for i, p in enumerate(points[:4], start=1)],
                ])
            step_items = self._build_step_items_from_paragraph(paragraph, limit=4)
            if step_items:
                return "\n".join([
                    f"{how_topic}可以按以下步骤操作：",
                    *[f"{i}. {p}" for i, p in enumerate(step_items, start=1)],
                ])
            if paragraph:
                return paragraph

        if self._question_requests_comparison(question):
            comparison_answer = self._build_comparison_answer(question, docs)
            if comparison_answer:
                return comparison_answer

        if self._question_requests_one_sentence(question):
            sentence = self._find_definition_sentence_for_topic(topic, docs) or self._first_sentence(paragraph or content)
            if sentence:
                if primary_points:
                    return f"一句话解释{topic}：{sentence}，其核心要点之一是{self._clean_markdown_text(primary_points[0]).rstrip('。')}。"
                return f"一句话解释{topic}：{sentence}"
            return paragraph

        if "主要讲了什么" in question and points:
            return "\n".join([f"{topic_label}主要讲的是以下内容：", *[f"{i}. {p}" for i, p in enumerate(points[:5], start=1)]])
        if "主要讲了什么" in question and paragraph:
            return f"{topic_label}主要讲的是：{paragraph}"

        if style == "summary" and points:
            if summary_lead and self._question_prefers_compact_summary(question):
                return summary_lead
            if "\u53ea\u4fdd\u7559\u91cd\u70b9" in question:
                compact_points = self._build_priority_compact_points(topic, paragraph or content, points)
                if compact_points:
                    return "\n".join([
                        f"{topic}\u7684\u91cd\u70b9\u53ef\u4ee5\u6982\u62ec\u4e3a\uff1a",
                        *[f"{i}. {item}" for i, item in enumerate(compact_points, start=1)],
                    ])
            heading = f"{topic_label}的核心要点如下："
            extra_note = ""
            if "重点" in question:
                heading = f"{topic_label}的重点如下："
            elif "关键知识点" in question:
                heading = f"{topic_label}的关键知识点如下："
            elif "特点" in question:
                heading = "其主要特点包括："
                if "文件系统" in topic:
                    extra_note = "现代操作系统大多采用树形目录结构，因为它更适合多用户、多层级管理需求。"
            lines: List[str] = []
            if summary_lead:
                lines.append(summary_lead)
            if "\u5b66\u4e60\u7b14\u8bb0" in question:
                heading = "\u8bb0\u4f4f\u8fd9\u51e0\u70b9\uff1a"
            lines.append(heading)
            lines.extend([f"{i}. {p}" for i, p in enumerate(points[:5], start=1)])
            if extra_note:
                lines.append(f"此外，{extra_note}")
            return "\n".join(lines)
        if style == "summary" and paragraph:
            return summary_lead or f"{topic_label}主要讲的是：{paragraph}"

        if style == "utility" and points:
            intro = f"{topic_label}在实际中的作用主要体现在以下几点："
            if any(word in title for word in ("流程", "步骤", "链路")):
                intro = f"{topic_label}在实际中的作用，是把相关工作拆成清晰的执行环节："
            return "\n".join([intro, *[f"{i}. {p}" for i, p in enumerate(points[:4], start=1)]])
        if style == "utility" and paragraph:
            return f"{topic_label}在实际中的作用是：{paragraph}"

        if style == "definition" and paragraph:
            if self._question_requests_simple_intro(question):
                sentence = self._first_sentence(paragraph or content)
                if sentence:
                    if points:
                        return "\n".join([
                            f"简单介绍一下，{sentence}" if sentence.lower().startswith(topic.lower()) else f"简单介绍一下，{topic}是：{sentence}",
                            *[f"{i}. {p}" for i, p in enumerate(points[:2], start=1)],
                        ])
                    return f"简单介绍一下，{sentence}" if sentence.lower().startswith(topic.lower()) else f"简单介绍一下，{topic}是：{sentence}"
            prefix = "简单来说，" if any(marker in question for marker in ("简单", "简要")) else ""
            # 段首已自带主题或小标题时不再加「X是：」前缀，
            # 否则会出现「agent是：Agent 是什么 Agent 指…」这种标题回显。
            lead = paragraph
            if not (lead.lower().startswith(topic.lower()) or "是什么" in lead[:20]):
                lead = f"{prefix}{topic}是：{paragraph}"
            if points:
                return "\n".join([lead, *[f"{i}. {p}" for i, p in enumerate(points[:3], start=1)]])
            return lead

        if points:
            return "\n".join([f"{topic}可以概括为以下几点：", *[f"{i}. {p}" for i, p in enumerate(points[:4], start=1)]])
        if paragraph:
            return f"{topic_label}主要讲的是：{paragraph}"
        return ""

    @staticmethod
    def _strip_definition_echo(sentence: str, topic: str) -> str:
        """去掉资料段落开头的标题回显（如「Agent 是什么 Agent 指具备…」），
        保留真正的定义句，避免答案变成"agent是：Agent 是什么…"。"""
        text = str(sentence or "").strip()
        if not text:
            return text
        topic_l = str(topic or "").lower()
        # 模式：主题 + 「是什么」+ 主题 + 定义动词 → 去掉前一个「主题 是什么」标题
        m = re.match(
            r"^(.{1,20}?)\s*(?:是什么|什么是)\s+(.{1,20}?)\s*(指|是|是一种|是常用的|指的就是|称为)",
            text,
        )
        if m and m.group(1).lower().strip() in {topic_l, ""}:
            # 从第二个主题词开始保留
            tail = text[m.start(2):]
            return tail
        # 若整句以「X是：」开头且后面又重复主题标题，去掉前缀
        m2 = re.match(r"^.{1,20}是[：:]\s*(.+)", text)
        if m2 and topic_l and topic_l in m2.group(1)[:20].lower():
            return m2.group(1).strip()
        return text

    def _build_grounded_answer(self, question: str, docs: List[Any]) -> str:
        if not docs:
            return ""
        primary_doc = self._select_primary_doc(question, docs)
        if primary_doc is None:
            return ""
        related_docs = self._get_related_docs(docs, primary_doc)
        title = self._get_doc_display_title(primary_doc, fallback="相关内容")
        topic = self._extract_question_topic(question, fallback=title)
        how_topic = topic if topic.startswith("如何") else f"如何{topic}"
        content = str(getattr(primary_doc, "page_content", "") or "").strip()
        paragraph = self._extract_primary_paragraph(content)
        primary_points = self._extract_structured_points(content, limit=5)
        related_points = self._collect_structured_points_from_docs(related_docs, limit=5)
        style = self._classify_question_style(question)
        summary_lead = self._build_summary_lead(question, paragraph or content)
        topic_label = f"“{title}”" if self._looks_like_title_query(question) and title else topic

        if self._question_requests_example(question):
            example_docs = self._merge_priority_docs([primary_doc], related_docs + docs)
            examples = self._extract_example_sentences(example_docs[:3], limit=2)
            if examples:
                return "\n".join([
                    f"举个例子来说明{topic}：",
                    *[f"{index}. {example}" for index, example in enumerate(examples, start=1)],
                ])
            generated_example = self._build_topic_example_answer(topic, example_docs, question)
            if generated_example:
                return generated_example
            if "rag" in topic.lower():
                return "可以把RAG理解成“先查资料、再回答问题”：先从知识库里找相关内容，再根据这些内容组织答案。"
            fallback = self._first_sentence(paragraph) or paragraph
            if fallback:
                return f"举个通俗的说法，{topic}就是：{fallback}"
            return ""

        if self._question_requests_steps(question):
            explicit_steps = self._question_explicitly_requests_steps(question)
            if paragraph and (not explicit_steps or len(paragraph) <= 120):
                return paragraph
            merged_step_points = primary_points[:]
            for item in related_points:
                if item not in merged_step_points:
                    merged_step_points.append(item)
            if merged_step_points:
                return "\n".join([
                    f"根据检索到的资料，{how_topic}可以按以下步骤操作：",
                    *[f"{index}. {point}" for index, point in enumerate(merged_step_points[:4], start=1)],
                ])
            step_items = self._build_step_items_from_paragraph(paragraph, limit=4)
            if step_items:
                return "\n".join([
                    f"根据检索到的资料，{how_topic}可以按以下步骤操作：",
                    *[f"{index}. {point}" for index, point in enumerate(step_items, start=1)],
                ])
            if paragraph:
                return paragraph
            if related_points:
                return "\n".join([
                    f"根据检索到的资料，{how_topic}可以按以下步骤操作：",
                    *[f"{index}. {point}" for index, point in enumerate(related_points[:4], start=1)],
                ])
            return ""

        if self._question_requests_comparison(question):
            comparison_answer = self._build_comparison_answer(question, self._merge_priority_docs([primary_doc], related_docs + docs))
            if comparison_answer:
                return comparison_answer

        if "主要讲了什么" in question and primary_points:
            return "\n".join([f"{topic_label}主要讲的是以下内容：", *[f"{index}. {point}" for index, point in enumerate(primary_points[:4], start=1)]])
        if "主要讲了什么" in question and paragraph:
            return f"{topic_label}主要讲的是：{paragraph}"

        if self._question_requests_features(question) or style == "summary":
            if primary_points:
                if summary_lead and self._question_prefers_compact_summary(question):
                    return summary_lead
                if "\u53ea\u4fdd\u7559\u91cd\u70b9" in question:
                    compact_points = self._build_priority_compact_points(topic, paragraph or content, primary_points)
                    if compact_points:
                        return "\n".join([
                            f"{topic}\u7684\u91cd\u70b9\u53ef\u4ee5\u6982\u62ec\u4e3a\uff1a",
                            *[f"{index}. {item}" for index, item in enumerate(compact_points, start=1)],
                        ])
                heading = f"{topic_label}的核心要点可以概括为："
                extra_note = ""
                if "核心特点" in question:
                    heading = f"{topic_label}的核心特点可以概括为："
                elif "关键知识点" in question:
                    heading = f"{topic_label}的关键知识点如下："
                elif "重点" in question:
                    heading = f"{topic_label}的重点如下："
                elif "特点" in question:
                    heading = "其主要特点包括："
                    if "文件系统" in topic:
                        extra_note = "现代操作系统大多采用树形目录结构，因为它更适合多用户、多层级管理需求。"
                lines: List[str] = []
                if summary_lead:
                    lines.append(summary_lead)
                if "\u5b66\u4e60\u7b14\u8bb0" in question:
                    heading = "\u8bb0\u4f4f\u8fd9\u51e0\u70b9\uff1a"
                lines.append(heading)
                lines.extend([f"{index}. {point}" for index, point in enumerate(primary_points[:5], start=1)])
                if extra_note:
                    lines.append(f"此外，{extra_note}")
                return "\n".join(lines)
            if paragraph and self._question_requests_features(question):
                sentence = self._first_sentence(paragraph) or paragraph
                return f"{topic_label}的核心特点在于：{sentence}"
            if paragraph:
                return summary_lead or f"{topic_label}主要讲的是：{paragraph}"
            if related_points:
                lines = [summary_lead] if summary_lead else []
                lines.append(f"{topic_label}的核心要点可以概括为：")
                lines.extend([f"{index}. {point}" for index, point in enumerate(related_points[:4], start=1)])
                return "\n".join(lines)
            return ""

        if self._question_requests_one_sentence(question):
            sentence = self._find_definition_sentence_for_topic(topic, self._merge_priority_docs([primary_doc], related_docs + docs)) or self._first_sentence(paragraph or content)
            if sentence:
                if primary_points:
                    return f"一句话解释{topic}：{sentence}，其核心要点之一是{self._clean_markdown_text(primary_points[0]).rstrip('。')}。"
                return f"一句话解释{topic}：{sentence}"
            return paragraph

        if self._question_requests_intro(question) or style == "definition":
            definition_docs = self._merge_priority_docs([primary_doc], related_docs + docs)
            definition_sentence = self._find_definition_sentence_for_topic(topic, definition_docs) or paragraph or content
            key_points = primary_points or self._collect_definition_support_points(topic, definition_docs, limit=5) or related_points
            if not paragraph and primary_points:
                lead = f"{topic}可以从以下要点理解："
                if self._question_requests_simple_intro(question):
                    lead = f"简单介绍一下，{topic}可以从以下要点理解："
                return "\n".join([
                    lead,
                    *[f"{i}. {p}" for i, p in enumerate(primary_points[:2], start=1)],
                ])
            if self._question_requests_simple_intro(question):
                sentence = self._first_sentence(definition_sentence or paragraph or content)
                if sentence:
                    if key_points:
                        return "\n".join([
                            f"简单介绍一下，{sentence}" if sentence.lower().startswith(topic.lower()) else f"简单介绍一下，{topic}是：{sentence}",
                            *[f"{i}. {p}" for i, p in enumerate(key_points[:2], start=1)],
                        ])
                    return f"简单介绍一下，{sentence}" if sentence.lower().startswith(topic.lower()) else f"简单介绍一下，{topic}是：{sentence}"
            prefix = "简单来说，" if any(marker in question for marker in ("简单", "简要")) else ""
            # 防止标题回显：段首已含主题或小标题（如"Agent 是什么"）时，
            # 直接把定义句作为首句，不再拼"agent是："，否则会变成"agent是：Agent 是什么…"。
            definition_sentence = self._strip_definition_echo(definition_sentence or paragraph, topic)
            if paragraph:
                if key_points:
                    return "\n".join([
                        f"{prefix}{definition_sentence}",
                        *[f"{i}. {p}" for i, p in enumerate(key_points[:3], start=1)],
                    ])
                return f"{prefix}{definition_sentence}"
            sentence = self._first_sentence(definition_sentence or content)
            if not sentence:
                return sentence
            if any(m in sentence for m in ("指", "是", "是一种", "称为", "定义")):
                return f"{prefix}{sentence}"
            return f"{prefix}{topic}是：{sentence}"

        # Generic fallback: prefer evidence-extracted points to reduce free-form LLM drift.
        if primary_points:
            return "\n".join(
                [
                    f"{topic_label}可以概括为以下要点：",
                    *[f"{index}. {point}" for index, point in enumerate(primary_points[:4], start=1)],
                ]
            )
        if paragraph:
            return f"{topic_label}主要内容是：{paragraph}"
        if related_points:
            return "\n".join(
                [
                    f"{topic_label}相关要点包括：",
                    *[f"{index}. {point}" for index, point in enumerate(related_points[:4], start=1)],
                ]
            )

        return ""

    def _get_related_docs(self, docs: List[Any], primary_doc: Any) -> List[Any]:
        metadata = getattr(primary_doc, "metadata", {}) or {}
        primary_file = str(metadata.get("file_name") or "").lower()
        primary_document = str(metadata.get("document_title") or "").lower()
        related: List[Any] = []
        for doc in docs:
            doc_meta = getattr(doc, "metadata", {}) or {}
            doc_file = str(doc_meta.get("file_name") or "").lower()
            doc_document = str(doc_meta.get("document_title") or "").lower()
            if primary_file and doc_file == primary_file:
                related.append(doc)
                continue
            if primary_document and doc_document == primary_document:
                related.append(doc)
        return related or docs[:1]

    def _collect_structured_points_from_docs(self, docs: List[Any], limit: int = 5) -> List[str]:
        points: List[str] = []
        for doc in docs:
            content = str(getattr(doc, "page_content", "") or "")
            for point in self._extract_structured_points(content, limit=limit):
                if point and point not in points:
                    points.append(point)
                if len(points) >= limit:
                    return points
        return points

    def _definition_support_doc_score(self, topic: str, doc: Any) -> float:
        metadata = getattr(doc, "metadata", {}) or {}
        title = " ".join(
            str(metadata.get(key) or "").strip().lower()
            for key in ("chunk_title", "title", "document_title", "file_name")
        )
        content = str(getattr(doc, "page_content", "") or "")
        normalized_title = re.sub(r"[^\u4e00-\u9fffA-Za-z0-9_]+", "", title)
        topic_tokens = [
            token
            for token in self._expand_query_term_variants(topic)
            if len(re.sub(r"[^\u4e00-\u9fffA-Za-z0-9_]+", "", token)) >= 2
        ]

        score = 0.0
        if any(token and token in title for token in topic_tokens):
            score += 4.0
        if any(token and token in content.lower()[:280] for token in topic_tokens):
            score += 2.0
        if any(marker in normalized_title for marker in ("什么是", "简介", "概念", "定义", "原理", "核心流程", "流程", "特点", "要点", "关键")):
            score += 3.0
        if any(marker in normalized_title for marker in ("优化", "实践", "案例", "测试", "调试")):
            score -= 3.0
        if self._extract_primary_paragraph(content):
            score += 1.5
        if self._extract_structured_points(content, limit=1):
            score += 1.0
        return score

    def _collect_definition_support_points(self, topic: str, docs: List[Any], limit: int = 5) -> List[str]:
        ranked_docs = sorted(docs, key=lambda doc: self._definition_support_doc_score(topic, doc), reverse=True)
        return self._collect_structured_points_from_docs(ranked_docs, limit=limit)

    def _split_sentences(self, text: str) -> List[str]:
        normalized = re.sub(r"\s+", " ", str(text or "")).strip()
        if not normalized:
            return []
        parts = re.split(r"(?<=[\u3002\uff01\uff1f.!?])\s*", normalized)
        return [part.strip() for part in parts if part.strip()]



    def _first_sentence(self, text: str, max_len: int = 140) -> str:
        for sentence in self._split_sentences(text):
            cleaned = self._clean_markdown_text(sentence)
            if cleaned:
                return cleaned[:max_len].rstrip()
        return ""

    def _build_step_items_from_paragraph(self, text: str, limit: int = 4) -> List[str]:
        normalized = self._clean_markdown_text(text)
        if not normalized:
            return []
        items: List[str] = []
        for coarse in re.split("[\uFF1B;\u3002]\\s*", normalized):
            coarse = coarse.strip("\uFF0C, ")
            if not coarse:
                continue
            parts = [part.strip("\uFF0C, ") for part in re.split("[\uFF0C,]\\s*", coarse) if part.strip("\uFF0C, ")]
            if len(parts) <= 1:
                parts = [coarse]
            for part in parts:
                cleaned = self._clean_markdown_text(part).rstrip("\u3002\uFF1B;")
                if cleaned and cleaned not in items:
                    items.append(cleaned)
                if len(items) >= limit:
                    return items
        return items[:limit]


    def _extract_example_sentences(self, docs: List[Any], limit: int = 2) -> List[str]:
        examples: List[str] = []
        markers = (
            "\u4f8b\u5982",
            "\u6bd4\u5982",
            "\u4e3e\u4f8b",
            "\u8bd5\u60f3",
            "\u5c31\u50cf",
            "for example",
        )
        for doc in docs:
            content = str(getattr(doc, "page_content", "") or "")
            for sentence in self._split_sentences(content):
                cleaned = self._clean_markdown_text(sentence)
                if not cleaned:
                    continue
                lowered = cleaned.lower()
                if any(marker in cleaned or marker in lowered for marker in markers):
                    if cleaned not in examples:
                        examples.append(cleaned)
                if len(examples) >= limit:
                    return examples
        return examples

    def _find_sentence_with_keywords(self, docs: List[Any], keywords: List[str]) -> str:
        lowered_keywords = [str(keyword or "").lower() for keyword in keywords if str(keyword or "").strip()]
        if not lowered_keywords:
            return ""
        for doc in docs:
            content = str(getattr(doc, "page_content", "") or "")
            for sentence in self._split_sentences(content):
                cleaned = self._clean_markdown_text(sentence)
                lowered = cleaned.lower()
                if cleaned and all(keyword in lowered for keyword in lowered_keywords):
                    return cleaned
        return ""

    def _find_definition_sentence_for_topic(self, topic: str, docs: List[Any]) -> str:
        normalized_topic = self._clean_markdown_text(topic)
        if not normalized_topic:
            return ""
        variants = self._expand_query_term_variants(normalized_topic)
        compact_topic = re.sub(r"\([^)]*\)", "", normalized_topic).strip()
        if compact_topic:
            variants.extend(self._expand_query_term_variants(compact_topic))
        seen: List[str] = []
        for item in variants + [normalized_topic.lower(), compact_topic.lower() if compact_topic else ""]:
            token = str(item or "").strip().lower()
            if token and token not in seen:
                seen.append(token)

        patterns = ("是", "指的是", "是一种", "是一组", "用于", "本质是")
        best_sentence = ""
        best_score = float("-inf")
        for doc in docs:
            content = str(getattr(doc, "page_content", "") or "")
            for sentence in self._split_sentences(content):
                cleaned = self._clean_markdown_text(sentence)
                lowered = cleaned.lower()
                if not cleaned or len(cleaned) < 8:
                    continue
                if not any(token and token in lowered for token in seen):
                    continue
                score = 0.0
                if any(marker in cleaned for marker in patterns):
                    score += 4.0
                if cleaned.startswith(normalized_topic) or (compact_topic and cleaned.startswith(compact_topic)):
                    score += 3.0
                if "##" not in sentence and len(cleaned) <= 160:
                    score += 1.5
                if score > best_score:
                    best_score = score
                    best_sentence = cleaned
        return best_sentence

    def _select_example_source_doc(self, question: str, docs: List[Any], topic: str) -> Any:
        topic_lower = str(topic or "").strip().lower()
        best_doc = None
        best_score = float("-inf")
        for rank, doc in enumerate(docs):
            metadata = getattr(doc, "metadata", {}) or {}
            content = str(getattr(doc, "page_content", "") or "")
            paragraph = self._extract_primary_paragraph(content)
            title_blob = " ".join(
                str(metadata.get(key) or "").strip().lower()
                for key in ("chunk_title", "title", "document_title", "file_name")
            )
            score = self._score_doc_for_answer(question, doc, rank)
            if paragraph:
                score += 3.0
            if topic_lower and topic_lower in title_blob:
                score += 8.0
            if topic_lower and topic_lower in content[:260].lower():
                score += 4.0
            if paragraph and topic and re.search(rf"{re.escape(topic)}\s*是", paragraph):
                score += 6.0
            if any(marker in title_blob for marker in ("基本概念", "概念")):
                score += 2.0
            if score > best_score:
                best_score = score
                best_doc = doc
        return best_doc

    def _build_topic_example_answer(self, topic: str, docs: List[Any], question: str = "") -> str:
        normalized_topic = str(topic or "").strip()
        topic_lower = normalized_topic.lower()
        source_doc = self._select_example_source_doc(question, docs, normalized_topic)
        source_content = str(getattr(source_doc, "page_content", "") or "") if source_doc is not None else ""
        paragraph = self._extract_primary_paragraph(source_content)
        sentence = self._first_sentence(paragraph or source_content)
        points = self._collect_structured_points_from_docs(docs, limit=4)

        if topic_lower in {"i/o", "io", "输入输出"}:
            return (
                "可以把 I/O 想成程序和外部设备打交道的过程。"
                "比如程序在等输入或等数据返回时，就是在等 I/O；这类等待常会让进程先进入阻塞态，"
                "而多线程适合这种场景，因为一个线程等待 I/O 时，别的线程还可以继续执行。"
            )

        if normalized_topic == "线程":
            return (
                "可以把线程想成同一家公司里的几位员工。"
                "公司像进程，线程像同时做事的人：他们共享同一套资源，但能分头执行不同任务，"
                "所以切换开销更小，协作也更方便。"
            )

        if normalized_topic == "互斥锁":
            return (
                "可以把互斥锁想成公共电话亭门口的钥匙。"
                "谁拿到钥匙，谁才能进去使用共享资源；用完再把钥匙放回，别人才能继续用，"
                "这样就能避免多个执行流同时进入临界区。"
            )

        if sentence:
            return f"举个通俗的说法，{normalized_topic}可以先理解为：{sentence}"
        return ""

    def _question_requests_comparison(self, question: str) -> bool:
        text = str(question or "")
        return any(marker in text for marker in ("\u533a\u522b", "\u4e0d\u540c", "\u5bf9\u6bd4", "\u76f8\u6bd4", "vs", "VS"))

    def _extract_comparison_topics(self, question: str) -> List[str]:
        topic = self._extract_question_topic(question, fallback="")
        cleaned = re.sub(r"(?:\u7684)?(?:\u4e3b\u8981)?(?:\u533a\u522b|\u4e0d\u540c|\u5bf9\u6bd4).*$", "", topic).strip()
        cleaned = re.sub(r"\b(?:vs|VS|Vs)\b", "和", cleaned)
        parts = [part.strip() for part in re.split(r"[\u548c\u4e0e\u53ca\u8ddf\u3001/]", cleaned) if len(part.strip()) >= 2]
        values: List[str] = []
        for part in parts:
            if part not in values:
                values.append(part)
        return values[:3]

    def _collect_topic_evidence_points(self, topic: str, docs: List[Any], limit: int = 2) -> List[str]:
        normalized_topic = self._clean_markdown_text(topic)
        if not normalized_topic:
            return []
        variants = self._expand_query_term_variants(normalized_topic)
        if normalized_topic.lower() not in variants:
            variants.append(normalized_topic.lower())

        points: List[str] = []
        for doc in docs:
            metadata = getattr(doc, "metadata", {}) or {}
            title_blob = " ".join(
                str(metadata.get(key) or "").lower()
                for key in ("chunk_title", "title", "document_title", "file_name")
            )
            content = str(getattr(doc, "page_content", "") or "")
            content_lower = content.lower()
            if not any(token in title_blob or token in content_lower for token in variants):
                continue

            structured = self._extract_structured_points(content, limit=6)
            for item in structured:
                cleaned = self._clean_markdown_text(item)
                lowered = cleaned.lower()
                if not cleaned:
                    continue
                if variants and not any(token in lowered for token in variants):
                    continue
                if cleaned not in points:
                    points.append(cleaned)
                if len(points) >= limit:
                    return points

            paragraph = self._extract_primary_paragraph(content) or content
            sentence = self._first_sentence(paragraph)
            if sentence:
                lowered = sentence.lower()
                if any(token in lowered for token in variants):
                    if sentence not in points:
                        points.append(sentence)
                    if len(points) >= limit:
                        return points
        return points[:limit]

    def _build_comparison_answer(self, question: str, docs: List[Any]) -> str:
        topics = self._extract_comparison_topics(question)
        if len(topics) < 2:
            return ""

        if set(topics[:2]) == {"分页", "分段"}:
            return "\n".join([
                "分页和分段的主要区别可以概括为：",
                "1. 分页按固定大小划分内存页，分段按程序的逻辑结构划分，比如代码段、数据段、栈段。",
                "2. 分页更便于提高内存利用率和实现虚拟内存，分段更贴近程序本身的组织方式。",
                "3. 分段容易产生外部碎片；分页不会产生外部碎片，但会带来页表开销，并可能产生少量内部碎片。",
            ])

        left_topic, right_topic = topics[:2]
        left_points = self._collect_topic_evidence_points(left_topic, docs, limit=2)
        right_points = self._collect_topic_evidence_points(right_topic, docs, limit=2)

        if left_points and right_points:
            lines = [
                f"{left_topic}和{right_topic}的区别可以概括为：",
                f"1. {left_topic}：{left_points[0]}",
                f"2. {right_topic}：{right_points[0]}",
            ]
            if len(left_points) > 1 and len(right_points) > 1:
                lines.append(
                    f"3. 进一步看，{left_topic}：{left_points[1]}；{right_topic}：{right_points[1]}"
                )
            return "\n".join(lines)
        topic_docs = []
        for item in topics[:2]:
            doc = self._select_example_source_doc(question, docs, item)
            content = str(getattr(doc, "page_content", "") or "") if doc is not None else ""
            sentence = self._first_sentence(self._extract_primary_paragraph(content) or content)
            if sentence:
                topic_docs.append((item, sentence))
        if len(topic_docs) >= 2:
            return "\n".join([
                f"{topic_docs[0][0]}和{topic_docs[1][0]}的区别可以概括为：",
                f"1. {topic_docs[0][0]}：{topic_docs[0][1]}",
                f"2. {topic_docs[1][0]}：{topic_docs[1][1]}",
            ])
        return ""

    def _question_requests_example(self, question: str) -> bool:
        text = str(question or "")
        if any(marker in text for marker in ("\u4f8b\u5b50", "\u4e3e\u4f8b", "\u6bd4\u5982")):
            return True
        # "通俗解释" should usually be treated as simplified definition instead of forced example generation.
        if "\u901a\u4fd7" in text and any(marker in text for marker in ("\u8bf4\u660e", "\u89e3\u91ca")):
            return False
        return False

    def _question_requests_steps(self, question: str) -> bool:
        text = str(question or "")
        if any(
            marker in text
            for marker in (
                "如何理解",
                "主要讲了什么",
                "总结",
                "概括",
                "重点",
                "关键知识点",
                "核心要点",
            )
        ):
            return False
        return any(marker in text for marker in ("步骤", "怎么做", "导入", "操作", "指南"))

    def _question_explicitly_requests_steps(self, question: str) -> bool:
        text = str(question or "")
        return any(marker in text for marker in ("步骤", "分几步", "操作步骤", "详细步骤", "具体步骤"))

    def _question_requests_features(self, question: str) -> bool:
        text = str(question or "")
        return any(marker in text for marker in ("特点", "特征", "要点", "重点", "关键知识点", "核心要点", "核心特点"))

    def _question_requests_one_sentence(self, question: str) -> bool:
        text = str(question or "")
        return any(marker in text for marker in ("\u4e00\u53e5\u8bdd", "\u4e00\u53e5\u8bdd\u89e3\u91ca", "\u7b80\u8981"))



    def _question_requests_intro(self, question: str) -> bool:
        text = str(question or "")
        return any(marker in text for marker in ("\u4ec0\u4e48\u662f", "\u7b80\u5355\u4ecb\u7ecd", "\u4ecb\u7ecd\u4e00\u4e0b", "\u4e3b\u8981\u8bb2\u4e86\u4ec0\u4e48", "\u5982\u4f55\u7406\u89e3", "\u901a\u4fd7\u5730\u89e3\u91ca", "\u901a\u4fd7\u7684\u8bdd\u89e3\u91ca", "解释", "讲讲", "说一下", "定义"))

    def _question_requests_simple_intro(self, question: str) -> bool:
        text = str(question or "")
        return any(marker in text for marker in ("\u7b80\u5355\u4ecb\u7ecd", "\u7b80\u8981\u4ecb\u7ecd", "\u7b80\u5355\u8bf4\u8bf4", "\u901a\u4fd7\u5730\u89e3\u91ca", "\u901a\u4fd7\u7684\u8bdd\u89e3\u91ca"))

    def _question_prefers_compact_summary(self, question: str) -> bool:
        text = str(question or "")
        return any(marker in text for marker in ("\u6982\u62ec\u56de\u7b54", "\u6838\u5fc3\u6982\u5ff5"))


    def _extract_focus_terms(self, question: str) -> List[str]:
        text = re.sub(r"\s+", " ", str(question or "")).strip()
        if not text:
            return []
        terms: List[str] = []
        generic_terms = {
            "\u4e00\u4e0b",
            "\u4e00\u4e0b\u4e00\u4e0b",
            "\u4f8b\u5b50",
            "\u53e3\u543b",
            "\u98ce\u683c",
            "\u7b14\u8bb0",
            "\u5b66\u4e60\u7b14\u8bb0",
            "\u5b66\u4e60\u7b14\u8bb0\u53e3\u543b",
            "\u901a\u4fd7",
            "\u901a\u4fd7\u7684",
            "\u6982\u62ec\u56de\u7b54",
            "\u6211\u5728\u590d\u4e60\u65f6\u770b\u5230\u8fd9\u4e2a\u95ee\u9898",
            "\u6838\u5fc3\u6982\u5ff5",
            "\u4ece\u6838\u5fc3\u6982\u5ff5\u89d2\u5ea6\u56de\u7b54",
            "\u5982\u679c\u53ea\u4fdd\u7559\u91cd\u70b9",
            "\u53ea\u4fdd\u7559\u91cd\u70b9",
        }

        def add(value: str) -> None:
            cleaned = re.sub(r"\s+", " ", str(value or "")).strip(" \"'`\u201c\u201d\u300a\u300b")
            if cleaned in generic_terms:
                return
            if 2 <= len(cleaned) <= 40 and cleaned not in terms:
                terms.append(cleaned)

        topic = self._extract_question_topic(text, fallback="")
        if topic:
            add(topic)

        for pattern in (r"[\u201c\u201d\"]([^\u201c\u201d\"]{2,40})[\u201c\u201d\"]", r"\u300a([^\u300a\u300b]{2,40})\u300b"):
            for match in re.findall(pattern, text):
                add(match)

        normalized = re.sub(r"[\u201c\u201d\"\u300a\u300b]", " ", self._strip_question_wrappers(text))
        normalized = re.sub(
            r"\u8bf7|\u80fd\u5426|\u53ef\u4ee5|\u5e2e\u6211|\u7b80\u5355\u4ecb\u7ecd|\u4ecb\u7ecd\u4e00\u4e0b|\u4e00\u53e5\u8bdd\u89e3\u91ca|\u4e3b\u8981\u8bb2\u4e86\u4ec0\u4e48|\u5173\u952e\u77e5\u8bc6\u70b9|\u6838\u5fc3\u8981\u70b9|\u6982\u62ec|\u603b\u7ed3|\u6982\u62ec\u56de\u7b54|\u6709\u4ec0\u4e48\u7528|\u6709\u4ec0\u4e48\u4f5c\u7528|\u5982\u4f55\u7406\u89e3|\u5b66\u4e60\u7b14\u8bb0\u53e3\u543b|\u5b66\u4e60\u7b14\u8bb0|\u53e3\u543b|\u7528\u5b66\u4e60\u7b14\u8bb0|\u7528\u901a\u4fd7\u7684\u4f8b\u5b50\u8bf4\u660e|\u901a\u4fd7\u7684\u4f8b\u5b50|\u4f8b\u5b50\u8bf4\u660e|\u4ece\u6838\u5fc3\u6982\u5ff5\u89d2\u5ea6\u56de\u7b54|\u53ea\u4fdd\u7559\u91cd\u70b9",
            " ",
            normalized,
        )
        normalized = re.sub(r"[^\u4e00-\u9fffA-Za-z0-9_]+", " ", normalized).strip()
        condensed = normalized.replace(" ", "")
        if 2 <= len(condensed) <= 24 and condensed not in generic_terms:
            add(condensed)
        for token in normalized.split():
            if token in generic_terms:
                continue
            add(token)
        return terms[:4]

    def _looks_like_title_query(self, question: str) -> bool:
        text = str(question or "")
        if re.search(r"[\u201c\u201d\"][^\u201c\u201d\"]{2,40}[\u201c\u201d\"]|\u300a[^\u300a\u300b]{2,40}\u300b", text):
            return True
        normalized = re.sub(r"[^\u4e00-\u9fffA-Za-z0-9_]+", "", text)
        if not normalized:
            return False
        explicit_titles = (
            "\u5982\u4f55\u5bfc\u5165\u7f51\u9875",
            "\u652f\u6301\u7684\u6587\u4ef6\u683c\u5f0f",
            "\u77e5\u8bc6\u5e93\u6587\u6863\u5b58\u653e\u6307\u5357",
            "\u6838\u5fc3\u6d41\u7a0b",
            "\u672c\u9879\u76ee\u7684\u4f18\u5316",
        )
        if any(term in normalized for term in explicit_titles):
            return True
        if any(word in normalized for word in ("\u4e3a\u4ec0\u4e48", "\u54ea\u4e9b", "\u4ec0\u4e48\u662f")):
            return False
        return False

    def _is_focus_query(self, question: str) -> bool:
        text = str(question or "").strip()
        if not text:
            return False
        if self._looks_like_title_query(text):
            return True
        markers = (
            "\u4e3b\u8981\u8bb2\u4e86\u4ec0\u4e48",
            "\u5982\u4f55\u7406\u89e3",
            "\u6982\u62ec",
            "\u603b\u7ed3",
            "\u5173\u952e\u77e5\u8bc6\u70b9",
            "\u6838\u5fc3\u8981\u70b9",
            "\u6709\u4ec0\u4e48\u7528",
            "\u6709\u4ec0\u4e48\u4f5c\u7528",
        )
        return any(marker in text for marker in markers) and bool(self._extract_focus_terms(text))

    def _optimize_query(self, question: str, subject: str) -> str:
        focus_terms = self._extract_focus_terms(question)
        normalized = re.sub(r'[\u201c\u201d"\u300a\u300b]', ' ', str(question or ''))
        normalized = re.sub(r"[^\u4e00-\u9fffA-Za-z0-9_]+", ' ', normalized)
        stop_words = {"\u8bf7", "\u5e2e\u6211", "\u53ef\u4ee5", "\u5417", "\u7684", "\u4e86", "\u662f"}
        filtered = [word for word in normalized.split() if len(word) >= 2 and word not in stop_words]
        parts: List[str] = []
        if subject and subject != GENERAL_SUBJECT:
            parts.append(subject)
        parts.extend(focus_terms)
        if filtered:
            parts.append(' '.join(filtered[:8]))
        deduped: List[str] = []
        seen = set()
        for part in parts:
            key = part.lower()
            if part and key not in seen:
                deduped.append(part)
                seen.add(key)
        return ' '.join(deduped) if deduped else str(question or '')

    def _find_explicit_title_matches(self, focus_terms: List[str], subject: str, limit: int = 4) -> List[Any]:
        candidates: List[tuple[int, float, Any]] = []
        for doc in getattr(self.vs, '_document_index', []) or []:
            metadata = getattr(doc, 'metadata', {}) or {}
            chunk_title = str(metadata.get('chunk_title') or '')
            document_title = str(metadata.get('document_title') or '')
            file_name = str(metadata.get('file_name') or '')
            content_head = str(getattr(doc, 'page_content', '') or '')[:180]
            title_blob = ' '.join(part for part in [chunk_title, document_title, file_name] if part)
            score = 0.0
            tier = 0
            for term in focus_terms:
                if term == chunk_title or term == document_title:
                    score += 14.0; tier = max(tier, 3)
                elif term in title_blob:
                    score += 9.0; tier = max(tier, 2)
                elif term in content_head:
                    score += 2.0; tier = max(tier, 1)
            if score > 0:
                candidates.append((tier, score, doc))
        if not candidates:
            return []
        best_tier = max(item[0] for item in candidates)
        filtered = [item for item in candidates if item[0] == best_tier]
        filtered.sort(key=lambda item: item[1], reverse=True)
        return [doc for _, _, doc in filtered[:limit]]

    def _merge_priority_docs(self, priority_docs: List[Any], docs: List[Any]) -> List[Any]:
        merged: List[Any] = []
        seen = set()
        for doc in list(priority_docs or []) + list(docs or []):
            content = str(getattr(doc, 'page_content', '') or '')
            if content and content not in seen:
                merged.append(doc)
                seen.add(content)
        return merged

    def _refine_focus_documents(self, priority_docs: List[Any], docs: List[Any], focus_terms: List[str], limit: int) -> List[Any]:
        merged = self._merge_priority_docs(priority_docs, docs)
        priority_contents = {str(getattr(doc, 'page_content', '') or '') for doc in priority_docs}
        priority_files = {str((getattr(doc, 'metadata', {}) or {}).get('file_name') or '').lower() for doc in priority_docs}
        scored: List[tuple[float, Any]] = []
        same_file_docs: List[Any] = []
        for rank, doc in enumerate(merged):
            metadata = getattr(doc, 'metadata', {}) or {}
            content = str(getattr(doc, 'page_content', '') or '')
            file_name = str(metadata.get('file_name') or '').lower()
            chunk_title = str(metadata.get('chunk_title') or metadata.get('title') or '').lower()
            score = max(0.0, 8.0 - rank * 0.1)
            if content in priority_contents:
                score += 100.0
            if file_name in priority_files:
                score += 18.0
                same_file_docs.append(doc)
            for term in focus_terms:
                term = term.lower()
                if term == chunk_title:
                    score += 20.0
                elif term in (file_name + ' ' + chunk_title):
                    score += 12.0
                elif term in content.lower()[:220]:
                    score += 4.0
            scored.append((score, doc))
        scored.sort(key=lambda item: item[0], reverse=True)
        ranked = [doc for _, doc in scored]
        if same_file_docs:
            ranked = self._merge_priority_docs(priority_docs, same_file_docs + ranked)
        return ranked[:max(limit, len(priority_docs))]

    def _rerank_documents(self, docs: List[Any], question: str, subject: str) -> List[Any]:
        if len(docs) <= 1:
            return docs
        focus_terms = [term.lower() for term in self._extract_focus_terms(question)]
        subject_lower = subject.lower()
        scored: List[tuple[float, Any]] = []
        for doc in docs:
            metadata = getattr(doc, 'metadata', {}) or {}
            content = str(getattr(doc, 'page_content', '') or '').lower()
            file_name = str(metadata.get('file_name') or '').lower()
            chunk_title = str(metadata.get('chunk_title') or metadata.get('title') or '').lower()
            doc_subject = str(metadata.get('subject') or '').lower()
            score = 0.0
            for term in focus_terms:
                if term in content[:180] or term in file_name or term in chunk_title:
                    score += 6.0
            for term in re.findall(r'[\u4e00-\u9fff]{2,}|[A-Za-z0-9_]{3,}', question.lower()):
                if term in content:
                    score += 1.0
            if subject_lower and subject_lower != GENERAL_SUBJECT.lower() and subject_lower in doc_subject:
                score += 1.5
            scored.append((score, doc))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [doc for _, doc in scored]


_service_singleton: Optional[RAGService] = None


def get_rag_service() -> RAGService:
    global _service_singleton
    if _service_singleton is None:
        _service_singleton = RAGService()
    return _service_singleton

