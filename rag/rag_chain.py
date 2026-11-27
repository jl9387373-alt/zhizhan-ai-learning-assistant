"""RAG 助手兼容层：封装 RAGService，保留旧接口"""

from typing import Optional, List, Dict, Any

from .service import get_rag_service

class RAGHelper:
    _instance: Optional["RAGHelper"] = None

    def __init__(self) -> None:
        self._service = get_rag_service()

    @classmethod
    def shared(cls) -> "RAGHelper":
        if cls._instance is None:
            cls._instance = RAGHelper()
        return cls._instance

    def ensure_ready(self) -> bool:
        return self._service.ensure_ready()

    def ask_with_details(self, question: str, subject: str = "通用", chat_history: str = "", image_data: str = None) -> Optional[Dict[str, Any]]:
        return self._service.ask_with_details(question, subject=subject, chat_history=chat_history, image_data=image_data)

    def ask_with_details_stream(self, question: str, subject: str = "通用", chat_history: str = "", image_data: str = None):
        return self._service.ask_with_details_stream(question, subject=subject, chat_history=chat_history, image_data=image_data)

    def ask(self, question: str, subject: str = "通用", chat_history: str = "") -> Optional[str]:
        details = self.ask_with_details(question, subject=subject, chat_history=chat_history)
        return details.get("answer") if details else None

    def ingest_texts(self, texts: List[str], subject: str = "通用", metadata: Optional[List[Dict[str, Any]]] = None) -> int:
        return self._service.ingest(texts, subject=subject, metadata=metadata)

    def rebuild(self) -> bool:
        return self._service.rebuild()

    def generate_mindmap(self, topic: str) -> str:
        return self._service.generate_mindmap(topic)


def get_rag_helper() -> RAGHelper:
    return RAGHelper.shared()
    

