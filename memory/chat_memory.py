"""
对话记忆管理器（LangChain风格的双层记忆架构）

实现了符合LangChain风格的"短期对话记忆 + 向量化长期记忆"混合架构：
- 短期记忆：维护当前会话上下文，使用类似ConversationBufferMemory的实现
- 长期记忆：跨会话知识沉淀，结合Embedding和向量数据库实现，本质上是可写入的RAG机制

在保证对话连贯性的同时，避免token膨胀，并支持长期知识复用。
"""

import os
import json
import datetime
import collections
from typing import Dict, List, Optional, Tuple, Union

# LangChain安全导入
def safe_import_langchain():
    try:
        from langchain_community.vectorstores import FAISS
        from langchain_core.documents import Document
        
        # 尝试导入最新版本的HuggingFaceEmbeddings
        try:
            from langchain_huggingface import HuggingFaceEmbeddings
        except ImportError:
            # 兼容旧版本
            from langchain_community.embeddings import HuggingFaceEmbeddings
            
        return {
            "FAISS": FAISS,
            "HuggingFaceEmbeddings": HuggingFaceEmbeddings,
            "Document": Document
        }
    except ImportError as e:
        print(f"LangChain导入失败: {e}")
        return None

# 导入项目内部模块
from memory.database_memory import DatabaseConnector, get_database_connector
from core.logger import logger


class ConversationMemoryManager:
    """管理多用户的对话记忆（LangChain风格的双层记忆架构）
    
    实现双层存储架构：
    - 短期记忆：使用双端队列存储当前会话的完整对话，类似LangChain的ConversationBufferMemory
    - 长期记忆：使用向量数据库存储对话摘要的embedding，本质上是可写入的RAG机制
    """

    def __init__(self, max_turns: int = 10, max_short_term: int = 20, memory_threshold: int = 10, db_type: str = "mongodb", **db_kwargs) -> None:
        # 短期记忆：user_id -> deque[((human, ai), ...)]
        self._short_term_memory: Dict[str, collections.deque] = {}
        # 长期记忆：user_id -> FAISS向量存储
        self._long_term_memory: Dict[str, Dict] = {}
        
        self.max_turns = max_turns  # 每个会话返回的最大轮数
        self.max_short_term = max_short_term  # 短期记忆的最大容量
        self.memory_threshold = memory_threshold  # 触发向长期记忆转换的阈值
        
        # 延迟加载LangChain组件
        self._langchain = None
        self._embeddings = None
        self._langchain_loaded = False
        
        # 初始化数据库连接器
        self.db_connector: Optional[DatabaseConnector] = None
        if db_type:
            try:
                self.db_connector = get_database_connector(db_type, **db_kwargs)
            except Exception as e:
                logger.warning(f"初始化数据库连接器失败: {e}")
                self.db_connector = None
        
        # 向量存储目录
        self._vector_store_dir = "data/vector_store/chat_memory"

    def _load_langchain(self):
        """延迟加载LangChain组件和Embeddings模型"""
        if self._langchain_loaded:
            return
            
        self._langchain_loaded = True
        
        # 加载LangChain组件
        self._langchain = safe_import_langchain()
        
        # 尝试初始化Embeddings（如果LangChain可用）
        if self._langchain:
            try:
                self._embeddings = self._langchain["HuggingFaceEmbeddings"](
                    model_name="sentence-transformers/all-MiniLM-L6-v2",
                    model_kwargs={'device': 'cpu'},
                    encode_kwargs={'normalize_embeddings': True}
                )
            except Exception as e:
                logger.warning(f"初始化Embeddings失败: {e}")
                self._embeddings = None
    
    def _ensure_user(self, user_id: str) -> None:
        """确保用户的记忆结构存在"""
        # 确保短期记忆存在
        if user_id not in self._short_term_memory:
            self._short_term_memory[user_id] = collections.deque(maxlen=self.max_short_term)
        
        # 确保长期记忆存在
        if user_id not in self._long_term_memory:
            self._long_term_memory[user_id] = {
                "store": None,
                "summaries": []  # 存储原始摘要用于序列化
            }
            # 尝试加载已存在的向量存储（仅在需要时加载LangChain）
            user_vector_dir = os.path.join(self._vector_store_dir, user_id)
            if os.path.exists(user_vector_dir):
                self._load_langchain()
                if self._embeddings and self._langchain:
                    try:
                        self._long_term_memory[user_id]["store"] = self._langchain["FAISS"].load_local(
                            user_vector_dir,
                            self._embeddings,
                            allow_dangerous_deserialization=True
                        )
                    except Exception as e:
                        logger.warning(f"加载用户向量存储失败: {e}")

    def get_history_as_text(self, user_id: str) -> str:
        """获取格式化的对话历史文本，类似LangChain的ConversationBufferMemory
        
        只返回当前会话的上下文，确保对话连贯性，避免token膨胀
        """
        self._ensure_user(user_id)
        
        # 从短期记忆获取最近的对话
        short_term_pairs = list(self._short_term_memory[user_id])
        if not short_term_pairs:
            return ""
        
        # 只返回最近的max_turns轮对话
        recent_pairs = short_term_pairs[-self.max_turns * 2 :]
        return "\n".join(
            f"{role.upper()}: {text}" for role, text in recent_pairs
        )
    
    def get_relevant_long_term_memory(self, user_id: str, query: str, k: int = 3) -> List[str]:
        """获取与查询相关的长期记忆，使用向量相似度搜索
        
        类似RAG机制，从长期记忆中检索相关的历史知识
        """
        self._ensure_user(user_id)
        
        long_term = self._long_term_memory[user_id]
        if not long_term["store"] or not query:
            return []
        
        try:
            # 使用向量相似度搜索相关文档
            docs = long_term["store"].similarity_search(query, k=k)
            return [doc.page_content for doc in docs]
        except Exception as e:
            logger.error(f"长期记忆检索失败: {e}")
            return []
    
    def get_recent_user_questions(self, user_id: str, limit: int = 5) -> List[str]:
        """返回同一会话中最近的用户提问（最近的在前）。
        只取 human 角色消息，不扫描全部历史，不读取 AI 回答。"""
        self._ensure_user(user_id)
        pairs = list(self._short_term_memory.get(user_id, []))
        questions = [text for role, text in pairs if role == "human"]
        return list(reversed(questions[-limit:]))

    def get_full_history(self, user_id: str) -> str:
        """获取完整的对话历史，包括短期记忆和长期记忆摘要"""
        self._ensure_user(user_id)
        
        # 合并短期记忆和长期记忆
        all_memory = []
        
        # 添加长期记忆摘要
        for summary in self._long_term_memory[user_id]["summaries"]:
            all_memory.append(f"HISTORY_SUMMARY: {summary['content']} (from {summary['timestamp']})")
        
        # 添加短期记忆的完整对话
        for role, text in self._short_term_memory[user_id]:
            all_memory.append(f"{role.upper()}: {text}")
        
        return "\n".join(all_memory)

    def append(self, user_id: str, human: str, ai: str, save_to_db: bool = True) -> None:
        """添加对话并可选保存到数据库"""
        self._ensure_user(user_id)
        
        # 添加到短期记忆
        short_term = self._short_term_memory[user_id]
        short_term.append(("human", human))
        short_term.append(("ai", ai))
        
        # 检查是否需要将短期记忆转换为长期记忆
        if len(short_term) >= self.memory_threshold:
            self._transfer_to_long_term(user_id)
        
        # 如果需要，自动保存到数据库
        if save_to_db:
            self.save_to_database(user_id)

    def _trim_if_needed(self, user_id: str) -> None:
        """短期记忆自动修剪（由deque的maxlen自动处理）"""
        # 短期记忆由deque的maxlen参数自动修剪
        pass
    
    def _create_memory_summary(self, text: str) -> str:
        """创建对话摘要（简单实现，可替换为LLM生成）"""
        # 简单实现：提取关键词和主要话题
        sentences = text.split("\n")
        if len(sentences) <= 3:
            return text  # 短文本不需要摘要
        
        # 简单提取用户问题和AI回答的核心
        user_questions = [s for s in sentences if s.startswith("human:")]
        ai_answers = [s for s in sentences if s.startswith("ai:")]
        
        if not user_questions or not ai_answers:
            return "对话摘要：" + text[:100] + "..." if len(text) > 100 else text
        
        # 取最近的几个问题和回答
        recent_questions = user_questions[-2:]
        recent_answers = ai_answers[-2:]
        
        summary = "对话摘要：\n"
        summary += "\n".join(recent_questions)
        summary += "\n"
        summary += "\n".join(recent_answers)
        
        return summary
    
    def _transfer_to_long_term(self, user_id: str) -> None:
        """将短期记忆转换为长期记忆（向量化存储）"""
        self._ensure_user(user_id)
        
        short_term = self._short_term_memory[user_id]
        if not short_term:
            return
        
        # 创建记忆摘要
        history_text = "\n".join([f"{role}: {text}" for role, text in short_term])
        summary = self._create_memory_summary(history_text)
        
        # 添加到长期记忆摘要列表
        long_term_entry = {
            "content": summary,
            "timestamp": datetime.datetime.now().isoformat(),
            "original_count": len(short_term)
        }
        self._long_term_memory[user_id]["summaries"].append(long_term_entry)
        
        # 如果有LangChain组件，将摘要向量化存储
        if self._embeddings and self._langchain:
            try:
                Document = self._langchain["Document"]
                doc = Document(page_content=summary, metadata={
                    "timestamp": long_term_entry["timestamp"],
                    "user_id": user_id
                })
                
                # 获取或创建向量存储
                user_long_term = self._long_term_memory[user_id]
                if user_long_term["store"] is None:
                    # 创建新的向量存储
                    user_long_term["store"] = self._langchain["FAISS"].from_documents([doc], self._embeddings)
                else:
                    # 向现有向量存储添加文档
                    user_long_term["store"].add_documents([doc])
                
                # 保存向量存储
                user_vector_dir = os.path.join(self._vector_store_dir, user_id)
                os.makedirs(user_vector_dir, exist_ok=True)
                user_long_term["store"].save_local(user_vector_dir)
                
            except Exception as e:
                logger.error(f"向量化长期记忆失败: {e}")
        
        # 清空短期记忆（可选，根据需求调整）
        # self._short_term_memory[user_id].clear()
    
    def _create_memory_summary(self, text: str) -> str:
        """创建对话摘要（简单实现，可替换为更复杂的算法如TextRank）"""
        # 简单实现：提取关键词和主要话题
        # 实际应用中可以使用NLP库如jieba、gensim等实现更复杂的摘要算法
        sentences = text.split("\n")
        if len(sentences) <= 3:
            return text  # 短文本不需要摘要
        
        # 简单提取用户问题和AI回答的核心
        user_questions = [s for s in sentences if s.startswith("human:")]
        ai_answers = [s for s in sentences if s.startswith("ai:")]
        
        if not user_questions or not ai_answers:
            return "对话摘要：" + text[:100] + "..." if len(text) > 100 else text
        
        # 取最近的几个问题和回答
        recent_questions = user_questions[-2:]
        recent_answers = ai_answers[-2:]
        
        summary = "对话摘要：\n"
        summary += "\n".join(recent_questions)
        summary += "\n"
        summary += "\n".join(recent_answers)
        
        return summary

    def serialize(self, user_id: str) -> Optional[Dict]:
        """序列化用户记忆，包含短期记忆和长期记忆"""
        self._ensure_user(user_id)
        
        # 序列化短期记忆
        short_term_memory = []
        for role, content in self._short_term_memory[user_id]:
            short_term_memory.append({
                "role": role,
                "content": content
            })
        
        return {
            "short_term_memory": short_term_memory,
            "long_term_memory": self._long_term_memory[user_id]["summaries"]
        }

    def load_from_serialized(self, user_id: str, data: Dict) -> None:
        """从序列化数据加载用户记忆"""
        self._ensure_user(user_id)
        
        # 加载短期记忆
        short_term_data = data.get("short_term_memory", [])
        if isinstance(short_term_data, list):
            short_term = self._short_term_memory[user_id]
            short_term.clear()  # 清空当前短期记忆
            
            for m in short_term_data:
                role = m.get("role")
                content = m.get("content")
                if role in ("human", "ai") and isinstance(content, str):
                    short_term.append((role, content))
        elif "messages" in data:  # 兼容旧格式
            msgs = data.get("messages", [])
            if isinstance(msgs, list):
                short_term = self._short_term_memory[user_id]
                short_term.clear()
                
                for m in msgs:
                    role = m.get("role")
                    content = m.get("content")
                    if role in ("human", "ai") and isinstance(content, str):
                        short_term.append((role, content))
        
        # 加载长期记忆
        long_term_data = data.get("long_term_memory", [])
        if isinstance(long_term_data, list):
            self._long_term_memory[user_id]["summaries"] = long_term_data
            
            # 如果有LangChain组件，重建向量存储
            if self._embeddings and self._langchain and long_term_data:
                try:
                    Document = self._langchain["Document"]
                    docs = []
                    
                    for summary in long_term_data:
                        docs.append(Document(page_content=summary["content"], metadata={
                            "timestamp": summary["timestamp"],
                            "user_id": user_id
                        }))
                    
                    # 创建向量存储
                    self._long_term_memory[user_id]["store"] = self._langchain["FAISS"].from_documents(docs, self._embeddings)
                    
                    # 保存向量存储
                    user_vector_dir = os.path.join(self._vector_store_dir, user_id)
                    os.makedirs(user_vector_dir, exist_ok=True)
                    self._long_term_memory[user_id]["store"].save_local(user_vector_dir)
                    
                except Exception as e:
                    logger.error(f"重建向量存储失败: {e}")

    def save_to_file(self, user_id, path="memory.json"):
        """保存记忆到文件"""
        data = self.serialize(user_id)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load_from_file(self, user_id, path="memory.json"):
        """从文件加载记忆"""
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.load_from_serialized(user_id, data)
    
    def get_long_term_memory(self, user_id: str) -> List[Dict]:
        """获取用户的长期记忆摘要"""
        self._ensure_user(user_id)
        return self._long_term_memory[user_id]
    
    def add_to_long_term_memory(self, user_id: str, content: str, save_to_db: bool = True) -> None:
        """直接将内容添加到用户的长期记忆"""
        self._ensure_user(user_id)
        
        # 创建记忆摘要
        summary = self._create_memory_summary(content)
        
        # 添加到长期记忆摘要列表
        long_term_entry = {
            "content": summary,
            "timestamp": datetime.datetime.now().isoformat(),
            "original_count": 1
        }
        self._long_term_memory[user_id]["summaries"].append(long_term_entry)
        
        # 如果有LangChain组件，将摘要向量化存储
        if self._embeddings and self._langchain:
            try:
                Document = self._langchain["Document"]
                doc = Document(page_content=summary, metadata={
                    "timestamp": long_term_entry["timestamp"],
                    "user_id": user_id
                })
                
                # 获取或创建向量存储
                user_long_term = self._long_term_memory[user_id]
                if user_long_term["store"] is None:
                    # 创建新的向量存储
                    user_long_term["store"] = self._langchain["FAISS"].from_documents([doc], self._embeddings)
                else:
                    # 向现有向量存储添加文档
                    user_long_term["store"].add_documents([doc])
                
                # 保存向量存储
                user_vector_dir = os.path.join(self._vector_store_dir, user_id)
                os.makedirs(user_vector_dir, exist_ok=True)
                user_long_term["store"].save_local(user_vector_dir)
                
            except Exception as e:
                logger.error(f"向量化长期记忆失败: {e}")
        
        # 如果需要，自动保存到数据库
        if save_to_db:
            self.save_to_database(user_id)
    
    def clear_short_term_memory(self, user_id: str) -> None:
        """清空用户的短期记忆"""
        self._ensure_user(user_id)
        self._short_term_memory[user_id].clear()
    
    def clear_long_term_memory(self, user_id: str) -> None:
        """清空用户的长期记忆"""
        self._ensure_user(user_id)
        
        # 清空摘要列表
        self._long_term_memory[user_id]["summaries"].clear()
        
        # 清空向量存储
        if self._long_term_memory[user_id]["store"] is not None:
            self._long_term_memory[user_id]["store"] = None
        
        # 删除向量存储文件
        user_vector_dir = os.path.join(self._vector_store_dir, user_id)
        if os.path.exists(user_vector_dir):
            import shutil
            try:
                shutil.rmtree(user_vector_dir)
            except Exception as e:
                logger.error(f"删除向量存储文件失败: {e}")
    
    def save_to_database(self, user_id: str) -> bool:
        """保存记忆到数据库"""
        if not self.db_connector:
            print("数据库连接器未初始化")
            return False
        
        data = self.serialize(user_id)
        return self.db_connector.save_memory(user_id, data)
    
    def load_from_database(self, user_id: str) -> bool:
        """从数据库加载记忆"""
        if not self.db_connector:
            print("数据库连接器未初始化")
            return False
        
        data = self.db_connector.load_memory(user_id)
        if data:
            self.load_from_serialized(user_id, data)
            return True
        return False
    
    def delete_from_database(self, user_id: str) -> bool:
        """从数据库删除记忆"""
        if not self.db_connector:
            return False
        
        result = self.db_connector.delete_memory(user_id)
        if result:
            if user_id in self._short_term_memory:
                del self._short_term_memory[user_id]
            if user_id in self._long_term_memory:
                del self._long_term_memory[user_id]
        return result
    
    def get_shared_memory(self, user_ids: List[str]) -> Dict[str, List[Dict]]:
        """获取多个用户的共享记忆"""
        if not self.db_connector:
            # 如果没有数据库连接，使用内存中的数据计算共享记忆
            shared_memory = {"messages": [], "summaries": []}
            all_messages = {}
            all_summaries = {}
            
            for user_id in user_ids:
                self._ensure_user(user_id)
                
                # 收集消息
                messages = [
                    {"role": role, "content": content}
                    for role, content in self._short_term_memory[user_id]
                ]
                
                for msg in messages:
                    msg_key = f"{msg.get('role')}:{msg.get('content', '')}"
                    all_messages[msg_key] = msg
                
                # 收集长期记忆摘要
                for summary in self._long_term_memory[user_id]["summaries"]:
                    summary_key = summary["content"][:100]  # 使用摘要前100字符作为键
                    all_summaries[summary_key] = summary
            
            shared_memory["messages"] = list(all_messages.values())
            shared_memory["summaries"] = list(all_summaries.values())
            return shared_memory
        else:
            # 使用数据库获取共享记忆
            return self.db_connector.get_shared_memory(user_ids)

    def clear(self, user_id: Optional[str] = None) -> None:
        """清空用户的所有记忆或全部记忆"""
        if user_id:
            if user_id in self._short_term_memory:
                del self._short_term_memory[user_id]
            if user_id in self._long_term_memory:
                del self._long_term_memory[user_id]
        else:
            self._short_term_memory.clear()
            self._long_term_memory.clear()


# 全局单例（使用内存存储，简化为课程设计/答辩场景）
conversation_memory = None

def get_conversation_memory():
    """获取或创建全局对话记忆单例"""
    global conversation_memory
    if conversation_memory is None:
        # 不使用MongoDB，直接使用内存存储，简化架构
        conversation_memory = ConversationMemoryManager(db_type=None)
    return conversation_memory


