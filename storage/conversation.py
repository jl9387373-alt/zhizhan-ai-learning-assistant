"""
会话存储模块
存储和管理用户的多轮对话会话
"""
from typing import Dict, Any, List, Optional
from datetime import datetime
from ._base import JsonStoreBase
import uuid


class ConversationStore(JsonStoreBase):
    """会话存储"""
    
    def __init__(self, data_dir: str = "data"):
        super().__init__(data_dir, "conversations.json", {"conversations": []})
    
    def create_conversation(self, user_id: str, title: str = None) -> str:
        """创建新会话，返回会话ID"""
        data = self._read() or {"conversations": []}
        conversation_id = str(uuid.uuid4())
        
        conversation = {
            "id": conversation_id,
            "user_id": user_id,
            "title": title or "新对话",
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "messages": []
        }
        
        data.setdefault("conversations", []).append(conversation)
        self._write(data)
        return conversation_id
    
    def add_message(self, conversation_id: str, role: str, content: str, metadata: Dict[str, Any] = None) -> bool:
        """向会话添加消息"""
        data = self._read() or {"conversations": []}
        conversations = data.get("conversations", [])
        
        for conv in conversations:
            if conv.get("id") == conversation_id:
                message = {
                    "role": role,  # "user" 或 "assistant"
                    "content": content,
                    "timestamp": datetime.now().isoformat()
                }
                if metadata:
                    message.update(metadata)
                
                conv["messages"].append(message)
                conv["updated_at"] = datetime.now().isoformat()
                
                # 如果是第一条用户消息，使用它作为标题
                if role == "user" and len(conv["messages"]) == 1:
                    conv["title"] = content[:50] + ("..." if len(content) > 50 else "")
                
                self._write(data)
                return True
        
        return False
    
    def get_conversation(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        """获取指定会话"""
        data = self._read() or {"conversations": []}
        conversations = data.get("conversations", [])
        
        for conv in conversations:
            if conv.get("id") == conversation_id:
                return conv
        
        return None
    
    def list_conversations(self, user_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        """获取用户的所有会话列表（按更新时间倒序）"""
        return self._list_conversations_impl(limit, user_id=user_id)

    def list_all_conversations(self, limit: int = 50) -> List[Dict[str, Any]]:
        """获取本机全部会话列表（跨账号，按更新时间倒序）。
        用于"最近对话"统一视图，解决切换账号后看不到历史的问题。"""
        return self._list_conversations_impl(limit, user_id=None)

    def _list_conversations_impl(self, limit: int, user_id: Optional[str] = None) -> List[Dict[str, Any]]:
        data = self._read() or {"conversations": []}
        conversations = data.get("conversations", [])

        # user_id 为 None 时不过滤账号
        selected = [
            conv for conv in conversations
            if user_id is None or conv.get("user_id") == user_id
        ]
        selected.sort(key=lambda x: x.get("updated_at", ""), reverse=True)

        result = []
        for conv in selected[:limit]:
            result.append({
                "id": conv.get("id"),
                "user_id": conv.get("user_id", "default"),
                "title": conv.get("title", "新对话"),
                "created_at": conv.get("created_at"),
                "updated_at": conv.get("updated_at"),
                "message_count": len(conv.get("messages", []))
            })
        return result
    
    def delete_conversation(self, conversation_id: str, user_id: str) -> bool:
        """删除会话"""
        data = self._read() or {"conversations": []}
        conversations = data.get("conversations", [])
        
        original_count = len(conversations)
        data["conversations"] = [
            conv for conv in conversations 
            if not (conv.get("id") == conversation_id and conv.get("user_id") == user_id)
        ]
        
        if len(data["conversations"]) < original_count:
            self._write(data)
            return True
        
        return False

    def delete_by_id(self, conversation_id: str) -> bool:
        """按会话ID删除，不校验账号（用于全部历史视图）"""
        data = self._read() or {"conversations": []}
        conversations = data.get("conversations", [])
        original_count = len(conversations)
        data["conversations"] = [
            conv for conv in conversations if conv.get("id") != conversation_id
        ]
        if len(data["conversations"]) < original_count:
            self._write(data)
            return True
        return False
    
    def update_conversation_title(self, conversation_id: str, title: str) -> bool:
        """更新会话标题"""
        data = self._read() or {"conversations": []}
        conversations = data.get("conversations", [])
        
        for conv in conversations:
            if conv.get("id") == conversation_id:
                conv["title"] = title
                conv["updated_at"] = datetime.now().isoformat()
                self._write(data)
                return True
        
        return False

    def get_global_stats(self) -> Dict[str, int]:
        """获取全局会话统计信息"""
        data = self._read() or {"conversations": []}
        conversations = data.get("conversations", [])
        
        total_conversations = len(conversations)
        total_messages = sum(len(c.get("messages", [])) for c in conversations)
        
        return {
            "conversation_count": total_conversations,
            "message_count": total_messages
        }


