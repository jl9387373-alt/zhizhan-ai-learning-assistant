"""
消息传递机制
用于Agent之间的通信
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional
from enum import Enum
from datetime import datetime


class MessageType(Enum):
    """消息类型"""
    REQUEST = "request"
    RESPONSE = "response"
    NOTIFICATION = "notification"
    QUERY = "query"
    ERROR = "error"


@dataclass
class Message:
    """消息类"""
    sender: str
    receiver: str
    msg_type: MessageType
    content: Dict[str, Any]
    timestamp: str = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "sender": self.sender,
            "receiver": self.receiver,
            "type": self.msg_type.value,
            "content": self.content,
            "timestamp": self.timestamp
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Message':
        """从字典创建消息"""
        return cls(
            sender=data["sender"],
            receiver=data["receiver"],
            msg_type=MessageType(data["type"]),
            content=data["content"],
            timestamp=data.get("timestamp")
        )
