"""
Agent 间通信协议
定义多智能体协作的消息格式和类型
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, List
from datetime import datetime
from enum import Enum
import uuid


class MessageAction(str, Enum):
    """消息动作类型"""
    # 协作相关
    REQUEST = "request"           # 请求某个能力
    RESPOND = "respond"           # 响应请求
    SHARE = "share"               # 分享信息
    ASK = "ask"                   # 询问
    NOTIFY = "notify"             # 通知
    
    # 任务相关
    TASK = "task"                 # 任务分发
    RESULT = "result"             # 结果返回
    COMPLETE = "complete"         # 任务完成
    FAIL = "fail"                 # 任务失败


class MessagePriority(str, Enum):
    """消息优先级"""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"


@dataclass
class AgentMessage:
    """
    Agent 间通信的统一消息格式
    """
    # 基础信息
    msg_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    # 通信双方
    sender: str = ""              # 发送者 agent_id
    receiver: str = ""            # 接收者 agent_id（空表示广播）
    
    # 消息内容
    action: MessageAction = MessageAction.SHARE
    content: Dict[str, Any] = field(default_factory=dict)
    
    # 元信息
    priority: MessagePriority = MessagePriority.NORMAL
    requires_response: bool = False
    correlation_id: str = ""      # 关联消息ID，用于请求-响应配对
    
    # 执行结果
    status: str = "pending"       # pending/success/error
    error: Optional[str] = None
    
    @classmethod
    def create_request(
        cls,
        sender: str,
        receiver: str,
        content: Dict[str, Any],
        action: MessageAction = MessageAction.REQUEST
    ) -> "AgentMessage":
        """创建请求消息"""
        msg_id = str(uuid.uuid4())
        return cls(
            msg_id=msg_id,
            sender=sender,
            receiver=receiver,
            action=action,
            content=content,
            requires_response=True,
            correlation_id=msg_id
        )
    
    @classmethod
    def create_response(
        cls,
        sender: str,
        receiver: str,
        original_msg: "AgentMessage",
        content: Dict[str, Any],
        status: str = "success"
    ) -> "AgentMessage":
        """创建响应消息"""
        return cls(
            sender=sender,
            receiver=receiver,
            action=MessageAction.RESPOND,
            content=content,
            correlation_id=original_msg.msg_id,
            status=status
        )
    
    @classmethod
    def create_broadcast(
        cls,
        sender: str,
        content: Dict[str, Any],
        action: MessageAction = MessageAction.NOTIFY
    ) -> "AgentMessage":
        """创建广播消息"""
        return cls(
            sender=sender,
            receiver="",  # 空表示广播
            action=action,
            content=content
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "msg_id": self.msg_id,
            "timestamp": self.timestamp,
            "sender": self.sender,
            "receiver": self.receiver,
            "action": self.action.value if isinstance(self.action, Enum) else self.action,
            "content": self.content,
            "priority": self.priority.value if isinstance(self.priority, Enum) else self.priority,
            "requires_response": self.requires_response,
            "correlation_id": self.correlation_id,
            "status": self.status,
            "error": self.error
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentMessage":
        """从字典创建"""
        action = data.get("action", MessageAction.SHARE)
        if isinstance(action, str):
            action = MessageAction(action)
        
        priority = data.get("priority", MessagePriority.NORMAL)
        if isinstance(priority, str):
            priority = MessagePriority(priority)
        
        return cls(
            msg_id=data.get("msg_id", str(uuid.uuid4())),
            timestamp=data.get("timestamp", datetime.now().isoformat()),
            sender=data.get("sender", ""),
            receiver=data.get("receiver", ""),
            action=action,
            content=data.get("content", {}),
            priority=priority,
            requires_response=data.get("requires_response", False),
            correlation_id=data.get("correlation_id", ""),
            status=data.get("status", "pending"),
            error=data.get("error")
        )


class MessageBus:
    """
    消息总线 - Agent 间消息传递的中间件
    """
    def __init__(self):
        # agent_id -> List[AgentMessage]
        self._inboxes: Dict[str, List[AgentMessage]] = {}
        # 全局消息日志
        self._message_log: List[AgentMessage] = []
        # 订阅者 (用于广播)
        self._subscribers: Dict[str, List[str]] = {}  # topic -> [agent_ids]
    
    def register_agent(self, agent_id: str):
        """注册 Agent 到消息总线"""
        if agent_id not in self._inboxes:
            self._inboxes[agent_id] = []
    
    def unregister_agent(self, agent_id: str):
        """注销 Agent"""
        if agent_id in self._inboxes:
            del self._inboxes[agent_id]
    
    def send(self, message: AgentMessage) -> bool:
        """
        发送消息
        - 如果 receiver 为空，广播给所有订阅者
        - 否则直接投递给指定接收者
        """
        self._message_log.append(message)
        
        if not message.receiver:
            # 广播
            for agent_id in self._inboxes:
                if agent_id != message.sender:  # 不发给自己
                    self._inboxes[agent_id].append(message)
            return True
        
        # 点对点发送
        if message.receiver in self._inboxes:
            self._inboxes[message.receiver].append(message)
            return True
        return False
    
    def receive(self, agent_id: str) -> List[AgentMessage]:
        """接收消息（原子性：获取后清空收件箱）"""
        if agent_id not in self._inboxes:
            return []
        messages = self._inboxes[agent_id].copy()
        self._inboxes[agent_id].clear()
        return messages
    
    def peek(self, agent_id: str) -> List[AgentMessage]:
        """查看消息（不删除）"""
        return self._inboxes.get(agent_id, []).copy()
    
    def subscribe(self, agent_id: str, topic: str):
        """订阅主题"""
        if topic not in self._subscribers:
            self._subscribers[topic] = []
        if agent_id not in self._subscribers[topic]:
            self._subscribers[topic].append(agent_id)
    
    def get_message_history(self, limit: int = 100) -> List[Dict]:
        """获取消息历史"""
        return [msg.to_dict() for msg in self._message_log[-limit:]]
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取消息统计"""
        return {
            "total_messages": len(self._message_log),
            "registered_agents": len(self._inboxes),
            "pending_messages": sum(len(msgs) for msgs in self._inboxes.values())
        }


# 全局消息总线实例
message_bus = MessageBus()
