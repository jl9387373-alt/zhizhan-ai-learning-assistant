"""
基础Agent类
所有Agent的基类 - 支持多智能体协作通信
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple, List
import time
import traceback
from datetime import datetime
from core.message import Message, MessageType
from core.storage import DataStorage
from core.logger import logger
from agents.message_protocol import (
    AgentMessage, MessageAction, MessagePriority,
    message_bus, MessageBus
)


class BaseAgent(ABC):
    """基础Agent类 - 增强版，支持工具调用(Skill)"""
    
    def __init__(self, agent_id: str, name: str):
        self.agent_id = agent_id
        self.name = name
        self.memory: Dict[str, Any] = {}
        self.tools: Dict[str, Dict[str, Any]] = {}
        self.storage = DataStorage()
        self.request_count = 0
        self.error_count = 0
        self.total_processing_time = 0.0
        self._init_time = datetime.now()
        
        # 多智能体协作：消息相关
        self.inbox: List[AgentMessage] = []      # 收件箱
        self.outbox: List[AgentMessage] = []     # 发件箱（待发送）
        self.pending_requests: Dict[str, AgentMessage] = {}  # 等待响应的请求
        self._collaboration_enabled = True      # 是否启用协作
        
        # 注册到消息总线
        message_bus.register_agent(self.agent_id)
    
    # ==================== 多智能体协作方法 ====================
    
    def send_to_agent(self, receiver: str, action: MessageAction, content: Dict[str, Any],
                      priority: MessagePriority = MessagePriority.NORMAL,
                      requires_response: bool = False) -> AgentMessage:
        """
        发送消息给另一个 Agent
        
        Args:
            receiver: 接收者 agent_id
            action: 消息动作类型
            content: 消息内容
            priority: 优先级
            requires_response: 是否需要响应
        
        Returns:
            AgentMessage: 发送的消息
        """
        msg = AgentMessage(
            sender=self.agent_id,
            receiver=receiver,
            action=action,
            content=content,
            priority=priority,
            requires_response=requires_response
        )
        
        if requires_response:
            self.pending_requests[msg.msg_id] = msg
        
        # 通过消息总线发送
        success = message_bus.send(msg)
        if success:
            logger.info(f"{self.name} -> {receiver}: {action.value} ({msg.msg_id[:8]})")
        else:
            logger.warning(f"{self.name} 发送消息到 {receiver} 失败")
        
        return msg
    
    def broadcast(self, action: MessageAction, content: Dict[str, Any],
                  priority: MessagePriority = MessagePriority.NORMAL) -> AgentMessage:
        """
        广播消息给所有其他 Agent
        """
        msg = AgentMessage(
            sender=self.agent_id,
            receiver="",  # 空表示广播
            action=action,
            content=content,
            priority=priority
        )
        message_bus.send(msg)
        logger.info(f"{self.name} 广播: {action.value}")
        return msg
    
    def receive_messages(self) -> List[AgentMessage]:
        """
        从消息总线接收所有待处理消息
        """
        messages = message_bus.receive(self.agent_id)
        self.inbox.extend(messages)
        return messages
    
    def process_inbox(self) -> List[AgentMessage]:
        """
        处理收件箱中的消息（子类可重写具体逻辑）
        默认实现：按类型分发到不同处理方法
        """
        processed = []
        remaining = []
        
        for msg in self.inbox:
            if self._should_process(msg):
                self._handle_message(msg)
                processed.append(msg)
            else:
                remaining.append(msg)
        
        self.inbox = remaining
        return processed
    
    def _should_process(self, msg: AgentMessage) -> bool:
        """
        判断是否应该处理该消息（可由子类重写）
        """
        # 处理需要响应的消息
        if msg.requires_response and msg.correlation_id in self.pending_requests:
            return True
        # 处理发给自己的消息
        if msg.receiver == self.agent_id:
            return True
        # 处理广播消息
        if not msg.receiver:
            return True
        return False
    
    def _handle_message(self, msg: AgentMessage):
        """
        处理单条消息（可由子类重写）
        """
        if msg.action == MessageAction.REQUEST:
            self._handle_request(msg)
        elif msg.action == MessageAction.NOTIFY:
            self._handle_notify(msg)
        elif msg.action == MessageAction.RESPOND:
            self._handle_response(msg)
    
    def _handle_request(self, msg: AgentMessage):
        """处理请求消息 - 子类应重写"""
        logger.debug(f"{self.name} 收到请求: {msg.content}")
    
    def _handle_notify(self, msg: AgentMessage):
        """处理通知消息 - 子类应重写"""
        logger.debug(f"{self.name} 收到通知: {msg.content}")
    
    def _handle_response(self, msg: AgentMessage):
        """处理响应消息"""
        original = self.pending_requests.pop(msg.correlation_id, None)
        if original:
            logger.info(f"{self.name} 收到响应 (关联: {msg.correlation_id[:8]})")
    
    def request_from(self, receiver: str, content: Dict[str, Any]) -> Optional[AgentMessage]:
        """
        向指定 Agent 发起请求并等待响应
        
        Args:
            receiver: 接收者 agent_id
            content: 请求内容
        
        Returns:
            响应消息，如果失败返回 None
        """
        msg = self.send_to_agent(
            receiver=receiver,
            action=MessageAction.REQUEST,
            content=content,
            requires_response=True
        )
        
        # 注意：实际响应是异步的，调用者需要通过 inbox 接收
        return msg
    
    def get_collaboration_status(self) -> Dict[str, Any]:
        """获取协作状态信息"""
        return {
            "agent_id": self.agent_id,
            "inbox_size": len(self.inbox),
            "outbox_size": len(self.outbox),
            "pending_requests": len(self.pending_requests),
            "collaboration_enabled": self._collaboration_enabled
        }
    
    # ==================== 原有方法 ====================
    
    def register_tool(self, name: str, func: Any, description: str, parameters: Dict[str, Any]):
        """注册一个工具(Skill)"""
        self.tools[name] = {
            "name": name,
            "func": func,
            "description": description,
            "parameters": parameters
        }
        logger.info(f"{self.name} 注册了工具: {name}")

    def execute_tool(self, name: str, arguments: Dict[str, Any]) -> Any:
        """执行已注册的工具"""
        if name not in self.tools:
            raise ValueError(f"未找到工具: {name}")
        
        tool = self.tools[name]
        logger.info(f"{self.name} 正在执行工具 {name}, 参数: {arguments}")
        try:
            return tool["func"](**arguments)
        except Exception as e:
            logger.error(f"工具 {name} 执行失败: {str(e)}")
            raise

    def get_tool_schemas(self) -> list:
        """获取所有工具的 JSON Schema 描述，用于 LLM Function Calling"""
        schemas = []
        for name, tool in self.tools.items():
            schemas.append({
                "name": name,
                "description": tool["description"],
                "parameters": tool["parameters"]
            })
        return schemas
    
    @abstractmethod
    def process(self, message: Message) -> Message:
        """处理消息（子类必须实现）"""
        pass
    
    def process_with_metrics(self, message: Message) -> Message:
        """带性能监控的消息处理"""
        start_time = time.time()
        self.request_count += 1
        
        try:
            logger.debug(f"{self.name} 开始处理消息: {message.content.get('action', 'unknown')}")
            result = self.process(message)
            
            elapsed_time = time.time() - start_time
            self.total_processing_time += elapsed_time
            
            # 添加处理元数据
            if result and isinstance(result, Message):
                if "metadata" not in result.content:
                    result.content["metadata"] = {}
                result.content["metadata"].update({
                    "processing_time": round(elapsed_time, 3),
                    "agent": self.agent_id,
                    "processed_at": datetime.now().isoformat()
                })
            
            logger.debug(f"{self.name} 处理完成，耗时: {elapsed_time:.3f}秒")
            return result
            
        except Exception as e:
            self.error_count += 1
            elapsed_time = time.time() - start_time
            error_msg = f"{self.name} 处理消息失败: {str(e)}"
            logger.error(f"{error_msg}\n{traceback.format_exc()}")
            
            # 返回错误响应
            return self.send_message(
                receiver=message.sender,
                msg_type=MessageType.ERROR,
                content={
                    "status": "error",
                    "message": error_msg,
                    "error_type": type(e).__name__,
                    "metadata": {
                        "processing_time": round(elapsed_time, 3),
                        "agent": self.agent_id,
                        "processed_at": datetime.now().isoformat()
                    }
                }
            )
    
    def send_message(self, receiver: str, msg_type: MessageType, content: Dict[str, Any]) -> Message:
        """发送消息"""
        # 确保content包含基本字段
        if "status" not in content:
            content["status"] = "success"
        
        return Message(
            sender=self.agent_id,
            receiver=receiver,
            msg_type=msg_type,
            content=content
        )
    
    def store_memory(self, key: str, value: Any):
        """存储记忆"""
        self.memory[key] = value
        logger.debug(f"{self.name} 存储记忆: {key}")
    
    def get_memory(self, key: str, default: Any = None) -> Any:
        """获取记忆"""
        return self.memory.get(key, default)
    
    def clear_memory(self):
        """清空记忆"""
        self.memory.clear()
        logger.debug(f"{self.name} 记忆已清空")
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取Agent统计信息"""
        avg_time = self.total_processing_time / self.request_count if self.request_count > 0 else 0
        error_rate = self.error_count / self.request_count if self.request_count > 0 else 0
        
        return {
            "agent_id": self.agent_id,
            "agent_name": self.name,
            "request_count": self.request_count,
            "error_count": self.error_count,
            "error_rate": round(error_rate, 3),
            "avg_processing_time": round(avg_time, 3),
            "total_processing_time": round(self.total_processing_time, 3),
            "uptime_seconds": (datetime.now() - self._init_time).total_seconds(),
            "memory_size": len(self.memory)
        }
    
    def validate_message(self, message: Message, required_fields: list) -> Tuple[bool, Optional[str]]:
        """验证消息内容是否包含必需字段"""
        if not message or not message.content:
            return False, "消息内容为空"
        
        for field in required_fields:
            if field not in message.content:
                return False, f"缺少必需字段: {field}"
        
        return True, None
    
    def __str__(self) -> str:
        return f"{self.name} ({self.agent_id})"
    
    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} id={self.agent_id} name={self.name}>"
