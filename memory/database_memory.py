"""
数据库记忆持久化模块
主要支持MongoDB存储方式，当MongoDB不可用时会自动降级到内存存储
"""

import json
from typing import Dict, List, Optional, Tuple, Any

# 导入MongoDB相关库
try:
    from pymongo import MongoClient
    from pymongo.collection import Collection
    MONGO_AVAILABLE = True
except ImportError:
    MONGO_AVAILABLE = False


class DatabaseConnector:
    """
    数据库连接器抽象基类
    """
    
    def __init__(self):
        self.connected = False
        
    def connect(self) -> bool:
        """建立数据库连接"""
        raise NotImplementedError
    
    def disconnect(self) -> None:
        """关闭数据库连接"""
        raise NotImplementedError
    
    def save_memory(self, user_id: str, memory_data: Dict) -> bool:
        """保存用户记忆"""
        raise NotImplementedError
    
    def load_memory(self, user_id: str) -> Optional[Dict]:
        """加载用户记忆"""
        raise NotImplementedError
    
    def delete_memory(self, user_id: str) -> bool:
        """删除用户记忆"""
        raise NotImplementedError
    
    def get_shared_memory(self, user_ids: List[str]) -> Dict:
        """获取多个用户共享的记忆"""
        raise NotImplementedError


class MongoDBConnector(DatabaseConnector):
    """
    MongoDB数据库连接器
    """
    
    def __init__(self, connection_string: str = "mongodb://localhost:27017", db_name: str = "learning_assistant", collection_name: str = "chat_memory"):
        super().__init__()
        
        self.connection_string = connection_string
        self.db_name = db_name
        self.collection_name = collection_name
        self.client = None
        self.db = None
        self.collection = None
        # 初始化时不立即连接，延迟到实际使用时再连接
        self._connected = False
    
    @property
    def connected(self):
        """检查连接状态"""
        return self._connected
    
    @connected.setter
    def connected(self, value):
        """设置连接状态"""
        self._connected = value
    
    def connect(self) -> bool:
        """建立MongoDB连接"""
        if self._connected:
            return True
            
        if not MONGO_AVAILABLE:
            print("MongoDB不可用: pymongo未安装")
            self._connected = False
            return False
            
        try:
            self.client = MongoClient(self.connection_string, serverSelectionTimeoutMS=3000)
            self.db = self.client[self.db_name]
            self.collection = self.db[self.collection_name]
            
            # 测试连接
            self.client.server_info()
            
            # 创建索引
            self.collection.create_index("user_id", unique=True)
            self.collection.create_index("updated_at")
            
            self._connected = True
            print("MongoDB连接成功")
            return True
        except Exception as e:
            print(f"MongoDB连接失败: {e}")
            self._connected = False
            return False
    
    def disconnect(self) -> None:
        """关闭MongoDB连接"""
        if self._connected and self.client:
            try:
                self.client.close()
            except Exception as e:
                print(f"断开连接失败: {e}")
        self.client = None
        self.db = None
        self.collection = None
        self._connected = False
    
    def save_memory(self, user_id: str, memory_data: Dict) -> bool:
        """保存用户记忆到MongoDB"""
        # 尝试连接（如果未连接）
        if not self._connected:
            if not self.connect():
                return False
        
        try:
            # 使用当前时间戳
            from datetime import datetime
            current_time = datetime.now()
            
            # 使用upsert操作
            self.collection.update_one(
                {"user_id": user_id},
                {
                    "$set": {
                        "memory_data": memory_data,
                        "updated_at": current_time
                    }
                },
                upsert=True
            )
            return True
        except Exception as e:
            print(f"保存记忆到MongoDB失败: {e}")
            # 连接可能已断开，重置连接状态
            self._connected = False
            return False
    
    def load_memory(self, user_id: str) -> Optional[Dict]:
        """从MongoDB加载用户记忆"""
        # 尝试连接（如果未连接）
        if not self._connected:
            if not self.connect():
                return None
        
        try:
            document = self.collection.find_one({"user_id": user_id})
            if document and "memory_data" in document:
                return document["memory_data"]
            return None
        except Exception as e:
            print(f"从MongoDB加载记忆失败: {e}")
            # 连接可能已断开，重置连接状态
            self._connected = False
            return None
    
    def delete_memory(self, user_id: str) -> bool:
        """从MongoDB删除用户记忆"""
        # 尝试连接（如果未连接）
        if not self._connected:
            if not self.connect():
                return False
        
        try:
            result = self.collection.delete_one({"user_id": user_id})
            return result.deleted_count > 0
        except Exception as e:
            print(f"从MongoDB删除记忆失败: {e}")
            # 连接可能已断开，重置连接状态
            self._connected = False
            return False
    
    def get_shared_memory(self, user_ids: List[str]) -> Dict:
        """获取多个用户的共享记忆"""
        # 尝试连接（如果未连接）
        if not self._connected:
            if not self.connect():
                return {"messages": []}
        
        try:
            shared_memory = {"messages": []}
            all_messages = {}
            
            for user_id in user_ids:
                memory = self.load_memory(user_id)
                if memory and "messages" in memory:
                    for msg in memory["messages"]:
                        # 使用内容作为键，避免重复
                        msg_key = f"{msg.get('role')}:{msg.get('content', '')}"
                        all_messages[msg_key] = msg
            
            # 将唯一消息添加到共享记忆中
            shared_memory["messages"] = list(all_messages.values())
            return shared_memory
        except Exception as e:
            print(f"获取共享记忆失败: {e}")
            # 连接可能已断开，重置连接状态
            self._connected = False
            return {"messages": []}


class MemoryConnector(DatabaseConnector):
    """
    内存数据库连接器（当MongoDB不可用时的降级方案）
    """
    
    def __init__(self):
        super().__init__()
        self.memory_store = {}
        self.connected = True
    
    def connect(self) -> bool:
        """内存连接总是成功"""
        self.connected = True
        return True
    
    def disconnect(self) -> None:
        """断开内存连接"""
        self.memory_store.clear()
        self.connected = False
    
    def save_memory(self, user_id: str, memory_data: Dict) -> bool:
        """保存用户记忆到内存"""
        if not self.connected:
            self.connect()
        
        self.memory_store[user_id] = memory_data
        return True
    
    def load_memory(self, user_id: str) -> Optional[Dict]:
        """从内存加载用户记忆"""
        if not self.connected:
            return None
        
        return self.memory_store.get(user_id)
    
    def delete_memory(self, user_id: str) -> bool:
        """从内存删除用户记忆"""
        if not self.connected:
            return False
        
        if user_id in self.memory_store:
            del self.memory_store[user_id]
            return True
        return False
    
    def get_shared_memory(self, user_ids: List[str]) -> Dict:
        """获取多个用户的共享记忆"""
        shared_memory = {"messages": []}
        all_messages = {}
        
        for user_id in user_ids:
            memory = self.load_memory(user_id)
            if memory and "messages" in memory:
                for msg in memory["messages"]:
                    # 使用内容作为键，避免重复
                    msg_key = f"{msg.get('role')}:{msg.get('content', '')}"
                    all_messages[msg_key] = msg
        
        # 将唯一消息添加到共享记忆中
        shared_memory["messages"] = list(all_messages.values())
        return shared_memory


def get_database_connector(db_type: str = "mongodb", **kwargs) -> DatabaseConnector:
    """
    获取数据库连接器实例
    
    Args:
        db_type: 数据库类型，只支持 'mongodb'
        **kwargs: 数据库连接参数
    
    Returns:
        数据库连接器实例（MongoDB或内存存储）
    """
    # 尝试使用MongoDB
    if db_type == "mongodb":
        mongo_connector = MongoDBConnector(**kwargs)
        # 显式调用connect()方法尝试连接MongoDB
        if mongo_connector.connect():
            return mongo_connector
        else:
            print("MongoDB不可用，降级到内存存储")
    
    # 如果MongoDB不可用，使用内存存储
    return MemoryConnector()