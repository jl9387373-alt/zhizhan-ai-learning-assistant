"""
配置管理模块
"""

# 增强的配置管理
import os
import json
from typing import Any, Dict, Optional

class ConfigManager:
    def __init__(self, config_path: str = "config.json"):
        self.config_path = config_path
        self._config = {}
        self._load_config()
        self._load_env_overrides()
    
    def _load_config(self):
        """加载配置文件"""
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    self._config = json.load(f)
            except Exception as e:
                print(f"加载配置文件失败: {e}")
    
    def reload(self):
        """重新加载配置文件"""
        self._load_config()
        self._load_env_overrides()
        print(f"配置已从 {self.config_path} 重新加载")
    
    def _load_env_overrides(self):
        """从环境变量加载配置覆盖"""
        # 支持RAG相关环境变量
        rag_prefix = "RAG_"
        for key, value in os.environ.items():
            if key.startswith(rag_prefix):
                config_key = key[len(rag_prefix):].lower()
                # 处理嵌套配置
                if "." in config_key:
                    parts = config_key.split(".")
                    if parts[0] not in self._config:
                        self._config[parts[0]] = {}
                    self._config[parts[0]][parts[1]] = self._parse_env_value(value)
                else:
                    self._config.setdefault("rag", {})[config_key] = self._parse_env_value(value)
    
    def _parse_env_value(self, value: str) -> Any:
        """解析环境变量值"""
        # 尝试解析为数字、布尔值等
        if value.lower() == "true":
            return True
        elif value.lower() == "false":
            return False
        
        try:
            return int(value)
        except ValueError:
            try:
                return float(value)
            except ValueError:
                return value
    
    def get(self, key: str, default: Any = None) -> Any:
        """获取配置值"""
        parts = key.split(".")
        value = self._config
        
        for part in parts:
            if isinstance(value, dict) and part in value:
                value = value[part]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any) -> None:
        """设置配置值"""
        parts = key.split(".")
        config = self._config
        
        for part in parts[:-1]:
            if part not in config:
                config[part] = {}
            config = config[part]
        
        config[parts[-1]] = value
    
    def save(self) -> None:
        """保存配置到文件"""
        try:
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(self._config, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"保存配置文件失败: {e}")

# 创建全局配置实例
config = ConfigManager()
