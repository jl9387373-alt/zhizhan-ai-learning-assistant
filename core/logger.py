"""
日志系统模块
"""

import logging
import os
import tempfile
from datetime import datetime
from typing import Optional


class Logger:
    """日志管理类"""
    
    def __init__(self, name: str = "MultiAgentSystem", log_dir: str = "logs"):
        self.name = name
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        # 创建logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)
        
        # 避免重复添加handler
        if not self.logger.handlers:
            self._setup_handlers()
    
    def _setup_handlers(self):
        """设置日志处理器"""
        # 文件处理器 - 所有日志
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        log_file = os.path.join(self.log_dir, f"{self.name}_{datetime.now().strftime('%Y%m%d')}.log")
        file_handler = None
        try:
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
        except OSError:
            fallback_dir = os.path.join(tempfile.gettempdir(), self.name)
            os.makedirs(fallback_dir, exist_ok=True)
            fallback_file = os.path.join(
                fallback_dir,
                f"{self.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
            )
            try:
                file_handler = logging.FileHandler(fallback_file, encoding='utf-8')
            except OSError:
                file_handler = None

        if file_handler is not None:
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(file_formatter)
        
        # 控制台处理器 - 只显示INFO及以上
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        )
        console_handler.setFormatter(console_formatter)
        
        if file_handler is not None:
            self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
    
    def debug(self, message: str, *args, **kwargs):
        """调试日志"""
        self.logger.debug(message, *args, **kwargs)
    
    def info(self, message: str, *args, **kwargs):
        """信息日志"""
        self.logger.info(message, *args, **kwargs)
    
    def warning(self, message: str, *args, **kwargs):
        """警告日志"""
        self.logger.warning(message, *args, **kwargs)
    
    def error(self, message: str, *args, **kwargs):
        """错误日志"""
        self.logger.error(message, *args, **kwargs)
    
    def exception(self, message: str, *args, **kwargs):
        """异常日志（包含堆栈信息）"""
        self.logger.exception(message, *args, **kwargs)


# 全局日志实例
logger = Logger()
