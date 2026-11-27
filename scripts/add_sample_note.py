import sys
import os

# Add project root to sys.path
sys.path.append(os.getcwd())

from storage.notes import NotesStore

store = NotesStore()

# Note 1: Python
title = "Python 列表推导式高效指南"
subject = "Python"
content = """列表推导式（List Comprehension）是 Python 中一种优雅且高效的创建列表的方式。

基本语法：
[expression for item in iterable if condition]

示例：
1. 生成平方数列表：
squares = [x**2 for x in range(10)]

2. 过滤偶数：
evens = [x for x in range(20) if x % 2 == 0]

优点：
- 代码更简洁
- 执行效率通常高于 for 循环

注意：
- 避免过于复杂的推导式，影响可读性
- 此时应考虑使用普通循环"""

# Add for user 'default'
store.create('default', subject, title, content)
print("Added Python note for 'default'")

# Add for user '1' (admin)
store.create('1', subject, title, content)
print("Added Python note for user '1'")

# Add for user '2'
store.create('2', subject, title, content)
print("Added Python note for user '2'")

# Note 2: Project Architecture
title2 = "学习助手项目架构笔记"
subject2 = "Project"
content2 = """本项目采用前后端分离架构，主要包含以下模块：

1. 前端 (static/):
- index.html: 单页应用入口
- app.js: 核心业务逻辑，使用原生 JS 实现路由和交互
- style.css: 样式定义

2. 后端 (Python/Flask):
- app.py: API 入口
- core/: 核心业务逻辑
  - coordinator.py: Agent 协调器
  - storage.py: 数据存储层
  - auth.py: 用户认证

3. 数据存储 (data/):
- users.db: 用户信息 (SQLite)
- notes.json: 云端笔记存储 (JSON)
- plans.json: 学习计划存储 (JSON)

4. 功能特性:
- 智能对话 (RAG)
- 学习计划生成
- 进度追踪
- 云端笔记同步"""

store.create('default', subject2, title2, content2)
store.create('1', subject2, title2, content2)
store.create('2', subject2, title2, content2)
print("Added project note for all users")
