"""
知识有向无环图 (Knowledge DAG) 与动态学习路径规划算法
使用拓扑排序 (Topological Sorting) 结合用户的 BKT 掌握度，动态生成个性化的学习路径。
"""

from typing import Dict, List, Set, Any
from collections import deque
from core.logger import logger
from core.bkt import tracer as bkt_tracer

class KnowledgeNode:
    """知识图谱中的节点"""
    def __init__(self, node_id: str, name: str, weight: int = 1):
        self.node_id = node_id
        self.name = name
        self.weight = weight  # 预计学习所需时间/权重
        
class KnowledgeDAG:
    """知识有向无环图"""
    def __init__(self):
        self.nodes: Dict[str, KnowledgeNode] = {}
        # 邻接表：节点 -> 其后继节点列表 (A -> B 表示必须先学 A 再学 B)
        self.adj_list: Dict[str, List[str]] = {}
        # 入度表：节点 -> 其前置节点数量
        self.in_degree: Dict[str, int] = {}
        
    def add_node(self, node_id: str, name: str, weight: int = 1):
        if node_id not in self.nodes:
            self.nodes[node_id] = KnowledgeNode(node_id, name, weight)
            self.adj_list[node_id] = []
            self.in_degree[node_id] = 0
            
    def add_edge(self, from_id: str, to_id: str):
        """添加依赖关系：from_id 是 to_id 的前置条件"""
        if from_id not in self.nodes or to_id not in self.nodes:
            raise ValueError(f"节点不存在: {from_id} 或 {to_id}")
            
        if to_id not in self.adj_list[from_id]:
            self.adj_list[from_id].append(to_id)
            self.in_degree[to_id] += 1

def generate_topological_path(dag: KnowledgeDAG, user_id: str, mastery_threshold: float = 0.8) -> List[Dict[str, Any]]:
    """
    基于拓扑排序 (Kahn算法) 和 BKT 掌握度生成个性化学习路径。
    如果用户对某个节点 (知识点) 的掌握度超过阈值，则自动跳过该节点，但其后置节点的基础依赖将被解除。
    """
    # 复制入度表，以免修改原图
    in_degree = dag.in_degree.copy()
    queue = deque()
    
    # 1. 找到所有初始入度为 0 的节点
    for node_id, degree in in_degree.items():
        if degree == 0:
            queue.append(node_id)
            
    learning_path = []
    
    # 2. 拓扑排序主循环
    while queue:
        current_id = queue.popleft()
        node = dag.nodes[current_id]
        
        # 检查用户的掌握度 (调用我们之前写的 BKT 模型)
        mastery = bkt_tracer.get_mastery(user_id, current_id)
        
        # 只有未掌握的节点才加入最终学习计划
        is_mastered = mastery >= mastery_threshold
        if not is_mastered:
            learning_path.append({
                "node_id": node.node_id,
                "name": node.name,
                "current_mastery": round(mastery, 2),
                "estimated_weight": node.weight
            })
        else:
            logger.info(f"动态图规划：用户 {user_id} 已掌握知识点 '{node.name}' (掌握度: {mastery:.2f})，自动跳过该节点。")
            
        # 无论是否加入计划，都将其对后继节点的依赖解除
        for neighbor_id in dag.adj_list[current_id]:
            in_degree[neighbor_id] -= 1
            if in_degree[neighbor_id] == 0:
                queue.append(neighbor_id)
                
    # 检查是否存在环（如果学习路径 + 已掌握节点数 != 总节点数）
    # 但由于我们在处理前已经默认 DAG 无环，此处省略严格的环检测。
    
    return learning_path

# --- 提供一个默认的图谱构建工厂 ---
def build_default_python_dag() -> KnowledgeDAG:
    """构建一个默认的 Python 知识图谱"""
    dag = KnowledgeDAG()
    
    # 添加节点
    dag.add_node("py_basic", "Python基础语法", 2)
    dag.add_node("py_types", "数据类型与变量", 2)
    dag.add_node("py_control", "控制流(if/for/while)", 3)
    dag.add_node("py_func", "函数与作用域", 3)
    dag.add_node("py_oop", "面向对象编程(OOP)", 5)
    dag.add_node("py_modules", "模块与包", 2)
    dag.add_node("py_advanced", "高级特性(装饰器/生成器)", 4)
    
    # 添加依赖边
    dag.add_edge("py_basic", "py_types")
    dag.add_edge("py_types", "py_control")
    dag.add_edge("py_control", "py_func")
    dag.add_edge("py_func", "py_oop")
    dag.add_edge("py_func", "py_modules")
    dag.add_edge("py_oop", "py_advanced")
    
    return dag

# 全局单例
python_dag = build_default_python_dag()
