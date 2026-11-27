"""
贝叶斯知识追踪 (Bayesian Knowledge Tracing, BKT) 算法实现
用于动态评估学习者对各个知识点的掌握概率，是教育数据挖掘领域的经典算法。
"""

from typing import Dict, List, Tuple
from core.logger import logger

class BKTModel:
    """
    标准贝叶斯知识追踪模型
    """
    def __init__(self, p_init: float = 0.3, p_transit: float = 0.1, p_slip: float = 0.1, p_guess: float = 0.2):
        """
        初始化 BKT 模型参数
        :param p_init: P(L0) - 初始掌握概率 (默认 0.3)
        :param p_transit: P(T) - 学习转化概率 (从不会到会的概率，默认 0.1)
        :param p_slip: P(S) - 失误概率 (会但做错的概率，默认 0.1)
        :param p_guess: P(G) - 猜测概率 (不会但蒙对的概率，默认 0.2)
        """
        self.p_init = p_init
        self.p_transit = p_transit
        self.p_slip = p_slip
        self.p_guess = p_guess

    def update_mastery(self, current_mastery: float, is_correct: bool) -> float:
        """
        根据单次答题结果更新知识点掌握概率
        :param current_mastery: 当前掌握概率 P(L_n-1)
        :param is_correct: 是否回答正确
        :return: 更新后的掌握概率 P(L_n)
        """
        if is_correct:
            # P(L_{n-1} | Obs = Correct)
            numerator = current_mastery * (1 - self.p_slip)
            denominator = numerator + (1 - current_mastery) * self.p_guess
        else:
            # P(L_{n-1} | Obs = Incorrect)
            numerator = current_mastery * self.p_slip
            denominator = numerator + (1 - current_mastery) * (1 - self.p_guess)

        # 避免除以 0
        if denominator == 0:
            p_l_given_obs = 0.0
        else:
            p_l_given_obs = numerator / denominator

        # P(L_n) = P(L_{n-1} | Obs) + (1 - P(L_{n-1} | Obs)) * P(T)
        new_mastery = p_l_given_obs + (1 - p_l_given_obs) * self.p_transit
        return min(max(new_mastery, 0.0), 1.0)

    def evaluate_sequence(self, responses: List[bool], initial_prob: float = None) -> List[float]:
        """
        评估一个答题序列，返回每次答题后的掌握概率轨迹
        """
        mastery = initial_prob if initial_prob is not None else self.p_init
        trajectory = [mastery]
        
        for response in responses:
            mastery = self.update_mastery(mastery, response)
            trajectory.append(mastery)
            
        return trajectory


class KnowledgeTracer:
    """
    知识追踪器，管理用户的多个知识点状态
    """
    def __init__(self):
        # 针对不同难度/类型的知识点可以设置不同的参数
        self.models = {
            "default": BKTModel(),
            "hard": BKTModel(p_init=0.1, p_transit=0.05, p_slip=0.2, p_guess=0.15),
            "easy": BKTModel(p_init=0.5, p_transit=0.2, p_slip=0.05, p_guess=0.25)
        }
        # 存储用户的知识点掌握度: {user_id: {concept: mastery_prob}}
        self.user_states: Dict[str, Dict[str, float]] = {}

    def get_mastery(self, user_id: str, concept: str) -> float:
        """获取当前掌握度"""
        if user_id not in self.user_states:
            self.user_states[user_id] = {}
        if concept not in self.user_states[user_id]:
            # 默认初始概率
            self.user_states[user_id][concept] = self.models["default"].p_init
        return self.user_states[user_id][concept]

    def record_interaction(self, user_id: str, concept: str, is_correct: bool, difficulty: str = "default") -> float:
        """
        记录一次交互（做题）并更新模型
        """
        current_mastery = self.get_mastery(user_id, concept)
        model = self.models.get(difficulty, self.models["default"])
        
        new_mastery = model.update_mastery(current_mastery, is_correct)
        self.user_states[user_id][concept] = new_mastery
        
        logger.info(f"BKT 更新: 用户 {user_id} 概念 {concept} 答题 {'正确' if is_correct else '错误'} -> 掌握度: {current_mastery:.2f} -> {new_mastery:.2f}")
        return new_mastery

# 全局单例
tracer = KnowledgeTracer()
