"""
复习Agent
负责管理学习复习计划，基于遗忘曲线安排复习
"""

from agents.base_agent import BaseAgent
from core.message import Message, MessageType
from core.storage import DataStorage
from core.logger import logger
from storage.analytics import track_event
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import uuid
import math


class ReviewService(BaseAgent):
    """复习Agent - 基于遗忘曲线的智能复习系统"""
    
    def __init__(self):
        super().__init__("review", "复习Agent")
        self.storage = DataStorage()
        
        # 艾宾浩斯遗忘曲线间隔（天）
        # 1小时后、1天后、2天后、4天后、7天后、15天后、30天后
        self.review_intervals = [1/24, 1, 2, 4, 7, 15, 30]
        
        # 重要性权重（影响复习频率）
        self.importance_weights = {
            "high": 1.2,      # 高重要性：更频繁复习
            "medium": 1.0,    # 中等重要性：标准频率
            "low": 0.8        # 低重要性：较少复习
        }
    
    def process(self, message: Message) -> Message:
        """处理复习相关请求"""
        # 确保 action 字段存在
        if "action" not in message.content:
            message.content["action"] = "schedule"
            
        # 验证消息
        valid, error_msg = self.validate_message(message, ["action"])
        if not valid:
            return self.send_message(
                receiver=message.sender,
                msg_type=MessageType.ERROR,
                content={"status": "error", "message": error_msg}
            )
        
        content = message.content
        action = content.get("action", "schedule")
        user_id = content.get("user_id", "default")
        
        try:
            if action == "schedule":
                result = self.schedule_review(
                    user_id=user_id,
                    subject=content.get("subject", "通用"),
                    content_text=content.get("content", ""),
                    importance=content.get("importance", "medium")
                )
            elif action == "next":
                result = self.get_next_review(user_id)
            elif action == "complete":
                result = self.complete_review(
                    user_id=user_id,
                    review_id=content.get("review_id"),
                    performance=content.get("performance", {})
                )
            elif action == "get":
                result = self.get_review(
                    user_id=user_id,
                    review_id=content.get("review_id")
                )
            elif action == "history":
                result = self.get_review_history(
                    user_id=user_id,
                    limit=content.get("limit", 50)
                )
            elif action == "analyze":
                result = self.analyze_review_performance(user_id)
            elif action == "delete":
                result = self.delete_review(
                    user_id=user_id,
                    review_id=content.get("review_id")
                )
            else:
                result = {
                    "status": "error",
                    "message": f"未知的操作: {action}"
                }
            
            return self.send_message(
                receiver=message.sender,
                msg_type=MessageType.RESPONSE,
                content=result
            )
        except Exception as e:
            logger.error(f"ReviewAgent处理失败: {str(e)}", exc_info=True)
            return self.send_message(
                receiver=message.sender,
                msg_type=MessageType.ERROR,
                content={
                    "status": "error",
                    "message": f"处理复习请求失败: {str(e)}"
                }
            )
    
    def schedule_review(self, user_id: str, subject: str, content_text: str = "", 
                       importance: str = "medium", first_review_at=None) -> Dict[str, Any]:
        """安排复习计划"""
        try:
            # 如果没有提供内容，尝试从学习进度中获取
            if not content_text:
                progress = self.storage.get_progress(user_id, subject)
                if progress:
                    # 从进度中提取需要复习的内容
                    notes = progress.get("notes", [])
                    if notes:
                        content_text = notes[-1].get("note", f"{subject}的学习内容")
                    else:
                        content_text = f"{subject}的基础知识"
            
            # 创建复习计划
            review_id = str(uuid.uuid4())
            now = datetime.now()
            
            # 计算第一次复习时间（基于重要性）
            weight = self.importance_weights.get(importance, 1.0)
            first_interval = self.review_intervals[0] * weight
            next_review_at = now + timedelta(days=first_interval)
            # 用户指定了首次复习时间就优先使用（前端传 datetime-local 原始字符串）
            if first_review_at:
                try:
                    parsed = datetime.fromisoformat(str(first_review_at))
                    next_review_at = parsed if parsed > now else now
                except Exception:
                    logger.warning(f"无法解析首次复习时间: {first_review_at}")
            
            review_plan = {
                "review_id": review_id,
                "user_id": user_id,
                "subject": subject,
                "content": content_text,
                "importance": importance,
                "scheduled_at": now.isoformat(),
                "next_review_at": next_review_at.isoformat(),
                "review_count": 0,
                "current_interval_index": 0,
                "status": "active",
                "last_reviewed_at": None,
                "mastery_level": 0.0,  # 掌握程度 0-1
                "performance_history": []
            }
            
            # 保存复习计划
            self.storage.save_review_plan(user_id, review_plan)
            
            logger.info(f"成功安排复习: {review_id}, 主题: {subject}, 下次复习: {next_review_at}")
            
            return {
                "status": "success",
                "review_id": review_id,
                "subject": subject,
                "next_review_at": next_review_at.isoformat(),
                "message": "复习计划已安排"
            }
        except Exception as e:
            logger.error(f"安排复习失败: {str(e)}", exc_info=True)
            return {
                "status": "error",
                "message": f"安排复习失败: {str(e)}"
            }
    
    def reschedule_review(self, user_id: str, review_id: str,
                          next_review_at=None, postpone_days=None) -> Dict[str, Any]:
        """调整复习计划的下次复习时间。

        postpone_days：从当前时间起推迟 N 天（用于逾期任务）。
        next_review_at：指定具体时间（ISO 字符串，前端 datetime-local 原值）。
        """
        try:
            plan = self._get_review_plan(user_id, review_id)
            if not plan:
                return {"status": "error", "message": "复习计划不存在"}

            now = datetime.now()
            target = None
            if postpone_days not in (None, ""):
                try:
                    target = now + timedelta(days=float(postpone_days))
                except (TypeError, ValueError):
                    return {"status": "error", "message": "推迟天数格式不正确"}
            elif next_review_at:
                try:
                    target = datetime.fromisoformat(str(next_review_at))
                except ValueError:
                    return {"status": "error", "message": "时间格式不正确"}

            if target is None:
                return {"status": "error", "message": "未提供新的复习时间"}
            if target < now:
                target = now

            plan["next_review_at"] = target.isoformat()
            plan["status"] = "active"
            self.storage.save_review_plan(user_id, plan)
            logger.info(f"复习计划已调整: {review_id} -> {plan['next_review_at']}")
            return {"status": "success", "review": plan}
        except Exception as exc:
            logger.error(f"调整复习时间失败: {exc}", exc_info=True)
            return {"status": "error", "message": str(exc)}

    def get_next_review(self, user_id: str) -> Dict[str, Any]:
        """获取下一次复习内容"""
        try:
            # 从存储中获取下一次复习
            next_review = self.storage.get_next_review(user_id)
            
            if not next_review:
                # 如果没有待复习内容，尝试自动生成
                return self._generate_auto_review(user_id)
            
            # 检查是否到了复习时间
            next_review_at_str = next_review.get("next_review_at", "")
            if not next_review_at_str:
                return {
                    "status": "success",
                    "message": "复习计划数据异常：缺少时间",
                    "review": None
                }
                
            next_review_at = datetime.fromisoformat(next_review_at_str)
            now = datetime.now()
            
            if now < next_review_at:
                return {
                    "status": "success",
                    "message": "还没有到复习时间",
                    "next_review_at": next_review_at.isoformat(),
                    "review": None
                }
            
            # 准备复习内容
            review_content = {
                "review_id": next_review.get("review_id"),
                "subject": next_review.get("subject"),
                "content": next_review.get("content"),
                "importance": next_review.get("importance"),
                "review_count": next_review.get("review_count", 0),
                "mastery_level": next_review.get("mastery_level", 0.0),
                "scheduled_at": next_review.get("scheduled_at"), # 补充 scheduled_at
                "suggested_focus": self._get_suggested_focus(next_review)
            }
            
            return {
                "status": "success",
                "review": review_content,
                "next_review_at": next_review_at.isoformat()
            }
        except Exception as e:
            logger.error(f"获取复习内容失败: {str(e)}", exc_info=True)
            return {
                "status": "error",
                "message": f"获取复习内容失败: {str(e)}"
            }
    
    def complete_review(self, user_id: str, review_id: str, 
                       performance: Dict[str, Any]) -> Dict[str, Any]:
        """完成复习"""
        try:
            # 获取复习计划
            review_plan = self._get_review_plan(user_id, review_id)
            
            if not review_plan:
                return {
                    "status": "error",
                    "message": "复习计划不存在"
                }
            
            # 更新复习记录
            now = datetime.now()
            review_plan["last_reviewed_at"] = now.isoformat()
            review_plan["review_count"] = review_plan.get("review_count", 0) + 1
            
            # 记录表现
            performance_entry = {
                "timestamp": now.isoformat(),
                "score": performance.get("score", 0),  # 0-100
                "confidence": performance.get("confidence", 0.5),  # 0-1
                "time_spent": performance.get("time_spent", 0)  # 分钟
            }
            
            performance_history = review_plan.get("performance_history", [])
            performance_history.append(performance_entry)
            review_plan["performance_history"] = performance_history[-10:]  # 保留最近10次
            
            # 更新掌握程度
            recent_scores = [p.get("score", 0) for p in performance_history[-3:]]
            if recent_scores:
                review_plan["mastery_level"] = sum(recent_scores) / len(recent_scores) / 100.0
            
            # 计算下次复习时间（基于表现和遗忘曲线）
            current_index = review_plan.get("current_interval_index", 0)
            score = performance.get("score", 50)
            
            # 如果表现好，可以延长间隔；表现差，缩短间隔
            if score >= 80:
                # 表现好：进入下一个间隔
                current_index = min(current_index + 1, len(self.review_intervals) - 1)
            elif score < 60:
                # 表现差：回到上一个间隔或保持当前
                current_index = max(current_index - 1, 0)
            
            review_plan["current_interval_index"] = current_index
            
            # 计算下次复习时间
            weight = self.importance_weights.get(review_plan.get("importance", "medium"), 1.0)
            interval_days = self.review_intervals[current_index] * weight
            
            # 根据掌握程度调整
            mastery = review_plan.get("mastery_level", 0.5)
            if mastery > 0.8:
                interval_days *= 1.5  # 掌握得好，延长间隔
            elif mastery < 0.5:
                interval_days *= 0.7  # 掌握得差，缩短间隔
            
            next_review_at = now + timedelta(days=interval_days)
            review_plan["next_review_at"] = next_review_at.isoformat()
            
            # 如果已经复习足够多次且掌握程度高，可以标记为完成
            if review_plan["review_count"] >= len(self.review_intervals) and mastery >= 0.8:
                review_plan["status"] = "mastered"
            
            # 保存更新
            self.storage.save_review_plan(user_id, review_plan)
            track_event("review_completed", user_id, {
                "review_id": review_id,
                "subject": review_plan.get("subject", "通用"),
                "score": score,
                "review_count": review_plan["review_count"],
                "mastery_level": round(review_plan["mastery_level"], 2),
            })
            
            logger.info(f"复习完成: {review_id}, 得分: {score}, 下次复习: {next_review_at}")
            
            return {
                "status": "success",
                "review_id": review_id,
                "mastery_level": round(review_plan["mastery_level"], 2),
                "next_review_at": next_review_at.isoformat(),
                "review_count": review_plan["review_count"],
                "message": "复习已完成"
            }
        except Exception as e:
            logger.error(f"完成复习失败: {str(e)}", exc_info=True)
            return {
                "status": "error",
                "message": f"完成复习失败: {str(e)}"
            }
    
    def get_review(self, user_id: str, review_id: str) -> Dict[str, Any]:
        """获取复习详情"""
        try:
            review_plan = self._get_review_plan(user_id, review_id)
            
            if not review_plan:
                return {
                    "status": "error",
                    "message": "复习计划不存在"
                }
            
            return {
                "status": "success",
                "review": review_plan
            }
        except Exception as e:
            logger.error(f"获取复习详情失败: {str(e)}", exc_info=True)
            return {
                "status": "error",
                "message": f"获取复习详情失败: {str(e)}"
            }
    
    def get_review_history(self, user_id: str, limit: int = 50) -> Dict[str, Any]:
        """获取复习历史"""
        try:
            history = self.storage.get_review_history(user_id, limit)
            
            return {
                "status": "success",
                "history": history,
                "count": len(history)
            }
        except Exception as e:
            logger.error(f"获取复习历史失败: {str(e)}", exc_info=True)
            return {
                "status": "error",
                "message": f"获取复习历史失败: {str(e)}"
            }
    
    def delete_review(self, user_id: str, review_id: str) -> Dict[str, Any]:
        """删除复习计划"""
        try:
            if not review_id:
                return {
                    "status": "error",
                    "message": "缺少复习计划ID"
                }
            
            # 检查复习计划是否存在
            review_plan = self._get_review_plan(user_id, review_id)
            if not review_plan:
                return {
                    "status": "error",
                    "message": "复习计划不存在"
                }
            
            # 删除复习计划
            success = self.storage.delete_review_plan(user_id, review_id)
            
            if success:
                logger.info(f"成功删除复习计划: {review_id}, 主题: {review_plan.get('subject')}")
                return {
                    "status": "success",
                    "review_id": review_id,
                    "message": "复习计划已删除"
                }
            else:
                return {
                    "status": "error",
                    "message": "删除复习计划失败"
                }
        except Exception as e:
            logger.error(f"删除复习计划失败: {str(e)}", exc_info=True)
            return {
                "status": "error",
                "message": f"删除复习计划失败: {str(e)}"
            }
    
    def analyze_review_performance(self, user_id: str) -> Dict[str, Any]:
        """分析复习表现"""
        try:
            history = self.storage.get_review_history(user_id, limit=1000)
            
            if not history:
                return {
                    "status": "success",
                    "message": "暂无复习记录",
                    "statistics": {}
                }
            
            # 计算统计信息
            active_reviews = [r for r in history if r.get("status") == "active"]
            mastered_reviews = [r for r in history if r.get("status") == "mastered"]
            
            total_reviews = len(history)
            total_review_sessions = sum(r.get("review_count", 0) for r in history)
            
            # 计算平均掌握程度
            mastery_levels = [r.get("mastery_level", 0) for r in history if r.get("mastery_level")]
            avg_mastery = sum(mastery_levels) / len(mastery_levels) if mastery_levels else 0
            
            # 按主题统计
            subject_stats = {}
            for review in history:
                subject = review.get("subject", "未知")
                if subject not in subject_stats:
                    subject_stats[subject] = {
                        "count": 0,
                        "mastery_levels": [],
                        "review_counts": []
                    }
                subject_stats[subject]["count"] += 1
                if review.get("mastery_level"):
                    subject_stats[subject]["mastery_levels"].append(review.get("mastery_level"))
                subject_stats[subject]["review_counts"].append(review.get("review_count", 0))
            
            # 计算每个主题的平均值
            for subject in subject_stats:
                stats = subject_stats[subject]
                if stats["mastery_levels"]:
                    stats["avg_mastery"] = sum(stats["mastery_levels"]) / len(stats["mastery_levels"])
                else:
                    stats["avg_mastery"] = 0
                if stats["review_counts"]:
                    stats["avg_review_count"] = sum(stats["review_counts"]) / len(stats["review_counts"])
                else:
                    stats["avg_review_count"] = 0
                del stats["mastery_levels"]
                del stats["review_counts"]
            
            return {
                "status": "success",
                "statistics": {
                    "total_reviews": total_reviews,
                    "active_reviews": len(active_reviews),
                    "mastered_reviews": len(mastered_reviews),
                    "total_review_sessions": total_review_sessions,
                    "average_mastery": round(avg_mastery, 2),
                    "mastery_rate": round(len(mastered_reviews) / total_reviews * 100, 2) if total_reviews > 0 else 0,
                    "subject_statistics": subject_stats
                }
            }
        except Exception as e:
            logger.error(f"分析复习表现失败: {str(e)}", exc_info=True)
            return {
                "status": "error",
                "message": f"分析复习表现失败: {str(e)}"
            }
    
    # === 私有辅助方法 ===
    
    def _get_review_plan(self, user_id: str, review_id: str) -> Optional[Dict[str, Any]]:
        """获取复习计划（内部方法）"""
        # 这里需要从存储中获取，暂时使用存储接口
        # 如果ReviewStore没有实现，可以从历史中查找
        history = self.storage.get_review_history(user_id, limit=1000)
        for review in history:
            if review.get("review_id") == review_id:
                return review
        return None
    
    def _generate_auto_review(self, user_id: str) -> Dict[str, Any]:
        """自动生成复习内容（基于学习进度）"""
        try:
            # 获取所有学习进度
            all_progress = self.storage.get_all_progress(user_id)
            
            if not all_progress:
                return {
                    "status": "success",
                    "message": "暂无学习内容需要复习",
                    "review": None
                }
            
            # 找到最近学习但未安排复习的主题
            for subject, progress in all_progress.items():
                if isinstance(progress, dict):
                    # 检查是否已有复习计划
                    history = self.storage.get_review_history(user_id, limit=1000)
                    has_review = any(
                        r.get("subject") == subject and r.get("status") == "active"
                        for r in history
                    )
                    
                    if not has_review:
                        # 自动安排复习
                        notes = progress.get("notes", [])
                        content = notes[-1].get("note", f"{subject}的学习内容") if notes else f"{subject}的基础知识"
                        
                        result = self.schedule_review(
                            user_id=user_id,
                            subject=subject,
                            content_text=content,
                            importance="medium"
                        )
                        
                        if result.get("status") == "success":
                            return {
                                "status": "success",
                                "message": "已自动生成复习计划",
                                "review": {
                                    "review_id": result.get("review_id"),
                                    "subject": subject,
                                    "content": content,
                                    "next_review_at": result.get("next_review_at")
                                }
                            }
            
            return {
                "status": "success",
                "message": "暂无待复习内容",
                "review": None
            }
        except Exception as e:
            logger.warning(f"自动生成复习失败: {str(e)}")
            return {
                "status": "success",
                "message": "暂无待复习内容",
                "review": None
            }
    
    def _get_suggested_focus(self, review: Dict[str, Any]) -> List[str]:
        """获取建议的复习重点"""
        suggestions = []
        
        mastery = review.get("mastery_level", 0)
        review_count = review.get("review_count", 0)
        
        if mastery < 0.5:
            suggestions.append("建议重点复习基础概念")
            suggestions.append("可以多做相关练习")
        elif mastery < 0.7:
            suggestions.append("建议巩固核心知识点")
            suggestions.append("尝试应用所学内容")
        else:
            suggestions.append("掌握程度良好，可以适当拓展")
        
        if review_count == 0:
            suggestions.append("这是第一次复习，建议仔细回顾")
        elif review_count < 3:
            suggestions.append("建议加强记忆巩固")
        
        return suggestions

