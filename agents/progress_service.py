"""
进度跟踪Agent
负责跟踪和记录学习进度
"""

from agents.base_agent import BaseAgent
from core.message import Message, MessageType
from core.storage import DataStorage
from core.plan_progress import plan_progress_data
from core.logger import logger
from typing import Dict, Any, List
from datetime import datetime, timedelta
from collections import defaultdict
from contextlib import nullcontext


class ProgressService(BaseAgent):
    """进度跟踪Agent"""
    
    def __init__(self):
        super().__init__("progress", "进度跟踪Agent")
        self.storage = DataStorage()
    
    def process(self, message: Message) -> Message:
        """处理进度跟踪请求"""
        content = message.content
        action = content.get("action", "get")
        user_id = content.get("user_id", "default")
        subject = content.get("subject", "")
        
        if action == "update":
            progress_data = content.get("progress_data", {})
            progress = self.update_progress(user_id, subject, progress_data)
            response_content = {
                "status": "success",
                "progress": progress,
                "message": "进度已更新"
            }
        elif action == "get":
            progress = self.get_progress(user_id, subject)
            response_content = {
                "status": "success",
                "progress": progress
            }
        elif action == "get_all":
            all_progress = self.get_all_progress(user_id)
            response_content = {
                "status": "success",
                "progress": all_progress
            }
        elif action == "dashboard":
            progress = self.get_progress(user_id, subject)
            dashboard = self._build_dashboard(progress)
            response_content = {
                "status": "success",
                "dashboard": dashboard,
                "progress": progress
            }
        elif action == "report":
            period = content.get("period", "weekly")
            progress = self.get_progress(user_id, subject)
            stats = self._compute_statistics(progress)
            report = self._generate_report(progress, stats, period=period)
            response_content = {
                "status": "success",
                "report": report,
                "statistics": stats
            }
        else:
            response_content = {
                "status": "error",
                "message": "未知的操作"
            }
        
        return self.send_message(
            receiver=message.sender,
            msg_type=MessageType.RESPONSE,
            content=response_content
        )
    
    def update_progress(self, user_id: str, subject: str, progress_data: Dict[str, Any]) -> Dict[str, Any]:
        """更新学习进度；同文件的更新串行，防止计时和任务状态互相覆盖。"""
        store = getattr(self.storage, 'progress_store', None)
        with (store._lock if store is not None else nullcontext()):
            return self._update_progress(user_id, subject, progress_data)

    def _update_progress(self, user_id, subject, progress_data):
        if progress_data.get("progress_scope") == "detailed_tasks":
            plan = self.storage.get_plan(user_id, subject)
            if plan:
                progress_data = {**progress_data, **plan_progress_data(plan)}
        # 从存储中获取现有进度
        progress = self.storage.get_progress(user_id, subject)
        
        if not progress:
            progress = {
                "user_id": user_id,
                "subject": subject,
                "total_tasks": 0,
                "completed_tasks": 0,
                "progress_percentage": 0,
                "created_at": datetime.now().isoformat(),
                "study_time": 0,  # 学习时间（分钟）
                "notes": [],
                "weekly_progress": {}
            }
        
        # 更新进度数据
        if "completed_tasks" in progress_data:
            # coordinator已经处理好累加逻辑，这里直接使用传入的值
            progress["completed_tasks"] = progress_data["completed_tasks"]
        if "total_tasks" in progress_data:
            # coordinator已经处理好总任务数，这里直接使用传入的值
            # 但确保总任务数不会小于已完成任务数
            new_total = progress_data["total_tasks"]
            completed = progress.get("completed_tasks", 0)
            progress["total_tasks"] = max(new_total, completed)  # 确保总任务数 >= 已完成任务数
        if "study_time" in progress_data:
            progress["study_time"] = progress.get("study_time", 0) + progress_data["study_time"]
        if "note" in progress_data:
            if "notes" not in progress:
                progress["notes"] = []
            progress["notes"].append({
                "note": progress_data["note"],
                "timestamp": datetime.now().isoformat()
            })
        if "week" in progress_data and "week_progress" in progress_data:
            if "weekly_progress" not in progress:
                progress["weekly_progress"] = {}
            progress["weekly_progress"][str(progress_data["week"])] = progress_data["week_progress"]
        if "weekly_progress" in progress_data and isinstance(progress_data.get("weekly_progress"), dict):
            progress["weekly_progress"] = progress_data["weekly_progress"]
        if "tasks" in progress_data and isinstance(progress_data.get("tasks"), list):
            progress["tasks"] = progress_data["tasks"]
        
        progress["progress_scope"] = progress_data.get("progress_scope") or (
            "manual" if "completed_tasks" in progress_data or "total_tasks" in progress_data
            else progress.get("progress_scope", "manual"))
        if "plan_progress_summary" in progress_data:
            progress["plan_progress_summary"] = progress_data["plan_progress_summary"]
        elif progress["progress_scope"] != "detailed_tasks":
            progress.pop("plan_progress_summary", None)

        # 计算进度百分比：必须同时具备可信的 total_tasks 依据才算百分比
        basis_total = progress_data.get("calculation_basis", {}) or {}
        has_verifiable_total = bool(basis_total.get("verifiable")) and progress.get("total_tasks", 0) > 0
        if has_verifiable_total:
            progress["progress_percentage"] = int(
                (progress.get("completed_tasks", 0) / progress["total_tasks"]) * 100
            )
            progress["percentage_verifiable"] = True
        else:
            # 保留旧百分比数值，但标注无法核实，避免"1/10=10%"被当作真实进度
            progress["percentage_verifiable"] = False
            if "progress_percentage" not in progress:
                progress["progress_percentage"] = 0

        # 记录本次更新的溯源信息：匹配的计划、任务、计算依据
        if basis_total:
            progress["last_update_basis"] = {
                "matched_plan_subject": basis_total.get("plan_subject"),
                "total_tasks_source": basis_total.get("total_tasks_source"),
                "completed_tasks_source": basis_total.get("completed_tasks_source"),
                "matched_task": basis_total.get("matched_task"),
                "verifiable": has_verifiable_total,
                "explain": basis_total.get("explain"),
                "updated_at": datetime.now().isoformat(),
            }

        progress["last_updated"] = datetime.now().isoformat()

        # 活跃日志
        study_time_increment = progress_data.get("study_time", 0)
        log_entry = {
            "timestamp": progress["last_updated"],
            "date": progress_data.get("date") or progress["last_updated"][:10],
            "study_time": study_time_increment,
            "completed_tasks": progress.get("completed_tasks", 0)
        }
        progress.setdefault("activity_log", [])
        progress["activity_log"].append(log_entry)
        progress["activity_log"] = progress["activity_log"][-200:]
        
        # 保存到存储
        self.storage.save_progress(user_id, subject, progress)
        return progress
    
    def get_progress(self, user_id: str, subject: str) -> Dict[str, Any]:
        """获取学习进度，包括详细任务信息"""
        progress = self.storage.get_progress(user_id, subject)
        if not progress:
            # 返回默认进度并添加友好提示
            return {
                "user_id": user_id,
                "subject": subject,
                "total_tasks": 0,
                "completed_tasks": 0,
                "progress_percentage": 0,
                "study_time": 0,
                "notes": [],
                "weekly_progress": {},
                "tasks": [],  # 添加空任务列表
                "message": f"暂无{subject}的学习进度记录，请通过发送'我完成了{subject}的学习'这样的消息来更新您的学习进度。"
            }
        
        # 确保返回详细的任务信息
        if "tasks" not in progress:
            # 如果没有任务列表，基于现有数据生成任务信息
            progress["tasks"] = self._generate_task_list(progress)
        
        # 添加学习建议
        if "suggestions" not in progress or progress.get('progress_scope') == 'detailed_tasks':
            progress["suggestions"] = self._generate_suggestions(progress)
        
        # 计算每日学习时间分布
        if "daily_study_time" not in progress:
            progress["daily_study_time"] = self._calculate_daily_time(progress)
        
        return progress
    
    def _generate_task_list(self, progress: Dict[str, Any]) -> List[Dict[str, Any]]:
        """基于进度数据生成任务列表"""
        tasks = []
        completed_tasks = progress.get("completed_tasks", 0)
        total_tasks = progress.get("total_tasks", 0)
        
        # 生成任务列表
        for i in range(1, total_tasks + 1):
            task = {
                "id": f"task_{i}",
                "name": f"任务 {i}",
                "completed": i <= completed_tasks,
                "created_at": progress.get("created_at")
            }
            
            # 如果任务已完成，尝试设置完成时间
            if task["completed"] and progress.get("activity_log"):
                activity_log = progress.get("activity_log", [])
                # 查找最后完成任务的记录时间
                for entry in reversed(activity_log):
                    if entry.get("completed_tasks", 0) >= i:
                        task["completed_at"] = entry.get("timestamp")
                        break
            
            tasks.append(task)
        
        return tasks
    
    def _generate_suggestions(self, progress: Dict[str, Any]) -> List[str]:
        """基于进度生成学习建议"""
        percentage = progress.get("progress_percentage", 0)
        suggestions = []
        
        if (progress.get("plan_progress_summary") or {}).get("has_unplanned_weeks"):
            return ["当前完成率只统计已安排任务，请结合成果检查，并在临近时继续安排后续周。",
                    "保持每日学习与复习，未完成任务可在安排下一周时补学。"]
        if percentage < 30:
            suggestions = [
                "建议制定详细的学习计划，每天安排固定学习时间",
                "从基础知识点开始，循序渐进地学习",
                "可以尝试将大任务分解为小任务，逐个完成"
            ]
        elif percentage < 70:
            suggestions = [
                "学习进度良好，继续保持当前的学习节奏",
                "建议定期复习已学内容，巩固知识点",
                "可以尝试做一些练习题，检验学习效果"
            ]
        else:
            suggestions = [
                "学习即将完成，建议进行全面复习",
                "可以尝试应用所学知识解决实际问题",
                "考虑记录学习心得，总结学习经验"
            ]
        
        return suggestions
    
    def _calculate_daily_time(self, progress: Dict[str, Any]) -> Dict[str, int]:
        """计算每日学习时间分布"""
        daily_time = {}
        activity_log = progress.get("activity_log", [])
        
        for entry in activity_log:
            date = entry.get("date") or entry.get("timestamp", "")[:10]
            if date:
                daily_time[date] = daily_time.get(date, 0) + entry.get("study_time", 0)
        
        return dict(sorted(daily_time.items(), key=lambda x: x[0], reverse=True))
    
    def get_all_progress(self, user_id: str) -> Dict[str, Any]:
        """获取用户所有学习进度"""
        return self.storage.get_all_progress(user_id)

    # === 新增能力 ===
    def _build_dashboard(self, progress: Dict[str, Any]) -> Dict[str, Any]:
        activity_log = progress.get("activity_log", [])
        daily_time = defaultdict(int)
        cumulative_completion = []
        completion = 0
        for entry in activity_log:
            date = entry.get("date") or (entry.get("timestamp", "")[:10])
            if not date:
                continue
            daily_time[date] += entry.get("study_time", 0)
            completion = max(completion, entry.get("completed_tasks", 0))
            cumulative_completion.append({"date": date, "value": completion})
        line_study = [{"date": d, "value": daily_time[d]} for d in sorted(daily_time.keys())]
        stats = self._compute_statistics(progress)
        weekly_progress = progress.get("weekly_progress", {})
        weekly_series = []
        for week, detail in weekly_progress.items():
            total = detail.get("total_tasks", detail.get("total", 0)) or 1
            completed = detail.get("completed_tasks", detail.get("completed", 0))
            weekly_series.append({
                "week": week,
                "completion": round((completed / total) * 100, 1)
            })
        return {
            "study_time_series": line_study,
            "completion_series": cumulative_completion,
            "weekly_completion": sorted(weekly_series, key=lambda x: int(x["week"])),
            "statistics": stats
        }

    def _compute_statistics(self, progress: Dict[str, Any]) -> Dict[str, Any]:
        activity_log = progress.get("activity_log", [])
        unique_days = {entry.get("date") or entry.get("timestamp", "")[:10] for entry in activity_log if entry.get("timestamp")}
        active_days = len({d for d in unique_days if d})
        created_at = progress.get("created_at")
        total_days = 1
        if created_at:
            try:
                start = datetime.fromisoformat(created_at).date()
                total_days = max(1, (datetime.now().date() - start).days + 1)
            except Exception:
                total_days = max(1, len(unique_days))
        persistence_rate = round(active_days / total_days, 2) if total_days else 0.0
        streak = self._calculate_streak(activity_log)
        average_completion = progress.get("progress_percentage", 0)
        study_minutes = progress.get("study_time", 0)
        return {
            "active_days": active_days,
            "total_days": total_days,
            "persistence_rate": persistence_rate,
            "current_streak_days": streak,
            "progress_percentage": average_completion,
            "study_hours": round(study_minutes / 60, 1)
        }

    def _calculate_streak(self, activity_log: List[Dict[str, Any]]) -> int:
        dates = sorted(
            {entry.get("date") or entry.get("timestamp", "")[:10] for entry in activity_log if entry.get("timestamp")},
            reverse=True
        )
        streak = 0
        previous = None
        for date_str in dates:
            try:
                current = datetime.fromisoformat(date_str).date()
            except ValueError:
                continue
            if previous is None:
                streak += 1
                previous = current
                continue
            delta = (previous - current).days
            if delta in (0, 1):
                streak += 1
                previous = current
            else:
                break
        return streak

    def _generate_report(self, progress: Dict[str, Any], stats: Dict[str, Any], period: str = "weekly") -> Dict[str, Any]:
        percentage = progress.get("progress_percentage", 0)
        study_hours = stats.get("study_hours", 0)
        active_days = stats.get("active_days", 0)
        streak = stats.get("current_streak_days", 0)
        summary = (
            f"本{self._period_label(period)}累计学习约 {study_hours} 小时，活跃 {active_days} 天，"
            f"已记录任务完成率 {percentage}%。"
        )
        if (progress.get("plan_progress_summary") or {}).get("has_unplanned_weeks"):
            summary += " 后续周尚未全部安排，此完成率不表示全周期目标完成。"
        if percentage >= 80:
            highlight = "学习进度良好，继续保持输出与复盘节奏。"
        elif percentage >= 50:
            highlight = "完成度稳步提升，建议保持每周至少3次学习频率。"
        else:
            highlight = "当前进度稍显滞后，建议重新分配学习时间并设置每日小目标。"
        suggestions = [
            "结合自动调整后的学习计划完成补课任务",
            "维持学习打卡，增强学习节奏感",
            "输出阶段总结或小项目巩固所学"
        ]
        notes = progress.get("notes", [])
        recent_notes = []
        if notes:
            sorted_notes = sorted(notes, key=lambda n: n.get("timestamp", ""), reverse=True)
            recent_notes = [n.get("note", "") for n in sorted_notes[:3]]
        return {
            "period": period,
            "summary": summary,
            "highlight": highlight,
            "suggestions": suggestions,
            "streak_days": streak,
            "recent_notes": recent_notes
        }

    def _period_label(self, period: str) -> str:
        mapping = {"daily": "日度", "weekly": "周度", "monthly": "月度"}
        return mapping.get(period, period)
    
    def analyze_learning_patterns(self, user_id: str, subject: str) -> Dict[str, Any]:
        """分析学习模式"""
        progress = self.get_progress(user_id, subject)
        if not progress:
            return {"message": "暂无进度数据"}
        
        activity_log = progress.get("activity_log", [])
        if not activity_log:
            return {"message": "暂无活动记录"}
        
        # 分析学习时间模式
        hourly_pattern = defaultdict(int)
        weekday_pattern = defaultdict(int)
        
        for entry in activity_log:
            timestamp = entry.get("timestamp", "")
            if timestamp:
                try:
                    dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                    hour = dt.hour
                    weekday = dt.strftime('%A')
                    study_time = entry.get("study_time", 0)
                    hourly_pattern[hour] += study_time
                    weekday_pattern[weekday] += study_time
                except:
                    continue
        
        # 找出最佳学习时间
        best_hour = max(hourly_pattern.items(), key=lambda x: x[1])[0] if hourly_pattern else None
        best_weekday = max(weekday_pattern.items(), key=lambda x: x[1])[0] if weekday_pattern else None
        
        # 计算学习频率
        dates = sorted(set(entry.get("date") or entry.get("timestamp", "")[:10] 
                          for entry in activity_log if entry.get("timestamp")))
        frequency = len(dates) / max(len(activity_log), 1)
        
        return {
            "best_hour": best_hour,
            "best_weekday": best_weekday,
            "hourly_pattern": dict(hourly_pattern),
            "weekday_pattern": dict(weekday_pattern),
            "learning_frequency": round(frequency, 2),
            "total_study_sessions": len(activity_log)
        }
    
    def predict_completion(self, user_id: str, subject: str) -> Dict[str, Any]:
        """预测学习完成时间"""
        progress = self.get_progress(user_id, subject)
        if not progress:
            return {"message": "暂无进度数据"}
        
        progress_percentage = progress.get("progress_percentage", 0)
        study_time = progress.get("study_time", 0)  # 分钟
        activity_log = progress.get("activity_log", [])
        
        if (progress.get("plan_progress_summary") or {}).get("has_unplanned_weeks"):
            return {"message": "后续周任务尚未全部安排，暂不能据此预测整个目标的完成时间"}
        if progress_percentage <= 0 or study_time <= 0:
            return {"message": "进度数据不足"}
        
        # 计算平均学习速度（每分钟完成的百分比）
        avg_speed = progress_percentage / max(study_time, 1)
        
        # 预测剩余时间（分钟）
        remaining_percentage = 100 - progress_percentage
        predicted_time = remaining_percentage / max(avg_speed, 0.001)  # 避免除零
        
        # 根据学习频率预测完成日期
        if activity_log:
            recent_dates = sorted(set(entry.get("date") or entry.get("timestamp", "")[:10] 
                                     for entry in activity_log[-10:] if entry.get("timestamp")))
            if len(recent_dates) >= 2:
                # 计算平均学习天数间隔
                date_objects = [datetime.fromisoformat(d) for d in recent_dates]
                intervals = [(date_objects[i] - date_objects[i-1]).days 
                            for i in range(1, len(date_objects))]
                avg_interval = sum(intervals) / len(intervals) if intervals else 7
                
                # 预测完成日期
                predicted_days = predicted_time / (60 * avg_interval)  # 假设每天学习avg_interval小时
                predicted_date = datetime.now() + timedelta(days=int(predicted_days))
            else:
                predicted_date = None
        else:
            predicted_date = None
        
        return {
            "current_progress": progress_percentage,
            "predicted_time_minutes": round(predicted_time, 0),
            "predicted_time_hours": round(predicted_time / 60, 1),
            "predicted_completion_date": predicted_date.isoformat() if predicted_date else None,
            "confidence": "high" if len(activity_log) >= 10 else "medium"
        }
    
    def generate_smart_suggestions(self, user_id: str, subject: str) -> List[str]:
        """生成智能学习建议"""
        progress = self.get_progress(user_id, subject)
        if not progress:
            return ["开始记录学习进度，以便获得个性化建议"]
        
        suggestions = []
        progress_percentage = progress.get("progress_percentage", 0)
        streak = self._calculate_streak(progress.get("activity_log", []))
        study_time = progress.get("study_time", 0)
        
        # 基于进度的建议
        if progress_percentage < 20:
            suggestions.append("学习初期建议制定详细的学习计划，逐步建立学习习惯")
        elif progress_percentage < 50:
            suggestions.append("保持当前学习节奏，建议增加实践练习以巩固知识")
        elif progress_percentage < 80:
            suggestions.append("学习进度良好，可以尝试更复杂的项目来提升能力")
        else:
            suggestions.append("已安排任务接近完成，建议检查阶段成果并继续安排后续任务")
        
        # 基于学习习惯的建议
        if streak < 3:
            suggestions.append(f"当前连续学习{streak}天，建议每天固定时间学习以形成习惯")
        elif streak >= 7:
            suggestions.append(f"连续学习{streak}天很棒！保持这个节奏，注意适当休息")
        
        # 基于学习时间的建议
        study_hours = study_time / 60
        if study_hours < 10:
            suggestions.append("学习时间较少，建议增加每日学习时长以提高效率")
        elif study_hours > 100:
            suggestions.append("学习时间充足，建议注重学习质量和实际应用")
        
        # 学习模式分析
        patterns = self.analyze_learning_patterns(user_id, subject)
        if patterns.get("best_hour") is not None:
            suggestions.append(f"建议在{patterns['best_hour']}点学习，这是您的效率高峰期")
        
        return suggestions
