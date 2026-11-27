"""
学习计划Agent
负责制定和管理学习计划
"""

from agents.base_agent import BaseAgent
from core.message import Message, MessageType
from core.storage import DataStorage
from core.plan_progress import summarize_plan_progress, plan_progress_data
from core.logger import logger
from typing import Dict, Any, List, Tuple, Optional
import re
import math
from copy import deepcopy
from datetime import datetime, timedelta
from nlp.plan_constraints import plan_today, extract_plan_constraints
from rag.rag_chain import get_rag_helper


from storage.profiles import ProfilesStore
from storage.analytics import track_event
from agents.plan_generation import generate_goal_path, generate_learning_week, PlanGenerationError

class PlannerService(BaseAgent):
    """学习计划Agent"""
    
    def __init__(self):
        super().__init__("planner", "学习计划Agent")
        self.storage = DataStorage()
        self.profile_store = ProfilesStore()
        self.plan_templates = self._load_plan_templates()
    
    def process(self, message: Message) -> Message:
        """处理学习计划相关请求"""
        # 确保 action 字段存在
        if "action" not in message.content:
            message.content["action"] = "create"
            
        # 验证消息
        valid, error_msg = self.validate_message(message, ["subject"])
        if not valid and message.content.get("action") != "get_all":
            return self.send_message(
                receiver=message.sender,
                msg_type=MessageType.ERROR,
                content={"status": "error", "message": error_msg or "缺少必需字段: subject"}
            )
        
        content = message.content
        action = content.get("action", "create")
        user_id = content.get("user_id", "default")
        subject = content.get("subject", "")
        
        if action == "create":
            duration = content.get("duration", "4周")
            try:
                plan = self.create_learning_plan(subject, user_id, duration, content.get("constraints"))
            except PlanGenerationError as exc:
                return self.send_message(
                    receiver=message.sender,
                    msg_type=MessageType.ERROR,
                    content={"status": "error", "message": str(exc), "code": exc.code,
                             "http_status": exc.http_status, "retryable": exc.retryable},
                )
            response_content = {
                "status": "success",
                "intent": "plan_creation",
                "plan": plan,
                "message": f"已为您创建{subject}的{plan.get('requested_duration', plan['duration'])}学习计划"
            }
        elif action == "get":
            plan = self.get_learning_plan(user_id, subject)
            response_content = {
                "status": "success",
                "plan": plan
            }
        elif action == "generate_week":
            response_content = self.refine_learning_week(user_id, subject, int(content.get("week", 1)))
        elif action == "timeline":
            plan = self.get_learning_plan(user_id, subject)
            response_content = {
                "status": "success",
                "timeline": plan.get("timeline", []),
                "plan": plan
            }
        elif action == "update":
            plan_data = content.get("plan_data", {})
            plan = self.update_learning_plan(user_id, subject, plan_data)
            response_content = {
                "status": "success",
                "plan": plan,
                "message": "学习计划已更新"
            }
        elif action == "adjust":
            plan, adjustments = self.adjust_plan(user_id, subject)
            response_content = {
                "status": "success",
                "plan": plan,
                "adjustments": adjustments,
                "message": "学习计划已根据进度自动调整"
            }
        elif action == "review":
            review = self.generate_phase_review(user_id, subject)
            response_content = {
                "status": "success",
                "review": review
            }
        elif action == "delete":
            # 删除学习计划
            success = self.delete_learning_plan(user_id, subject)
            response_content = {
                "status": "success" if success else "error",
                "message": "学习计划已删除" if success else "学习计划不存在"
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
    
    def _load_plan_templates(self) -> Dict[str, Any]:
        """加载学习计划模板"""
        return {
            "python": {
                "4周": [
                    {"week": 1, "topics": ["Python基础", "环境搭建", "数据类型"], 
                     "tasks": ["安装Python和IDE", "学习变量和数据类型", "掌握基本语法", "完成10个基础练习"],
                     "goals": ["能够编写简单的Python程序", "理解基本数据类型"]},
                    {"week": 2, "topics": ["控制结构", "函数", "模块"], 
                     "tasks": ["学习if/for/while", "编写自定义函数", "使用标准库模块", "完成小型项目"],
                     "goals": ["掌握程序流程控制", "能够编写函数和模块"]},
                    {"week": 3, "topics": ["面向对象", "文件操作", "异常处理"], 
                     "tasks": ["学习类和对象", "文件读写操作", "异常处理机制", "完成中等项目"],
                     "goals": ["理解面向对象编程", "能够处理文件和异常"]},
                    {"week": 4, "topics": ["高级特性", "项目实战", "总结提升"], 
                     "tasks": ["学习装饰器、生成器", "完成综合项目", "代码优化", "总结学习成果"],
                     "goals": ["掌握Python高级特性", "能够独立完成项目"]}
                ],
                "8周": [
                    {"week": 1, "topics": ["Python基础", "环境搭建"], "tasks": ["安装环境", "学习基础语法"]},
                    {"week": 2, "topics": ["数据类型", "运算符"], "tasks": ["掌握各种数据类型", "练习运算符"]},
                    {"week": 3, "topics": ["控制结构"], "tasks": ["学习条件语句", "学习循环语句"]},
                    {"week": 4, "topics": ["函数", "模块"], "tasks": ["编写函数", "使用模块"]},
                    {"week": 5, "topics": ["面向对象"], "tasks": ["学习类和对象", "继承和多态"]},
                    {"week": 6, "topics": ["文件操作", "异常处理"], "tasks": ["文件读写", "异常处理"]},
                    {"week": 7, "topics": ["高级特性"], "tasks": ["装饰器", "生成器", "迭代器"]},
                    {"week": 8, "topics": ["项目实战"], "tasks": ["完成综合项目", "代码优化"]}
                ]
            },
            "机器学习": {
                "4周": [
                    {"week": 1, "topics": ["机器学习基础", "数学基础"], 
                     "tasks": ["学习基本概念", "复习线性代数和概率论", "安装相关库"],
                     "goals": ["理解机器学习基本概念", "掌握必要的数学基础"]},
                    {"week": 2, "topics": ["监督学习", "回归和分类"], 
                     "tasks": ["学习线性回归", "学习逻辑回归", "实现简单模型"],
                     "goals": ["掌握监督学习算法", "能够实现基础模型"]},
                    {"week": 3, "topics": ["无监督学习", "深度学习入门"], 
                     "tasks": ["学习聚类算法", "学习神经网络基础", "使用TensorFlow/PyTorch"],
                     "goals": ["理解无监督学习", "入门深度学习"]},
                    {"week": 4, "topics": ["项目实战", "模型评估"], 
                     "tasks": ["完成完整项目", "学习模型评估方法", "优化模型性能"],
                     "goals": ["能够完成端到端项目", "掌握模型评估"]}
                ]
            },
            "java": {
                "4周": [
                    {"week": 1, "topics": ["Java基础", "环境搭建"], "tasks": ["安装JDK", "学习基本语法", "理解面向对象"]},
                    {"week": 2, "topics": ["集合框架", "异常处理"], "tasks": ["学习集合类", "异常处理", "IO操作"]},
                    {"week": 3, "topics": ["多线程", "网络编程"], "tasks": ["学习线程", "Socket编程", "完成网络应用"]},
                    {"week": 4, "topics": ["框架学习", "项目实战"], "tasks": ["学习Spring框架", "完成Web项目", "部署应用"]}
                ]
            },
            "通用": {
                "4周": [
                    {"week": 1, "topics": ["基础知识", "环境搭建"], "tasks": ["搭建学习环境", "学习基础概念"]},
                    {"week": 2, "topics": ["核心概念", "实践练习"], "tasks": ["深入学习核心概念", "完成练习"]},
                    {"week": 3, "topics": ["进阶内容", "项目实战"], "tasks": ["学习进阶内容", "开始项目"]},
                    {"week": 4, "topics": ["复习巩固", "总结提升"], "tasks": ["复习所有内容", "完成项目"]}
                ]
            }
        }
    
    def create_learning_plan(self, subject: str, user_id: str, duration: str = "4周",
                             constraints: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """创建学习计划（增强版：智能推荐 + DAG动态规划 + 用户画像注入）"""
        plan_key = f"{user_id}_{subject}"
        
        # 获取用户画像信息
        profile = deepcopy(self.profile_store.get_profile(user_id))
        constraints = dict(constraints or {})
        if constraints.get("goal"):
            profile["goal"] = constraints["goal"]
        if constraints.get("daily_hours") is not None:
            profile["daily_hours"] = float(constraints["daily_hours"])
        daily_hours = float(profile.get("daily_hours", 2))
        if not math.isfinite(daily_hours) or not 0 < daily_hours <= 24:
            raise ValueError("每天可用时间应大于 0，且不超过 24 小时")
        if constraints.get("duration_months"):
            period, _ = extract_plan_constraints(f"{constraints['duration_months']}个月", reference_date=plan_today())
            constraints.update(period)
        days = int(constraints.get("duration_days") or self._parse_weeks(duration) * 7)
        if not 1 <= days <= 728:
            raise ValueError("计划周期应为 1 天到 104 周")
        weeks = math.ceil(days / 7)
        goal = constraints.get("goal") or self._resolve_topic_goal(subject, profile) or f"掌握{subject}的核心知识并完成实践"
        
        generated = self._generate_goal_path(subject, goal, days, daily_hours, profile)
        if not generated:
            raise PlanGenerationError("学习计划暂未生成，请稍后重试。", "model_unavailable")
        weekly_plan = generated["source"]

        # 计算开始和结束日期
        start_date = datetime.combine(plan_today(), datetime.min.time())
        if generated.get("rolling"):
            daily_plan = deepcopy(generated['daily_entries'])
            weekly_plan = self._aggregate_daily(daily_plan, days, goal)
            deferred_tasks = []
        else:
            daily_plan, weekly_plan, deferred_tasks = self._schedule_with_budget(
                weekly_plan, days, daily_hours, goal, add_acceptance=False
            )
        for item in daily_plan:
            item["date"] = (start_date + timedelta(days=item["day"] - 1)).date().isoformat()
        end_date = start_date + timedelta(days=days)
        
        plan = {
            "subject": subject,
            "user_id": user_id,
            "duration": f"{weeks}周",
            "requested_duration": constraints.get("requested_duration") or duration,
            "duration_days": days,
            "goal": goal,
            "daily_hours": daily_hours,
            "daily_hours_source": constraints.get("daily_hours_source", "message"),
            "duration_months": constraints.get("duration_months"),
            "planned_start_date": start_date.date().isoformat(),
            "planned_end_date": (start_date + timedelta(days=days - 1)).date().isoformat(),
            "total_available_hours": round(days * daily_hours, 2),
            "deferred_tasks": deferred_tasks,
            "schedule_note": (
                f"每天 {daily_hours * 60:g} 分钟，共 {days} 天；按预计耗时安排，实际耗时可调整。"
                + (" 先安排本周，后续结合完成情况逐周细化；至少20%时间留给自主练习、复习和补进度。" if days > 14 else " 可用时间不必填满。")
                + (f" 有 {len(deferred_tasks)} 项任务超出本期时间预算，已列为后续任务。" if deferred_tasks else "")
            ),
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "weekly_plan": weekly_plan,
            "daily_plan": daily_plan,
            "stage_plan": generated.get('phases', []),
            "planning_mode": "rolling" if days > 14 else "daily",
            "detailed_weeks": [1] if days > 14 else list(range(1, weeks + 1)),
            "outcome": ("本期时间不足，请继续完成后续任务。" if deferred_tasks else "") + generated["outcome"],
            "assumptions": generated["assumptions"],
            "goal_feasible": generated["goal_feasible"] and not deferred_tasks,
            "generation_source": generated["generation_source"],
            "task_completion": self._build_task_completion(weekly_plan),
            "status": "active",
            "progress": 0,
            "created_at": datetime.now().isoformat()
        }
        
        # 保存到存储
        self.storage.save_plan(user_id, subject, plan)
        task_total = sum(len(week.get("tasks", []) or []) for week in weekly_plan)
        track_event("plan_created", user_id, {
            "plan_key": plan_key,
            "subject": subject,
            "duration": plan["duration"],
            "task_total": task_total,
        })
        enriched = self._attach_dynamic_fields(plan, user_id, subject)
        return enriched

    def _build_rolling_draft(self, subject, goal, days, daily_hours):
        source = self._build_goal_draft(subject, goal)
        titles = ['基础与目标', '跟做案例', '独立实践', '检验与总结']
        objectives = [f'梳理{subject}的必要基础和目标范围', '理解并跟做一个相关案例',
                      f'围绕「{goal}」独立实践', '检查成果并补齐主要差距']
        phases = [{'title': title, 'start_day': idx * days // 4 + 1, 'end_day': (idx + 1) * days // 4,
                   'objective': objectives[idx], 'deliverable': ['基础清单', '案例记录', '实践成果', '验收记录'][idx]}
                  for idx, title in enumerate(titles)]
        daily = []
        for idx, entry in enumerate(source[:7]):
            hours = min(.75, daily_hours * .8)
            daily.append({'day':idx + 1, 'topics': entry['topics'], 'tasks':entry['tasks'],
                          'task_estimates':[hours], 'deliverables':entry['deliverables'],
                          'estimated_hours':hours, 'available_hours':daily_hours, 'goals':[goal]})
        return {'rolling':True, 'source':[], 'phases':phases, 'daily_entries':daily,
                'outcome':f'围绕「{goal}」完成基础梳理、实践和自检；具体专业步骤需要核对',
                'assumptions':['模型未能生成通过校验的具体计划，当前显示通用草案；后续周在临近时结合进度细化。'],
                'goal_feasible':False, 'generation_source':'draft'}

    def _aggregate_daily(self, daily, days, goal):
        weekly = [{'week': idx + 1, 'topics':[], 'tasks':[], 'task_estimates':[], 'goals':[goal],
                   'available_days':min(7, days - idx * 7),
                   'available_hours':0.0, 'estimated_hours':0.0} for idx in range(math.ceil(days / 7))]
        for day in daily:
            number = (day['day'] - 1) // 7 + 1
            week = weekly[number - 1]
            week['available_hours'] = round(week['available_hours'] + day['available_hours'], 4)
            week['estimated_hours'] = round(week['estimated_hours'] + day['estimated_hours'], 4)
            day['task_refs'] = []
            for task, estimate in zip(day['tasks'], day['task_estimates']):
                day['task_refs'].append({'week':str(number), 'task_index':len(week['tasks'])})
                week['tasks'].append(f"第{day['day']}天：{task}")
                week['task_estimates'].append(estimate)
            for topic in day['topics']:
                if topic not in week['topics']:
                    week['topics'].append(topic)
        return weekly

    def refine_learning_week(self, user_id, subject, week):
        plan = self.storage.get_plan(user_id, subject)
        if not plan or not 1 <= week <= math.ceil(plan.get('duration_days', 0) / 7):
            return {'status':'error', 'message':'未找到计划或周数不正确'}
        if plan.get('duration_days', 0) <= 14:
            return {'status':'error', 'message':'短计划已逐日安排，可在修改条件中调整。'}
        completed = plan.get('task_completion', {}).get(str(week), [])
        if any(completed):
            return {'status':'error', 'message':'这周已有完成记录，请保留现有任务，在修改条件时重新安排未完成内容。'}
        detailed_weeks = plan.get('detailed_weeks') or sorted(set((d['day'] - 1) // 7 + 1 for d in plan.get('daily_plan', [])))
        if week > 1 and week - 1 not in detailed_weeks:
            return {'status':'error', 'message':'请先安排前一周，再结合完成情况细化本周。'}
        current = max(1, (plan_today() - datetime.fromisoformat(plan['start_date']).date()).days // 7 + 1)
        previous_done = plan.get('task_completion', {}).get(str(week - 1), [])
        if week > current + 1 and not (previous_done and all(previous_done)):
            return {'status':'error','message':'后续周在临近时再细化；如果提前完成前一周，也可以继续安排。'}
        progress = []
        tc = plan.get('task_completion', {})
        for item in plan.get('weekly_plan', []):
            if item.get('week', 1) >= week:
                break
            done = tc.get(str(item['week']), [])
            progress.extend({'task': task, 'completed':bool(done[idx]) if idx < len(done) else False}
                            for idx, task in enumerate(item.get('tasks', [])))
        try:
            detailed = self._generate_learning_week(plan, week, {'previous_tasks':progress[-40:]})
        except PlanGenerationError as exc:
            return {'status':'error', 'message':str(exc), 'error_code':exc.code,
                    'http_status':exc.http_status, 'retryable':exc.retryable}
        if not detailed:
            return {'status':'error','message':'本周具体安排暂未生成，请稍后重试；原计划和完成记录已保留。',
                    'error_code':'model_unavailable', 'http_status':503, 'retryable':True}
        if self.storage.get_plan(user_id, subject) != plan:
            return {'status':'error','message':'生成期间计划或完成记录已变化，请重新尝试，以使用最新进度。'}
        expected = deepcopy(plan)
        plan = deepcopy(plan)
        start = datetime.fromisoformat(plan['start_date'])
        for item in detailed:
            item['date'] = (start + timedelta(days=item['day'] - 1)).date().isoformat()
            item['goals'] = [plan['goal']]
        daily = [d for d in plan.get('daily_plan', []) if (d['day'] - 1) // 7 + 1 != week]
        daily.extend(detailed)
        daily.sort(key=lambda d:d['day'])
        plan['daily_plan'] = daily
        weekly = self._aggregate_daily(daily, plan['duration_days'], plan['goal'])
        # 保持其他周的索引、任务和完成记录，替换仅限未开始的目标周。
        old_weekly = {item['week']: item for item in plan.get('weekly_plan', [])}
        plan['weekly_plan'] = [entry if entry['week'] == week else old_weekly.get(entry['week'], entry) for entry in weekly]
        plan['task_completion'][str(week)] = [False] * len(plan['weekly_plan'][week - 1]['tasks'])
        plan['detailed_weeks'] = sorted(set(detailed_weeks + [week]))
        plan['planning_mode'] = 'rolling'
        plan['progress_summary'] = summarize_plan_progress(plan)
        plan['progress'] = plan['progress_summary']['detailed_task_percentage'] or 0
        plan['updated_at'] = datetime.now().isoformat()
        if hasattr(self.storage, 'save_plan_if_unchanged'):
            if not self.storage.save_plan_if_unchanged(user_id, subject, expected, plan):
                return {'status':'error','message':'生成期间计划或完成记录已变化，请使用最新进度重试。'}
        else:
            self.storage.save_plan(user_id, subject, plan)
        return {'status':'success','plan':self._attach_dynamic_fields(plan, user_id, subject),
                'message':f'已结合完成记录安排第{week}周'}

    def _generate_learning_week(self, plan, week, progress):
        return generate_learning_week(plan, week, progress)

    def _generate_goal_path(self, subject, goal, days, daily_hours, profile):
        return generate_goal_path(subject, goal, days, daily_hours, profile)

    def _build_goal_draft(self, subject: str, goal: str) -> List[Dict[str, Any]]:
        """模型不可用时明确展示通用草案，覆盖学习、实践和验收，不冒充领域方案。"""
        stages = [
            ("目标与范围", f"把「{goal}」拆成可检查的结果，确定本期最小范围和必要基础", "目标清单与基础差距"),
            ("准备资料与工具", f"选择一份{subject}入门资料和一个与目标有关的案例，准备所需工具", "入门资料和案例"),
            ("核心概念", f"阅读{subject}的核心概念，用自己的话解释3个关键点并举例", "3个概念解释"),
            ("理解关键步骤", f"拆解案例中实现「{goal}」所需的步骤，标出不懂的环节", "操作步骤与疑问清单"),
            ("跟做示例", f"跟做一个与{subject}目标有关的最小示例，记录操作和结果", "示例结果与操作记录"),
            ("完成第一次实践", f"围绕「{goal}」独立完成最小范围的实践，保存可展示的结果", "第一次实践成果"),
            ("检验薄弱环节", f"对照目标检查{subject}实践结果，定位并补齐一个主要薄弱环节", "问题与修正记录"),
            ("整理完整成果", f"将实践结果整理成「{goal}」的最小演示或成果材料", "可展示成果"),
            ("测试与自检", f"按目标清单检查成果，验证典型情况和一个失败情况并修正", "自检记录"),
            ("成果验收", f"展示「{goal}」相关成果，解释关键步骤，记录未达到的部分和下一步", "成果与后续改进清单"),
        ]
        return [{"week": idx+1, "topics": [topic], "tasks": [task], "task_estimates": [.75],
                 "deliverables": [deliverable], "goals": [goal]}
                for idx, (topic, task, deliverable) in enumerate(stages)]

    def _schedule_with_budget(self, source: List[Dict[str, Any]], days: int,
                              daily_hours: float, goal: str, add_acceptance: bool = True):
        """先生成逐日执行安排，再聚合兼容原有任务完成/进度接口的周索引。"""
        entries = []
        for week in source:
            estimates = week.get("task_estimates", [])
            for idx, task in enumerate(week.get("tasks", [])):
                hours = estimates[idx] if idx < len(estimates) else 2
                entries.append((task, float(hours), week.get("topics", []), week.get("deliverables", [])))
        if add_acceptance:
            entries.append((f"围绕目标「{goal}」提交学习成果并自测验收", 2.0, ["成果验收"], []))
        scheduled = []
        # 任务按实际需要安排，不把短任务均摊成一整个周期。
        load_ratio = 1.0
        for idx in range(days):
            scheduled.append({"day": idx + 1, "topics": [], "tasks": [], "task_estimates": [], "deliverables": [],
                              "goals": [goal], "available_hours": round(daily_hours, 4), "estimated_hours": 0.0})
        cursor, deferred = 0, []
        for task, estimate, topics, deliverables in entries:
            # 一个任务允许拆成连续多天的学习片段，保持前置知识顺序。
            remaining = estimate
            allocations = []
            # 能在一天完成的任务保持完整，不从前一天剩下的几分钟拆出一个“部分”。
            if estimate <= daily_hours and cursor < len(scheduled):
                free = scheduled[cursor]['available_hours'] - scheduled[cursor]['estimated_hours']
                if free + .0001 < estimate:
                    cursor += 1
            while cursor < len(scheduled) and remaining > 0.0001:
                week = scheduled[cursor]
                capacity = round(week["available_hours"] * load_ratio, 4)
                free = capacity - week["estimated_hours"]
                if free <= 0.0001:
                    cursor += 1
                    continue
                part = min(free, remaining)
                allocations.append((week, part))
                week["estimated_hours"] = round(week["estimated_hours"] + part, 4)
                remaining -= part
            if remaining > 0.0001:
                deferred.append({"task": task, "estimated_hours": round(remaining, 2)})
            for part_idx, (week, part) in enumerate(allocations):
                label = task if len(allocations) == 1 else f"{task}（第 {part_idx + 1}/{len(allocations)} 部分）"
                if remaining > 0.0001:
                    label += f"（本期学习 {part:g} 小时，后续继续）"
                week["tasks"].append(label)
                week["task_estimates"].append(round(part, 4))
                if part_idx == len(allocations) - 1 and remaining <= 0.0001:
                    for deliverable in deliverables:
                        if deliverable not in week["deliverables"]:
                            week["deliverables"].append(deliverable)
                for topic in topics:
                    if topic not in week["topics"]:
                        week["topics"].append(topic)
        for week in scheduled:
            week["estimated_hours"] = round(week["estimated_hours"], 4)
        weekly = []
        for day in scheduled:
            week_num = (day["day"] - 1) // 7 + 1
            if len(weekly) < week_num:
                weekly.append({"week": week_num, "topics": [], "tasks": [], "task_estimates": [], "goals": [goal],
                               "available_days": 0, "available_hours": 0.0, "estimated_hours": 0.0})
            week = weekly[-1]
            week["available_days"] += 1
            week["available_hours"] = round(week["available_hours"] + day["available_hours"], 4)
            week["estimated_hours"] = round(week["estimated_hours"] + day["estimated_hours"], 4)
            day["task_refs"] = []
            for task, estimate in zip(day["tasks"], day["task_estimates"]):
                day["task_refs"].append({"week": str(week_num), "task_index": len(week["tasks"])})
                week["tasks"].append(f"第{day['day']}天：{task}")
                week["task_estimates"].append(estimate)
            for topic in day["topics"]:
                if topic not in week["topics"]:
                    week["topics"].append(topic)
        return scheduled, weekly, deferred

    def _match_template_subject(self, subject: str) -> Optional[str]:
        """判断主题是否有内置模板；没则返回 None（走自由主题生成）。"""
        subject_lower = (subject or "").lower()
        for key in self.plan_templates.keys():
            if key == "通用":
                continue
            if key in subject_lower or subject_lower in key:
                return key
        return None

    def _get_template_plan(self, subject: str, duration: str) -> List[Dict[str, Any]]:
        """从模板获取计划"""
        subject_lower = subject.lower()
        template_subject = "通用"
        for key in self.plan_templates.keys():
            if key in subject_lower:
                template_subject = key
                break
        
        templates = self.plan_templates.get(template_subject, self.plan_templates["通用"])
        template_plan = templates.get(duration, templates.get("4周", []))
        return [deepcopy(item) for item in template_plan]

    def _interview_role(self, subject: str) -> str:
        role = str(subject or "")
        for key in ("面试", "求职", "应聘"):
            role = role.replace(key, "")
        return role.strip(" 　的")

    def _is_interview_subject(self, subject: str) -> bool:
        text = str(subject or "")
        if not any(key in text for key in ("面试", "求职", "应聘")):
            return False
        return bool(self._interview_role(text))

    def _resolve_topic_goal(self, subject: str, profile: Dict[str, Any]) -> str:
        """画像目标只在与主题相关时才写进计划，避免"产品经理面试"挂上"春招后端开发"。
        画像目标不相关时返回空串，由调用方使用主题默认目标。"""
        goal = str((profile or {}).get("goal") or "").strip()
        if not goal:
            return ""
        if self._is_interview_subject(subject):
            role = self._interview_role(subject)
            return goal if role and role in goal else f"完成{role}面试准备"
        subject_lower = str(subject or "").lower()
        goal_lower = goal.lower()
        if subject_lower and (subject_lower in goal_lower or goal_lower in subject_lower):
            return goal
        return ""

    def _build_topic_plan(self, subject: str, duration: str, profile: Dict[str, Any], constraints: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """为不在固定模板中的明确主题（如 RAG、经济学）生成围绕该主题的具体计划。
        必要信息缺失时使用明确标注的默认值，不伪造个性化。"""
        constraints = constraints or {}
        weeks = self._parse_weeks(duration)
        level = constraints.get("level") or profile.get("level") or "零基础"
        goal = constraints.get("goal") or self._resolve_topic_goal(subject, profile)
        daily_hours = constraints.get("daily_hours") or profile.get("daily_hours") or 2

        # 面试/求职主题：阶段任务围绕岗位认知、简历、案例和模拟面试，而不是"梳理定义"
        if self._is_interview_subject(subject):
            return self._build_interview_plan(subject, weeks, goal)

        goal = goal or "掌握该主题的核心知识并能实践"
        # 用主题名构造具体、可执行的阶段任务，而不是通用"基础知识/环境搭建"
        stages = [
            (f"{subject}核心概念与术语", [
                f"梳理{subject}的定义、整体框架与关键术语",
                f"通读{subject}的入门材料并做笔记",
            ]),
            (f"{subject}关键原理深入", [
                f"逐个攻克{subject}的核心原理并写出自己的理解",
                f"针对{subject}的每个原理各做一次复述练习",
            ]),
            (f"{subject}实践应用", [
                f"围绕{subject}动手完成一个小型实践/案例",
                f"用{subject}知识解决 2-3 个具体问题",
            ]),
            (f"{subject}复盘与检验", [
                f"整理{subject}的知识脑图并自测关键概念",
                f"总结{subject}学习成果与薄弱点",
            ]),
        ]
        weekly_plan: List[Dict[str, Any]] = []
        for week in range(1, weeks + 1):
            if week <= len(stages):
                topic, tasks = stages[week - 1]
            else:
                topic, tasks = f"{subject}综合提升（第{week}周）", [
                    f"继续深入{subject}的薄弱部分", f"完成{subject}阶段性练习"
                ]
            weekly_plan.append({
                "week": week,
                "topics": [topic],
                "tasks": list(tasks),
                "goals": [goal],
                "generation_note": (
                    f"按主题《{subject}》生成；基础水平={level}（默认/画像），"
                    f"每日可用时间={daily_hours}小时（默认/画像）。"
                ),
            })
        return weekly_plan

    def _build_interview_plan(self, subject: str, weeks: int, goal: str) -> List[Dict[str, Any]]:
        """面试准备计划：岗位认知 → 简历 → 案例分析 → 模拟面试。"""
        role = self._interview_role(subject) or "目标岗位"
        weekly_goal = goal or f"完成{role}面试准备"
        stages = [
            ("岗位与能力认知", [
                f"梳理{role}的岗位职责与能力要求，列出自己的差距清单",
                "调研 2-3 家目标公司的业务与产品，记录面试重点",
            ]),
            ("简历与经历梳理", [
                "用 STAR 结构化 2-3 段项目/校园经历，并量化结果",
                f"对照{role}要求改一版简历，准备 1 分钟自我介绍",
            ]),
            ("产品分析与案例", [
                f"拆解 1 款目标公司的产品，写 1 页优化建议",
                "练习竞品分析、需求优先级、数据分析常见题型",
            ]),
            ("模拟面试与复盘", [
                "准备高频行为面试问题与反问清单",
                "完成 2 次模拟面试，记录问题并补齐材料",
            ]),
        ]
        weekly_plan: List[Dict[str, Any]] = []
        for week in range(1, weeks + 1):
            if week <= len(stages):
                topic, tasks = stages[week - 1]
            else:
                topic, tasks = f"{role}面试强化（第{week}周）", [
                    "复盘模拟面试中的薄弱环节", "补充新的产品分析案例",
                ]
            weekly_plan.append({
                "week": week,
                "topics": [topic],
                "tasks": list(tasks),
                "goals": [weekly_goal],
                "generation_note": f"按面试准备主题《{subject}》生成，不套用画像中的无关目标。",
            })
        return weekly_plan

    def _parse_weeks(self, duration: str) -> int:
        match = re.search(r"(\d+)", str(duration or ""))
        return max(1, int(match.group(1))) if match else 4
    
    def get_learning_plan(self, user_id: str, subject: str) -> Dict[str, Any]:
        """获取学习计划"""
        plan = self.storage.get_plan(user_id, subject)
        if not plan:
            return {}
        updated, plan = self._ensure_task_completion(plan)
        if updated:
            self.storage.save_plan(user_id, subject, plan)
        return self._attach_dynamic_fields(plan, user_id, subject)
    
    def update_learning_plan(self, user_id: str, subject: str, plan_data: Dict[str, Any]) -> Dict[str, Any]:
        """更新学习计划"""
        plan = self.storage.get_plan(user_id, subject)
        if plan:
            plan.update(plan_data)
            _, plan = self._ensure_task_completion(plan)
            plan["updated_at"] = datetime.now().isoformat()
            self.storage.save_plan(user_id, subject, plan)
            return self._attach_dynamic_fields(plan, user_id, subject)
        return {}

    def _build_task_completion(self, weekly_plan: List[Dict[str, Any]]) -> Dict[str, List[bool]]:
        completion: Dict[str, List[bool]] = {}
        for idx, item in enumerate(weekly_plan or []):
            week_key = str(item.get("week", idx + 1))
            tasks = item.get("tasks", []) or []
            completion[week_key] = [False for _ in range(len(tasks))]
        return completion

    def _ensure_task_completion(self, plan: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        changed = False
        if "task_completion" not in plan or not isinstance(plan.get("task_completion"), dict):
            plan["task_completion"] = {}
            changed = True
        tc: Dict[str, Any] = plan.get("task_completion") or {}
        for idx, item in enumerate(plan.get("weekly_plan", []) or []):
            week_key = str(item.get("week", idx + 1))
            tasks = item.get("tasks", []) or []
            existing = tc.get(week_key)
            if not isinstance(existing, list):
                tc[week_key] = [False for _ in range(len(tasks))]
                changed = True
                continue
            norm = [bool(x) for x in existing]
            if len(norm) < len(tasks):
                norm.extend([False for _ in range(len(tasks) - len(norm))])
                changed = True
            if len(norm) > len(tasks):
                norm = norm[:len(tasks)]
                changed = True
            tc[week_key] = norm
        plan["task_completion"] = tc
        return changed, plan
    
    def get_all_plans(self, user_id: str) -> Dict[str, Any]:
        """获取用户所有学习计划"""
        return self.storage.get_all_plans(user_id)

    def delete_learning_plan(self, user_id: str, subject: str) -> bool:
        """删除学习计划"""
        # 检查计划是否存在
        if self.storage.get_plan(user_id, subject):
            # 调用存储层删除计划
            return self.storage.delete_plan(user_id, subject)
        return False

    # === 高级能力 ===
    def adjust_plan(self, user_id: str, subject: str) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        plan = self.storage.get_plan(user_id, subject)
        if not plan:
            return {}, []
        progress = self.storage.get_progress(user_id, subject)
        adjusted_plan, adjustments = self._adapt_plan_with_progress(plan, progress)
        self.storage.save_plan(user_id, subject, adjusted_plan)
        carry_over_count = sum(
            len(week.get("carry_over", []) or [])
            for week in adjusted_plan.get("weekly_plan", [])
        )
        track_event("plan_adjusted", user_id, {
            "plan_key": f"{user_id}_{subject}",
            "subject": subject,
            "adjustment_count": len(adjustments),
            "carry_over_count": carry_over_count,
        })
        if carry_over_count:
            track_event("task_carried_over", user_id, {
                "plan_key": f"{user_id}_{subject}",
                "subject": subject,
                "carry_over_count": carry_over_count,
            })
        enriched = self._attach_dynamic_fields(adjusted_plan, user_id, subject, progress=progress)
        return enriched, adjustments

    def generate_phase_review(self, user_id: str, subject: str) -> Dict[str, Any]:
        plan = self.storage.get_plan(user_id, subject)
        progress = self.storage.get_progress(user_id, subject)
        if not plan:
            return {"message": "未找到学习计划"}
        return self._build_phase_review(plan, progress)

    # === 私有辅助方法 ===
    def _attach_dynamic_fields(self, plan: Dict[str, Any], user_id: str, subject: str, progress: Dict[str, Any] = None) -> Dict[str, Any]:
        plan = deepcopy(plan)
        plan['progress_summary'] = summarize_plan_progress(plan)
        plan['progress'] = plan['progress_summary']['detailed_task_percentage'] or 0
        progress = progress or self.storage.get_progress(user_id, subject)
        if progress and progress.get('progress_scope') == 'detailed_tasks':
            progress = {**progress, **plan_progress_data(plan)}
        plan["timeline"] = self._build_timeline(plan)
        plan["current_week"] = plan["progress_summary"]["current_week"]
        if progress:
            plan["adaptive_status"] = self._summarize_progress(progress)
            plan["weekly_plan"] = self._merge_progress_into_plan(plan["weekly_plan"], progress)
        return plan

    def _build_timeline(self, plan: Dict[str, Any]) -> List[Dict[str, Any]]:
        timeline = []
        start = datetime.fromisoformat(plan["start_date"])
        for idx, item in enumerate(plan.get("weekly_plan", [])):
            week_start = start + timedelta(weeks=idx)
            week_end = week_start + timedelta(days=item.get("available_days", 7) - 1)
            timeline.append({
                "id": f"week-{item.get('week', idx + 1)}",
                "label": f"第{item.get('week', idx + 1)}周",
                "start": week_start.date().isoformat(),
                "end": week_end.date().isoformat(),
                "topics": item.get("topics", []),
                "tasks": item.get("tasks", []),
                "goals": item.get("goals", [])
            })
        return timeline

    def _current_week(self, plan: Dict[str, Any]) -> int:
        start = datetime.fromisoformat(plan["start_date"])
        delta = datetime.now() - start
        weeks = max(1, min(len(plan.get("weekly_plan", [])), delta.days // 7 + 1))
        return weeks

    def _merge_progress_into_plan(self, weekly_plan: List[Dict[str, Any]], progress: Dict[str, Any]) -> List[Dict[str, Any]]:
        merge_plan = []
        weekly_progress = progress.get("weekly_progress", {})
        for idx, item in enumerate(weekly_plan):
            week_key = str(item.get("week", idx + 1))
            status = deepcopy(item)
            prog = weekly_progress.get(week_key, {})
            completed = prog.get("completed_tasks", prog.get("completed", 0))
            total = prog.get("total_tasks", prog.get("total", len(item.get("tasks", []))))
            completion = 0
            if total:
                completion = int((completed / total) * 100)
            status["progress_snapshot"] = {
                "completed_tasks": completed,
                "total_tasks": total,
                "completion": completion
            }
            if completion >= 80:
                status["status"] = "on_track"
            elif completion >= 40:
                status["status"] = "attention"
                status["recommended_actions"] = ["复习关键知识点", "增加针对性练习"]
            else:
                status["status"] = "off_track"
                status["recommended_actions"] = ["安排补课时段", "与导师沟通学习策略"]
            merge_plan.append(status)
        return merge_plan

    def _adapt_plan_with_progress(self, plan: Dict[str, Any], progress: Dict[str, Any]) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        if not progress:
            return plan, [{"message": "暂无学习进度，建议及时更新进度以便自动调整"}]
        plan = deepcopy(plan)
        if plan.get("daily_plan"):
            # 每日任务和周索引一一对应，不能单独改周任务后让勾选指向另一项。
            return plan, [{"message": "已保留每日任务与时间预算；未完成内容可在剩余日期内补学，超出预算时需要重新规划。"}]
        adjustments = []
        weekly_progress = progress.get("weekly_progress", {})
        carry_over_tasks = []
        for idx, week in enumerate(plan.get("weekly_plan", [])):
            week_key = str(week.get("week", idx + 1))
            prog = weekly_progress.get(week_key, {})
            completed = prog.get("completed_tasks", 0)
            total = prog.get("total_tasks", len(week.get("tasks", [])) or 1)
            completion = completed / total if total else 0
            if completion < 0.7:
                missed_tasks = [t for t in week.get("tasks", []) if t not in prog.get("completed_list", [])]
                carry_over_tasks.extend(missed_tasks)
                adjustments.append({
                    "week": week_key,
                    "status": "lagging",
                    "completion": round(completion * 100, 1),
                    "action": "将未完成任务滚动到下一周，并安排额外复习时间"
                })
            else:
                adjustments.append({
                    "week": week_key,
                    "status": "on_track",
                    "completion": round(completion * 100, 1),
                    "action": "保持当前节奏，适当拓展进阶内容"
                })
            # 附加carry-over
            if carry_over_tasks:
                week.setdefault("carry_over", [])
                week["carry_over"].extend(carry_over_tasks)
        if carry_over_tasks and plan.get("weekly_plan"):
            plan["weekly_plan"][-1].setdefault("tasks", [])
            plan["weekly_plan"][-1]["tasks"].extend(carry_over_tasks)
        plan["last_adapted_at"] = datetime.now().isoformat()
        return plan, adjustments

    def _summarize_progress(self, progress: Dict[str, Any]) -> Dict[str, Any]:
        total = progress.get("total_tasks", 0)
        completed = progress.get("completed_tasks", 0)
        percentage = progress.get("progress_percentage", 0)
        study_time = progress.get("study_time", 0)
        return {
            "total_tasks": total,
            "completed_tasks": completed,
            "progress_percentage": percentage,
            "study_time_minutes": study_time
        }

    def _build_phase_review(self, plan: Dict[str, Any], progress: Dict[str, Any]) -> Dict[str, Any]:
        review = {
            "subject": plan.get("subject", ""),
            "duration": plan.get("duration", ""),
            "generated_at": datetime.now().isoformat(),
            "summary": "",
            "highlights": [],
            "improvements": [],
            "next_focus": []
        }
        if not progress:
            review["summary"] = "暂无进度数据，建议尽快在学习过程中记录任务完成度与学习时间。"
            review["next_focus"] = ["制定学习打卡机制", "每周回顾学习成效"]
            return review
        percentage = progress.get("progress_percentage", 0)
        study_minutes = progress.get("study_time", 0)
        total_hours = round(study_minutes / 60, 1)
        review["summary"] = f"当前整体完成度约为 {percentage}% ，累计学习约 {total_hours} 小时。"
        if percentage >= 80:
            review["highlights"].append("学习进度良好，已基本掌握阶段目标。")
        elif percentage >= 50:
            review["highlights"].append("学习节奏稳定，建议保持输出与复盘。")
        else:
            review["improvements"].append("当前进度滞后，建议重新分配学习时间与任务优先级。")
        notes = progress.get("notes", [])
        if notes:
            latest_notes = sorted(notes, key=lambda x: x.get("timestamp", ""), reverse=True)[:3]
            review["highlights"].append("近期重点记录：" + "；".join(n.get("note", "") for n in latest_notes))
        review["next_focus"] = [
            "结合自动调整后的学习计划完成补课任务",
            "维持每周不少于3次的学习频率",
            "阶段结束前整理输出一份学习总结"
        ]
        return review
    
    def _enhance_plan_with_rag(self, weekly_plan: List[Dict[str, Any]], subject: str, duration: str, profile: Dict[str, Any] = None) -> Optional[List[Dict[str, Any]]]:
        """使用RAG增强学习计划，获取更详细的学习内容，并注入用户画像信息"""
        try:
            import threading
            import time
            
            # 定义一个内部函数来执行RAG调用
            def rag_call():
                nonlocal enhanced_plan
                try:
                    rag = get_rag_helper()
                    
                    # 构建用户画像上下文
                    profile_context = ""
                    if profile:
                        profile_context = f"用户画像: 专业={profile.get('major', '未知')}, 目标={profile.get('goal', '未知')}, 薄弱点={','.join(profile.get('weak_areas', []))}, 每日可用时间={profile.get('daily_hours', 2)}小时。"
                    
                    for week_item in enhanced_plan:
                        week_num = week_item.get("week", 1)
                        topics = week_item.get("topics", [])

                        # 为每个主题获取RAG建议，并把结果写回该周的tasks
                        enhanced_tasks = []
                        for topic in topics:
                            # 查询只围绕当前主题，不带画像目标（避免检索到"春招后端/软件工程"等无关资料）
                            query = f"请给出学习 {topic} 的具体步骤和任务，只围绕 {topic} 本身。"
                            rag_result = rag.ask_with_details(
                                query, subject=subject,
                                chat_history=profile_context + f" 本次计划周期：{duration}。请按目标和每日时间给出可执行任务。"
                            )

                            if rag_result and rag_result.get("answer"):
                                answer = rag_result.get("answer", "")
                                # 只保留真正围绕主题的RAG答案，通用知识兜底答案不写回计划
                                if rag_result.get("mode") in {"general_llm", "no_context", "error"}:
                                    continue
                                # 从答案中提取任务建议，并要求与主题相关
                                suggestions = self._extract_task_suggestions(answer, topic)
                                enhanced_tasks.extend(suggestions)

                        # 把与主题相关的任务建议写回周计划
                        if enhanced_tasks:
                            existing = [t for t in week_item.get("tasks", []) if isinstance(t, str)]
                            week_item["tasks"] = existing + enhanced_tasks
                except Exception as e:
                    logger.warning(f"RAG调用失败: {e}")
                    # 如果RAG调用失败，保持enhanced_plan不变
            
            enhanced_plan = deepcopy(weekly_plan)

            # 同步执行RAG增强，确保提取到的任务建议能够写回计划。
            # 原线程+join(timeout=10)方案在超时或竞争时会丢弃写回结果，导致增强永远不生效。
            rag_call()
            return enhanced_plan
        except Exception as e:
            logger.error(f"RAG增强计划失败: {e}")
            return None
    
    def _extract_task_suggestions(self, answer: str, topic: str) -> List[str]:
        """从RAG答案中提取任务建议，只保留与当前主题相关的条目。"""
        import re
        suggestions = []
        # 字母词（RAG/Python）与中文词（核心概念）分开提取，避免粘连成一个匹配不到的长词
        topic_text = str(topic or "")
        topic_keywords = [k.lower() for k in re.findall(r"[a-zA-Z]{2,}", topic_text)]
        topic_keywords += [k for k in re.findall(r"[\u4e00-\u9fa5]{2,}", topic_text)]

        def relevant(task: str) -> bool:
            task_lower = task.lower()
            return any(k in task_lower for k in topic_keywords)

        # 查找列表项（1. 2. 3. 或 - *）
        list_patterns = [
            r'\d+[\.\)]\s*([^。\n]+)',
            r'[-•]\s*([^。\n]+)',
            r'（\d+）\s*([^。\n]+)'
        ]

        for pattern in list_patterns:
            matches = re.findall(pattern, answer)
            for match in matches:
                task = match.strip()
                if 5 < len(task) < 100 and relevant(task):
                    suggestions.append(task)

        # 如果没有找到列表，尝试从句子中提取
        if not suggestions:
            sentences = re.split(r'[。\n]', answer)
            for sentence in sentences:
                sentence = sentence.strip()
                if 10 < len(sentence) < 80 and ('学习' in sentence or '练习' in sentence or '完成' in sentence) and relevant(sentence):
                    suggestions.append(sentence)

        return suggestions[:5]
    
    def _smart_adjust_schedule(self, plan: Dict[str, Any], progress: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
        """智能调整学习计划（改进版）"""
        adjusted_plan = deepcopy(plan)
        adjustments = []
        if adjusted_plan.get("daily_plan"):
            return adjusted_plan, ["已保留每日任务与时间预算，请按完成情况重新规划需要调整的内容。"]
        
        if not progress:
            return adjusted_plan, ["暂无进度数据，无法进行智能调整"]
        
        weekly_progress = progress.get("weekly_progress", {})
        weekly_plan = adjusted_plan.get("weekly_plan", [])
        
        for week_item in weekly_plan:
            week_key = str(week_item.get("week", 1))
            week_prog = weekly_progress.get(week_key, {})
            
            completed = week_prog.get("completed_tasks", 0)
            total = week_prog.get("total_tasks", len(week_item.get("tasks", [])))
            
            if total == 0:
                continue
            
            completion_rate = completed / total
            
            # 智能调整逻辑
            if completion_rate < 0.5:
                # 进度落后：简化任务，延长时间
                original_tasks = week_item.get("tasks", [])
                # 保留核心任务，移除辅助任务
                core_tasks = original_tasks[:max(1, len(original_tasks) // 2)]
                week_item["tasks"] = core_tasks
                week_item["status"] = "simplified"
                adjustments.append(f"第{week_key}周：简化任务列表，聚焦核心内容")
            
            elif completion_rate > 0.9:
                # 进度超前：添加扩展任务
                if not week_item.get("extra_tasks"):
                    week_item["extra_tasks"] = self._generate_extra_tasks(week_item.get("topics", []))
                    week_item["status"] = "extended"
                    adjustments.append(f"第{week_key}周：添加扩展任务，深化学习")
        
        return adjusted_plan, adjustments
    
    def _generate_extra_tasks(self, topics: List[str]) -> List[str]:
        """为超前学习者生成扩展任务"""
        extra_tasks = []
        
        for topic in topics:
            if "基础" in topic or "入门" in topic:
                extra_tasks.append(f"深入探索{topic}的高级应用")
            elif "项目" in topic:
                extra_tasks.append(f"尝试{topic}的更复杂场景")
            else:
                extra_tasks.append(f"阅读{topic}相关的进阶资料")
        
        return extra_tasks[:3]  # 限制数量
