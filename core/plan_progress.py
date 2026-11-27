"""从任务勾选计算进度；本周、已安排任务和全周期覆盖分别表示。"""
from __future__ import annotations

import math
from datetime import date, datetime
from typing import Any, Dict

from nlp.plan_constraints import plan_today


def summarize_plan_progress(plan: Dict[str, Any], today: date | None = None):
    weeks = plan.get('weekly_plan') or []
    completion = plan.get('task_completion') if isinstance(plan.get('task_completion'), dict) else {}
    total_weeks = math.ceil(plan['duration_days'] / 7) if plan.get('duration_days') else max(1, len(weeks))
    current_week = 1
    try:
        start = plan.get('planned_start_date') or plan.get('start_date')
        if start:
            elapsed = ((today or plan_today()) - datetime.fromisoformat(start).date()).days
            current_week = max(1, min(total_weeks, elapsed // 7 + 1))
    except (ValueError, TypeError):
        pass
    summaries, tasks = {}, []
    for idx, item in enumerate(weeks):
        key = str(item.get('week', idx + 1))
        names = item.get('tasks') or []
        values = completion.get(key) if isinstance(completion.get(key), list) else []
        flags = [bool(values[i]) if i < len(values) else False for i in range(len(names))]
        done = sum(flags)
        summaries[key] = {'completed_tasks': done, 'total_tasks': len(names),
                          'percentage': int(done / len(names) * 100) if names else None}
        for i, name in enumerate(names):
            tasks.append({'id': f'{key}_{i}', 'name': name if plan.get('daily_plan') else f'第{key}周：{name}',
                          'completed': flags[i], 'week': key, 'index': i})
    total = len(tasks)
    done = sum(task['completed'] for task in tasks)
    detailed_weeks = sum(s['total_tasks'] > 0 for s in summaries.values())
    current = summaries.get(str(current_week), {'completed_tasks': 0, 'total_tasks': 0, 'percentage': None})
    return {'completed_tasks': done, 'total_tasks': total,
            'detailed_task_percentage': int(done / total * 100) if total else None,
            'current_week': current_week, 'current_week_completed': current['completed_tasks'],
            'current_week_total': current['total_tasks'], 'current_week_percentage': current['percentage'],
            'detailed_weeks': detailed_weeks, 'total_weeks': total_weeks,
            'has_unplanned_weeks': detailed_weeks < total_weeks,
            'weekly_progress': summaries, 'tasks': tasks}


def plan_progress_data(plan: Dict[str, Any]):
    summary = summarize_plan_progress(plan)
    return {'completed_tasks': summary['completed_tasks'], 'total_tasks': summary['total_tasks'],
            'weekly_progress': summary['weekly_progress'], 'tasks': summary['tasks'], 'study_time': 0,
            'plan_progress_summary': summary, 'progress_scope': 'detailed_tasks',
            'calculation_basis': {'verifiable': summary['total_tasks'] > 0, 'plan_subject': plan.get('subject'),
                                  'total_tasks_source': '已细化的计划任务', 'completed_tasks_source': '当前勾选状态',
                                  'explain': '按当前实际任务和勾选计算；未来尚未细化的周不计入分母，不表示整个学习目标完成率。'}}


def refresh_plan_progress(progress, plan):
    """读取勾选进度时以计划为准，保留独立自报记录、时长和笔记。"""
    if not progress or not plan:
        return progress
    scope = progress.get('progress_scope')
    is_task_record = any(isinstance(task, dict) and 'week' in task and 'index' in task
                         for task in (progress.get('tasks') or []))
    if scope != 'detailed_tasks' and (scope or not is_task_record):
        return progress
    result = dict(progress)
    payload = plan_progress_data(plan)
    result.update({key: value for key, value in payload.items() if key not in {'study_time', 'calculation_basis'}})
    result['progress_percentage'] = payload['plan_progress_summary']['detailed_task_percentage'] or 0
    result['percentage_verifiable'] = payload['calculation_basis']['verifiable']
    result['last_update_basis'] = payload['calculation_basis']
    return result
