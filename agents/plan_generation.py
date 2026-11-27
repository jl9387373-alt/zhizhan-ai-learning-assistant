"""按目标生成具体学习任务；主题、时间预算和基础条件统一传入模型。"""
from __future__ import annotations

import json
import math
import os
import urllib.request
import urllib.error
import socket
import re
from typing import Any, Dict

from core.config import config
from core.logger import logger


class PlanGenerationError(RuntimeError):
    def __init__(self, message: str, code: str, http_status: int = 503, retryable: bool = True):
        super().__init__(message)
        self.code = code
        self.http_status = http_status
        self.retryable = retryable


def validate_daily_entries(items: Any, first_day: int, last_day: int, daily_hours: float):
    if not isinstance(items, list) or len(items) != last_day - first_day + 1:
        raise ValueError("近期安排必须逐日覆盖当前周")
    result = []
    for day, item in zip(range(first_day, last_day + 1), items):
        if not isinstance(item, dict) or item.get("day") != day:
            raise ValueError("每日序号不连续")
        topic, task, deliverable = (str(item.get(key) or "").strip() for key in ("topic", "task", "deliverable"))
        hours = float(item.get("hours", 0))
        if not topic or len(task) < 8 or not deliverable or not math.isfinite(hours) or not 0 < hours <= daily_hours:
            raise ValueError("每日任务或时间无效")
        if re.search(r"第\s*\d+\s*/\s*\d+\s*部|第[一二三四五六七八九十\d]+部分", task):
            raise ValueError("每天需要具体小目标，不能只编号拆分")
        result.append({"day": day, "topics": [topic], "tasks": [task], "task_estimates": [hours],
                       "deliverables": [deliverable], "estimated_hours": hours, "available_hours": daily_hours})
    if len({item['tasks'][0] for item in result}) != len(result):
        raise ValueError("每日任务不能完全重复")
    return result


def validate_rolling_path(payload: Any, days: int, daily_hours: float):
    phases = payload.get("phases")
    if not isinstance(phases, list) or not 2 <= len(phases) <= 6:
        raise ValueError("长期计划需要2～6个阶段")
    normalized, expected = [], 1
    for phase in phases:
        if not isinstance(phase, dict) or phase.get("start_day") != expected:
            raise ValueError("阶段日期必须连续")
        end = phase.get("end_day")
        title, objective, deliverable = (str(phase.get(key) or '').strip() for key in ('title','objective','deliverable'))
        if not isinstance(end, int) or not expected <= end <= days or not all((title, objective, deliverable)):
            raise ValueError("阶段范围或成果缺失")
        normalized.append({"title": title, "objective": objective, "deliverable": deliverable,
                           "start_day": expected, "end_day": end})
        expected = end + 1
    if expected != days + 1:
        raise ValueError("阶段没有覆盖完整周期")
    daily = validate_daily_entries(payload.get('first_week'), 1, min(7, days), daily_hours)
    if any(day['estimated_hours'] > daily_hours * .8 + .0001 for day in daily):
        raise ValueError("主任务最多占每日时间80%，需要留出复习与补进度时间")
    outcome = str(payload.get('outcome') or '').strip()
    assumptions = payload.get('assumptions', [])
    if not outcome or not isinstance(payload.get('goal_feasible'), bool) or not isinstance(assumptions, list) or any(not isinstance(s,str) for s in assumptions):
        raise ValueError("成果或前提缺失")
    return {"phases": normalized, "daily_entries": daily, "rolling": True, "source": [], "outcome": outcome,
            "assumptions": assumptions, "goal_feasible": payload['goal_feasible'], "generation_source": "model"}


def validate_goal_path(payload: Any, days: int, daily_hours: float) -> Dict[str, Any]:
    if days > 14:
        if not isinstance(payload, dict):
            raise ValueError("长期计划格式错误")
        return validate_rolling_path(payload, days, daily_hours)
    if not isinstance(payload, dict) or not isinstance(payload.get("milestones"), list):
        raise ValueError("计划缺少任务列表")
    milestones = payload["milestones"]
    if not 2 <= len(milestones) <= 64:
        raise ValueError("计划任务数量不合理")
    source, total = [], 0.0
    for idx, item in enumerate(milestones):
        if not isinstance(item, dict):
            raise ValueError("计划任务格式错误")
        topic, task, deliverable = (str(item.get(key) or "").strip() for key in ("topic", "task", "deliverable"))
        hours = float(item.get("hours", 0))
        if not topic or len(task) < 8 or not deliverable or not math.isfinite(hours) or hours <= 0:
            raise ValueError("计划任务需要具体操作、产出和有效耗时")
        if hours > daily_hours or re.search(r"第\s*\d+\s*/\s*\d+\s*部", task):
            raise ValueError("短期任务应能在一天完成，且写明具体操作")
        total += hours
        source.append({"week": idx + 1, "topics": [topic], "tasks": [task],
                       "task_estimates": [hours], "deliverables": [deliverable]})
    if total > days * daily_hours + .001:
        raise ValueError("计划预计耗时超过可用时间")
    outcome = str(payload.get("outcome") or "").strip()
    if not outcome or not isinstance(payload.get("goal_feasible"), bool):
        raise ValueError("计划缺少成果范围或可行性说明")
    assumptions = payload.get("assumptions", [])
    if not isinstance(assumptions, list) or any(not isinstance(s, str) for s in assumptions):
        raise ValueError("基础条件格式错误")
    return {"source": source, "outcome": outcome, "assumptions": assumptions,
            "goal_feasible": payload["goal_feasible"], "generation_source": "model"}


def generate_goal_path(subject: str, goal: str, days: int, daily_hours: float,
                       profile: Dict[str, Any]) -> Dict[str, Any]:
    """复用项目模型配置；不初始化向量库，不把检索原文当作待执行任务。"""
    api_key = str(config.get("rag.api_key", "") or os.getenv("OPENAI_API_KEY") or os.getenv("DASHSCOPE_API_KEY") or "")
    if not api_key:
        raise PlanGenerationError("请先配置可用的模型服务，再生成学习计划。", "model_not_configured", 503, False)
    budget = round(days * daily_hours, 4)
    instructions = (
        "你是学习计划设计者。根据用户主题和目标，设计完整、可执行的学习任务路径。只返回JSON对象。"
        "所有领域使用同样的规则：从必要基础到实践、检查和最终验收，不能把实践与验收全部排除。"
        "不要生成固定四周模板。预计总耗时必须不超过用户总预算；减少范围，不能虚构学习速度。"
        "任务必须围绕实际主题，描述具体操作和产出，不能写‘按以下步骤执行’、‘第一部分’等空话。"
        "若目标不能在预算内完成，goal_feasible=false，outcome写明本期可达到的有限成果。"
        "若依赖已有基础、现成模板或工具，在assumptions写明。最终任务验收outcome。"
        "milestones按先后顺序排列；10天左右尽量10个里程碑，每个最多当天可用时长。"
        "每天可用时长是上限，不必填满，不得为了铺满周期增加任务耗时。不要输出Markdown。"
        '结构：{"outcome":"本期成果","goal_feasible":true,"assumptions":["前提"],'
        '"milestones":[{"topic":"阶段主题","task":"具体操作","hours":0.75,"deliverable":"可检查产出"}]}。'
    )
    if days > 14:
        instructions = (
            "你是学习计划设计者。长期计划只生成阶段方向和首周具体任务，返回JSON。"
            "所有领域按目标设计2～6个阶段，连续覆盖完整天数，阶段写明目标与成果；最后阶段验收最终成果。"
            "仅为第1～7天生成first_week，每天一个具体操作、小目标与产出，实际耗时不得超过每天上限。"
            "不要输出后续每一天的任务，不要用第1/7部分等重复句子代替小目标。"
            "每天时间是上限，不必填满；预留练习、复习和补进度时间。不要把验收拉长成一周的相同任务。"
            "首周主任务hours最多为每日可用时间的80%，至少20%留给自主练习、复习和补进度。"
            "若最终目标不可达，goal_feasible=false并缩小outcome，必要基础写在assumptions。"
            '结构：{"outcome":"本期成果","goal_feasible":true,"assumptions":[],"phases":'
            '[{"title":"阶段","start_day":1,"end_day":14,"objective":"阶段目标","deliverable":"产出"}],'
            '"first_week":[{"day":1,"topic":"当日主题","task":"具体操作","hours":1,"deliverable":"检查结果"}]}。'
        )
    requirements = {"subject": subject, "goal": goal, "days": days, "daily_hours": daily_hours,
                    "total_hours": budget, "known_skills": profile.get("known_skills", []),
                    "weak_areas": profile.get("weak_areas", []), "major": profile.get("major", "")}
    if days > 14:
        requirements['daily_core_hours_limit'] = round(daily_hours * .8, 4)
    content = _request_plan_json(instructions, requirements, strict=True)
    try:
        return validate_goal_path(content, days, daily_hours)
    except (ValueError, TypeError, KeyError) as exc:
        logger.warning("目标计划输出未通过约束校验: %s", type(exc).__name__)
        raise PlanGenerationError("模型生成的计划未通过任务和时间校验，请重试。", "plan_validation_failed", 502) from exc


def _request_plan_json(instructions: str, requirements: Dict[str, Any], *, strict: bool = False,
                       attempts: int = 1, max_tokens: int = 5000):
    api_key = str(config.get("rag.api_key", "") or os.getenv("OPENAI_API_KEY") or os.getenv("DASHSCOPE_API_KEY") or "")
    if not api_key:
        if strict:
            raise PlanGenerationError("请先配置可用的模型服务，再安排下一周。", "model_not_configured", 503, False)
        return None
    base_url = str(config.get("rag.base_url", "https://api.openai.com/v1")).rstrip("/")
    model = str(config.get("rag.model_name", "gpt-3.5-turbo"))
    body = json.dumps({"model": model, "messages": [
        {"role": "system", "content": instructions},
        {"role": "user", "content": json.dumps(requirements, ensure_ascii=False)}
    ], "temperature": .2, "max_tokens": max_tokens}, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(base_url + "/chat/completions", data=body,
                                     headers={"Content-Type": "application/json", "Authorization": "Bearer " + api_key})
    attempts = max(1, min(2, attempts))
    for attempt in range(attempts):
        try:
            timeout = max(45 if strict else 5, min(60, int(config.get("rag.request_timeout", 30))))
            with urllib.request.urlopen(request, timeout=timeout) as response:
                result = json.loads(response.read().decode("utf-8"))
            content = result["choices"][0]["message"]["content"].strip()
            if content.startswith("```"):
                content = content.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
            return json.loads(content)
        except (TimeoutError, socket.timeout) as exc:
            failure = PlanGenerationError("模型响应超时，请重试安排下一周；现有计划和完成记录已保留。", "model_timeout", 504)
        except urllib.error.HTTPError as exc:
            retryable = exc.code == 429 or exc.code >= 500
            failure = PlanGenerationError("模型服务暂不可用，请稍后重试。" if retryable else "模型服务配置或权限异常，请检查模型配置。",
                                          "model_unavailable" if retryable else "model_configuration_error", 503, retryable)
        except urllib.error.URLError as exc:
            failure = (PlanGenerationError("模型响应超时，请重试安排下一周；现有计划和完成记录已保留。", "model_timeout", 504)
                       if isinstance(exc.reason, (TimeoutError, socket.timeout)) else
                       PlanGenerationError("暂时无法连接模型服务，请稍后重试。", "model_connection_error", 503))
        except (ValueError, TypeError, KeyError, IndexError, AttributeError) as exc:
            failure = PlanGenerationError("模型没有返回有效的计划数据，请重试；原计划已保留。", "model_invalid_response", 502)
        # 只记录分类和尝试次数，不记录请求头、密钥、用户内容或服务端响应。
        logger.warning(f"计划模型请求失败: {failure.code}（{attempt + 1}/{attempts}）")
        if not failure.retryable or attempt + 1 >= attempts:
            if strict:
                raise failure
            return None


def generate_learning_week(plan: Dict[str, Any], week: int, progress: Dict[str, Any]):
    first, last = (week - 1) * 7 + 1, min(week * 7, plan['duration_days'])
    requirements = {"subject": plan['subject'], "goal": plan['goal'], "outcome": plan.get('outcome'),
                    "first_day": first, "last_day": last, "daily_hours": plan['daily_hours'],
                    "phases": plan.get('stage_plan', []), "progress": progress}
    requirements['daily_core_hours_limit'] = round(plan['daily_hours'] * .8, 4)
    instructions = (
        "你是学习计划设计者。结合阶段方向和真实完成记录，细化本周每日任务，只返回JSON。"
        "优先处理前周未完成的必要任务，然后推进当前阶段；已完成内容不要原样重复。"
        "days必须覆盖first_day～last_day，按day递增。每天一个具体操作、小目标和产出，hours不得超过每日上限。"
        "每天时间是上限，不必填满。任务可包含具体练习、复习与补进度，不能把同一句话重复编号拆分。"
        "hours必须小于等于daily_core_hours_limit，不能等于daily_hours；这是每日主任务时间硬上限。至少20%留给自主练习、复习和补进度。"
        "task每条40～100字，deliverable每条不超过40字。day必须使用给出的实际日期序号。"
        '结构：{"days":[{"day":8,"topic":"主题","task":"具体操作","hours":1,"deliverable":"可检查产出"}]}。'
    )
    for attempt in range(2):
        payload = _request_plan_json(instructions, requirements, strict=True, attempts=2 if attempt == 0 else 1, max_tokens=2500)
        if payload is None:
            raise PlanGenerationError("本周安排暂未生成，请稍后重试；原计划已保留。", "model_unavailable")
        try:
            daily = normalize_week_response(payload, first, last, plan['daily_hours'])
            if any(day['estimated_hours'] > plan['daily_hours'] * .8 + .0001 for day in daily):
                raise ValueError("主任务时长超过daily_core_hours_limit")
            return daily
        except (ValueError, TypeError, KeyError) as exc:
            logger.warning(f"本周计划内容校验失败（{attempt + 1}/2）: {str(exc)}")
            if attempt == 0:
                # 不放宽时间与任务质量要求，给模型一个具体的修正原因，再生成一次。
                instructions += f" 上次输出未通过校验：{str(exc)}。请修正，重新返回完整days列表。"
            else:
                raise PlanGenerationError("模型生成的任务未满足日期或时间要求，请重试；原计划已保留。", "plan_validation_failed", 502)


def normalize_week_response(payload: Any, first_day: int, last_day: int, daily_hours: float):
    """兼容字段格式变化，不修改实际任务内容、时间预算或质量校验要求。"""
    if not isinstance(payload, dict):
        raise ValueError("本周计划不是JSON对象")
    items = payload.get('days', payload.get('daily_plan'))
    if not isinstance(items, list):
        raise ValueError("本周计划缺少days列表")
    normalized = []
    for item in items:
        if not isinstance(item, dict):
            raise ValueError("每日任务格式无效")
        day_value = str(item.get('day', '')).strip()
        match = re.fullmatch(r'(?:第)?\s*(\d+)\s*(?:天)?', day_value)
        if not match:
            raise ValueError("每日序号必须是数字")
        task = item.get('task', item.get('tasks'))
        deliverable = item.get('deliverable', item.get('deliverables'))
        topic = item.get('topic', item.get('topics'))
        def text(value):
            return '；'.join(value) if isinstance(value, list) and all(isinstance(s, str) for s in value) else value
        hours = item.get('hours')
        if hours is None and item.get('minutes') is not None:
            hours = float(item['minutes']) / 60
        if isinstance(hours, str):
            hour_match = re.fullmatch(r'\s*(\d+(?:\.\d+)?)\s*(小时|h|分钟|分)?\s*', hours, re.I)
            if not hour_match:
                raise ValueError("任务时长格式无效")
            hours = float(hour_match.group(1)) / (60 if hour_match.group(2) in {'分钟', '分'} else 1)
        normalized.append({'day':int(match.group(1)), 'topic':text(topic), 'task':text(task),
                           'deliverable':text(deliverable), 'hours':hours})
    numbers = [item['day'] for item in normalized]
    # 模型把“这周第1～7天”写成相对序号时，只在完整连续覆盖的情况下转换。
    if first_day > 1 and numbers == list(range(1, last_day - first_day + 2)):
        for item in normalized:
            item['day'] += first_day - 1
    return validate_daily_entries(normalized, first_day, last_day, daily_hours)
