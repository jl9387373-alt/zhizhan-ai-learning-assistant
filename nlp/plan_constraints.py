"""对话计划参数提取；只提取用户明确填写的值，不用默认值覆盖待办参数。"""
from __future__ import annotations

import re
import calendar
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, Tuple

from nlp.intent_detection import _cn_number_to_int, extract_subject

NUMBER = r"(?:\d+(?:\.\d+)?|[一二两三四五六七八九十]+|半)"
PERIOD = re.compile(rf"({NUMBER})\s*(天|周|星期|个月|月)")
DAILY = re.compile(
    rf"(?:每天|每日|一天|日均)\s*(?:改成|改为|调整为|可以|能够|能|可用|可投入|可支配|可)?\s*"
    rf"(?:学习|学|投入|安排|有|用|花)?\s*(?:时间|时长)?\s*[:：=]?\s*"
    rf"({NUMBER})\s*(小时|个小时|分钟|分|h)(?:\s*(?:零|又)?\s*({NUMBER})\s*分钟)?",
    re.IGNORECASE,
)
GOAL = re.compile(
    r"(?:学习目标|目标|希望|为了|想要|我想|想|达到)\s*(?:是|为|要)?\s*[:：=]?\s*([^，,。；;\n]+)"
)


def _number(token: str) -> float:
    if token == "半":
        return 0.5
    if re.fullmatch(r"\d+(?:\.\d+)?", token):
        return float(token)
    return float(_cn_number_to_int(token) or 0)


def plan_today() -> date:
    return datetime.now(timezone(timedelta(hours=8))).date()


def _add_calendar_months(start: date, months: int) -> date:
    year, month = divmod(start.year * 12 + start.month - 1 + months, 12)
    month += 1
    return date(year, month, min(start.day, calendar.monthrange(year, month)[1]))


def extract_plan_constraints(query: str, reference_date: date | None = None) -> Tuple[Dict[str, Any], Dict[str, str]]:
    text = str(query or "").strip()
    params: Dict[str, Any] = {}
    errors: Dict[str, str] = {}
    period = PERIOD.search(text)
    if period:
        value, unit = _number(period.group(1)), period.group(2)
        start = reference_date or plan_today()
        if unit in {"月", "个月"}:
            if not value.is_integer() or not 1 <= value <= 24:
                errors["duration"] = "按月安排请填写完整月份，例如两个月；最长104周，也可以直接填写天数。"
                days = 0.0
            else:
                days = float((_add_calendar_months(start, int(value)) - start).days)
        else:
            days = value * (1 if unit == "天" else 7)
        if not 1 <= days <= 728 or not days.is_integer():
            errors.setdefault("duration", "周期请填写 1 天到 104 周之间的完整天数，例如 10 天或 4 周。")
        else:
            days = int(days)
            params.update(duration=f"{(days + 6) // 7}周", duration_days=days,
                          requested_duration=f"{period.group(1)}{unit}",
                          planned_start_date=start.isoformat(),
                          planned_end_date=(start + timedelta(days=days - 1)).isoformat())
            if unit in {"月", "个月"}:
                params["duration_months"] = int(value)
    daily = DAILY.search(text)
    if daily:
        hours = _number(daily.group(1))
        if daily.group(2) in {"分钟", "分"}:
            hours /= 60
        elif daily.group(3):
            hours += _number(daily.group(3)) / 60
        elif text[daily.end():].startswith("半"):
            hours += 0.5
        if not 0 < hours <= 24:
            errors["daily_hours"] = "每天可用时间请填写大于 0、且不超过 24 小时的时长。"
        else:
            params["daily_hours"] = round(hours, 4)
    if re.search(r"(?:每天|每日|一天)[^，,。；;]*[:：]?\s*-\s*\d", text):
        params.pop("daily_hours", None)
        errors["daily_hours"] = "每天可用时间请填写大于 0、且不超过 24 小时的时长。"
    if re.search(r"-\s*\d+(?:\.\d+)?\s*(?:天|周|星期|个月|月)", text):
        errors["duration"] = "计划周期必须大于 0。"
    for match in GOAL.finditer(text):
        goal = match.group(1).strip()
        goal = PERIOD.sub("", goal)
        goal = DAILY.sub("", goal).strip(" ，,的")
        goal = re.sub(r"^(?:在|用|为期)?\s*内\s*", "", goal)
        goal = re.sub(r"^(?:能够|能)\s*", "", goal)
        if goal and not re.search(r"制定|创建|生成|计划|每天|每日|周期", goal):
            params["goal"] = goal
            break
    if "goal" not in params and re.match(r"^(?:能|能够|掌握|学会|完成|通过|考过|准备|独立)", text):
        params["goal"] = re.split(r"[，,。；;\n]", text)[0].strip()
    return params, errors


def extract_plan_subject(query: str, allow_bare: bool = False) -> str | None:
    """先移除目标/时间描述，避免目标中的 Python、AI 等词覆盖实际学习主题。"""
    parts = re.split(r"[，,。；;\n]", str(query or ""))
    topic_parts = [p for p in parts if not re.match(
        r"\s*(?:学习目标|目标|周期|时长|每天|每日|一天|日均|希望|为了|想要|达到|其余|默认)", p
    )]
    text = PERIOD.sub("", "，".join(topic_parts))
    text = DAILY.sub("", text).strip()
    labeled = re.search(r"(?:学习主题|主题|学习内容)\s*(?:是|为|[:：=])\s*([^，,。；;]+)", text)
    if labeled:
        return labeled.group(1).strip()
    subject = extract_subject(text)
    if subject != "通用" and not re.search(r"计划|制定|创建|生成|默认|[，,。；;]", subject):
        return subject
    if allow_bare and re.fullmatch(r"[\w\u4e00-\u9fff +#.-]{2,24}", text):
        if not re.search(r"计划|制定|目标|周期|小时|分钟|每天|默认|随便|好的", text):
            return text
    return None
