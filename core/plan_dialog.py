"""计划缺项追问的展示与快捷回复；画像时间只有确认后才进入生成条件。"""
from __future__ import annotations

import re
from typing import Any, Dict


def goal_choices(subject: str) -> list[Dict[str, str]]:
    name = {"java": "Java", "python": "Python", "javascript": "JavaScript", "rag": "RAG"}.get(
        subject.lower(), subject
    )
    if re.search(r"面试|求职", name):
        pairs = [("了解岗位", f"了解{name}所需能力与准备方向"),
                 ("准备案例", f"为{name}准备简历和项目案例"), ("模拟面试", f"完成{name}模拟面试并复盘")]
    elif re.search(r"英语|日语|法语|德语|口语|语言", name):
        pairs = [("掌握基础", f"掌握{name}基础表达"), ("日常交流", f"能用{name}完成日常交流"),
                 ("准备考试", f"准备{name}考试，需补充考试名称与目标分数")]
    elif re.search(r"java|python|rag|前端|后端|开发|编程|智能体|agent", name, re.I):
        pairs = [("掌握基础", f"掌握{name}基础并完成入门练习"),
                 ("做项目", f"能用{name}独立完成一个入门项目"), ("准备面试", f"准备{name}相关岗位面试")]
    else:
        pairs = [("掌握基础", f"掌握{name}基础并完成练习"),
                 ("实践应用", f"能将{name}用于一个实际案例"), ("备考复习", f"复习{name}并完成自测")]
    return [{"label": label, "message": f"目标：{goal}"} for label, goal in pairs]


def build_plan_clarification(params: Dict[str, Any], errors: Dict[str, str]) -> Dict[str, Any]:
    missing = params.get("missing_fields", [])
    subject = params.get("subject") or ""
    display = {"java": "Java", "python": "Python", "rag": "RAG"}.get(subject.lower(), subject)
    period = params.get("requested_duration") or params.get("duration")
    summary = "，".join(str(value) for value in (display, period) if value)
    intro = f"{summary}，已经记下。" if summary else "可以，我们先确定这次想学什么。"
    if params.get("goal"):
        intro += f"目标是{params['goal']}。"
    prompts = []
    if "subject" in missing:
        prompts.append({"field": "subject", "text": "你想学习哪个主题？", "placeholder": "例如：Java", "options": []})
    if "goal" in missing and subject:
        prompts.append({"field": "goal", "text": f"你希望学完{display}后能做到什么？",
                        "placeholder": f"目标：能用{display}完成一个实际案例",
                        "options": goal_choices(subject) + [{"label": "自定义", "prefill": "目标："}]})
    if "duration" in missing:
        prompts.append({"field": "duration", "text": "你打算用多长时间学习？", "placeholder": "周期：两个月",
                        "options": [{"label": "两周", "message": "周期：两周"}, {"label": "一个月", "message": "周期：一个月"},
                                    {"label": "自定义", "prefill": "周期："}]})
    if "daily_hours" in missing:
        suggestion = params.get("suggested_daily_hours")
        if suggestion is not None and not params.get("daily_hours_declined"):
            prompts.append({"field": "daily_hours", "text": f"你之前设置每天可学 {suggestion:g} 小时，本次沿用吗？",
                            "source": "profile", "options": [{"label": "沿用", "message": "沿用之前的每天时间"},
                                                              {"label": "修改", "message": "修改每天可用时间", "prefill_after": "每天"}]})
        else:
            prompts.append({"field": "daily_hours", "text": "这次每天能留出多少学习时间？", "placeholder": "每天1小时",
                            "options": [{"label": "30分钟", "message": "每天30分钟"}, {"label": "1小时", "message": "每天1小时"},
                                        {"label": "自定义", "prefill": "每天"}]})
    known_hours = params.get("daily_hours")
    if known_hours is not None:
        intro += f"每天按 {known_hours:g} 小时安排。"
    if params.get("planned_start_date") and params.get("planned_end_date"):
        intro += f"日期为 {params['planned_start_date']} 至 {params['planned_end_date']}。"
    example_goal = "做项目" if any(x["label"] == "做项目" for x in goal_choices(subject)) else "掌握基础"
    hint = f"也可以直接回复：“{example_goal}，每天1小时”。" if "goal" in missing and "daily_hours" in missing and subject else "可以点选，也可以直接回复。"
    defaults = {"label": "其余用默认值", "message": "其余用默认值"}
    default_parts = []
    if "duration" in missing:
        default_parts.append("周期4周")
    if "daily_hours" in missing:
        default_parts.append("每天2小时")
    if "goal" in missing:
        default_parts.append("目标为基础入门")
    default_note = "未填项默认：" + "、".join(default_parts) + "；已填写的条件保留。" if default_parts else "学习主题需要由你指定。"
    message = "\n\n".join([intro] + list(errors.values()) + [p["text"] for p in prompts] + [hint, default_note])
    return {"status": "success", "type": "plan_clarification", "message": message,
            "missing_fields": list(missing), "plan_params": dict(params),
            "dialog": {"intro": intro, "errors": list(errors.values()), "prompts": prompts,
                       "hint": hint, "defaults": defaults, "default_note": default_note}}
