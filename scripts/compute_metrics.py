"""从本地埋点事件计算核心指标。

用法：
    python scripts/compute_metrics.py
    python scripts/compute_metrics.py path/to/events.jsonl
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from storage.analytics import AnalyticsStore, DEFAULT_EVENTS_PATH

LEARNING_EVENTS = {
    "plan_created",
    "task_completed",
    "qa_answered",
    "review_completed",
}


def _props(event: Dict[str, Any]) -> Dict[str, Any]:
    props = event.get("properties")
    return props if isinstance(props, dict) else {}


def _rate(numerator: int, denominator: int):
    if not denominator:
        return None
    return round(numerator / denominator, 4)


def _metric(numerator: int, denominator: int) -> Dict[str, Any]:
    return {
        "value": _rate(numerator, denominator),
        "numerator": numerator,
        "denominator": denominator,
    }


def compute_metrics(events: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    events = list(events)
    event_counts = Counter(event.get("event", "unknown") for event in events)

    plan_totals: Dict[tuple, int] = {}
    for event in events:
        if event.get("event") != "plan_created":
            continue
        props = _props(event)
        key = (event.get("user_id", "default"), str(props.get("plan_key") or event.get("event_id")))
        plan_totals[key] = int(props.get("task_total") or 0)

    completed_tasks = set()
    for event in events:
        if event.get("event") != "task_completed":
            continue
        props = _props(event)
        task_key = props.get("task_key") or "|".join(
            str(props.get(k, "")) for k in ("plan_key", "week", "task_index")
        )
        completed_tasks.add((event.get("user_id", "default"), str(task_key)))

    qa_asked_ids = set()
    qa_answered_ids = set()
    for event in events:
        name = event.get("event")
        if name not in ("qa_asked", "qa_answered"):
            continue
        props = _props(event)
        qa_id = str(props.get("qa_id") or event.get("event_id"))
        if name == "qa_asked":
            qa_asked_ids.add(qa_id)
        elif event.get("status") != "error":
            qa_answered_ids.add(qa_id)

    helpful_feedback = set()
    unhelpful_feedback = set()
    for event in events:
        name = event.get("event")
        if name not in ("qa_helpful", "qa_unhelpful"):
            continue
        props = _props(event)
        feedback_key = str(props.get("feedback_key") or event.get("event_id"))
        if name == "qa_helpful":
            helpful_feedback.add(feedback_key)
        else:
            unhelpful_feedback.add(feedback_key)

    review_due_ids = set()
    review_completed_ids = set()
    for event in events:
        name = event.get("event")
        props = _props(event)
        if name == "review_due":
            for review_id in props.get("review_ids") or []:
                review_due_ids.add(str(review_id))
        elif name == "review_completed" and props.get("review_id"):
            review_completed_ids.add(str(props["review_id"]))

    app_open_users = {
        event.get("user_id", "default") for event in events if event.get("event") == "app_opened"
    }
    plan_created_users = {
        event.get("user_id", "default") for event in events if event.get("event") == "plan_created"
    }

    active_by_week: Dict[str, set] = defaultdict(set)
    for event in events:
        if event.get("event") not in LEARNING_EVENTS:
            continue
        week = event.get("week")
        if week:
            active_by_week[str(week)].add(event.get("user_id", "default"))

    weeks = sorted(active_by_week)
    retention_pair = None
    retention = {"value": None, "numerator": 0, "denominator": 0, "weeks": []}
    if len(weeks) >= 2:
        previous_week, current_week = weeks[-2], weeks[-1]
        previous_users = active_by_week[previous_week]
        retained_users = previous_users & active_by_week[current_week]
        retention_pair = [previous_week, current_week]
        retention = {
            "value": _rate(len(retained_users), len(previous_users)),
            "numerator": len(retained_users),
            "denominator": len(previous_users),
            "weeks": retention_pair,
        }

    return {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "event_total": len(events),
        "unique_users": len({event.get("user_id", "default") for event in events}),
        "event_counts": dict(sorted(event_counts.items())),
        "metrics": {
            "plan_completion_rate": _metric(len(completed_tasks), sum(plan_totals.values())),
            "activation_rate": _metric(len(plan_created_users), len(app_open_users)),
            "qa_answer_rate": _metric(len(qa_answered_ids), len(qa_asked_ids)),
            "qa_resolution_rate": _metric(len(helpful_feedback), len(qa_asked_ids)),
            "review_completion_rate": _metric(len(review_completed_ids), len(review_due_ids)),
            "weekly_retention": retention,
        },
    }


def main() -> None:
    events_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_EVENTS_PATH
    store = AnalyticsStore(events_path)
    result = compute_metrics(store.read_events())
    result["source"] = str(store.path)

    output_path = PROJECT_ROOT / "data" / "analytics" / "metrics.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"\n指标结果已写入: {output_path}")


if __name__ == "__main__":
    main()
