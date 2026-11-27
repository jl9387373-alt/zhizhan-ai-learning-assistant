import json
import os
import threading
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence

from core.config import config
from core.logger import logger
from rag.metrics import calculate_answer_quality
from rag.rag_chain import get_rag_helper


def _data_dir() -> str:
    return os.path.abspath(config.get("storage.data_dir", "data"))


def _eval_dir() -> str:
    path = os.path.join(_data_dir(), "rag_evals")
    os.makedirs(path, exist_ok=True)
    return path


def _index_path() -> str:
    return os.path.join(_eval_dir(), "index.json")


def _load_index() -> Dict[str, Any]:
    path = _index_path()
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    return data
        except Exception:
            pass
    return {"jobs": []}


def _save_index(data: Dict[str, Any]) -> None:
    with open(_index_path(), "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _job_path(job_id: str) -> str:
    return os.path.join(_eval_dir(), f"{job_id}.json")


def _write_job(job_id: str, job: Dict[str, Any]) -> None:
    with open(_job_path(job_id), "w", encoding="utf-8") as f:
        json.dump(job, f, ensure_ascii=False, indent=2)


def _read_job(job_id: str) -> Optional[Dict[str, Any]]:
    path = _job_path(job_id)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, dict):
                return data
    except Exception:
        return None
    return None


def _tokenize(text: str) -> List[str]:
    import re

    return re.findall(r"[\u4e00-\u9fff]{2,}|[a-zA-Z0-9_]{3,}", str(text or "").lower())


def _extract_expected_keywords(ground_truth: str, limit: int = 10) -> List[str]:
    stop = {
        "我们",
        "你们",
        "他们",
        "它们",
        "这个",
        "那个",
        "以及",
        "一个",
        "一种",
        "可以",
        "用于",
        "包括",
        "主要",
        "通过",
        "进行",
    }
    uniq: List[str] = []
    for token in _tokenize(ground_truth):
        if token in stop:
            continue
        if token not in uniq:
            uniq.append(token)
        if len(uniq) >= limit:
            break
    return uniq


def _normalize_sources(meta: Dict[str, Any]) -> List[str]:
    sources: List[str] = []
    primary = os.path.basename(str(meta.get("primary_source") or meta.get("source_file") or "")).strip()
    if primary:
        sources.append(primary)

    acceptable_raw = meta.get("acceptable_sources")
    if isinstance(acceptable_raw, list):
        for item in acceptable_raw:
            name = os.path.basename(str(item or "")).strip()
            if name and name not in sources:
                sources.append(name)
    return sources


def _match_source_files(snippets: Sequence[Dict[str, Any]], acceptable_sources: Sequence[str]) -> List[str]:
    acceptable = {os.path.basename(str(item or "")).strip() for item in acceptable_sources if str(item).strip()}
    matched: List[str] = []
    for snippet in snippets:
        metadata = snippet.get("metadata") if isinstance(snippet, dict) else {}
        if not isinstance(metadata, dict):
            continue
        source = os.path.basename(str(metadata.get("file_name") or metadata.get("file_path") or "")).strip()
        if source and source in acceptable and source not in matched:
            matched.append(source)
    return matched


def load_jsonl_dataset(path: str) -> List[Dict[str, Any]]:
    cases: List[Dict[str, Any]] = []
    if not os.path.exists(path):
        return cases
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = (line or "").strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if isinstance(obj, dict) and obj.get("question"):
                    cases.append(obj)
            except Exception:
                continue
    return cases


def _aggregate_metrics(cases: Sequence[Dict[str, Any]]) -> Dict[str, float]:
    metric_values: Dict[str, List[float]] = {}
    for case in cases:
        metrics = case.get("metrics") if isinstance(case.get("metrics"), dict) else {}
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                metric_values.setdefault(key, []).append(float(value))
    return {
        key: round(sum(values) / len(values), 4)
        for key, values in metric_values.items()
        if values
    }


_running_jobs: Dict[str, threading.Thread] = {}
_lock = threading.Lock()


def create_eval_job(dataset_path: str, limit: int = 50, subject: str = "") -> Dict[str, Any]:
    job_id = str(uuid.uuid4())
    now = datetime.now().isoformat()
    job = {
        "id": job_id,
        "status": "queued",
        "created_at": now,
        "updated_at": now,
        "dataset_path": dataset_path,
        "limit": limit,
        "subject": subject,
        "progress": {"done": 0, "total": 0},
        "summary": {},
        "cases": [],
        "error": "",
    }
    _write_job(job_id, job)

    idx = _load_index()
    idx.setdefault("jobs", [])
    idx["jobs"].append({"id": job_id, "created_at": now, "dataset_path": dataset_path, "limit": limit})
    idx["jobs"].sort(key=lambda x: x.get("created_at", ""), reverse=True)
    idx["jobs"] = idx["jobs"][:50]
    _save_index(idx)
    return job


def _run_job(job_id: str) -> None:
    job = _read_job(job_id) or {"id": job_id}
    job["status"] = "running"
    job["updated_at"] = datetime.now().isoformat()
    _write_job(job_id, job)

    try:
        dataset_path = str(job.get("dataset_path") or "")
        raw_limit = job.get("limit")
        limit = int(raw_limit) if raw_limit is not None else 50
        subject = str(job.get("subject") or "").strip()

        cases = load_jsonl_dataset(dataset_path)
        if subject:
            cases = [case for case in cases if str((case.get("meta") or {}).get("subject") or "") == subject]
        if limit > 0:
            cases = cases[:limit]

        rag = get_rag_helper()
        total = len(cases)
        job["progress"] = {"done": 0, "total": total}
        job["cases"] = []
        response_times: List[float] = []
        conf_scores: List[float] = []
        _write_job(job_id, job)

        for index, case in enumerate(cases, start=1):
            question = str(case.get("question") or "").strip()
            ground_truth = str(case.get("ground_truth") or "").strip()
            meta = case.get("meta") if isinstance(case.get("meta"), dict) else {}
            case_subject = str(meta.get("subject") or "通用")

            details = rag.ask_with_details(question, subject=case_subject, chat_history="") or {}
            answer = str(details.get("answer") or "")
            mode = str(details.get("mode") or "")
            fallback = bool(details.get("fallback"))
            response_time = float(details.get("response_time") or 0.0)
            snippets = details.get("context_snippets") or []
            if not isinstance(snippets, list):
                snippets = []
            contexts = [str(snip.get("content")) for snip in snippets if isinstance(snip, dict) and snip.get("content")]

            acceptable_sources = _normalize_sources(meta)
            expected_keywords = _extract_expected_keywords(ground_truth, limit=10) if ground_truth else []
            quality = calculate_answer_quality(
                answer=answer,
                context=contexts,
                question=question,
                expected_keywords=expected_keywords,
                expected_aspects=[],
                ground_truth=ground_truth,
                context_snippets=snippets,
                acceptable_sources=acceptable_sources,
            )
            metrics = dict(quality)
            matched_files = _match_source_files(snippets, acceptable_sources)

            item = {
                "question": question,
                "subject": case_subject,
                "meta": meta,
                "ground_truth": ground_truth,
                "answer": answer,
                "mode": mode,
                "fallback": fallback,
                "confidence": details.get("confidence"),
                "response_time": response_time,
                "metrics": metrics,
                "retrieval_hit": bool(quality.get("hit_rate", 0.0) > 0),
                "matched_files": matched_files,
                "context_snippets": snippets[:5],
            }

            job["cases"].append(item)
            job["progress"]["done"] = index
            job["updated_at"] = datetime.now().isoformat()

            if response_time > 0:
                response_times.append(response_time)
            try:
                conf_scores.append(float(details.get("confidence") or 0.0))
            except Exception:
                pass
            _write_job(job_id, job)

        avg_metrics = _aggregate_metrics(job["cases"])
        avg_rt = round(sum(response_times) / len(response_times), 2) if response_times else 0.0
        avg_conf = round(sum(conf_scores) / len(conf_scores), 4) if conf_scores else 0.0

        retrieval_metrics = {
            "context_precision": avg_metrics.get("context_precision", 0.0),
            "context_recall": avg_metrics.get("context_recall", 0.0),
            "hit_rate": avg_metrics.get("hit_rate", 0.0),
            "mrr": avg_metrics.get("mrr", 0.0),
        }
        generation_metrics = {
            "faithfulness": avg_metrics.get("faithfulness", 0.0),
            "answer_relevancy": avg_metrics.get("answer_relevancy", 0.0),
            "answer_correctness": avg_metrics.get("answer_correctness", 0.0),
        }
        job["summary"] = {
            "total_cases": total,
            "avg_response_time": avg_rt,
            "avg_confidence": avg_conf,
            "retrieval_metrics": retrieval_metrics,
            "generation_metrics": generation_metrics,
            "avg_metrics": {**retrieval_metrics, **generation_metrics},
            "hit_rate": retrieval_metrics["hit_rate"],
        }
        job["status"] = "done"
        job["updated_at"] = datetime.now().isoformat()
        _write_job(job_id, job)
    except Exception as exc:
        logger.error(f"RAG 评测任务失败: {exc}", exc_info=True)
        job = _read_job(job_id) or {"id": job_id}
        job["status"] = "error"
        job["error"] = str(exc)
        job["updated_at"] = datetime.now().isoformat()
        _write_job(job_id, job)


def start_eval_job(job_id: str) -> None:
    with _lock:
        if job_id in _running_jobs and _running_jobs[job_id].is_alive():
            return
        thread = threading.Thread(target=_run_job, args=(job_id,), daemon=True)
        _running_jobs[job_id] = thread
        thread.start()


def list_eval_jobs() -> List[Dict[str, Any]]:
    idx = _load_index()
    jobs = idx.get("jobs", [])
    if not isinstance(jobs, list):
        return []
    result: List[Dict[str, Any]] = []
    for item in jobs:
        if not isinstance(item, dict) or not item.get("id"):
            continue
        job = _read_job(item["id"]) or item
        result.append(
            {
                "id": job.get("id"),
                "status": job.get("status"),
                "created_at": job.get("created_at"),
                "updated_at": job.get("updated_at"),
                "dataset_path": job.get("dataset_path"),
                "limit": job.get("limit"),
                "progress": job.get("progress"),
                "summary": job.get("summary"),
                "error": job.get("error"),
            }
        )
    return result


def get_eval_job(job_id: str) -> Optional[Dict[str, Any]]:
    return _read_job(job_id)

