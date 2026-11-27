import argparse
import json
import sys
from pathlib import Path
from statistics import mean, median


ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


from rag.eval_lab import _job_path, _run_job, create_eval_job, get_eval_job


DEFAULT_DATASET = ROOT_DIR / "data" / "rag_testcases.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run RAG answer quality evaluation and persist a job report.")
    parser.add_argument(
        "--dataset",
        type=str,
        default=str(DEFAULT_DATASET),
        help="Path to the JSONL dataset used for answer evaluation.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=50,
        help="Maximum number of samples to evaluate. Use 0 to evaluate all rows in the dataset.",
    )
    parser.add_argument(
        "--subject",
        type=str,
        default="",
        help="Optional subject filter.",
    )
    return parser.parse_args()


def summarize_cases(cases: list[dict]) -> dict:
    metric_values: dict[str, list[float]] = {}
    response_times: list[float] = []
    confidences: list[float] = []

    for case in cases:
        metrics = case.get("metrics") if isinstance(case.get("metrics"), dict) else {}
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                metric_values.setdefault(key, []).append(float(value))

        response_time = case.get("response_time")
        if isinstance(response_time, (int, float)) and float(response_time) > 0:
            response_times.append(float(response_time))

        confidence = case.get("confidence")
        if isinstance(confidence, (int, float)):
            confidences.append(float(confidence))

    trimmed_times = [value for value in response_times if value <= 60]

    return {
        "completed_cases": len(cases),
        "avg_metrics": {
            key: round(mean(values), 4)
            for key, values in metric_values.items()
            if values
        },
        "avg_response_time_s": round(mean(response_times), 2) if response_times else 0.0,
        "median_response_time_s": round(median(response_times), 2) if response_times else 0.0,
        "avg_response_time_s_le_60": round(mean(trimmed_times), 2) if trimmed_times else 0.0,
        "avg_confidence": round(mean(confidences), 4) if confidences else 0.0,
    }


def main() -> None:
    args = parse_args()
    dataset_path = Path(args.dataset).resolve()
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    limit = int(args.limit)
    if limit < 0:
        raise ValueError("--limit must be greater than or equal to 0")

    job = create_eval_job(
        dataset_path=str(dataset_path),
        limit=limit,
        subject=args.subject,
    )
    job_id = str(job["id"])

    print("=" * 72)
    print("RAG answer quality evaluation started")
    print("=" * 72)
    print(f"job_id: {job_id}")
    print(f"dataset: {dataset_path}")
    print(f"limit: {limit}")
    print(f"subject: {args.subject or '(all)'}")
    print(f"job_file: {_job_path(job_id)}")

    _run_job(job_id)

    result = get_eval_job(job_id)
    if not result:
        raise RuntimeError(f"Failed to read evaluation job: {job_id}")

    print("=" * 72)
    print(f"status: {result.get('status')}")
    if result.get("error"):
        print(f"error: {result['error']}")

    summary = result.get("summary") or {}
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if result.get("status") != "done":
        partial = summarize_cases(result.get("cases") or [])
        print("=" * 72)
        print("partial_summary:")
        print(json.dumps(partial, ensure_ascii=False, indent=2))

    if result.get("status") != "done":
        raise RuntimeError(f"Evaluation did not finish successfully. job_id={job_id}")


if __name__ == "__main__":
    main()
