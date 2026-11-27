# -*- coding: utf-8 -*-
"""校验一次架构对比是否真的调用了模型。

用法：
    python scripts/check_arch_run.py                # 自动找最新的 architecture_comparison_*.json
    python scripts/check_arch_run.py <路径>

判定依据：
    真模型跑多步任务，单条耗时应 > 1000ms；
    若 avg_elapsed_ms < 500ms，基本可以判定仍在走兜底/stub。
"""
import glob
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVAL_DIR = os.path.join(ROOT, "data", "eval_results")


def pick_latest():
    files = sorted(glob.glob(os.path.join(EVAL_DIR, "architecture_comparison_eval_*.json")))
    return files[-1] if files else None


def main() -> int:
    path = sys.argv[1] if len(sys.argv) > 1 else pick_latest()
    if not path or not os.path.exists(path):
        print("找不到结果文件。先跑：python scripts/run_architecture_comparison_eval.py --enable-rag ...")
        return 1

    with open(path, encoding="utf-8-sig") as f:
        d = json.load(f)

    print("=" * 70)
    print("文件:", os.path.basename(path))
    print("生成时间:", d.get("generated_at"))

    verdicts = []
    for label in ("single_agent", "multi_agent"):
        c = (d.get(label) or {}).get("complex") or {}
        ms = c.get("avg_elapsed_ms", 0) or 0
        print(f"\n[{label}]")
        print(f"  完成 {c.get('completed_cases')}/{c.get('total_cases')} = {c.get('completion_rate')}")
        print(f"  子任务命中 {c.get('matched_subtasks')}/{c.get('expected_subtasks')}")
        print(f"  平均耗时 {ms} ms")
        verdicts.append(ms)

    # 单任务部分的首条答案，用来判断是不是模板
    for label in ("single_agent", "multi_agent"):
        s = (d.get(label) or {}).get("single") or {}
        details = s.get("details") or []
        if details:
            preview = str((details[0].get("response") or {}).get("answer_preview") or "")
            print(f"\n[{label}.single 首条答案前 80 字]\n  {preview[:80]}")
            break

    print("\n" + "-" * 70)
    avg = sum(verdicts) / len(verdicts) if verdicts else 0
    if avg >= 1000:
        print(f"判定：真模型（平均 {avg:.0f} ms）—— 这组数字可以用")
    elif avg >= 500:
        print(f"判定：存疑（平均 {avg:.0f} ms）—— 可能部分走了缓存或兜底，看答案再定")
    else:
        print(f"判定：仍走兜底（平均 {avg:.0f} ms）—— 不要引用这组数字")
    print("-" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
