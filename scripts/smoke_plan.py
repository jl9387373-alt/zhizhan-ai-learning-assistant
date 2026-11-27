import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.coordinator import AgentCoordinator


def main() -> None:
    user_id = "default"
    c = AgentCoordinator()
    r1 = c.process_request("为我制定为期四周的python学习计划", user_id)
    print("create keys:", sorted(list(r1.keys())))
    r2 = c.process_request("查看计划详情: python", user_id)
    print("detail has plan:", bool(r2.get("plan")))
    if r2.get("plan"):
        print("weekly_plan len:", len(r2["plan"].get("weekly_plan", [])))


if __name__ == "__main__":
    main()

