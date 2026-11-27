import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agents.qa_agent import QAAgent
from core.message import Message, MessageType


def main() -> None:
    agent = QAAgent()
    msg = Message(
        sender="smoke",
        receiver="qa",
        msg_type=MessageType.REQUEST,
        content={
            "action": "generate",
            "question": "什么是进程",
            "subject": "通用",
            "user_id": "default",
        },
    )
    resp = agent.process(msg)
    print(resp.content.get("answer", ""))


if __name__ == "__main__":
    main()
