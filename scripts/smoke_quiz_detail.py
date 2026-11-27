import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app import app


def main() -> None:
    client = app.test_client()

    gen = client.post(
        "/api/quiz/generate",
        json={
            "user_id": "default",
            "subject": "Python基础",
            "difficulty": "简单",
            "questions_count": 2,
        },
    ).get_json() or {}
    quiz = gen.get("quiz") or {}
    quiz_id = quiz.get("id")
    assert gen.get("status") == "success" and quiz_id

    answers = []
    for q in quiz.get("questions") or []:
        answers.append({"question_id": q.get("id"), "selected": 0, "user_answer": ""})

    sub = client.post(
        "/api/quiz/submit",
        json={"user_id": "default", "quiz_id": quiz_id, "answers": answers},
    ).get_json() or {}
    assert sub.get("status") == "success"

    detail = client.get(f"/api/quiz/detail?user_id=default&quiz_id={quiz_id}").get_json() or {}
    assert detail.get("status") == "success"
    result = detail.get("result") or {}
    assert result.get("quiz_id") == quiz_id
    assert isinstance(result.get("questions"), list) and len(result["questions"]) == 2
    print("ok", quiz_id)


if __name__ == "__main__":
    main()

