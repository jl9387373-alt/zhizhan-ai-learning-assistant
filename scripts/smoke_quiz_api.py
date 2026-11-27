import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

print("smoke_quiz_api start", flush=True)

from app import app

print("smoke_quiz_api imported app", flush=True)


def main() -> None:
    try:
        client = app.test_client()

        resp = client.post(
            "/api/quiz/generate",
            json={
                "user_id": "default",
                "subject": "Python基础",
                "difficulty": "简单",
                "questions_count": 3,
            },
        )
        data = resp.get_json(silent=True) or {}
        print("generate:", resp.status_code, data.get("status"), flush=True)
        quiz = data.get("quiz") or {}
        print("quiz_id:", quiz.get("id"), flush=True)
        print("questions:", len(quiz.get("questions") or []), flush=True)

        answers = []
        for q in quiz.get("questions") or []:
            answers.append({"question_id": q.get("id"), "selected": 0, "user_answer": ""})

        resp2 = client.post(
            "/api/quiz/submit",
            json={"user_id": "default", "quiz_id": quiz.get("id"), "answers": answers},
        )
        data2 = resp2.get_json(silent=True) or {}
        print("submit:", resp2.status_code, data2.get("status"), flush=True)
        result = data2.get("result") or {}
        print("result_questions:", len(result.get("questions") or []), flush=True)
    except Exception as e:
        print("smoke failed:", repr(e), flush=True)


if __name__ == "__main__":
    main()
