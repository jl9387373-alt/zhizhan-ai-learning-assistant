import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app import app

print("client init...", flush=True)
client = app.test_client()

print("posting generate...", flush=True)
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
print("generate:", resp.status_code, data.get("status"))
quiz = data.get("quiz") or {}
print("quiz_id:", quiz.get("id"))
print("questions:", len(quiz.get("questions") or []))

answers = []
for q in quiz.get("questions") or []:
    answers.append({"question_id": q.get("id"), "selected": 0, "user_answer": ""})

resp2 = client.post(
    "/api/quiz/submit",
    json={"user_id": "default", "quiz_id": quiz.get("id"), "answers": answers},
)
data2 = resp2.get_json(silent=True) or {}
print("submit:", resp2.status_code, data2.get("status"))
result = data2.get("result") or {}
print("result_questions:", len(result.get("questions") or []))
