import os
import sys
import unittest

# 确保项目根目录在 sys.path，便于导入顶层包（nlp / storage / rag）
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nlp.intent_detection import (
    _detect_intent_rules,
    extract_subject,
    extract_duration,
    SUBJECT_KEYWORDS,
)


class TestExtractSubject(unittest.TestCase):
    """主题提取是纯关键词匹配，确定性可测。"""

    def test_known_subject(self):
        self.assertEqual(extract_subject("我想学 python 的装饰器"), "python")
        self.assertEqual(extract_subject("复习操作系统进程调度"), "操作系统")
        self.assertEqual(extract_subject("什么是机器学习"), "机器学习")

    def test_default_subject(self):
        self.assertEqual(extract_subject("随便聊聊今天天气"), "通用")
        self.assertEqual(extract_subject(""), "通用")


class TestDetectIntentRules(unittest.TestCase):
    """直接测规则兜底函数，避免依赖 LLM / API key，结果确定。

    detect_intent() 的线上行为是「LLM 优先 + 规则兜底」，
    这里锁定兜底规则，保证无 key 环境下意图分类依然可用。
    """

    def test_plan(self):
        self.assertEqual(_detect_intent_rules("帮我制定一个学习计划"), "plan")

    def test_qa(self):
        self.assertEqual(_detect_intent_rules("什么是装饰器"), "qa")

    def test_resource(self):
        self.assertEqual(_detect_intent_rules("推荐一些数据结构的学习资料"), "resource")

    def test_progress(self):
        self.assertEqual(_detect_intent_rules("查看我的学习进度"), "progress")

    def test_quiz(self):
        self.assertEqual(_detect_intent_rules("出几道题考我"), "quiz")

    def test_review(self):
        self.assertEqual(_detect_intent_rules("帮我安排复习"), "review")

    def test_empty(self):
        self.assertEqual(_detect_intent_rules(""), "qa")


class TestExtractDuration(unittest.TestCase):
    def test_weeks(self):
        self.assertEqual(extract_duration("用6周学完"), "6周")

    def test_days(self):
        self.assertEqual(extract_duration("30天复习"), "5周")

    def test_months(self):
        self.assertEqual(extract_duration("3个月掌握"), "12周")

    def test_default(self):
        self.assertEqual(extract_duration("随便看看"), "4周")


class TestSubjectKeywords(unittest.TestCase):
    def test_coverage(self):
        self.assertIn("python", SUBJECT_KEYWORDS)
        self.assertIn("操作系统", SUBJECT_KEYWORDS)
        self.assertTrue(all(isinstance(v, list) for v in SUBJECT_KEYWORDS.values()))


if __name__ == "__main__":
    unittest.main()
