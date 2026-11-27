import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    _HAS_SPLITTER = True
except Exception:
    _HAS_SPLITTER = False

try:
    from rag.metrics import _split_sentences

    _HAS_METRICS = True
except Exception:
    _HAS_METRICS = False


class TestTextSplitSentences(unittest.TestCase):
    """rag/metrics._split_sentences 是纯中文分句函数，零依赖。"""

    @unittest.skipUnless(_HAS_METRICS, "rag.metrics._split_sentences unavailable")
    def test_basic(self):
        self.assertEqual(
            _split_sentences("你好。世界！今天天气怎么样？"),
            ["你好", "世界", "今天天气怎么样"],
        )

    @unittest.skipUnless(_HAS_METRICS, "rag.metrics._split_sentences unavailable")
    def test_empty(self):
        self.assertEqual(_split_sentences(""), [])
        self.assertEqual(_split_sentences(None), [])


@unittest.skipUnless(_HAS_SPLITTER, "langchain_text_splitters not installed")
class TestRAGChunking(unittest.TestCase):
    """RAG 预处理：长文本按 chunk_size / overlap 切分（Hybrid Search 前置步骤）。"""

    def test_split_long_text(self):
        splitter = RecursiveCharacterTextSplitter(chunk_size=80, chunk_overlap=20)
        text = "。".join(["机器学习是人工智能的一个重要分支"] * 50)
        chunks = splitter.split_text(text)
        self.assertGreater(len(chunks), 1)
        for chunk in chunks:
            self.assertLessEqual(len(chunk), 80 + 20)


if __name__ == "__main__":
    unittest.main()
