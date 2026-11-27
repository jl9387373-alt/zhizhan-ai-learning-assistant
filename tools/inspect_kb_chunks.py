import os
import sys
from collections import Counter

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_dir not in sys.path:
    sys.path.append(root_dir)

from rag.vector_store import VectorStoreService


def main():
    vs = VectorStoreService()
    docs = vs._collect_documents()

    splitter = vs.imports["RecursiveCharacterTextSplitter"](
        chunk_size=vs.cfg["chunk_size"],
        chunk_overlap=vs.cfg["chunk_overlap"],
        separators=["\n\n", "\n", "。", "！", "？", " ", ""],
    )
    split_docs = splitter.split_documents(docs)

    counter = Counter()
    length_counter = Counter()

    for doc in split_docs:
        meta = getattr(doc, "metadata", {}) or {}
        file_path = meta.get("file_path") or meta.get("file_name") or meta.get("source", "unknown")
        label = os.path.relpath(file_path, root_dir) if os.path.isabs(str(file_path)) else str(file_path)
        counter[label] += 1
        length_counter[label] += len(getattr(doc, "page_content", "") or "")

    print("=" * 60)
    print("知识库分块统计")
    print("=" * 60)
    print(f"分块参数: chunk_size={vs.cfg['chunk_size']}, chunk_overlap={vs.cfg['chunk_overlap']}")
    print(f"收集文档数: {len(docs)}")
    print(f"生成文档块数: {len(split_docs)}")
    print("-" * 60)
    print(f"{'文档':50} {'块数':>6} {'总字符数':>10}")
    print("-" * 60)

    for label, count in sorted(counter.items(), key=lambda x: (-x[1], x[0])):
        print(f"{label[:50]:50} {count:>6} {length_counter[label]:>10}")

    print("=" * 60)


if __name__ == "__main__":
    main()
