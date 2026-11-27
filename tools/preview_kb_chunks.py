import json
import os
import sys
from collections import defaultdict

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

    grouped = defaultdict(list)

    for i, doc in enumerate(split_docs, 1):
        meta = getattr(doc, "metadata", {}) or {}
        file_path = meta.get("file_path") or meta.get("file_name") or meta.get("source", "unknown")
        label = os.path.relpath(file_path, root_dir) if os.path.isabs(str(file_path)) else str(file_path)
        grouped[label].append(
            {
                "chunk_id": i,
                "length": len(getattr(doc, "page_content", "") or ""),
                "content": getattr(doc, "page_content", "") or "",
            }
        )

    output_dir = os.path.join(root_dir, "data")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "kb_chunk_preview.json")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(grouped, f, ensure_ascii=False, indent=2)

    print("=" * 60)
    print("知识库分块预览已生成")
    print("=" * 60)
    print(f"输出文件: {output_path}")
    print(f"分块参数: chunk_size={vs.cfg['chunk_size']}, chunk_overlap={vs.cfg['chunk_overlap']}")
    print(f"总文档数: {len(docs)}")
    print(f"总块数: {len(split_docs)}")
    print("-" * 60)

    for label, chunks in sorted(grouped.items(), key=lambda x: (-len(x[1]), x[0])):
        print(f"{label} -> {len(chunks)} 块")
        for chunk in chunks[:2]:
            preview = chunk["content"].replace("\n", " ")[:80]
            print(f"  - chunk_id={chunk['chunk_id']}, length={chunk['length']}, preview={preview}")
        if len(chunks) > 2:
            print("  - ...")

    print("=" * 60)


if __name__ == "__main__":
    main()
