import os
import sys
import time

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_dir not in sys.path:
    sys.path.append(root_dir)

from rag.rag_chain import get_rag_helper
from core.logger import logger


def rebuild_kb():
    print("=" * 50)
    print("开始重建知识库 (向量库)...")
    print("=" * 50)

    start_time = time.time()

    try:
        rag = get_rag_helper()

        print("正在扫描 data 目录并生成向量，这可能需要一些时间，请稍候...")
        success = rag.rebuild()

        duration = time.time() - start_time

        if success:
            print("\n" + "=" * 50)
            print("成功：知识库已完成重建！")
            print(f"总耗时: {duration:.2f} 秒")
            print("=" * 50)
            print("现在启动应用后，提问将可以立即获得回答。")
        else:
            print("\n" + "!" * 50)
            print("错误：知识库重建失败。")
            print("请检查网络连接或 API Key 是否正确。")
            print("!" * 50)
    except Exception as e:
        print(f"\n发生异常: {e}")
        logger.error(f"重建脚本执行失败: {e}", exc_info=True)


if __name__ == "__main__":
    rebuild_kb()
