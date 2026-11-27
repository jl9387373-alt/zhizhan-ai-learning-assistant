try:
    from ddgs import DDGS
except ImportError:
    try:
        from duckduckgo_search import DDGS
    except ImportError:
        DDGS = None
from core.logger import logger
from typing import List, Dict

def web_search(query: str, max_results: int = 5) -> List[Dict[str, str]]:
    """
    使用 DuckDuckGo 进行网页搜索
    :param query: 搜索关键词
    :param max_results: 最大结果数量
    :return: 搜索结果列表，包含 title, href, body
    """
    logger.info(f"正在进行网页搜索: {query}")
    results = []
    if DDGS is None:
        logger.warning("DuckDuckGo search dependency is unavailable; returning no web results")
        return []
    try:
        with DDGS() as ddgs:
            seen_urls = set()
            for region in ("cn-zh", "wt-wt"):
                ddgs_generator = ddgs.text(query, region=region, safesearch='off', timelimit='y')
                for r in ddgs_generator:
                    url = r.get("href", "")
                    if not url or url in seen_urls:
                        continue
                    seen_urls.add(url)
                    results.append({
                        "title": r.get("title", ""),
                        "url": url,
                        "content": r.get("body", "")
                    })
                    if len(results) >= max_results:
                        return results
        return results
    except Exception as e:
        logger.error(f"网页搜索失败: {str(e)}")
        return [{"error": f"搜索失败: {str(e)}"}]
