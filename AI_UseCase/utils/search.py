from typing import List, Dict, Optional

from config.config import SEARCH_TOP_K


def web_search(query: str, k: Optional[int] = None) -> List[Dict]:
    """
    Perform a lightweight web search and return a list of {title, href, body} dicts.
    Uses DuckDuckGo (no API key).
    """
    top_k = k or SEARCH_TOP_K

    DDGS = None
    try:
        from ddgs import DDGS  # modern package name
    except Exception:
        try:
            from duckduckgo_search import DDGS  # legacy package
        except Exception as e:
            raise RuntimeError(f"Install 'ddgs' package for web search: {e}")

    results: List[Dict] = []
    try:
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=top_k):
                results.append({
                    "title": r.get("title"),
                    "href": r.get("href"),
                    "body": r.get("body"),
                })
    except Exception as e:
        # Return empty on failure; caller can choose to ignore
        return []

    return results
