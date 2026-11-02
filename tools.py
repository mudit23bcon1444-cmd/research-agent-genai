from typing import Optional
import requests
from bs4 import BeautifulSoup
from ddgs import DDGS
from ddgs.exceptions import RatelimitException
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential


def _is_ratelimit_error(exception: BaseException) -> bool:
    """Check if the exception is a DuckDuckGo rate limit error."""
    if isinstance(exception, RatelimitException):
        return True
    msg = str(exception).lower()
    return "ratelimit" in msg or "202" in msg


@retry(
    retry=retry_if_exception(_is_ratelimit_error),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=2, min=2, max=6),
    reraise=True,
)
def _perform_ddgs_search(query: str, max_results: int = 5) -> list[dict]:
    """Helper function to execute DDGS search with retries for rate limits."""
    search = DDGS()
    return list(search.text(query=query, max_results=max_results))


class WebSearchInput(BaseModel):
    query: Optional[str] = Field(default="", description="The search query text to search on DuckDuckGo")


@tool("web_search", args_schema=WebSearchInput)
def web_search(query: Optional[str] = None) -> str:
    """Searches the web for recent, reliable info and returns titles, URLs and snippets."""
    if not query or not query.strip():
        return "Search query cannot be empty."

    try:
        results = _perform_ddgs_search(query.strip(), max_results=5)

        if not results:
            return f"No search results found for query: '{query.strip()}'."

        formatted_blocks = []
        for r in results:
            title = r.get("title", "No Title")
            url = r.get("href", "No URL")
            snippet = r.get("body", "")[:300]
            formatted_blocks.append(
                f"Title: {title}\n"
                f"URL: {url}\n"
                f"Snippet: {snippet}"
            )

        return "\n\n----\n\n".join(formatted_blocks)

    except Exception as e:
        return f"Web search failed for '{query}': {str(e)}"


class ScrapeUrlInput(BaseModel):
    url: Optional[str] = Field(default=None, description="The full HTTP or HTTPS URL to fetch and scrape (e.g. https://example.com)")
    cursor: Optional[int] = Field(default=0, description="Optional cursor offset for reading")
    loc: Optional[int] = Field(default=0, description="Optional character location offset")


@tool("scrape_url", args_schema=ScrapeUrlInput)
def scrape_url(url: Optional[str] = None, cursor: Optional[int] = 0, loc: Optional[int] = 0) -> str:
    """Scrapes the given URL and returns up to 3000 characters of clean text content."""
    if not url or not url.strip():
        return "URL was not provided. Please provide a valid URL or proceed to summarize the retrieved findings."

    clean_url = url.strip()
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        )
    }

    try:
        resp = requests.get(clean_url, timeout=8, headers=headers)
        resp.raise_for_status()

        try:
            soup = BeautifulSoup(resp.text, "lxml")
        except Exception:
            soup = BeautifulSoup(resp.text, "html.parser")

        # Decompose unwanted tags
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()

        text = soup.get_text(separator=" ", strip=True)
        if not text:
            return f"Failed to scrape URL {clean_url}: Page returned no readable text."

        return text[:3000]

    except Exception as e:
        return f"Failed to scrape URL {clean_url}: {str(e)}"
