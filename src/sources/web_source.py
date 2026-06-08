"""WebSource — HTTP URL fetch and index.

Maps to agent_os_initial_plan.md §10.1 (Web input source).
"""

from urllib.request import urlopen, Request
from urllib.error import URLError
from src.models.chunk import TrustLevel


class WebSource:
    """Fetch web page content and index it."""

    def fetch_and_index(self, url: str, file_store, source_name: str = "") -> dict:
        """Fetch a URL and index its text content."""
        try:
            req = Request(url, headers={"User-Agent": "Agent-OS/0.2"})
            with urlopen(req, timeout=15) as resp:
                content = resp.read().decode("utf-8", errors="replace")
        except URLError as e:
            return {"error": f"Fetch failed: {e}"}
        except Exception as e:
            return {"error": str(e)}

        # Strip HTML tags for basic text extraction
        import re
        text = re.sub(r"<[^>]+>", " ", content)
        text = re.sub(r"\s+", " ", text).strip()

        name = source_name or url.split("/")[-1] or "web_page"
        source_id = file_store.ingest_text(
            content=text,
            source_name=name,
            source_type="web",
            trust_level=TrustLevel.EXTERNAL_UNTRUSTED,
        )

        return {"source_id": source_id, "text_length": len(text)}
