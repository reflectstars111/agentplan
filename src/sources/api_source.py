"""ApiSource — HTTP JSON API data source.

Maps to agent_os_initial_plan.md §10.1 (API input).
"""

import json
from urllib.request import Request, urlopen
from urllib.error import URLError
from src.models.chunk import TrustLevel


class ApiSource:
    """Fetch data from external JSON APIs and index it."""

    def fetch_and_index(
        self,
        url: str,
        file_store,
        method: str = "GET",
        headers: dict | None = None,
        body: str = "",
        json_path: str = "",
        source_name: str = "",
    ) -> dict:
        """Fetch JSON from an API endpoint and index the result.

        Args:
            url: API endpoint URL.
            file_store: FileStore instance for ingestion.
            method: HTTP method (GET, POST, etc.).
            headers: Optional dict of HTTP headers.
            body: Request body for POST/PUT.
            json_path: Dot-separated path to extract nested JSON
                       (e.g. "data.items" → response["data"]["items"]).
            source_name: Name for the source identifier.

        Returns:
            dict with source_id, item_count, or error.
        """
        try:
            data = body.encode("utf-8") if body else None
            req = Request(
                url,
                data=data,
                headers=headers or {},
                method=method,
            )
            if "User-Agent" not in (headers or {}):
                req.add_header("User-Agent", "Agent-OS/0.3")
            with urlopen(req, timeout=30) as resp:
                content = resp.read().decode("utf-8", errors="replace")
        except URLError as e:
            return {"error": f"API fetch failed: {e}"}
        except Exception as e:
            return {"error": str(e)}

        try:
            parsed = json.loads(content)
        except json.JSONDecodeError as e:
            return {"error": f"JSON parse error: {e}"}

        # Extract nested path
        if json_path:
            for key in json_path.split("."):
                if isinstance(parsed, list):
                    try:
                        idx = int(key)
                        parsed = parsed[idx]
                    except (ValueError, IndexError):
                        return {"error": f"Invalid json_path index: {key}"}
                elif isinstance(parsed, dict):
                    parsed = parsed.get(key)
                    if parsed is None:
                        return {"error": f"Key '{key}' not found in JSON"}
                else:
                    return {"error": f"Cannot traverse path at key: {key}"}

        # Handle both single objects and arrays
        if isinstance(parsed, list):
            items = parsed
            text = "\n\n".join(json.dumps(item, ensure_ascii=False) for item in items)
            item_count = len(items)
        else:
            text = json.dumps(parsed, indent=2, ensure_ascii=False)
            item_count = 1

        name = source_name or url.rstrip("/").split("/")[-1] or "api_data"
        source_id = file_store.ingest_text(
            content=text,
            source_name=name,
            source_type="api",
            trust_level=TrustLevel.EXTERNAL_UNTRUSTED,
        )
        return {"source_id": source_id, "item_count": item_count}
