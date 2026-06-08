"""DbSource — SQLite/PostgreSQL database query source.

Maps to agent_os_initial_plan.md §10.1 (Database input).
"""

import sqlite3
import re
from src.models.chunk import TrustLevel


class DbSource:
    """Query a database and index the results."""

    def query_and_index(
        self,
        db_path: str,
        query: str,
        file_store,
        db_type: str = "sqlite",
        source_name: str = "",
    ) -> dict:
        """Execute a SQL SELECT query and index the results.

        Args:
            db_path: SQLite file path or PostgreSQL connection string.
            query: SQL SELECT query to execute.
            file_store: FileStore instance for ingestion.
            db_type: "sqlite" (default) or "postgresql".
            source_name: Name for the source identifier.

        Returns:
            dict with source_id, row_count, columns, or error.
        """
        # Security: only allow SELECT statements
        clean = query.strip()
        if not re.match(r"^\s*SELECT\b", clean, re.IGNORECASE):
            return {"error": "Only SELECT queries are allowed"}

        try:
            if db_type == "sqlite":
                rows, columns = self._query_sqlite(db_path, query)
            elif db_type == "postgresql":
                rows, columns = self._query_postgres(db_path, query)
            else:
                return {"error": f"Unsupported db_type: {db_type}"}
        except Exception as e:
            return {"error": str(e)}

        if not rows:
            return {"source_id": "", "row_count": 0, "columns": columns}

        # Convert rows to text: "col1: val1 | col2: val2"
        text_lines = []
        for row in rows:
            parts = []
            for col, val in zip(columns, row):
                parts.append(f"{col}: {val}")
            text_lines.append(" | ".join(parts))

        text = "\n".join(text_lines)
        name = source_name or f"db:{db_path}"
        source_id = file_store.ingest_text(
            content=text,
            source_name=name,
            source_type="database",
            trust_level=TrustLevel.TOOL_OBSERVATION,
        )
        return {
            "source_id": source_id,
            "row_count": len(rows),
            "columns": columns,
        }

    def _query_sqlite(self, db_path: str, query: str) -> tuple[list, list]:
        """Execute query on SQLite database."""
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        try:
            cursor = conn.execute(query)
            columns = [d[0] for d in cursor.description] if cursor.description else []
            rows = [tuple(r) for r in cursor.fetchall()]
            return rows, columns
        finally:
            conn.close()

    def _query_postgres(self, conn_str: str, query: str) -> tuple[list, list]:
        """Execute query on PostgreSQL database."""
        try:
            import psycopg2
        except ImportError:
            raise ImportError(
                "psycopg2 not installed. pip install psycopg2-binary"
            )
        conn = psycopg2.connect(conn_str)
        try:
            cursor = conn.cursor()
            cursor.execute(query)
            columns = [d[0] for d in cursor.description] if cursor.description else []
            rows = cursor.fetchall()
            return rows, columns
        finally:
            conn.close()
