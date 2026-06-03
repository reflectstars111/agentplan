"""FileStore — file ingestion, chunking, and chunk persistence."""

import json
from pathlib import Path
from typing import Optional
from datetime import datetime, timezone
from src.db.connection import Database
from src.models.chunk import DocumentChunk, ChunkType, TrustLevel, ChunkLocation
from src.storage.chunker import chunk_text, ChunkerConfig


class FileStore:
    """Manages file ingestion, chunking, and chunk persistence."""

    def __init__(self, db: Database, file_store_path: str = "data/files"):
        self.db = db
        self.file_store_path = Path(file_store_path)
        self.file_store_path.mkdir(parents=True, exist_ok=True)

    def ingest_file(self, file_path: Path, source_type: str | None = None) -> str:
        """Ingest a file from disk. Returns source_id."""
        if source_type is None:
            source_type = self._guess_type(file_path)

        source_id = f"file:{file_path.name}"

        if source_type == "pdf":
            return self._ingest_pdf(file_path, source_id)
        elif source_type in ("markdown", "md", "text", "txt", "py", "js", "ts"):
            content = file_path.read_text(encoding="utf-8", errors="replace")
            return self.ingest_text(content, source_name=file_path.name, source_type=source_type)
        else:
            # Treat unknown as text
            content = file_path.read_text(encoding="utf-8", errors="replace")
            return self.ingest_text(content, source_name=file_path.name, source_type="text")

    def ingest_text(
        self,
        content: str,
        source_name: str,
        source_type: str = "text",
        trust_level: TrustLevel = TrustLevel.EXTERNAL_UNTRUSTED,
    ) -> str:
        """Ingest text content. Returns source_id."""
        source_id = f"file:{source_name}"

        # Delete existing chunks for this source (re-ingest)
        self.delete_source(source_id)

        chunks = chunk_text(
            content,
            source_id=source_id,
            source_type=source_type,
            trust_level=trust_level,
        )

        for chunk in chunks:
            self._insert_chunk(chunk)

        return source_id

    def get_chunks(self, source_id: str) -> list[DocumentChunk]:
        """Retrieve all chunks for a given source."""
        rows = self.db.execute(
            "SELECT * FROM chunks WHERE source_id = ? ORDER BY chunk_id", (source_id,)
        ).fetchall()
        return [self._row_to_chunk(dict(r)) for r in rows]

    def get_chunk(self, chunk_id: str) -> Optional[DocumentChunk]:
        """Retrieve a single chunk by ID."""
        row = self.db.execute(
            "SELECT * FROM chunks WHERE chunk_id = ?", (chunk_id,)
        ).fetchone()
        if row is None:
            return None
        return self._row_to_chunk(dict(row))

    def list_sources(self) -> list[str]:
        """List all distinct source IDs."""
        rows = self.db.execute(
            "SELECT DISTINCT source_id FROM chunks"
        ).fetchall()
        return [r["source_id"] for r in rows]

    def delete_source(self, source_id: str) -> None:
        """Delete all chunks for a source."""
        self.db.execute("DELETE FROM chunks WHERE source_id = ?", (source_id,))
        self.db.commit()

    def count_chunks(self) -> int:
        row = self.db.execute("SELECT COUNT(*) as cnt FROM chunks").fetchone()
        return row["cnt"] if row else 0

    def _insert_chunk(self, chunk: DocumentChunk) -> None:
        now = datetime.now(timezone.utc).isoformat()
        self.db.execute(
            """INSERT INTO chunks
               (chunk_id, source_id, source_type, text, summary, keywords,
                location_page, location_section, location_line_start, location_line_end,
                chunk_type, embedding_id, trust_level, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                chunk.chunk_id, chunk.source_id, chunk.source_type,
                chunk.text, chunk.summary,
                json.dumps(chunk.keywords),
                chunk.location.page, chunk.location.section,
                chunk.location.line_start, chunk.location.line_end,
                chunk.chunk_type.value, chunk.embedding_id,
                chunk.trust_level.value, now,
            ),
        )
        self.db.commit()

    def _ingest_pdf(self, file_path: Path, source_id: str) -> str:
        """Ingest a PDF file using PyMuPDF."""
        try:
            import fitz  # PyMuPDF
        except ImportError:
            raise ImportError(
                "PyMuPDF (fitz) is required for PDF ingestion. pip install PyMuPDF"
            )

        doc = fitz.open(str(file_path))
        full_text_parts = []

        for page_num, page in enumerate(doc):
            text = page.get_text()
            if text.strip():
                full_text_parts.append(f"[Page {page_num + 1}]\n{text}")

        doc.close()
        full_text = "\n\n".join(full_text_parts)

        return self.ingest_text(
            content=full_text,
            source_name=file_path.name,
            source_type="pdf",
            trust_level=TrustLevel.EXTERNAL_UNTRUSTED,
        )

    def _guess_type(self, file_path: Path) -> str:
        ext = file_path.suffix.lower()
        mapping = {
            ".pdf": "pdf", ".md": "markdown", ".txt": "text",
            ".py": "code", ".js": "code", ".ts": "code",
            ".rs": "code", ".go": "code", ".java": "code",
            ".html": "text", ".css": "code", ".json": "text",
            ".yaml": "text", ".yml": "text", ".toml": "text",
        }
        return mapping.get(ext, "text")

    def _row_to_chunk(self, row: dict) -> DocumentChunk:
        return DocumentChunk(
            chunk_id=row["chunk_id"],
            source_id=row["source_id"],
            source_type=row["source_type"],
            text=row["text"],
            summary=row.get("summary") or "",
            keywords=json.loads(row.get("keywords", "[]")),
            location=ChunkLocation(
                page=row.get("location_page"),
                section=row.get("location_section"),
                line_start=row.get("location_line_start"),
                line_end=row.get("location_line_end"),
            ),
            chunk_type=ChunkType(row.get("chunk_type", "paragraph")),
            embedding_id=row.get("embedding_id"),
            trust_level=TrustLevel(row.get("trust_level", "external_untrusted")),
            created_at=datetime.fromisoformat(row["created_at"]) if row.get("created_at") else datetime.now(timezone.utc),
        )
