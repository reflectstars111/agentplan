"""OpenDataLoader PDF parser wrapper with PyMuPDF fallback.

Primary: OpenDataLoader PDF (pure CPU, structured Markdown+JSON, needs Java 11+)
Fallback: PyMuPDF flat text (zero dependencies beyond Python)
"""

import re
from pathlib import Path
from dataclasses import dataclass, field
from src.models.chunk import DocumentChunk, ChunkType, ChunkLocation, TrustLevel


@dataclass
class PDFParseResult:
    """Result of PDF parsing."""
    chunks: list[DocumentChunk] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    parser_used: str = "unknown"


class PDFParser:
    """PDF parser with automatic OpenDataLoader/PyMuPDF dispatch."""

    def __init__(self, mode: str = "auto"):
        """mode: 'auto' (try OpenDataLoader, fallback PyMuPDF) |
                 'native' (OpenDataLoader only) |
                 'pymupdf' (PyMuPDF only, no Java needed)
        """
        self.mode = mode

    def parse(self, file_path: Path, source_id: str) -> PDFParseResult:
        """Parse a PDF file into structured chunks.

        Returns PDFParseResult with chunks and metadata.
        """
        if self.mode == "pymupdf":
            return self._parse_pymupdf(file_path, source_id)

        if self.mode in ("auto", "native"):
            if _has_opendataloader():
                try:
                    return self._parse_opendataloader(file_path, source_id)
                except Exception:
                    if self.mode == "native":
                        raise
            # Fallback
            if self.mode == "auto":
                return self._parse_pymupdf(file_path, source_id)

        return self._parse_pymupdf(file_path, source_id)

    def _parse_opendataloader(self, file_path: Path, source_id: str) -> PDFParseResult:
        """Parse using OpenDataLoader PDF."""
        import os
        import opendataloader_pdf
        import tempfile
        import json

        # Ensure Java 11+ is on PATH
        env = os.environ.copy()
        java_homes = [
            r"C:/Program Files/Java/jdk-11.0.30/bin",
            r"C:/Program Files/Microsoft/jdk-11.0.31.11-hotspot/bin",
        ]
        for jh in java_homes:
            if os.path.isdir(jh):
                env["PATH"] = jh + ";" + env.get("PATH", "")
                env["JAVA_HOME"] = jh.replace("/bin", "")
                break

        with tempfile.TemporaryDirectory() as tmpdir:
            opendataloader_pdf.convert(
                input_path=[str(file_path)],
                output_dir=tmpdir,
                format="json",
            )
            json_files = list(Path(tmpdir).glob("*.json"))
            if not json_files:
                raise FileNotFoundError(f"No JSON output from OpenDataLoader for {file_path.name}")
            json_path = json_files[0]

            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)

        chunks = self._odl_to_chunks(data, source_id)
        pages = data.get("number of pages", len(data.get("kids", [])))
        return PDFParseResult(
            chunks=chunks,
            metadata={
                "pages": pages,
                "parser": "opendataloader",
                "title": data.get("title", ""),
                "author": data.get("author", ""),
            },
            parser_used="opendataloader",
        )

    def _parse_pymupdf(self, file_path: Path, source_id: str) -> PDFParseResult:
        """Fallback: flat text via PyMuPDF."""
        try:
            import fitz
        except ImportError:
            return PDFParseResult(
                chunks=[],
                metadata={"error": "PyMuPDF not installed", "parser": "none"},
                parser_used="none",
            )

        doc = fitz.open(str(file_path))
        parts = []
        page_count = doc.page_count

        for page_num, page in enumerate(doc):
            text = page.get_text()
            if text.strip():
                parts.append(f"[Page {page_num + 1}]\n{text}")

        doc.close()

        full_text = "\n\n".join(parts)
        chunks = _text_to_chunks(full_text, source_id, "pdf", page_count)

        return PDFParseResult(
            chunks=chunks,
            metadata={"pages": page_count, "parser": "pymupdf"},
            parser_used="pymupdf",
        )

    def _odl_to_chunks(self, data: dict, source_id: str) -> list[DocumentChunk]:
        """Convert OpenDataLoader JSON output to DocumentChunks.

        OpenDataLoader JSON schema:
          { "file name": ..., "number of pages": N,
            "kids": [ { "type": "paragraph"|"heading"|"table"|..., "content": "...",
                        "page number": N, "font": "...", "font size": N, ... } ] }
        """
        chunks = []
        idx = 0
        kids = data.get("kids", [])

        for kid in kids:
            elem_type = kid.get("type", "paragraph")
            text = kid.get("content", "").strip()
            if not text:
                continue

            chunk_type = {
                "heading": ChunkType.HEADING,
                "table": ChunkType.TABLE,
                "figure": ChunkType.FIGURE,
            }.get(elem_type, ChunkType.PARAGRAPH)

            page = kid.get("page number")
            font_size = kid.get("font size")
            section_name = text if elem_type == "heading" else None

            chunks.append(DocumentChunk(
                chunk_id=f"chunk_{source_id}_{idx:04d}",
                source_id=source_id,
                source_type="pdf",
                text=text,
                chunk_type=chunk_type,
                trust_level=TrustLevel.EXTERNAL_UNTRUSTED,
                location=ChunkLocation(page=page, section=section_name),
            ))
            idx += 1

        return chunks

    @staticmethod
    def is_available() -> bool:
        """Check if OpenDataLoader PDF is available."""
        return _has_opendataloader()


def _has_opendataloader() -> bool:
    """Check if opendataloader_pdf can be imported."""
    try:
        import opendataloader_pdf  # noqa: F401
        return True
    except ImportError:
        return False


def _text_to_chunks(text: str, source_id: str, source_type: str, pages: int) -> list[DocumentChunk]:
    """Convert flat text to PARAGRAPH chunks (PyMuPDF fallback path)."""
    if not text.strip():
        return []
    paragraphs = re.split(r'\n\s*\n', text.strip())
    chunks = []
    for idx, para in enumerate(paragraphs):
        para = para.strip()
        if not para:
            continue
        chunks.append(DocumentChunk(
            chunk_id=f"chunk_{source_id}_{idx:04d}",
            source_id=source_id,
            source_type=source_type,
            text=para,
            chunk_type=ChunkType.PARAGRAPH,
            trust_level=TrustLevel.EXTERNAL_UNTRUSTED,
        ))
    return chunks
