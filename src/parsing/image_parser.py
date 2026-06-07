"""ImageParser — OCR text extraction from images.

Uses pytesseract + Pillow for OCR. Maps to agent_os_initial_plan.md §10.1 (Image input).
"""

from pathlib import Path
from dataclasses import dataclass, field
from src.models.chunk import DocumentChunk, ChunkType, TrustLevel


@dataclass
class ImageParseResult:
    """Result of image OCR parsing."""
    chunks: list[DocumentChunk] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    parser_used: str = "none"


class ImageParser:
    """OCR text extraction from image files using pytesseract + Pillow."""

    SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tiff", ".webp"}

    def parse(self, file_path: Path, source_id: str) -> ImageParseResult:
        """Extract text from an image file via OCR.

        Args:
            file_path: Path to the image file.
            source_id: Source identifier for chunk metadata.

        Returns:
            ImageParseResult with text chunks and metadata.
        """
        if not self.is_available():
            return ImageParseResult(
                chunks=[],
                metadata={"error": "pytesseract or Pillow not installed"},
                parser_used="none",
            )

        try:
            from PIL import Image
            import pytesseract

            img = Image.open(str(file_path))
            width, height = img.size

            # OCR with language auto-detect
            text = pytesseract.image_to_string(img, lang="eng+chi_sim")
            if not text or not text.strip():
                return ImageParseResult(
                    chunks=[],
                    metadata={"width": width, "height": height, "parser": "tesseract",
                              "text_length": 0},
                    parser_used="tesseract",
                )

            # Split into paragraph-level chunks
            paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
            chunks = []
            for idx, para in enumerate(paragraphs):
                chunks.append(DocumentChunk(
                    chunk_id=f"chunk_{source_id}_{idx:04d}",
                    source_id=source_id,
                    source_type="image",
                    text=para,
                    chunk_type=ChunkType.PARAGRAPH,
                    trust_level=TrustLevel.USER_PROVIDED_DATA,
                ))

            return ImageParseResult(
                chunks=chunks,
                metadata={
                    "width": width, "height": height,
                    "parser": "tesseract",
                    "paragraphs": len(chunks),
                    "text_length": len(text),
                },
                parser_used="tesseract",
            )

        except Exception as e:
            return ImageParseResult(
                chunks=[],
                metadata={"error": str(e)},
                parser_used="none",
            )

    @staticmethod
    def is_available() -> bool:
        """Check if pytesseract and Pillow are installed."""
        try:
            import pytesseract  # noqa: F401
            from PIL import Image  # noqa: F401
            return True
        except ImportError:
            return False
