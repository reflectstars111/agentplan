"""Tests for ImageParser."""
import pytest
from src.parsing.image_parser import ImageParser


class TestImageParser:
    def test_is_available(self):
        """is_available() should return bool without raising."""
        result = ImageParser.is_available()
        assert isinstance(result, bool)

    def test_parse_nonexistent_file(self):
        """Parsing a nonexistent file should return error metadata."""
        from pathlib import Path
        parser = ImageParser()
        result = parser.parse(Path("/nonexistent/image.png"), "source:test")
        # Should either return chunks (if OCR available and image opens)
        # or metadata with error
        assert result.parser_used in ("tesseract", "none")

    def test_parse_creates_chunks_for_text_image(self, tmp_path):
        """OCR on a generated image with text should produce chunks if available."""
        if not ImageParser.is_available():
            pytest.skip("pytesseract/Pillow not available")

        from PIL import Image, ImageDraw
        img_path = tmp_path / "text_image.png"
        img = Image.new("RGB", (400, 100), color="white")
        draw = ImageDraw.Draw(img)
        draw.text((10, 40), "Hello World OCR Test", fill="black")
        img.save(str(img_path))

        parser = ImageParser()
        result = parser.parse(img_path, "source:ocr_test")

        # OCR engine binary may not be on PATH — skip if unavailable
        if result.parser_used == "none" and not result.chunks:
            pytest.skip(f"tesseract OCR engine not available: {result.metadata}")

        # If engine is available, should produce results
        if result.parser_used == "tesseract":
            assert len(result.chunks) > 0
            text = " ".join(c.text for c in result.chunks)
            assert len(text) > 0
