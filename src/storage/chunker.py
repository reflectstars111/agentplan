"""Text chunker with configurable size and overlap. Produces DocumentChunk objects."""

import re
from dataclasses import dataclass
from src.models.chunk import DocumentChunk, ChunkType, ChunkLocation, TrustLevel


@dataclass
class ChunkerConfig:
    """Configuration for the chunker."""
    chunk_size: int = 500       # characters per chunk
    chunk_overlap: int = 50     # character overlap between adjacent chunks


DEFAULT_CONFIG = ChunkerConfig()


def chunk_text(
    text: str,
    source_id: str,
    source_type: str = "text",
    config: ChunkerConfig | None = None,
    trust_level: TrustLevel = TrustLevel.EXTERNAL_UNTRUSTED,
) -> list[DocumentChunk]:
    """Split plain text into overlapping DocumentChunks.

    Uses sentence-boundary-aware splitting: splits on paragraph breaks first,
    then on sentence boundaries, falling back to character-level splitting
    for very long sentences.
    """
    if config is None:
        config = DEFAULT_CONFIG

    if not text.strip():
        return []

    # Step 1: Split into paragraphs
    paragraphs = re.split(r'\n\s*\n', text.strip())

    chunks: list[DocumentChunk] = []
    current = ""
    line_counter = 0

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        # If adding this paragraph stays within chunk_size, accumulate
        if len(current) + len(para) + 2 <= config.chunk_size:
            current = (current + "\n\n" + para).strip() if current else para
        else:
            # Flush current chunk
            if current:
                chunks.append(_make_chunk(
                    text=current,
                    source_id=source_id,
                    source_type=source_type,
                    chunk_index=len(chunks),
                    line_start=line_counter,
                    trust_level=trust_level,
                ))
                line_counter += current.count('\n') + 2
                current = ""

            # If paragraph itself exceeds chunk_size, split at sentence boundaries
            if len(para) > config.chunk_size:
                sub_chunks = _split_long_paragraph(
                    para, config, source_id, source_type, trust_level,
                    start_index=len(chunks), start_line=line_counter,
                )
                chunks.extend(sub_chunks)
                line_counter += para.count('\n') + 2
                current = ""
            else:
                current = para

    # Flush remaining
    if current:
        chunks.append(_make_chunk(
            text=current,
            source_id=source_id,
            source_type=source_type,
            chunk_index=len(chunks),
            line_start=line_counter,
            trust_level=trust_level,
        ))

    return chunks


def _split_long_paragraph(
    para: str,
    config: ChunkerConfig,
    source_id: str,
    source_type: str,
    trust_level: TrustLevel,
    start_index: int,
    start_line: int,
) -> list[DocumentChunk]:
    """Split a paragraph that exceeds chunk_size into sentence-boundary chunks with overlap."""
    sentences = re.split(r'(?<=[.!?])\s+', para)
    chunks = []
    current = ""
    line_offset = start_line

    for sent in sentences:
        if len(current) + len(sent) + 1 <= config.chunk_size:
            current = (current + " " + sent).strip() if current else sent
        else:
            if current:
                chunks.append(_make_chunk(
                    text=current, source_id=source_id, source_type=source_type,
                    chunk_index=start_index + len(chunks),
                    line_start=line_offset, trust_level=trust_level,
                ))
                line_offset += current.count('\n') + 1
                # Overlap: keep last `overlap` chars of previous chunk
                if config.chunk_overlap > 0:
                    overlap_text = current[-config.chunk_overlap:]
                    current = overlap_text + " " + sent
                else:
                    current = sent
            else:
                # Single sentence exceeds chunk_size — hard split
                for i in range(0, len(sent), config.chunk_size - config.chunk_overlap):
                    piece = sent[i:i + config.chunk_size]
                    chunks.append(_make_chunk(
                        text=piece, source_id=source_id, source_type=source_type,
                        chunk_index=start_index + len(chunks),
                        line_start=line_offset, trust_level=trust_level,
                    ))
                current = ""
                line_offset += 1

    if current:
        chunks.append(_make_chunk(
            text=current, source_id=source_id, source_type=source_type,
            chunk_index=start_index + len(chunks),
            line_start=line_offset, trust_level=trust_level,
        ))

    return chunks


def _make_chunk(
    text: str,
    source_id: str,
    source_type: str,
    chunk_index: int,
    line_start: int,
    trust_level: TrustLevel,
) -> DocumentChunk:
    """Create a DocumentChunk with computed metadata."""
    keywords = list(set(
        t.lower() for t in re.findall(r'[\w一-鿿]{2,}', text)
    ))[:20]  # top 20 keywords

    return DocumentChunk(
        chunk_id=f"chunk_{source_id}_{chunk_index:04d}",
        source_id=source_id,
        source_type=source_type,
        text=text,
        summary="",  # Summary generated later by LLM or simple heuristic
        keywords=keywords,
        location=ChunkLocation(
            line_start=line_start,
            line_end=line_start + text.count('\n'),
        ),
        chunk_type=ChunkType.PARAGRAPH,
        trust_level=trust_level,
    )
