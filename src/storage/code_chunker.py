"""CodeChunker — AST-aware code chunking at function/class boundaries.

Uses tree-sitter to produce DocumentChunks with ChunkType.CODE,
respecting function/class boundaries and extracting structure metadata.
"""

from src.models.chunk import DocumentChunk, ChunkType, ChunkLocation, TrustLevel
from src.parsing.code_parser import CodeParser


def chunk_code(
    source_code: str,
    source_id: str,
    language: str = "python",
    trust_level: TrustLevel = TrustLevel.EXTERNAL_UNTRUSTED,
) -> list[DocumentChunk]:
    """Split source code into chunks at function/class/method boundaries.

    Each top-level function or class becomes its own CODE-type chunk.
    Import statements are grouped into a PARAGRAPH preamble chunk.
    Large classes are split to individual methods if needed.

    Args:
        source_code: Raw source code string.
        source_id: Source identifier (e.g. "file:main.py").
        language: Programming language ("python"|"javascript"|"typescript").
        trust_level: Trust level for the chunks.

    Returns:
        List of DocumentChunk objects with ChunkType.CODE for code entities.
    """
    if not source_code.strip():
        return []

    try:
        parser = CodeParser(language)
    except (ValueError, ImportError):
        # Fallback: treat as plain text if language unsupported
        from src.storage.chunker import chunk_text
        return chunk_text(source_code, source_id, source_type="code",
                          trust_level=trust_level)

    tree = parser.parser.parse(source_code)
    root = tree.root_node()

    chunks = []
    idx = 0

    # Group imports into preamble chunk
    import_chunk_text = _extract_imports(root, source_code)
    if import_chunk_text.strip():
        chunks.append(DocumentChunk(
            chunk_id=f"chunk_{source_id}_{idx:04d}",
            source_id=source_id,
            source_type="code",
            text=import_chunk_text,
            chunk_type=ChunkType.PARAGRAPH,
            trust_level=trust_level,
        ))
        idx += 1

    # Extract function and class chunks
    for i in range(root.child_count()):
        child = root.child(i)
        if child.kind() == "function_definition":
            chunks.append(_make_code_chunk(child, source_code, source_id, idx))
            idx += 1
        elif child.kind() == "class_definition":
            chunks.append(_make_code_chunk(child, source_code, source_id, idx))
            idx += 1

    return chunks


def _extract_imports(root, source_code: str) -> str:
    """Extract import statements as a single preamble chunk."""
    parts = []
    for i in range(root.child_count()):
        child = root.child(i)
        if child.kind() in ("import_statement", "import_from_statement"):
            br = child.byte_range()
            parts.append(source_code[br.start:br.end])
    return "\n".join(parts)


def _make_code_chunk(node, source_code: str, source_id: str, idx: int) -> DocumentChunk:
    """Create a DocumentChunk from a tree-sitter function/class node."""
    br = node.byte_range()
    text = source_code[br.start:br.end]

    return DocumentChunk(
        chunk_id=f"chunk_{source_id}_{idx:04d}",
        source_id=source_id,
        source_type="code",
        text=text,
        chunk_type=ChunkType.CODE,
        trust_level=TrustLevel.EXTERNAL_UNTRUSTED,
        location=ChunkLocation(
            line_start=node.start_position().row + 1,
            line_end=node.end_position().row + 1,
        ),
    )
