"""DocumentChunk — a slice of an external document. Maps to agent_os_initial_plan.md §4.4."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional
from dataclasses_json import dataclass_json, config


class ChunkType(str, Enum):
    PARAGRAPH = "paragraph"
    TABLE = "table"
    CODE = "code"
    FIGURE = "figure"
    HEADING = "heading"


class TrustLevel(str, Enum):
    TRUSTED_INSTRUCTION = "trusted_instruction"
    USER_INSTRUCTION = "user_instruction"
    INTERNAL_MEMORY = "internal_memory"
    USER_PROVIDED_DATA = "user_provided_data"
    EXTERNAL_UNTRUSTED = "external_untrusted"
    TOOL_OBSERVATION = "tool_observation"
    AGENT_GENERATED = "agent_generated"


@dataclass_json
@dataclass
class ChunkLocation:
    """Where this chunk lives in its source document."""
    page: Optional[int] = None
    section: Optional[str] = None
    line_start: Optional[int] = None
    line_end: Optional[int] = None


@dataclass_json
@dataclass
class DocumentChunk:
    """A chunk from an external document (PDF, code, web, etc.)."""

    chunk_id: str
    source_id: str               # e.g. "file:paper_001.pdf"
    source_type: str              # pdf | markdown | code | web | text
    text: str
    summary: str = ""
    keywords: list[str] = field(default_factory=list)
    location: ChunkLocation = field(default_factory=ChunkLocation)
    chunk_type: ChunkType = ChunkType.PARAGRAPH
    embedding_id: Optional[str] = None  # index in FAISS
    trust_level: TrustLevel = TrustLevel.EXTERNAL_UNTRUSTED
    created_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc),
        metadata=config(encoder=lambda d: d.isoformat(), decoder=lambda s: datetime.fromisoformat(s)),
    )
