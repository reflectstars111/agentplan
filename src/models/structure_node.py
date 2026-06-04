"""StructureNode — hierarchical structure tree node.

Maps to agent_os_initial_plan.md §5.1 (Structure Index).
"""

from dataclasses import dataclass, field
from typing import Optional
from dataclasses_json import dataclass_json


@dataclass_json
@dataclass
class StructureNode:
    """A node in the hierarchical structure tree of a document or codebase."""

    node_id: str                               # e.g. "node_main_py_0001"
    source_id: str                             # owning source file
    node_type: str                             # "file" | "section" | "heading" | "function" | "class" | "method" | "page"
    name: str                                  # display name
    parent_id: Optional[str] = None            # NULL for root nodes
    depth: int = 0                             # nesting level (0 = root)
    location_page: Optional[int] = None
    location_section: Optional[str] = None
    location_line_start: Optional[int] = None
    location_line_end: Optional[int] = None
    chunk_ids: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    created_at: str = ""                       # ISO datetime
