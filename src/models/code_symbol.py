"""CodeSymbol — a code symbol extracted from source files via tree-sitter.

Maps to agent_os_initial_plan.md §4.2 (CodeSymbol storage object).
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional
from dataclasses_json import dataclass_json, config


@dataclass_json
@dataclass
class CodeSymbol:
    """A code symbol (function, class, method) extracted from source files."""

    symbol_id: str                             # e.g. "sym_main_py_0001"
    source_id: str                             # e.g. "file:main.py"
    name: str                                  # function/class/method name
    symbol_type: str                           # "function" | "class" | "method" | "variable"
    language: str                              # "python" | "javascript" | "typescript"
    signature: str = ""                        # full signature line
    body: str = ""                             # source code text of the symbol
    docstring: str = ""                        # extracted docstring
    location_line_start: int = 0
    location_line_end: int = 0
    parent_symbol_id: Optional[str] = None     # enclosing class for methods
    chunk_id: Optional[str] = None             # reference to corresponding DocumentChunk
    created_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc),
        metadata=config(encoder=lambda d: d.isoformat(), decoder=lambda s: datetime.fromisoformat(s)),
    )
