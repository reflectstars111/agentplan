"""DependencyGraph — code import/call relationship extraction.

Maps to agent_os_initial_plan.md §5.1 (Graph Index) and §19 Phase 4 (Dependency Index).
"""

import json
import re
import uuid
from datetime import datetime, timezone
from src.db.connection import Database
from src.models.code_symbol import CodeSymbol


class DependencyGraph:
    """Extract and query code dependency relationships."""

    def __init__(self, db: Database):
        self.db = db

    def extract_from_symbols(self, symbols: list[CodeSymbol], source_id: str) -> int:
        """Extract import dependencies from code symbols and store in dependency_edges."""
        now = datetime.now(timezone.utc).isoformat()
        count = 0

        for sym in symbols:
            imports = self._extract_imports(sym.body)
            for imp in imports:
                edge_id = f"edge_{uuid.uuid4().hex[:12]}"
                self.db.execute(
                    """INSERT OR IGNORE INTO dependency_edges
                       (edge_id, source_symbol_id, target_symbol_id, edge_type, source_file, created_at)
                       VALUES (?,?,?,?,?,?)""",
                    (edge_id, sym.symbol_id, f"module:{imp}", "imports", source_id, now),
                )
                count += 1

        self.db.commit()
        return count

    def get_dependencies(self, symbol_id: str) -> list[str]:
        """Get all modules/symbols that this symbol depends on."""
        rows = self.db.execute(
            "SELECT target_symbol_id FROM dependency_edges WHERE source_symbol_id = ?",
            (symbol_id,),
        ).fetchall()
        return [r["target_symbol_id"] for r in rows]

    def get_dependents(self, symbol_id: str) -> list[str]:
        """Get all symbols that depend on this symbol."""
        rows = self.db.execute(
            "SELECT source_symbol_id FROM dependency_edges WHERE target_symbol_id = ?",
            (symbol_id,),
        ).fetchall()
        return [r["source_symbol_id"] for r in rows]

    def _extract_imports(self, source_code: str) -> list[str]:
        """Extract Python import targets from source code."""
        imports = []
        for match in re.finditer(r'(?:from\s+(\S+)\s+import|import\s+(\S+))', source_code):
            imp = match.group(1) or match.group(2)
            if imp:
                imports.append(imp)
        return imports
