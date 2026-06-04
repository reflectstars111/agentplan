"""StructureIndex — hierarchical structure node index.

SQLite-backed index of sections, symbols, headings, and pages with
parent-child hierarchy and name-based search.

Maps to agent_os_initial_plan.md §5.1 (Structure Index).
"""

import json
from datetime import datetime, timezone
from src.db.connection import Database
from src.models.structure_node import StructureNode


class StructureIndex:
    """Hierarchical index of structure nodes in SQLite.

    Supports CRUD, tree traversal (subtree), name search, and
    structural relevance scoring for hybrid retrieval.
    """

    def __init__(self, db: Database):
        self.db = db

    def index_nodes(self, nodes: list[StructureNode]) -> int:
        """Insert or replace structure nodes. Returns count inserted."""
        now = datetime.now(timezone.utc).isoformat()
        for node in nodes:
            if not node.created_at:
                node.created_at = now
            self.db.execute(
                """INSERT OR REPLACE INTO structure_nodes
                   (node_id, source_id, node_type, name, parent_id, depth,
                    location_page, location_section, location_line_start,
                    location_line_end, chunk_ids, metadata, created_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (node.node_id, node.source_id, node.node_type, node.name,
                 node.parent_id, node.depth, node.location_page,
                 node.location_section, node.location_line_start,
                 node.location_line_end, json.dumps(node.chunk_ids),
                 json.dumps(node.metadata), node.created_at),
            )
        self.db.commit()
        return len(nodes)

    def get_children(self, parent_id: str | None = None,
                     source_id: str | None = None) -> list[StructureNode]:
        """Get direct children. If parent_id is None, returns root nodes."""
        if parent_id is None and source_id is None:
            sql = "SELECT * FROM structure_nodes WHERE parent_id IS NULL"
            rows = self.db.execute(sql).fetchall()
        elif parent_id is None:
            sql = "SELECT * FROM structure_nodes WHERE parent_id IS NULL AND source_id = ?"
            rows = self.db.execute(sql, (source_id,)).fetchall()
        elif source_id is None:
            sql = "SELECT * FROM structure_nodes WHERE parent_id = ?"
            rows = self.db.execute(sql, (parent_id,)).fetchall()
        else:
            sql = "SELECT * FROM structure_nodes WHERE parent_id = ? AND source_id = ?"
            rows = self.db.execute(sql, (parent_id, source_id)).fetchall()
        return [self._row_to_node(dict(r)) for r in rows]

    def get_by_source(self, source_id: str) -> list[StructureNode]:
        """Get all nodes for a file."""
        rows = self.db.execute(
            "SELECT * FROM structure_nodes WHERE source_id = ? ORDER BY depth",
            (source_id,),
        ).fetchall()
        return [self._row_to_node(dict(r)) for r in rows]

    def get_subtree(self, node_id: str) -> list[StructureNode]:
        """Get a node and all its descendants using recursive CTE."""
        rows = self.db.execute("""
            WITH RECURSIVE subtree AS (
                SELECT * FROM structure_nodes WHERE node_id = ?
                UNION ALL
                SELECT sn.* FROM structure_nodes sn
                JOIN subtree st ON sn.parent_id = st.node_id
            )
            SELECT * FROM subtree ORDER BY depth
        """, (node_id,)).fetchall()
        return [self._row_to_node(dict(r)) for r in rows]

    def search_by_name(self, name: str,
                       node_type: str | None = None) -> list[StructureNode]:
        """Case-insensitive name search with optional type filter."""
        name_lower = name.lower()
        if node_type:
            rows = self.db.execute(
                "SELECT * FROM structure_nodes WHERE LOWER(name) LIKE ? AND node_type = ?",
                (f"%{name_lower}%", node_type),
            ).fetchall()
        else:
            rows = self.db.execute(
                "SELECT * FROM structure_nodes WHERE LOWER(name) LIKE ?",
                (f"%{name_lower}%",),
            ).fetchall()
        return [self._row_to_node(dict(r)) for r in rows]

    def structural_relevance(self, chunk_id: str,
                             query_terms: list[str]) -> float:
        """Score 0.0-1.0: how well does the chunk's structural context match query terms?

        Checks if the parent section/heading/function name contains any query terms.
        Higher score when chunk is under a relevant structural node.
        """
        if not query_terms:
            return 0.0

        # Find nodes that contain this chunk_id
        rows = self.db.execute(
            "SELECT * FROM structure_nodes WHERE chunk_ids LIKE ?",
            (f"%{chunk_id}%",),
        ).fetchall()

        if not rows:
            return 0.0

        hits = 0
        for row in rows:
            node = self._row_to_node(dict(row))
            # Check if any query term appears in the node name or its ancestors
            for term in query_terms:
                if term.lower() in node.name.lower():
                    hits += 1
                    break

        return min(1.0, hits / max(1, len(rows)))

    def _row_to_node(self, row: dict) -> StructureNode:
        return StructureNode(
            node_id=row["node_id"],
            source_id=row["source_id"],
            node_type=row["node_type"],
            name=row["name"],
            parent_id=row.get("parent_id"),
            depth=row.get("depth", 0),
            location_page=row.get("location_page"),
            location_section=row.get("location_section"),
            location_line_start=row.get("location_line_start"),
            location_line_end=row.get("location_line_end"),
            chunk_ids=json.loads(row.get("chunk_ids", "[]")),
            metadata=json.loads(row.get("metadata", "{}")),
            created_at=row.get("created_at", ""),
        )
