"""CodeParser — tree-sitter based code symbol and structure extraction.

Maps to agent_os_initial_plan.md §4.2 (CodeSymbol) and §5.1 (Structure Index).
"""

import uuid
from dataclasses import dataclass, field
from pathlib import Path
from src.models.code_symbol import CodeSymbol
from src.models.structure_node import StructureNode


@dataclass
class ParseResult:
    """Result of a full code parse."""
    symbols: list[CodeSymbol] = field(default_factory=list)
    structure: list[StructureNode] = field(default_factory=list)
    source_id: str = ""
    language: str = ""


# Language extension mapping
_EXT_LANG_MAP = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
}


class CodeParser:
    """tree-sitter based code parser for Python, JavaScript, and TypeScript."""

    SUPPORTED_LANGUAGES = {"python", "javascript", "typescript"}

    def __init__(self, language: str = "python"):
        if language not in self.SUPPORTED_LANGUAGES:
            raise ValueError(
                f"Unsupported language: {language}. "
                f"Supported: {self.SUPPORTED_LANGUAGES}"
            )
        self.language = language
        self._parser = None

    @property
    def parser(self):
        """Lazy-load the tree-sitter parser."""
        if self._parser is None:
            from tree_sitter_language_pack import get_parser
            self._parser = get_parser(self.language)
        return self._parser

    def parse(self, source_code: str, source_id: str) -> ParseResult:
        """Full parse: extract symbols and structure."""
        symbols = self.extract_symbols(source_code, source_id)
        structure = self.extract_structure(source_code, source_id)
        return ParseResult(
            symbols=symbols,
            structure=structure,
            source_id=source_id,
            language=self.language,
        )

    def extract_symbols(self, source_code: str, source_id: str) -> list[CodeSymbol]:
        """Extract functions, classes, and methods from source code."""
        if not source_code.strip():
            return []

        tree = self.parser.parse(source_code)
        root = tree.root_node()
        symbols = []
        classes = {}  # node_id -> CodeSymbol for parent lookup

        self._walk_symbols(root, source_code, source_id, symbols, classes)
        return symbols

    def _walk_symbols(self, node, source_code, source_id, symbols, classes,
                      parent_id=None):
        """Recursively walk AST nodes to find function/class/method definitions."""
        for i in range(node.child_count()):
            child = node.child(i)

            if child.kind() == "function_definition":
                sym = self._extract_function(child, source_code, source_id,
                                             parent_id=parent_id,
                                             enclosing_class=parent_id)
                symbols.append(sym)

            elif child.kind() == "class_definition":
                sym = self._extract_class(child, source_code, source_id)
                symbols.append(sym)
                classes[child] = sym
                # Walk class body to find methods
                body = child.child_by_field_name("body")
                if body:
                    self._walk_symbols(body, source_code, source_id,
                                       symbols, classes, parent_id=sym.symbol_id)

    def _extract_function(self, node, source_code, source_id,
                          parent_id=None, enclosing_class=None) -> CodeSymbol:
        name_node = node.child_by_field_name("name")
        name = self._node_text(source_code, name_node) if name_node else "unknown"

        # Determine symbol_type
        if enclosing_class:
            sym_type = "method"
        else:
            sym_type = "function"

        # Get docstring from body (new tree-sitter: string is direct child of block)
        docstring = ""
        body_node = node.child_by_field_name("body")
        if body_node and body_node.kind() in ("block", "suite"):
            for i in range(body_node.child_count()):
                stmt = body_node.child(i)
                # Check for expression_statement containing string (old tree-sitter)
                if stmt.kind() == "expression_statement":
                    for j in range(stmt.child_count()):
                        expr_child = stmt.child(j)
                        if expr_child.kind() == "string":
                            docstring = self._extract_string_text(source_code, expr_child)
                            break
                # Also check for direct string child (new tree-sitter 0.24+)
                elif stmt.kind() == "string":
                    docstring = self._extract_string_text(source_code, stmt)
                if docstring:
                    break

        # Get signature text
        params = node.child_by_field_name("parameters")
        sig_start = name_node.byte_range().start if name_node else node.byte_range().start
        sig_end = params.end_byte() if params else node.child_by_field_name("body").start_byte() if body_node else node.byte_range().end
        signature = source_code[sig_start:sig_end].strip()

        # Get body text
        body_text = source_code[node.byte_range().start:node.byte_range().end]

        return CodeSymbol(
            symbol_id=f"sym_{uuid.uuid4().hex[:12]}",
            source_id=source_id,
            name=name,
            symbol_type=sym_type,
            language=self.language,
            signature=signature,
            body=body_text,
            docstring=docstring,
            location_line_start=self._node_line(node),
            location_line_end=self._node_end_line(node),
            parent_symbol_id=enclosing_class,
        )

    def _extract_class(self, node, source_code, source_id) -> CodeSymbol:
        name_node = node.child_by_field_name("name")
        name = self._node_text(source_code, name_node) if name_node else "unknown"

        # Get docstring from body
        docstring = ""
        body_node = node.child_by_field_name("body")
        if body_node:
            for i in range(body_node.child_count()):
                stmt = body_node.child(i)
                if stmt.kind() == "expression_statement":
                    for j in range(stmt.child_count()):
                        expr_child = stmt.child(j)
                        if expr_child.kind() == "string":
                            docstring = self._extract_string_text(source_code, expr_child)
                            break
                    if docstring:
                        break

        body_text = source_code[node.byte_range().start:node.byte_range().end]

        return CodeSymbol(
            symbol_id=f"sym_{uuid.uuid4().hex[:12]}",
            source_id=source_id,
            name=name,
            symbol_type="class",
            language=self.language,
            signature=f"class {name}",
            body=body_text,
            docstring=docstring,
            location_line_start=self._node_line(node),
            location_line_end=self._node_end_line(node),
        )

    def _extract_string_text(self, source_code, node) -> str:
        """Extract clean text from a string node, stripping quotes."""
        text = self._node_text(source_code, node)
        text = text.strip()
        if (text.startswith('"""') and text.endswith('"""')) or \
           (text.startswith("'''") and text.endswith("'''")):
            text = text[3:-3]
        elif (text.startswith('"') and text.endswith('"')) or \
             (text.startswith("'") and text.endswith("'")):
            text = text[1:-1]
        return text.strip()

    def extract_structure(self, source_code: str, source_id: str) -> list[StructureNode]:
        """Build hierarchical structure: file > class/function nodes."""
        nodes = []

        # Root file node
        file_node_id = f"node_{uuid.uuid4().hex[:12]}"
        nodes.append(StructureNode(
            node_id=file_node_id,
            source_id=source_id,
            node_type="file",
            name=source_id,
            depth=0,
            metadata={"language": self.language},
        ))

        if not source_code.strip():
            return nodes

        tree = self.parser.parse(source_code)
        root = tree.root_node()
        self._walk_structure(root, source_code, source_id, nodes, file_node_id, 1)

        return nodes

    def _walk_structure(self, node, source_code, source_id, nodes, parent_id, depth):
        """Walk AST to build structure nodes for functions and classes."""
        for i in range(node.child_count()):
            child = node.child(i)

            if child.kind() in ("function_definition", "class_definition"):
                name_node = child.child_by_field_name("name")
                name = self._node_text(source_code, name_node) if name_node else "unknown"
                node_type = "class" if child.kind() == "class_definition" else "function"

                struct = StructureNode(
                    node_id=f"node_{uuid.uuid4().hex[:12]}",
                    source_id=source_id,
                    node_type=node_type,
                    name=name,
                    parent_id=parent_id,
                    depth=depth,
                    location_line_start=child.start_position().row + 1,
                    location_line_end=child.end_position().row + 1,
                    created_at="",
                )
                nodes.append(struct)

                # For classes, walk body for methods
                if child.kind() == "class_definition":
                    body = child.child_by_field_name("body")
                    if body:
                        self._walk_structure(body, source_code, source_id,
                                             nodes, struct.node_id, depth + 1)

    @staticmethod
    def _node_text(source_code: str, node) -> str:
        """Extract text of a tree-sitter node from source code using byte range."""
        br = node.byte_range()
        return source_code[br.start:br.end]

    @staticmethod
    def _node_line(node) -> int:
        """Get 1-based start line number for a node."""
        return node.start_position().row + 1

    @staticmethod
    def _node_end_line(node) -> int:
        """Get 1-based end line number for a node."""
        return node.end_position().row + 1

    @staticmethod
    def detect_language(file_path: str) -> str | None:
        """Guess language from file extension."""
        ext = Path(file_path).suffix.lower()
        return _EXT_LANG_MAP.get(ext)
