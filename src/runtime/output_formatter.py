"""OutputFormatter — structured output helpers (Mermaid, LaTeX, JSON).

Maps to agent_os_initial_plan.md §10.4 (Output System).
"""

import json


class OutputFormatter:
    """Structured output formatting utilities for Agent-OS responses."""

    @staticmethod
    def mermaid(code: str, diagram_type: str = "flowchart") -> str:
        """Wrap diagram code in a Mermaid fenced block.

        Args:
            code: Mermaid diagram source.
            diagram_type: "flowchart", "sequence", "class", "er", etc.

        Returns:
            Markdown fenced code block with ```mermaid.
        """
        return f"```mermaid\n{diagram_type}\n{code}\n```"

    @staticmethod
    def latex(expr: str, display: bool = True) -> str:
        """Wrap LaTeX expression in math delimiters.

        Args:
            expr: LaTeX expression.
            display: True for display math ($$), False for inline ($).

        Returns:
            LaTeX math-block string.
        """
        if display:
            return f"$$\n{expr}\n$$"
        return f"${expr}$"

    @staticmethod
    def table(headers: list[str], rows: list[list[str]]) -> str:
        """Build a Markdown table.

        Args:
            headers: Column header strings.
            rows: List of row lists.

        Returns:
            Markdown table string.
        """
        if not headers:
            return ""
        header_line = "| " + " | ".join(headers) + " |"
        sep_line = "|" + "|".join(" --- " for _ in headers) + "|"
        body_lines = [
            "| " + " | ".join(str(c) for c in row) + " |"
            for row in rows
        ]
        return "\n".join([header_line, sep_line] + body_lines)

    @staticmethod
    def code_block(code: str, language: str = "") -> str:
        """Wrap code in a Markdown fenced code block."""
        return f"```{language}\n{code}\n```"

    @staticmethod
    def json_output(data: dict | list) -> str:
        """Format data as indented JSON."""
        return json.dumps(data, indent=2, ensure_ascii=False)

    @staticmethod
    def report(
        title: str,
        sections: list[tuple[str, str]],
    ) -> str:
        """Build a structured Markdown report.

        Args:
            title: Report title (H1).
            sections: List of (heading, body) tuples.

        Returns:
            Markdown report string.
        """
        lines = [f"# {title}\n"]
        for heading, body in sections:
            lines.append(f"## {heading}\n")
            lines.append(body)
            lines.append("")
        return "\n".join(lines)
