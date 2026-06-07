"""FileWriter — write Agent-OS responses to disk.

Maps to agent_os_initial_plan.md §10.4 (Output System — file output).
"""

from pathlib import Path


class FileWriter:
    """Write responses, reports, and data to the filesystem."""

    def __init__(self, output_dir: str = "data/output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def write(
        self, content: str, filename: str, mode: str = "w"
    ) -> Path:
        """Write content to a file in the output directory.

        Args:
            content: Text content to write.
            filename: Target filename (relative to output_dir).
            mode: File open mode ("w" for overwrite, "a" for append).

        Returns:
            Path to the written file.

        Raises:
            ValueError: If filename contains path traversal.
        """
        # Security: reject path traversal
        name = Path(filename).name
        if name != filename or ".." in filename:
            raise ValueError(
                f"Invalid filename: '{filename}'. "
                "Only simple filenames (no paths) are allowed."
            )

        filepath = self.output_dir / name
        with open(filepath, mode, encoding="utf-8") as f:
            f.write(content)

        return filepath

    def write_report(
        self,
        title: str,
        sections: list[tuple[str, str]],
        filename: str,
    ) -> Path:
        """Write a Markdown report using OutputFormatter.

        Args:
            title: Report title.
            sections: List of (heading, body) tuples.
            filename: Output filename.

        Returns:
            Path to the written file.
        """
        from src.runtime.output_formatter import OutputFormatter
        content = OutputFormatter.report(title, sections)
        return self.write(content, filename)

    def list_files(self) -> list[str]:
        """List all files currently in the output directory."""
        if not self.output_dir.exists():
            return []
        return sorted(
            f.name for f in self.output_dir.iterdir() if f.is_file()
        )
