"""GitHub repository clone + index via GitPython."""

import tempfile
from pathlib import Path


class GithubSource:
    """Clone a GitHub repository and index all files."""

    def clone_and_index(
        self,
        repo_url: str,
        file_store,
        branch: str = "main",
    ) -> dict:
        """Clone repo and index all code/doc files.

        Returns: {repo_name, files_indexed, source_ids, error?}
        """
        try:
            from git import Repo
        except ImportError:
            return {"error": "GitPython not installed. pip install GitPython"}

        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                repo = Repo.clone_from(
                    repo_url, tmpdir, branch=branch, depth=1,
                )
            except Exception as e:
                return {"error": f"Clone failed: {e}"}

            repo_name = repo_url.rstrip("/").split("/")[-1].replace(".git", "")
            files_indexed = 0
            source_ids = []

            for file_path in Path(tmpdir).rglob("*"):
                if not file_path.is_file():
                    continue
                if file_path.name.startswith("."):
                    continue
                if any(p.startswith(".") for p in file_path.parts):
                    continue
                # Skip binary and large files
                if file_path.suffix.lower() in {
                    ".py", ".js", ".ts", ".md", ".txt", ".json",
                    ".yaml", ".yml", ".toml", ".html", ".css", ".rs", ".go",
                    ".java", ".rb", ".c", ".h", ".cpp", ".hpp",
                }:
                    try:
                        source_id = file_store.ingest_file(file_path)
                        source_ids.append(source_id)
                        files_indexed += 1
                    except Exception:
                        pass

            return {
                "repo_name": repo_name,
                "files_indexed": files_indexed,
                "source_ids": source_ids,
            }
