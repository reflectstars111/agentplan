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

        # Try specified branch, fallback to common defaults
        branches_to_try = [branch]
        if branch != "master":
            branches_to_try.append("master")
        if branch != "main":
            branches_to_try.append("main")

        repo_name = repo_url.rstrip("/").split("/")[-1].replace(".git", "")
        clone_error = ""

        with tempfile.TemporaryDirectory() as tmpdir:
            repo = None
            for b in branches_to_try:
                try:
                    repo = Repo.clone_from(repo_url, tmpdir, branch=b, depth=1)
                    break
                except Exception as e:
                    clone_error = str(e)
                    continue

            if repo is None:
                return {
                    "error": f"Clone failed. Tried branches {branches_to_try}: {clone_error}"
                }

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
