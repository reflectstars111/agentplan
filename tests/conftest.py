"""Shared test fixtures."""

import os
import tempfile
import pytest
from pathlib import Path
from src.config import Config


@pytest.fixture
def temp_config() -> Config:
    """Config pointing at temp directories for isolated tests."""
    tmp = tempfile.mkdtemp()
    return Config(
        db_path=f"{tmp}/test.db",
        file_store_path=f"{tmp}/files",
        vector_index_path=f"{tmp}/vec.index",
    )


@pytest.fixture
def sample_pdf_path() -> Path:
    """Path to a minimal test PDF. Created on first use."""
    return Path(__file__).parent / "fixtures" / "sample.pdf"


@pytest.fixture
def sample_markdown_path(tmp_path: Path) -> Path:
    """Create a temporary sample markdown file."""
    p = tmp_path / "sample.md"
    p.write_text("""# Test Document

## Section 1

This is the first section. It contains important information about the project.

## Section 2

This is the second section. It contains more details about implementation.

### Subsection 2.1

Here are some code examples.

```python
def hello():
    print("Hello, World!")
```

## Section 3

Final conclusions and next steps.
""")
    return p
