import pytest

from ard.infra.db import Database
from semstate.runtime import SemStateRuntime


@pytest.fixture
def runtime():
    db = Database(":memory:")
    instance = SemStateRuntime(db)
    yield instance
    db.close()
