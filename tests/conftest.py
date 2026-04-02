from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture
def tmp_path(request) -> Path:
    root = Path("data/test_tmp")
    root.mkdir(parents=True, exist_ok=True)
    safe_name = "".join(character if character.isalnum() else "_" for character in request.node.name)
    target = root / safe_name
    target.mkdir(parents=True, exist_ok=True)
    return target
