from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def fixture_dir():
    return FIXTURES


@pytest.fixture
def env_paths():
    return {
        "valid_words_path": FIXTURES / "valids_small.json",
        "solution_words_path": FIXTURES / "words_small.json",
    }
