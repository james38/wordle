import json


def test_fixture_lists_are_disjoint_five_letter_words(env_paths):
    valids = json.load(open(env_paths["valid_words_path"]))
    solutions = json.load(open(env_paths["solution_words_path"]))
    assert all(len(w) == 5 for w in valids + solutions)
    assert set(valids) & set(solutions) == {"stale"}
    assert len(set(valids)) == len(valids)
    assert len(set(solutions)) == len(solutions)


def test_dependencies_import():
    import gymnasium
    import numpy
    import torch

    assert gymnasium.__version__ >= "1.0"
