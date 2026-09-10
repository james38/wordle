import random

import numpy as np
import pytest
import torch

from gym_wordle.envs.feedback import GREEN, GREY, YELLOW, colour
from gym_wordle.envs.wordle_env import (
    DEFAULT_SOLUTIONS,
    DEFAULT_VALIDS,
    LETTER_INDEX,
    letter_matrix,
    load_word_lists,
)


def colour_ref(guess, secret):
    """Independent plain-Python Wordle colouring."""
    n = len(guess)
    out = [GREY] * n
    remaining = {}
    for g, s in zip(guess, secret):
        if g != s:
            remaining[s] = remaining.get(s, 0) + 1
    for i, (g, s) in enumerate(zip(guess, secret)):
        if g == s:
            out[i] = GREEN
        elif remaining.get(g, 0) > 0:
            out[i] = YELLOW
            remaining[g] -= 1
    return out


def enc(*words):
    return torch.tensor([[LETTER_INDEX[c] for c in w] for w in words], dtype=torch.int8)


@pytest.mark.parametrize(
    "guess,secret,expected",
    [
        ("paper", "apple", [1, 1, 2, 1, 0]),
        ("label", "llama", [2, 1, 0, 0, 1]),
        ("eerie", "there", [1, 0, 1, 0, 2]),
        ("speed", "abide", [0, 0, 1, 0, 1]),
        ("crane", "crane", [2, 2, 2, 2, 2]),
        ("zzzzz", "crane", [0, 0, 0, 0, 0]),
    ],
)
def test_colour_hand_cases(guess, secret, expected):
    out = colour(enc(guess), enc(secret))
    assert out.dtype == torch.int8
    assert out.tolist() == [expected]
    assert colour_ref(guess, secret) == expected


def test_colour_matches_reference_on_real_lists():
    words, _ = load_word_lists(DEFAULT_VALIDS, DEFAULT_SOLUTIONS)
    rng = random.Random(0)
    pairs = [(rng.choice(words), rng.choice(words)) for _ in range(2000)]
    guesses = enc(*(g for g, _ in pairs))
    secrets = enc(*(s for _, s in pairs))
    out = colour(guesses, secrets).tolist()
    for (g, s), row in zip(pairs, out):
        assert row == colour_ref(g, s), (g, s)


def test_colour_broadcasts_one_guess_against_many_secrets():
    words, _ = load_word_lists(DEFAULT_VALIDS, DEFAULT_SOLUTIONS)
    letters, _ = letter_matrix(words, 5)
    W = torch.from_numpy(letters)
    g = enc("paper")
    out = colour(g.unsqueeze(1), W.unsqueeze(0))          # (1,V,5)
    assert out.shape == (1, len(words), 5)
    i = words.index("apple")
    assert out[0, i].tolist() == [1, 1, 2, 1, 0]
