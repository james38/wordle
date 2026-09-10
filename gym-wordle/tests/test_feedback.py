import random

import numpy as np
import pytest
import torch

from gym_wordle.envs.feedback import GREEN, GREY, YELLOW, colour, consistent_mask, hard_mode_mask
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


def _real_matrices():
    words, _ = load_word_lists(DEFAULT_VALIDS, DEFAULT_SOLUTIONS)
    letters, counts = letter_matrix(words, 5)
    return words, torch.from_numpy(letters), torch.from_numpy(counts)


def _play(words, secret, guesses):
    """Return (green_pos, min_count, guess_letters (T,5), colours (T,5)) after playing guesses."""
    green_pos = [-1] * 5
    min_count = [0] * 26
    gl, cl = [], []
    for g in guesses:
        c = colour_ref(g, secret)
        gl.append([LETTER_INDEX[x] for x in g])
        cl.append(c)
        for i, (ch, col) in enumerate(zip(g, c)):
            if col == GREEN:
                green_pos[i] = LETTER_INDEX[ch]
        for ch in set(g):
            k = sum(1 for x, col in zip(g, c) if x == ch and col != GREY)
            min_count[LETTER_INDEX[ch]] = max(min_count[LETTER_INDEX[ch]], k)
    return green_pos, min_count, gl, cl


def hard_ok_ref(word, green_pos, min_count):
    for i, g in enumerate(green_pos):
        if g >= 0 and LETTER_INDEX[word[i]] != g:
            return False
    for l, m in enumerate(min_count):
        if m and sum(1 for ch in word if LETTER_INDEX[ch] == l) < m:
            return False
    return True


def test_hard_mode_mask_matches_brute_force():
    words, W_letters, W_counts = _real_matrices()
    rng = random.Random(1)
    B = 50
    gp, mc = [], []
    for _ in range(B):
        secret = rng.choice(words)
        guesses = [rng.choice(words) for _ in range(rng.randint(1, 3))]
        g, m, _, _ = _play(words, secret, guesses)
        gp.append(g)
        mc.append(m)
    mask = hard_mode_mask(
        torch.tensor(gp, dtype=torch.int8), torch.tensor(mc, dtype=torch.int8), W_letters, W_counts
    )
    assert mask.shape == (B, len(words)) and mask.dtype == torch.bool
    sample = rng.sample(range(len(words)), 300)
    for b in range(B):
        for w in sample:
            assert bool(mask[b, w]) == hard_ok_ref(words[w], gp[b], mc[b]), (b, words[w])


def test_hard_mode_mask_all_true_before_any_guess():
    words, W_letters, W_counts = _real_matrices()
    mask = hard_mode_mask(
        torch.full((3, 5), -1, dtype=torch.int8), torch.zeros(3, 26, dtype=torch.int8), W_letters, W_counts
    )
    assert mask.all()


def test_hard_mode_mask_grey_letters_not_forbidden():
    words, W_letters, W_counts = _real_matrices()
    gp, mc, _, _ = _play(words, "crane", ["slate"])   # s,l,t grey; a,e green
    mask = hard_mode_mask(
        torch.tensor([gp], dtype=torch.int8), torch.tensor([mc], dtype=torch.int8), W_letters, W_counts
    )
    assert bool(mask[0, words.index("state")])        # reuses grey s and t, keeps a/e greens


def test_consistent_mask_matches_brute_force_colouring():
    words, W_letters, W_counts = _real_matrices()
    rng = random.Random(2)
    B, T = 20, 3
    gl = torch.zeros(B, T, 5, dtype=torch.int8)
    cl = torch.zeros(B, T, 5, dtype=torch.int8)
    n_guesses = torch.zeros(B, dtype=torch.long)
    games = []
    for b in range(B):
        secret = rng.choice(words)
        k = rng.randint(1, T)
        guesses = [rng.choice(words) for _ in range(k)]
        _, _, g, c = _play(words, secret, guesses)
        gl[b, :k] = torch.tensor(g, dtype=torch.int8)
        cl[b, :k] = torch.tensor(c, dtype=torch.int8)
        n_guesses[b] = k
        games.append((secret, guesses))
    mask = consistent_mask(gl, cl, n_guesses, W_letters, W_counts)
    assert mask.shape == (B, len(words))
    sample = rng.sample(range(len(words)), 300)
    for b, (secret, guesses) in enumerate(games):
        assert bool(mask[b, words.index(secret)]), "secret must stay consistent"
        for w in sample:
            ref = all(colour_ref(g, words[w]) == colour_ref(g, secret) for g in guesses)
            assert bool(mask[b, w]) == ref, (b, words[w])


def test_consistent_count_non_increasing_along_a_game():
    words, W_letters, W_counts = _real_matrices()
    secret, guesses = "crane", ["slate", "brine", "crane"]
    _, _, g, c = _play(words, secret, guesses)
    gl = torch.tensor([g], dtype=torch.int8)
    cl = torch.tensor([c], dtype=torch.int8)
    counts = [
        int(consistent_mask(gl, cl, torch.tensor([k]), W_letters, W_counts).sum())
        for k in range(0, 4)
    ]
    assert counts[0] == len(words)
    assert counts == sorted(counts, reverse=True)
    assert counts[-1] == 1
