"""Wordle as a gymnasium environment with a strict hard mode.

Observation layout (flat int8 vector, length 1 + 26*n_letters + 2*26):
  [0]                        guesses used so far
  [POS + 5*letter + pos]     0 = other, 1 = possible yellow, 2 = green
  [EXCEEDED + letter]        1 if the letter was guessed more times than it
                             appears in the solution
  [YELLOWS + letter]         max yellows seen for the letter, less any
                             greens found later
"""
import json
import math
import string
from pathlib import Path

import gymnasium as gym
import numpy as np
from gymnasium import spaces

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_VALIDS = REPO_ROOT / "valids.json"
DEFAULT_SOLUTIONS = REPO_ROOT / "words.json"

ALPHABET = string.ascii_lowercase
N_ALPHABET = len(ALPHABET)
LETTER_INDEX = {c: i for i, c in enumerate(ALPHABET)}


def load_word_lists(valid_words_path, solution_words_path):
    """Return (words, solutions). `words` is the sorted union of both files."""
    with open(valid_words_path) as f:
        valids = json.load(f)
    with open(solution_words_path) as f:
        solutions = json.load(f)
    words = sorted(set(valids) | set(solutions))
    return words, list(solutions)


def letter_matrix(words, n_letters):
    """(N, n_letters) int8 letter indices and (N, 26) int8 letter counts."""
    letters = np.array(
        [[LETTER_INDEX[c] for c in w] for w in words], dtype=np.int8
    ).reshape(len(words), n_letters)
    counts = np.zeros((len(words), N_ALPHABET), dtype=np.int8)
    rows = np.arange(len(words))
    for col in range(n_letters):
        np.add.at(counts, (rows, letters[:, col]), 1)
    return letters, counts


class WordleEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        valid_words_path=DEFAULT_VALIDS,
        solution_words_path=DEFAULT_SOLUTIONS,
        n_letters=5,
        max_attempts=6,
        L=4.0,
        min_guesses=True,
    ):
        super().__init__()
        assert isinstance(n_letters, int), "n_letters must be int"
        assert isinstance(max_attempts, int), "max_attempts must be int"
        self.n_letters = n_letters
        self.max_attempts = max_attempts

        self.words, self.solutions = load_word_lists(
            valid_words_path, solution_words_path
        )
        for w in self.words:
            assert len(w) == n_letters and w.isalpha() and w.islower(), w
        self.word_to_action = {w: i for i, w in enumerate(self.words)}
        self.word_letters, self.word_counts = letter_matrix(self.words, n_letters)

        self.POS = 1
        self.EXCEEDED = self.POS + N_ALPHABET * n_letters
        self.YELLOWS = self.EXCEEDED + N_ALPHABET
        self.obs_size = self.YELLOWS + N_ALPHABET
        nvec = (
            [1 + max_attempts]
            + [3] * (N_ALPHABET * n_letters)
            + [2] * N_ALPHABET
            + [1 + n_letters // 2] * N_ALPHABET
        )
        self.observation_space = spaces.MultiDiscrete(nvec, dtype=np.int8)
        self.action_space = spaces.Discrete(len(self.words))
        self.rewards = self._gen_rewards(L, min_guesses)
        self.reward_range = (0.0, 1.0)

        self.secret_word = None
        self.n_guesses = 0
        self.terminated = False
        self.overused_letters = set()
        self._state = np.zeros(self.obs_size, dtype=np.int8)
        self._legal = np.ones(len(self.words), dtype=bool)
        self._guessed = np.zeros(len(self.words), dtype=bool)

    # ------------------------------------------------------------------ api

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        secret = (options or {}).get("secret_word")
        if secret is None:
            secret = self.solutions[self.np_random.integers(len(self.solutions))]
        assert isinstance(secret, str), "secret_word must be str"
        assert len(secret) == self.n_letters, "secret_word has wrong length"
        assert secret.isalpha() and secret.islower(), "secret_word must be a-z"
        self.secret_word = secret
        self.n_guesses = 0
        self.terminated = False
        self.overused_letters = set()
        self._state = np.zeros(self.obs_size, dtype=np.int8)
        self._legal = np.ones(len(self.words), dtype=bool)
        self._guessed = np.zeros(len(self.words), dtype=bool)
        return self._state.copy(), self._info(valid=True)

    def step(self, action):
        action = int(action)
        assert self.action_space.contains(action), f"action {action} out of range"
        assert self.secret_word is not None, "call reset() first"
        assert not self.terminated, "episode is over; call reset()"
        guess = self.words[action]

        if not self._action_mask()[action]:
            return self._state.copy(), 0.0, False, False, self._info(valid=False)

        self.n_guesses += 1
        self._state[0] = self.n_guesses
        self._guessed[action] = True
        self._update_state(guess)
        self._legal = self._hard_mode_mask()

        reward = 0.0
        if guess == self.secret_word:
            reward = float(self.rewards[self.n_guesses - 1])
            self.terminated = True
        elif self.n_guesses >= self.max_attempts:
            self.terminated = True
        return self._state.copy(), reward, self.terminated, False, self._info(valid=True)

    # -------------------------------------------------------------- helpers

    def _info(self, valid):
        mask = self._action_mask()
        return {
            "valid": valid,
            "valid_words": {int(i): self.words[i] for i in np.flatnonzero(mask)},
            "action_mask": mask,
            "secret_word": self.secret_word,
            "n_guesses": self.n_guesses,
        }

    def _action_mask(self):
        return self._legal & ~self._guessed

    def _hard_mode_mask(self):
        """Boolean mask over all words: which satisfy hard mode right now.

        1. A green letter must be kept at its position.
        2. An overused letter (grey, or every copy already green) may only
           appear at positions where it is green.
        3. A letter with a known yellow must appear at least
           (its tracked yellows, at least 1) + (its greens) times.
        The first guess is exempt: before any guess the state is all zeros
        and every rule is vacuous, so the mask is all True.
        """
        ok = np.ones(len(self.words), dtype=bool)
        pos = self._state[self.POS : self.EXCEEDED].reshape(N_ALPHABET, self.n_letters)
        overused = np.array(
            sorted(LETTER_INDEX[c] for c in self.overused_letters), dtype=np.int8
        )
        for i in range(self.n_letters):
            greens = np.flatnonzero(pos[:, i] == 2)
            if greens.size:
                ok &= self.word_letters[:, i] == greens[0]
            elif overused.size:
                ok &= ~np.isin(self.word_letters[:, i], overused)
        has_yellow = (pos == 1).any(axis=1)
        n_green = (pos == 2).sum(axis=1)
        yellows = self._state[self.YELLOWS : self.YELLOWS + N_ALPHABET]
        for letter in np.flatnonzero(has_yellow):
            need = max(int(yellows[letter]), 1) + int(n_green[letter])
            ok &= self.word_counts[:, letter] >= need
        return ok

    def _pos_slice(self, letter):
        """Writable view of the n_letters positional slots for `letter`."""
        start = self.POS + self.n_letters * LETTER_INDEX[letter]
        return self._state[start : start + self.n_letters]

    def _position_column(self, pos):
        """Writable strided view over all 26 letters at board position `pos`."""
        return self._state[self.POS + pos : self.EXCEEDED : self.n_letters]

    def _update_state(self, guess):
        secret = self.secret_word

        # Greens first, so yellow bookkeeping below sees them.
        for i, c in enumerate(guess):
            if c != secret[i]:
                continue
            self._position_column(i)[:] = 0
            self._pos_slice(c)[i] = 2
            y = self.YELLOWS + LETTER_INDEX[c]
            if self._state[y] == 1:
                # The only yellow we knew about is now placed: forget the
                # "possible yellow" marks for this letter.
                self._state[y] = 0
                sl = self._pos_slice(c)
                sl[sl == 1] = 0
            elif self._state[y] == 2:
                self._state[y] = 1

        # Yellows and greys.
        for i, c in enumerate(guess):
            if c == secret[i]:
                continue
            sl = self._pos_slice(c)
            n_green = int((sl == 2).sum())
            n_secret_extra = secret.count(c) - n_green
            n_guess_extra = guess.count(c) - n_green
            if n_secret_extra > 0:
                n_yellows = min(n_secret_extra, n_guess_extra)
                y = self.YELLOWS + LETTER_INDEX[c]
                self._state[y] = max(int(self._state[y]), n_yellows)
                if not (sl == 1).any():
                    sl[sl == 0] = 1
                sl[i] = 0
            else:
                # Not in the secret, or every copy is already green.
                self.overused_letters.add(c)
                self._state[self.EXCEEDED + LETTER_INDEX[c]] = 1
            if guess.count(c) > secret.count(c):
                self._state[self.EXCEEDED + LETTER_INDEX[c]] = 1

    # -------------------------------------------------------------- rewards

    @staticmethod
    def _gt_n_poisson(L, n):
        """P(X >= n) for X ~ Poisson(L)."""
        return 1 - math.exp(-L) * sum(L**i / math.factorial(i) for i in range(n))

    def _gen_rewards(self, L, min_guesses):
        if not min_guesses:
            return [1.0] * self.max_attempts
        scale = (1 - 1 / len(self.solutions)) / self._gt_n_poisson(L, 1)
        return [
            self._gt_n_poisson(L, i) * scale
            for i in range(1, self.max_attempts + 1)
        ]
