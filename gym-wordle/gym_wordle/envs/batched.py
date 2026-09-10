"""N Wordle games stepped in lockstep as torch tensors, with auto-reset.

The env knows the solution list (it draws secrets from it). Nothing it
returns to the agent encodes that list: observations are letters, colours,
turn counters and the legal mask over the valid list.
"""
import math

import torch
import torch.nn.functional as F

from gym_wordle.envs.feedback import GREEN, GREY, N_ALPHABET, colour, consistent_mask, hard_mode_mask
from gym_wordle.envs.wordle_env import letter_matrix


def poisson_rewards(L, max_attempts):
    """r_k = P(X >= k) / P(X >= 1) for X ~ Poisson(L), k = 1..max_attempts."""
    def tail(k):
        return 1.0 - math.exp(-L) * sum(L**i / math.factorial(i) for i in range(k))
    return [tail(k) / tail(1) for k in range(1, max_attempts + 1)]


class BatchedWordle:
    def __init__(
        self,
        words,
        solutions,
        n_games,
        *,
        hard_mode=True,
        max_attempts=6,
        L=4.0,
        shaping_coef=0.0,
        device="cpu",
        seed=0,
    ):
        self.words = list(words)
        self.solutions = list(solutions)
        self.word_to_action = {w: i for i, w in enumerate(self.words)}
        self.n_letters = len(self.words[0])
        self.n_games = int(n_games)
        self.hard_mode = bool(hard_mode)
        self.max_attempts = int(max_attempts)
        self.shaping_coef = float(shaping_coef)
        self.device = torch.device(device)

        letters, counts = letter_matrix(self.words, self.n_letters)
        self.W_letters = torch.from_numpy(letters).to(self.device)          # (V,n) int8
        self.W_counts = torch.from_numpy(counts).to(self.device)            # (V,26) int8
        self.solution_idx = torch.tensor(
            [self.word_to_action[w] for w in self.solutions], dtype=torch.long, device=self.device
        )
        self.rewards = torch.tensor(poisson_rewards(L, self.max_attempts), dtype=torch.float32, device=self.device)
        self.gen = torch.Generator().manual_seed(int(seed))

        N, V, n, T = self.n_games, len(self.words), self.n_letters, self.max_attempts
        self.secret = torch.zeros(N, dtype=torch.long, device=self.device)
        self.turn = torch.zeros(N, dtype=torch.long, device=self.device)
        self.guess_letters = torch.zeros(N, T, n, dtype=torch.int8, device=self.device)
        self.colours = torch.zeros(N, T, n, dtype=torch.int8, device=self.device)
        self.guessed = torch.zeros(N, V, dtype=torch.bool, device=self.device)
        self.green_pos = torch.full((N, n), -1, dtype=torch.int8, device=self.device)
        self.min_count = torch.zeros(N, N_ALPHABET, dtype=torch.int8, device=self.device)
        self._rows = torch.arange(N, device=self.device)
        self._cand = None

    # ------------------------------------------------------------------ api

    def reset(self):
        self._reset_rows(torch.ones(self.n_games, dtype=torch.bool, device=self.device), self._draw_secrets())
        self._cand = None
        return self.observation()

    def set_secrets(self, indices):
        """Evaluation hook: restart every game with the given word indices."""
        indices = torch.as_tensor(indices, dtype=torch.long, device=self.device)
        assert indices.shape == (self.n_games,), indices.shape
        self._reset_rows(torch.ones(self.n_games, dtype=torch.bool, device=self.device), indices)
        self._cand = None
        return self.observation()

    def legal_mask(self):
        mask = ~self.guessed
        if self.hard_mode:
            mask = mask & hard_mode_mask(self.green_pos, self.min_count, self.W_letters, self.W_counts)
        return mask

    def step(self, actions):
        actions = torch.as_tensor(actions, dtype=torch.long, device=self.device)
        assert actions.shape == (self.n_games,), actions.shape
        if not bool(self.legal_mask()[self._rows, actions].all()):
            raise ValueError("illegal action passed to BatchedWordle.step")

        t = self.turn
        g = self.W_letters[actions]                                   # (N,n) int8
        c = colour(g, self.W_letters[self.secret])                    # (N,n) int8
        self.guess_letters[self._rows, t] = g
        self.colours[self._rows, t] = c
        self.guessed[self._rows, actions] = True
        self.turn = t + 1

        is_green = c == GREEN
        self.green_pos = torch.where(is_green, g, self.green_pos)
        onehot = F.one_hot(g.long(), N_ALPHABET).to(torch.int8)      # (N,n,26)
        seen = (onehot * (c != GREY).unsqueeze(-1).to(torch.int8)).sum(1, dtype=torch.int8)
        self.min_count = torch.maximum(self.min_count, seen)

        solved = actions == self.secret
        done = solved | (self.turn >= self.max_attempts)
        reward = torch.where(solved, self.rewards[self.turn - 1], torch.zeros((), device=self.device))
        if self.shaping_coef > 0:
            before = self._cand if self._cand is not None else torch.full_like(self.turn, len(self.words))
            after = self.candidates()
            reward = reward + self.shaping_coef * (torch.log2(before.float()) - torch.log2(after.float()))

        info = {"solved": solved, "n_guesses": self.turn.clone(), "secret": self.secret.clone()}
        self._reset_rows(done, self._draw_secrets())
        if self.shaping_coef > 0:
            self._cand = self.candidates()
        return self.observation(), reward, done, info

    def observation(self):
        N, T, n = self.guess_letters.shape
        letters = self.guess_letters.reshape(N, T * n).long()
        cols = self.colours.reshape(N, T * n).long()
        turn_idx = torch.arange(T, device=self.device).repeat_interleave(n).unsqueeze(0).expand(N, T * n)
        tokens = torch.stack([letters, cols, turn_idx], dim=-1)          # (N, T*n, 3)
        pad = turn_idx >= self.turn.unsqueeze(1)                          # cells of guesses not yet made
        return {"tokens": tokens, "pad": pad, "turn": self.turn.clone(), "mask": self.legal_mask()}

    def candidates(self):
        """Number of valid words consistent with every past guess, per game."""
        return consistent_mask(
            self.guess_letters, self.colours, self.turn, self.W_letters, self.W_counts
        ).sum(1)

    # -------------------------------------------------------------- helpers

    def _draw_secrets(self):
        idx = torch.randint(len(self.solutions), (self.n_games,), generator=self.gen)
        return self.solution_idx[idx.to(self.device)]

    def _reset_rows(self, rows, secrets):
        """Restart the games where rows is True, giving them `secrets`."""
        self.secret = torch.where(rows, secrets, self.secret)
        self.turn = torch.where(rows, torch.zeros_like(self.turn), self.turn)
        self.guess_letters[rows] = 0
        self.colours[rows] = 0
        self.guessed[rows] = False
        self.green_pos[rows] = -1
        self.min_count[rows] = 0
