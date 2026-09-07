"""Pieces shared by the DQN trainers: masking, replay, checkpoints, word features."""
import json
import os

import numpy as np
import torch

from gym_wordle.envs.wordle_env import LETTER_INDEX, N_ALPHABET


def invalid_from_mask(action_mask):
    """Indices of actions that are NOT allowed, from a boolean allow-mask."""
    return np.flatnonzero(~np.asarray(action_mask, dtype=bool)).astype(np.int64)


def masked_argmax(q, invalid):
    """Greedy action and its value after masking invalid actions with -inf.

    q: (B, N) tensor.
    invalid: one 1-D integer array applied to every row, a list/tuple of B
    such arrays (one per row), or a (B, N) bool tensor (True = invalid).
    Dispatch is on the type of `invalid`, so a batch of size one with a
    one-element list works.
    Returns (actions (B,1) int64, values (B,1)).
    """
    q = q.clone()
    if isinstance(invalid, torch.Tensor) and invalid.dtype == torch.bool:
        q.masked_fill_(invalid, float("-inf"))
    elif isinstance(invalid, (list, tuple)):
        assert len(invalid) == q.shape[0], "one invalid array per row"
        for row, inv in enumerate(invalid):
            inv = np.asarray(inv, dtype=np.int64)
            if inv.size:
                q[row, torch.as_tensor(inv, device=q.device)] = float("-inf")
    else:
        inv = np.asarray(invalid, dtype=np.int64)
        if inv.size:
            q[:, torch.as_tensor(inv, device=q.device)] = float("-inf")
    actions = torch.argmax(q, dim=1, keepdim=True)
    return actions, torch.gather(q, 1, actions)


class ReplayBuffer:
    """Ring buffer of transitions with optional proportional prioritization.

    `legal_next[i]` (bit-packed with `np.packbits`) holds the actions that
    are legal in `next_state[i]` (not yet guessed and not hard-mode
    illegal). It is used only to mask the bootstrap argmax over
    `next_state`; unpack a row (or rows) with `legal_mask`. Packing keeps
    per-row storage at ceil(n_actions / 8) bytes instead of one int64 per
    invalid action, which matters at the real ~13k-word action space.
    """

    def __init__(self, capacity, obs_size, n_actions, alpha=0.6, eps=1e-6, rng=None):
        self.capacity = capacity
        self.n_actions = n_actions
        self.alpha = alpha
        self.eps = eps
        self.rng = rng if rng is not None else np.random.default_rng()
        self.state = np.zeros((capacity, obs_size), dtype=np.int8)
        self.next_state = np.zeros((capacity, obs_size), dtype=np.int8)
        self.action = np.zeros(capacity, dtype=np.int64)
        self.reward = np.zeros(capacity, dtype=np.float32)
        self.done = np.zeros(capacity, dtype=bool)
        self.priority = np.zeros(capacity, dtype=np.float64)
        self.legal_next = np.zeros(
            (capacity, -(-n_actions // 8)), dtype=np.uint8
        )
        self.n_seen = 0

    def __len__(self):
        return min(self.n_seen, self.capacity)

    def add(self, state, action, reward, next_state, done, legal_next):
        n = len(self)
        i = self.n_seen % self.capacity
        self.state[i] = state
        self.next_state[i] = next_state
        self.action[i] = int(action)
        self.reward[i] = reward
        self.done[i] = bool(done)
        self.legal_next[i] = np.packbits(np.asarray(legal_next, dtype=bool))
        # Standard PER: a new transition gets the current max priority so it
        # is replayed at least once soon.
        self.priority[i] = self.priority[:n].max() if n > 0 else 1.0
        self.n_seen += 1
        return i

    def legal_mask(self, idx):
        """Unpack the legal-action mask for the given row indices.

        Returns a bool array of shape (len(idx), n_actions).
        """
        return np.unpackbits(
            self.legal_next[idx], axis=1, count=self.n_actions
        ).astype(bool)

    def sample_prioritized(self, batch_size, beta):
        n = len(self)
        p = self.priority[:n] ** self.alpha
        p /= p.sum()
        idx = self.rng.choice(n, size=batch_size, p=p)
        weights = (n * p[idx]) ** -beta
        weights /= (n * p.min()) ** -beta  # normalize by the largest possible weight
        return idx, weights.astype(np.float32)

    def sample_with_probs(self, batch_size, probs):
        n = len(self)
        probs = np.asarray(probs, dtype=np.float64)[:n]
        probs = probs / probs.sum()
        return self.rng.choice(n, size=batch_size, p=probs)

    def update_priority(self, idx, td_error):
        self.priority[idx] = np.maximum(self.eps, np.abs(np.asarray(td_error)))

    def keep_random_subset(self, k):
        """Drop everything except k rows chosen uniformly without replacement."""
        n = len(self)
        keep = self.rng.choice(n, size=min(k, n), replace=False)
        for name in ("state", "next_state", "action", "reward", "done", "priority", "legal_next"):
            arr = getattr(self, name)
            arr[: keep.size] = arr[keep]
        self.n_seen = int(keep.size)


def save_checkpoint(model, path):
    """state_dict at `path`, constructor kwargs at `path + '.json'`."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(model.state_dict(), path)
    with open(path + ".json", "w") as f:
        json.dump({"class": type(model).__name__, "kwargs": model.ctor_kwargs}, f)


def load_checkpoint(model_cls, path, device):
    with open(path + ".json") as f:
        meta = json.load(f)
    model = model_cls(**meta["kwargs"]).to(device)
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    return model


def word_feature_matrix(words, n_letters):
    """phi(w): one-hot letter per position (26*n_letters) then letter counts (26)."""
    n = len(words)
    phi = torch.zeros(n, N_ALPHABET * n_letters + N_ALPHABET)
    for row, w in enumerate(words):
        for pos, c in enumerate(w):
            phi[row, N_ALPHABET * pos + LETTER_INDEX[c]] = 1.0
            phi[row, N_ALPHABET * n_letters + LETTER_INDEX[c]] += 1.0
    return phi
