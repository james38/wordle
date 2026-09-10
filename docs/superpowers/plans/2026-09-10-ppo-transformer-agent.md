# PPO Transformer Agent Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A hand-rolled PPO agent with a transformer policy that learns Wordle from raw feedback tokens, trained on thousands of parallel games in a batched torch environment, finishing a run in minutes on one GPU.

**Architecture:** Pure batched feedback functions (`feedback.py`) feed a stateful `BatchedWordle` env that steps N games at once with auto-reset. A pre-norm transformer with a readout token consumes (letter, colour, slot) tokens and scores all 12947 words through a factored lexical head; a value MLP shares the readout. `ppo.py` holds storage, GAE and the clipped update; `train.py` holds the loop, logging, checkpoints, evaluation and CLI.

**Tech Stack:** Python ≥3.10, torch ≥2.2, numpy, gymnasium (already present), pytest. uv + hatchling packaging. No new runtime dependencies.

**Spec:** `docs/superpowers/specs/2026-09-10-ppo-transformer-agent-design.md`

## Global Constraints

- **The agent never sees the solution list.** No is-solution feature, no per-word parameter learned over the solution list, no candidate set or reward term computed over the solutions. Everything the policy consumes must be derivable from the valid-guess list plus observed feedback. The env may sample secrets from the solutions.
- **Hand-rolled PPO.** No RL library.
- **Nothing existing is modified** except `README.md`, `.gitignore` (add `runs/`), `pyproject.toml` (pytest marker + description) and package registration. `WordleEnv`, `dqn.py`, `transformer.py`, `common.py` and their tests are untouched.
- **No new runtime dependencies.** Dev group stays `pytest` only.
- **Hard mode is real Wordle's rule:** greens fixed in place, per-letter count ≥ max over past guesses of greens+yellows for that letter. Greys are not forbidden.
- **Reward:** solve on guess k pays `P(X≥k)/P(X≥1)` for X ~ Poisson(L), L = 4.0 default; fail pays 0; optional shaping `c·Δlog2(consistent valid words)`, default off.
- **Tests run on CPU** and finish fast; the one learning test is marked `slow` and stays under about 30 s.
- All work happens in `gym-wordle/` (the uv project). Run tests with `cd gym-wordle && uv run pytest -q`.
- **Commits:** one per task, message ends with the two trailers
  `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>` and
  `Claude-Session: https://claude.ai/code/session_01UiaAuvPC2EccbmCvHw1WTQ`. Never push.

## Reused interfaces (already in the repo, do not modify)

```python
from gym_wordle.envs.wordle_env import (
    load_word_lists,      # (valid_path, solution_path) -> (words: sorted union list[str], solutions: list[str])
    letter_matrix,        # (words, n_letters) -> (np.int8 (V,n) letter indices, np.int8 (V,26) counts)
    LETTER_INDEX, N_ALPHABET, DEFAULT_VALIDS, DEFAULT_SOLUTIONS,
)
from gym_wordle.agents.common import (
    save_checkpoint,      # (model, path): state_dict at path, {"class","kwargs": model.ctor_kwargs} at path+".json"
    load_checkpoint,      # (model_cls, path, device) -> model
    word_feature_matrix,  # (words, n_letters) -> torch.float (V, 26*n+26): one-hot letter per position, then counts
)
```

Test fixtures (already present): `tests/fixtures/words_small.json` =
`["apple","crane","stale","maple","delta"]`;
`tests/fixtures/valids_small.json` = 15 words; the sorted union has 19 words.
`tests/conftest.py` provides `env_paths` = `{"valid_words_path", "solution_words_path"}`.

## File map

| File | Responsibility |
|---|---|
| `gym_wordle/envs/feedback.py` | `colour`, `hard_mode_mask`, `consistent_mask` — pure, stateless |
| `gym_wordle/envs/batched.py` | `poisson_rewards`, `BatchedWordle` |
| `gym_wordle/agents/ppo/__init__.py` | exports |
| `gym_wordle/agents/ppo/model.py` | `WordlePolicy` |
| `gym_wordle/agents/ppo/ppo.py` | `PPOConfig`, `RolloutBuffer`, `ppo_update` |
| `gym_wordle/agents/ppo/train.py` | `collect`, `train`, `evaluate`, `main` |
| `tests/test_feedback.py`, `tests/test_batched_env.py`, `tests/test_ppo_model.py`, `tests/test_ppo.py` | tests |
| `README.md`, `.gitignore`, `pyproject.toml` | docs, ignore `runs/`, `slow` marker |

---

### Task 1: `feedback.colour`

**Files:**
- Create: `gym-wordle/gym_wordle/envs/feedback.py`
- Test: `gym-wordle/tests/test_feedback.py`

**Interfaces:**
- Produces: `GREY, YELLOW, GREEN = 0, 1, 2`; `colour(guess_letters, secret_letters) -> int8` with shapes broadcast to a common `(..., n)`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_feedback.py
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
        ("eerie", "there", [1, 0, 0, 0, 2]),
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
```

- [ ] **Step 2: Run to verify failure**

Run: `cd gym-wordle && uv run pytest tests/test_feedback.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'gym_wordle.envs.feedback'`

- [ ] **Step 3: Implement `colour`**

```python
# gym_wordle/envs/feedback.py
"""Pure batched Wordle feedback functions. Torch, device-agnostic, stateless.

Every function takes tensors and returns tensors on the same device. None
of them look at the solution list: they see letters, colours and the
valid-word matrices only.
"""
import torch

GREY, YELLOW, GREEN = 0, 1, 2
N_ALPHABET = 26


def colour(guess_letters, secret_letters):
    """Standard Wordle colouring, batched. Shapes broadcast to (..., n).

    Greens first. Then, scanning positions left to right, a non-green cell
    is yellow iff the secret still has an unmatched copy of that letter;
    each yellow consumes one copy. Returns int8 in {GREY, YELLOW, GREEN}.
    """
    guess, secret = torch.broadcast_tensors(guess_letters.long(), secret_letters.long())
    shape = guess.shape
    n = shape[-1]
    guess = guess.reshape(-1, n)
    secret = secret.reshape(-1, n)
    b = guess.shape[0]
    dev = guess.device

    green = guess == secret
    remaining = torch.zeros(b, N_ALPHABET, dtype=torch.int32, device=dev)
    remaining.scatter_add_(1, secret, torch.ones_like(secret, dtype=torch.int32))
    remaining.scatter_add_(1, guess, -green.to(torch.int32))

    out = torch.full((b, n), GREY, dtype=torch.int8, device=dev)
    out[green] = GREEN
    rows = torch.arange(b, device=dev)
    for p in range(n):
        letter = guess[:, p]
        avail = (remaining[rows, letter] > 0) & ~green[:, p]
        out[avail, p] = YELLOW
        remaining[rows, letter] -= avail.to(torch.int32)
    return out.reshape(shape)
```

- [ ] **Step 4: Run tests**

Run: `cd gym-wordle && uv run pytest tests/test_feedback.py -q`
Expected: 8 passed

- [ ] **Step 5: Commit**

```bash
git add gym-wordle/gym_wordle/envs/feedback.py gym-wordle/tests/test_feedback.py
git commit -m "feat(env): batched Wordle colouring with independent reference test"
```
(append the two trailers)

---

### Task 2: `hard_mode_mask` and `consistent_mask`

**Files:**
- Modify: `gym-wordle/gym_wordle/envs/feedback.py`
- Test: `gym-wordle/tests/test_feedback.py`

**Interfaces:**
- Produces: `hard_mode_mask(green_pos (B,n) int8, min_count (B,26) int8, W_letters (V,n) int8, W_counts (V,26) int8) -> (B,V) bool`;
  `consistent_mask(guess_letters (B,T,n) int8, colours (B,T,n) int8, n_guesses (B,) long, W_letters, W_counts) -> (B,V) bool`.

- [ ] **Step 1: Write the failing tests** (append to `tests/test_feedback.py`)

```python
from gym_wordle.envs.feedback import consistent_mask, hard_mode_mask


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
```

- [ ] **Step 2: Run to verify failure**

Run: `cd gym-wordle && uv run pytest tests/test_feedback.py -q`
Expected: FAIL with `ImportError: cannot import name 'consistent_mask'`

- [ ] **Step 3: Implement** (append to `feedback.py`)

```python
def hard_mode_mask(green_pos, min_count, W_letters, W_counts):
    """Real Wordle hard mode over the whole word list.

    green_pos (B,n) int8, -1 where no green is known; min_count (B,26) int8.
    Word w is legal for game b iff every known green matches and, for every
    letter, w has at least min_count[b, l] copies. Greys are not forbidden.
    Returns (B,V) bool.
    """
    has_green = green_pos >= 0                                        # (B,n)
    match = W_letters.unsqueeze(0) == green_pos.unsqueeze(1)          # (B,V,n)
    r1 = (match | ~has_green.unsqueeze(1)).all(-1)                    # (B,V)
    r2 = (W_counts.unsqueeze(0) >= min_count.unsqueeze(1)).all(-1)    # (B,V)
    return r1 & r2


def consistent_mask(guess_letters, colours, n_guesses, W_letters, W_counts):
    """Words whose colouring against every past guess reproduces the record.

    guess_letters, colours: (B,T,n) int8; n_guesses (B,) long says how many
    of the T rows are real. Uses the count characterisation of Wordle
    colouring: for a past guess g with colours c, a word w is consistent iff
      1. for each position p, (w[p] == g[p]) == (c[p] == GREEN), and
      2. for each letter l in g, with k = greens+yellows of l in g:
         count_w(l) == k if g has a grey copy of l, else count_w(l) >= k.
    Returns (B,V) bool. Never fed to the policy.
    """
    B, T, n = guess_letters.shape
    dev = guess_letters.device
    V = W_letters.shape[0]
    ok = torch.ones(B, V, dtype=torch.bool, device=dev)
    WC_T = W_counts.t().contiguous()                                  # (26,V) int8
    for t in range(T):
        active = n_guesses > t                                        # (B,)
        if not bool(active.any()):
            break
        g = guess_letters[:, t].long()                                # (B,n)
        c = colours[:, t]                                             # (B,n) int8
        is_green = c == GREEN
        match = W_letters.unsqueeze(0) == g.unsqueeze(1).to(W_letters.dtype)  # (B,V,n)
        r1 = (match == is_green.unsqueeze(1)).all(-1)                 # (B,V)

        onehot = torch.nn.functional.one_hot(g, N_ALPHABET).to(torch.int8)   # (B,n,26)
        need = (onehot * (c != GREY).unsqueeze(-1).to(torch.int8)).sum(1, dtype=torch.int8)    # (B,26)
        has_grey = (onehot * (c == GREY).unsqueeze(-1).to(torch.int8)).sum(1) > 0            # (B,26)
        need_g = need.gather(1, g)                                    # (B,n)
        exact_g = has_grey.gather(1, g)                               # (B,n)
        cnt_g = WC_T[g]                                               # (B,n,V) int8
        ge = cnt_g >= need_g.unsqueeze(-1)
        eq = cnt_g == need_g.unsqueeze(-1)
        r2 = torch.where(exact_g.unsqueeze(-1), eq, ge).all(1)        # (B,V)

        ok &= ~active.unsqueeze(1) | (r1 & r2)
    return ok
```

- [ ] **Step 4: Run tests**

Run: `cd gym-wordle && uv run pytest tests/test_feedback.py -q`
Expected: 13 passed

- [ ] **Step 5: Commit**

```bash
git add gym-wordle/gym_wordle/envs/feedback.py gym-wordle/tests/test_feedback.py
git commit -m "feat(env): vectorised real hard-mode mask and consistency mask"
```
(append the two trailers)

---

### Task 3: `BatchedWordle` core — reset, step, reward, auto-reset, legal mask

**Files:**
- Create: `gym-wordle/gym_wordle/envs/batched.py`
- Test: `gym-wordle/tests/test_batched_env.py`

**Interfaces:**
- Consumes: `colour`, `hard_mode_mask`, `GREEN`, `YELLOW` from `feedback.py`; `letter_matrix` from `wordle_env.py`.
- Produces: `poisson_rewards(L, max_attempts) -> list[float]`;
  `BatchedWordle(words, solutions, n_games, *, hard_mode=True, max_attempts=6, L=4.0, shaping_coef=0.0, device="cpu", seed=0)` with attributes `words, solutions, word_to_action, n_letters, n_games, max_attempts, device, W_letters, W_counts, solution_idx, rewards, secret, turn, guess_letters, colours, guessed, green_pos, min_count` and methods `reset() -> obs`, `set_secrets(indices) -> obs`, `step(actions) -> (obs, reward, done, info)`, `legal_mask() -> (N,V) bool`. `observation()` and `candidates()` arrive in Task 4; in this task `observation()` returns `{"mask": self.legal_mask()}` only.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_batched_env.py
import pytest
import torch

from gym_wordle.envs.batched import BatchedWordle, poisson_rewards
from gym_wordle.envs.feedback import hard_mode_mask
from gym_wordle.envs.wordle_env import load_word_lists


@pytest.fixture
def lists(env_paths):
    return load_word_lists(env_paths["valid_words_path"], env_paths["solution_words_path"])


def make(lists, n=4, **kw):
    words, solutions = lists
    return BatchedWordle(words, solutions, n, seed=0, **kw)


def test_poisson_rewards_table():
    r = poisson_rewards(4.0, 6)
    assert r[0] == pytest.approx(1.0)
    assert r == pytest.approx([1.0, 0.9254, 0.7761, 0.5771, 0.3781, 0.2189], abs=1e-4)
    assert r == sorted(r, reverse=True)


def test_construction_and_reset(lists):
    env = make(lists)
    words, solutions = lists
    assert env.W_letters.shape == (len(words), 5)
    env.reset()
    assert env.turn.tolist() == [0, 0, 0, 0]
    assert not env.guessed.any()
    assert (env.green_pos == -1).all() and not env.min_count.any()
    sol_set = set(env.solution_idx.tolist())
    assert all(s in sol_set for s in env.secret.tolist())


def test_same_seed_same_secrets(lists):
    a = make(lists, n=32); b = make(lists, n=32)
    a.reset(); b.reset()
    assert torch.equal(a.secret, b.secret)


def test_set_secrets(lists):
    env = make(lists, n=2)
    idx = torch.tensor([env.word_to_action["crane"], env.word_to_action["delta"]])
    env.set_secrets(idx)
    assert torch.equal(env.secret, idx)


def test_solve_on_first_guess_pays_one_and_resets(lists):
    env = make(lists, n=2)
    env.set_secrets(torch.tensor([env.word_to_action["crane"], env.word_to_action["delta"]]))
    a = torch.tensor([env.word_to_action["crane"], env.word_to_action["slate"]])
    obs, reward, done, info = env.step(a)
    assert done.tolist() == [True, False]
    assert info["solved"].tolist() == [True, False]
    assert reward[0].item() == pytest.approx(1.0) and reward[1].item() == 0.0
    assert info["n_guesses"][0].item() == 1
    assert info["secret"][0].item() == env.word_to_action["crane"]
    # row 0 auto-reset, row 1 untouched
    assert env.turn.tolist() == [0, 1]
    assert not env.guessed[0].any()
    assert env.guessed[1, env.word_to_action["slate"]]
    assert (env.green_pos[0] == -1).all()


def test_reward_schedule_by_guess_number(lists):
    env = make(lists, n=1, hard_mode=False)
    words, _ = lists
    secret = env.word_to_action["crane"]
    fillers = [w for w in words if w != "crane"]
    for k in range(1, 7):
        env.set_secrets(torch.tensor([secret]))
        for w in fillers[: k - 1]:
            _, r, d, _ = env.step(torch.tensor([env.word_to_action[w]]))
            assert r.item() == 0.0 and not d.item()
        _, r, d, info = env.step(torch.tensor([secret]))
        assert d.item() and info["solved"].item()
        assert r.item() == pytest.approx(env.rewards[k - 1].item())


def test_fail_pays_zero_and_resets(lists):
    env = make(lists, n=1, hard_mode=False)
    words, _ = lists
    env.set_secrets(torch.tensor([env.word_to_action["crane"]]))
    fillers = [w for w in words if w != "crane"][:6]
    for i, w in enumerate(fillers):
        _, r, d, info = env.step(torch.tensor([env.word_to_action[w]]))
        assert r.item() == 0.0
        assert d.item() == (i == 5)
    assert not info["solved"].item() and info["n_guesses"].item() == 6
    assert env.turn.item() == 0


def test_played_word_never_legal_again_and_illegal_raises(lists):
    env = make(lists, n=1, hard_mode=False)
    env.reset()
    a = torch.tensor([env.word_to_action["puppy"]])
    env.step(a)
    assert not env.legal_mask()[0, a.item()]
    with pytest.raises(ValueError):
        env.step(a)


def test_legal_mask_hard_vs_normal(lists):
    hard = make(lists, n=1, hard_mode=True)
    soft = make(lists, n=1, hard_mode=False)
    for env in (hard, soft):
        env.set_secrets(torch.tensor([env.word_to_action["crane"]]))
        env.step(torch.tensor([env.word_to_action["slate"]]))   # a, e green
    expected_hard = hard_mode_mask(hard.green_pos, hard.min_count, hard.W_letters, hard.W_counts) & ~hard.guessed
    assert torch.equal(hard.legal_mask(), expected_hard)
    assert torch.equal(soft.legal_mask(), ~soft.guessed)
    assert hard.legal_mask().sum() < soft.legal_mask().sum()
    assert hard.legal_mask()[0, hard.word_to_action["crane"]]


def test_hard_mode_bookkeeping(lists):
    env = make(lists, n=1)
    env.set_secrets(torch.tensor([env.word_to_action["apple"]]))
    env.step(torch.tensor([env.word_to_action["paper"]]))        # colours 1,1,2,1,0
    from gym_wordle.envs.wordle_env import LETTER_INDEX as L
    assert env.green_pos[0].tolist() == [-1, -1, L["p"], -1, -1]
    mc = env.min_count[0]
    assert mc[L["p"]] == 2 and mc[L["a"]] == 1 and mc[L["e"]] == 1 and mc[L["r"]] == 0
```

- [ ] **Step 2: Run to verify failure**

Run: `cd gym-wordle && uv run pytest tests/test_batched_env.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'gym_wordle.envs.batched'`

- [ ] **Step 3: Implement**

```python
# gym_wordle/envs/batched.py
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
        return {"mask": self.legal_mask()}          # extended in Task 4

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
```

- [ ] **Step 4: Run tests**

Run: `cd gym-wordle && uv run pytest tests/test_batched_env.py -q`
Expected: 10 passed

- [ ] **Step 5: Commit**

```bash
git add gym-wordle/gym_wordle/envs/batched.py gym-wordle/tests/test_batched_env.py
git commit -m "feat(env): BatchedWordle — N games in lockstep with auto-reset and Poisson reward"
```
(append the two trailers)

---

### Task 4: observation tokens, shaping reward, package exports

**Files:**
- Modify: `gym-wordle/gym_wordle/envs/batched.py` (replace `observation`)
- Modify: `gym-wordle/gym_wordle/envs/__init__.py` (export `BatchedWordle`)
- Test: `gym-wordle/tests/test_batched_env.py`

**Interfaces:**
- Produces: `observation() -> {"tokens": (N, T*n, 3) long, "pad": (N, T*n) bool, "turn": (N,) long, "mask": (N,V) bool}`. Slot s = 5·turn + position; `tokens[..., 0]` letter, `[..., 1]` colour, `[..., 2]` turn index.

- [ ] **Step 1: Write the failing tests** (append to `tests/test_batched_env.py`)

```python
def test_observation_layout(lists):
    env = make(lists, n=2)
    env.set_secrets(torch.tensor([env.word_to_action["apple"], env.word_to_action["crane"]]))
    obs = env.observation()
    assert obs["tokens"].shape == (2, 30, 3) and obs["tokens"].dtype == torch.long
    assert obs["pad"].shape == (2, 30) and obs["pad"].all()
    assert obs["turn"].tolist() == [0, 0]
    assert obs["mask"].shape == (2, len(lists[0]))

    obs, *_ = env.step(torch.tensor([env.word_to_action["paper"], env.word_to_action["slate"]]))
    from gym_wordle.envs.wordle_env import LETTER_INDEX as L
    row = obs["tokens"][0]
    assert row[:5, 0].tolist() == [L[c] for c in "paper"]
    assert row[:5, 1].tolist() == [1, 1, 2, 1, 0]
    assert row[:5, 2].tolist() == [0] * 5
    assert row[5:10, 2].tolist() == [1] * 5
    assert obs["pad"][0].tolist() == [False] * 5 + [True] * 25
    assert obs["turn"].tolist() == [1, 1]
    assert torch.equal(obs["mask"], env.legal_mask())


def test_observation_after_auto_reset_is_fresh(lists):
    env = make(lists, n=1)
    env.set_secrets(torch.tensor([env.word_to_action["crane"]]))
    obs, _, done, _ = env.step(torch.tensor([env.word_to_action["crane"]]))
    assert done.item()
    assert obs["pad"].all() and obs["turn"].item() == 0 and obs["mask"].all()


def test_candidates_and_shaping(lists):
    words, _ = lists
    env = make(lists, n=1, hard_mode=False, shaping_coef=0.5)
    env.set_secrets(torch.tensor([env.word_to_action["crane"]]))
    assert env.candidates().item() == len(words)
    _, r, _, _ = env.step(torch.tensor([env.word_to_action["slate"]]))
    after = env.candidates().item()
    assert 0 < after < len(words)
    import math
    assert r.item() == pytest.approx(0.5 * (math.log2(len(words)) - math.log2(after)))
    _, r2, d, _ = env.step(torch.tensor([env.word_to_action["crane"]]))
    assert d.item()
    assert r2.item() == pytest.approx(env.rewards[1].item() + 0.5 * (math.log2(after) - 0.0))


def test_shaping_off_gives_zero_intermediate_reward(lists):
    env = make(lists, n=1, hard_mode=False)
    env.set_secrets(torch.tensor([env.word_to_action["crane"]]))
    _, r, _, _ = env.step(torch.tensor([env.word_to_action["slate"]]))
    assert r.item() == 0.0


def test_batched_wordle_exported():
    from gym_wordle.envs import BatchedWordle as B
    assert B is BatchedWordle
```

- [ ] **Step 2: Run to verify failure**

Run: `cd gym-wordle && uv run pytest tests/test_batched_env.py -q`
Expected: FAIL — `KeyError: 'tokens'` and `ImportError` for the export

- [ ] **Step 3: Implement**

Replace `observation` in `batched.py`:

```python
    def observation(self):
        N, T, n = self.guess_letters.shape
        letters = self.guess_letters.reshape(N, T * n).long()
        cols = self.colours.reshape(N, T * n).long()
        turn_idx = torch.arange(T, device=self.device).repeat_interleave(n).unsqueeze(0).expand(N, T * n)
        tokens = torch.stack([letters, cols, turn_idx], dim=-1)          # (N, T*n, 3)
        pad = turn_idx >= self.turn.unsqueeze(1)                          # cells of guesses not yet made
        return {"tokens": tokens, "pad": pad, "turn": self.turn.clone(), "mask": self.legal_mask()}
```

Replace `gym_wordle/envs/__init__.py`:

```python
from gym_wordle.envs.batched import BatchedWordle
from gym_wordle.envs.wordle_env import WordleEnv

__all__ = ["BatchedWordle", "WordleEnv"]
```

- [ ] **Step 4: Run the whole suite**

Run: `cd gym-wordle && uv run pytest -q`
Expected: all pass (73 old + 13 + 15 new = 101)

- [ ] **Step 5: Commit**

```bash
git add gym-wordle/gym_wordle/envs/batched.py gym-wordle/gym_wordle/envs/__init__.py gym-wordle/tests/test_batched_env.py
git commit -m "feat(env): token observation, information-gain shaping, export BatchedWordle"
```
(append the two trailers)

---

### Task 5: `WordlePolicy`

**Files:**
- Create: `gym-wordle/gym_wordle/agents/ppo/__init__.py`
- Create: `gym-wordle/gym_wordle/agents/ppo/model.py`
- Test: `gym-wordle/tests/test_ppo_model.py`

**Interfaces:**
- Consumes: observation dict from Task 4; `word_feature_matrix`, `save_checkpoint`, `load_checkpoint` from `common.py`.
- Produces: `WordlePolicy(n_words, word_features=None, d_model=128, n_heads=4, d_ff=512, n_layers=3, max_turns=6, n_letters=5)`; `forward(obs) -> (logits (N,V), value (N,))`; `act(obs, greedy=False) -> (action, log_prob, entropy, value)`; `evaluate_actions(obs, actions) -> (log_prob, entropy, value)`; attribute `ctor_kwargs`; buffer `word_features`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_ppo_model.py
import pytest
import torch

from gym_wordle.agents.common import load_checkpoint, save_checkpoint, word_feature_matrix
from gym_wordle.agents.ppo.model import WordlePolicy
from gym_wordle.envs.batched import BatchedWordle
from gym_wordle.envs.wordle_env import load_word_lists


@pytest.fixture
def setup(env_paths):
    words, solutions = load_word_lists(env_paths["valid_words_path"], env_paths["solution_words_path"])
    env = BatchedWordle(words, solutions, 4, seed=0)
    torch.manual_seed(0)
    policy = WordlePolicy(len(words), word_feature_matrix(words, 5), d_model=32, n_heads=4, d_ff=64, n_layers=2)
    env.reset()
    env.step(torch.tensor([env.word_to_action[w] for w in ["slate", "crane", "paper", "least"]]))
    return words, env, policy


def test_shapes_and_masking(setup):
    words, env, policy = setup
    obs = env.observation()
    logits, value = policy(obs)
    assert logits.shape == (4, len(words)) and value.shape == (4,)
    assert torch.isinf(logits[~obs["mask"]]).all() and (logits[~obs["mask"]] < 0).all()
    assert torch.isfinite(logits[obs["mask"]]).all()
    probs = torch.softmax(logits, -1)
    assert (probs[~obs["mask"]] == 0).all()
    assert torch.allclose(probs.sum(-1), torch.ones(4))


def test_entropy_is_over_legal_subset(setup):
    words, env, policy = setup
    obs = env.observation()
    _, _, ent, _ = policy.act(obs)
    n_legal = obs["mask"].sum(-1).float()
    assert (ent <= torch.log(n_legal) + 1e-5).all()
    assert (ent >= 0).all()


def test_act_and_evaluate_actions_agree(setup):
    words, env, policy = setup
    obs = env.observation()
    a, lp, ent, v = policy.act(obs)
    assert obs["mask"][torch.arange(4), a].all(), "sampled actions must be legal"
    lp2, ent2, v2 = policy.evaluate_actions(obs, a)
    assert torch.allclose(lp, lp2) and torch.allclose(ent, ent2) and torch.allclose(v, v2)
    g, *_ = policy.act(obs, greedy=True)
    logits, _ = policy(obs)
    assert torch.equal(g, logits.argmax(-1))


def test_padding_tokens_do_not_affect_output(setup):
    words, env, policy = setup
    policy.eval()
    obs = env.observation()
    logits, value = policy(obs)
    noisy = dict(obs)
    tokens = obs["tokens"].clone()
    tokens[:, 5:, 0] = 25            # scribble on padded cells
    tokens[:, 5:, 1] = 2
    noisy["tokens"] = tokens
    logits2, value2 = policy(noisy)
    assert torch.allclose(logits[obs["mask"]], logits2[obs["mask"]], atol=1e-5)
    assert torch.allclose(value, value2, atol=1e-5)


def test_every_parameter_gets_a_gradient(setup):
    words, env, policy = setup
    obs = env.observation()
    a, lp, ent, v = policy.act(obs)
    (lp.mean() + ent.mean() + v.mean()).backward()
    for name, p in policy.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), name


def test_checkpoint_roundtrip(setup, tmp_path):
    words, env, policy = setup
    policy.eval()
    path = str(tmp_path / "policy.pt")
    save_checkpoint(policy, path)
    loaded = load_checkpoint(WordlePolicy, path, "cpu").eval()
    assert torch.equal(loaded.word_features, policy.word_features)
    obs = env.observation()
    l1, v1 = policy(obs)
    l2, v2 = loaded(obs)
    assert torch.allclose(l1[obs["mask"]], l2[obs["mask"]]) and torch.allclose(v1, v2)


def test_parameter_count_is_small(setup):
    words, env, policy = setup
    full = WordlePolicy(12947, torch.zeros(12947, 156))
    assert sum(p.numel() for p in full.parameters()) < 1_000_000
```

- [ ] **Step 2: Run to verify failure**

Run: `cd gym-wordle && uv run pytest tests/test_ppo_model.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'gym_wordle.agents.ppo'`

- [ ] **Step 3: Implement**

`gym_wordle/agents/ppo/__init__.py`:

```python
from gym_wordle.agents.ppo.model import WordlePolicy

__all__ = ["WordlePolicy"]
```

`gym_wordle/agents/ppo/model.py`:

```python
"""Transformer policy over raw feedback tokens with a factored lexical head.

The policy sees letters, colours and slot positions of its own guesses, and
a fixed lexical feature vector per valid word. It never sees which words
are solutions.
"""
import math

import torch
import torch.nn as nn
from torch.distributions import Categorical

from gym_wordle.envs.feedback import N_ALPHABET

N_COLOURS = 3


class WordlePolicy(nn.Module):
    def __init__(
        self,
        n_words,
        word_features=None,
        d_model=128,
        n_heads=4,
        d_ff=512,
        n_layers=3,
        max_turns=6,
        n_letters=5,
    ):
        super().__init__()
        n_feat = N_ALPHABET * n_letters + N_ALPHABET
        if word_features is None:
            word_features = torch.zeros(n_words, n_feat)
        assert tuple(word_features.shape) == (n_words, n_feat), word_features.shape
        self.register_buffer("word_features", word_features.float())

        self.letter_emb = nn.Embedding(N_ALPHABET, d_model)
        self.colour_emb = nn.Embedding(N_COLOURS, d_model)
        self.slot_emb = nn.Embedding(max_turns * n_letters, d_model)
        self.turn_emb = nn.Embedding(max_turns + 1, d_model)
        self.readout = nn.Parameter(torch.randn(d_model) * 0.02)

        layer = nn.TransformerEncoderLayer(
            d_model, n_heads, d_ff, dropout=0.0, activation="gelu",
            batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.word_tower = nn.Sequential(nn.Linear(n_feat, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        self.value_head = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, 1))
        self.d_model = d_model
        self.ctor_kwargs = dict(
            n_words=n_words, d_model=d_model, n_heads=n_heads, d_ff=d_ff,
            n_layers=n_layers, max_turns=max_turns, n_letters=n_letters,
        )

    def forward(self, obs):
        tokens, pad, turn, mask = obs["tokens"], obs["pad"], obs["turn"], obs["mask"]
        B, S, _ = tokens.shape
        slots = torch.arange(S, device=tokens.device).unsqueeze(0).expand(B, S)
        x = self.letter_emb(tokens[..., 0]) + self.colour_emb(tokens[..., 1]) + self.slot_emb(slots)
        r = (self.readout + self.turn_emb(turn)).unsqueeze(1)                   # (B,1,d)
        x = torch.cat([r, x], dim=1)                                             # (B,1+S,d)
        kpm = torch.cat([torch.zeros(B, 1, dtype=torch.bool, device=pad.device), pad], dim=1)
        h = self.encoder(x, src_key_padding_mask=kpm)
        z = self.norm(h[:, 0])                                                   # (B,d)
        u = self.word_tower(self.word_features)                                  # (V,d)
        logits = (z @ u.t()) / math.sqrt(self.d_model)
        logits = logits.masked_fill(~mask, float("-inf"))
        value = self.value_head(z).squeeze(-1)
        return logits, value

    def act(self, obs, greedy=False):
        logits, value = self(obs)
        dist = Categorical(logits=logits)
        action = logits.argmax(-1) if greedy else dist.sample()
        return action, dist.log_prob(action), dist.entropy(), value

    def evaluate_actions(self, obs, actions):
        logits, value = self(obs)
        dist = Categorical(logits=logits)
        return dist.log_prob(actions), dist.entropy(), value
```

- [ ] **Step 4: Run tests**

Run: `cd gym-wordle && uv run pytest tests/test_ppo_model.py -q`
Expected: 7 passed, no warnings about nested tensors

- [ ] **Step 5: Commit**

```bash
git add gym-wordle/gym_wordle/agents/ppo/__init__.py gym-wordle/gym_wordle/agents/ppo/model.py gym-wordle/tests/test_ppo_model.py
git commit -m "feat(ppo): transformer policy with readout token and factored lexical head"
```
(append the two trailers)

---

### Task 6: `RolloutBuffer`, GAE, `ppo_update`

**Files:**
- Create: `gym-wordle/gym_wordle/agents/ppo/ppo.py`
- Test: `gym-wordle/tests/test_ppo.py`

**Interfaces:**
- Consumes: `WordlePolicy.evaluate_actions`; observation dict keys.
- Produces:
  `PPOConfig` dataclass (`clip=0.2, vf_coef=0.5, ent_coef=0.01, epochs=4, minibatch=4096, max_grad_norm=0.5, target_kl=0.02, gamma=1.0, lam=0.95`);
  `RolloutBuffer(T, N, n_tokens, n_words, device)` with `store(t, obs, action, log_prob, value, reward, done)`, `compute_gae(last_value, gamma, lam)`, `flat_obs(idx) -> obs dict`, tensors `tokens, pad, turn, mask, actions, log_probs, values, rewards, dones, advantages, returns`;
  `ppo_update(policy, optimizer, buf, cfg) -> dict` with keys `policy_loss, value_loss, entropy, approx_kl, clip_frac, explained_variance, epochs_run`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_ppo.py
import pytest
import torch

from gym_wordle.agents.common import word_feature_matrix
from gym_wordle.agents.ppo.model import WordlePolicy
from gym_wordle.agents.ppo.ppo import PPOConfig, RolloutBuffer, ppo_update
from gym_wordle.envs.batched import BatchedWordle
from gym_wordle.envs.wordle_env import load_word_lists


def test_gae_hand_computed():
    buf = RolloutBuffer(T=3, N=1, n_tokens=30, n_words=5, device="cpu")
    buf.rewards[:, 0] = torch.tensor([0.0, 1.0, 0.0])
    buf.dones[:, 0] = torch.tensor([False, True, False])
    buf.values[:, 0] = torch.tensor([0.5, 0.2, 0.3])
    buf.compute_gae(last_value=torch.tensor([0.4]), gamma=1.0, lam=0.95)
    assert buf.advantages[:, 0].tolist() == pytest.approx([0.46, 0.8, 0.1], abs=1e-6)
    assert buf.returns[:, 0].tolist() == pytest.approx([0.96, 1.0, 0.4], abs=1e-6)


def test_gae_zero_when_value_is_perfect():
    buf = RolloutBuffer(T=2, N=1, n_tokens=30, n_words=5, device="cpu")
    buf.rewards[:, 0] = torch.tensor([0.0, 1.0])
    buf.dones[:, 0] = torch.tensor([False, True])
    buf.values[:, 0] = torch.tensor([1.0, 1.0])
    buf.compute_gae(last_value=torch.tensor([123.0]), gamma=1.0, lam=0.95)
    assert torch.allclose(buf.advantages, torch.zeros(2, 1))


@pytest.fixture
def small(env_paths):
    words, solutions = load_word_lists(env_paths["valid_words_path"], env_paths["solution_words_path"])
    env = BatchedWordle(words, solutions, 8, seed=0)
    torch.manual_seed(0)
    policy = WordlePolicy(len(words), word_feature_matrix(words, 5), d_model=32, n_heads=4, d_ff=64, n_layers=1)
    return words, env, policy


def fill_buffer(env, policy, T):
    buf = RolloutBuffer(T, env.n_games, env.max_attempts * env.n_letters, len(env.words), "cpu")
    obs = env.reset()
    with torch.no_grad():
        for t in range(T):
            a, lp, ent, v = policy.act(obs)
            next_obs, r, d, _ = env.step(a)
            buf.store(t, obs, a, lp, v, r, d)
            obs = next_obs
        _, last_v = policy(obs)
    buf.compute_gae(last_v, 1.0, 0.95)
    return buf


def test_store_and_flat_obs_roundtrip(small):
    words, env, policy = small
    buf = fill_buffer(env, policy, T=4)
    idx = torch.tensor([0, 9, 31])
    ob = buf.flat_obs(idx)
    assert ob["tokens"].shape == (3, 30, 3) and ob["mask"].shape == (3, len(words))
    assert torch.equal(ob["tokens"][1], buf.tokens[1, 1])         # flat 9 -> (t=1, n=1)
    assert torch.equal(ob["turn"][2], buf.turn[3, 7])
    assert buf.mask[torch.arange(4).unsqueeze(1), torch.arange(8).unsqueeze(0), buf.actions].all()


def test_one_update_runs_and_is_finite(small):
    words, env, policy = small
    buf = fill_buffer(env, policy, T=4)
    opt = torch.optim.Adam(policy.parameters(), lr=1e-3)
    cfg = PPOConfig(minibatch=16, epochs=2, target_kl=None)
    stats = ppo_update(policy, opt, buf, cfg)
    for k in ["policy_loss", "value_loss", "entropy", "approx_kl", "clip_frac", "explained_variance"]:
        assert k in stats and torch.isfinite(torch.tensor(stats[k])), k
    assert stats["epochs_run"] == 2
    assert 0.0 <= stats["clip_frac"] <= 1.0


def test_update_moves_logprobs_along_advantages(small):
    """One small step must raise log-probs where advantage is positive and lower them where negative."""
    words, env, policy = small
    buf = fill_buffer(env, policy, T=4)
    torch.manual_seed(1)
    buf.advantages = torch.randn_like(buf.advantages)
    n = buf.T * buf.N
    adv = buf.advantages.reshape(n)
    adv_norm = (adv - adv.mean()) / (adv.std() + 1e-8)
    before = [p.detach().clone() for p in policy.parameters()]
    with torch.no_grad():
        lp_before, _, _ = policy.evaluate_actions(buf.flat_obs(torch.arange(n)), buf.actions.reshape(n))
    opt = torch.optim.Adam(policy.parameters(), lr=1e-4)
    ppo_update(policy, opt, buf, PPOConfig(minibatch=n, epochs=1, ent_coef=0.0, vf_coef=0.0, target_kl=None))
    with torch.no_grad():
        lp_after, _, _ = policy.evaluate_actions(buf.flat_obs(torch.arange(n)), buf.actions.reshape(n))
    assert any(not torch.equal(a, b) for a, b in zip(before, policy.parameters()))
    assert ((lp_after - lp_before) * adv_norm).mean() > 0


def test_target_kl_stops_epochs_early(small):
    words, env, policy = small
    buf = fill_buffer(env, policy, T=4)
    opt = torch.optim.Adam(policy.parameters(), lr=1e-1)   # huge lr -> big KL
    stats = ppo_update(policy, opt, buf, PPOConfig(minibatch=32, epochs=8, target_kl=1e-6))
    assert stats["epochs_run"] < 8
```

- [ ] **Step 2: Run to verify failure**

Run: `cd gym-wordle && uv run pytest tests/test_ppo.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'gym_wordle.agents.ppo.ppo'`

- [ ] **Step 3: Implement**

```python
# gym_wordle/agents/ppo/ppo.py
"""Rollout storage, generalised advantage estimation and the clipped PPO update."""
from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class PPOConfig:
    clip: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    epochs: int = 4
    minibatch: int = 4096
    max_grad_norm: float = 0.5
    target_kl: float | None = 0.02
    gamma: float = 1.0
    lam: float = 0.95


class RolloutBuffer:
    """Preallocated (T, N, ...) tensors on the training device."""

    def __init__(self, T, N, n_tokens, n_words, device):
        self.T, self.N = int(T), int(N)
        dev = torch.device(device)
        self.tokens = torch.zeros(T, N, n_tokens, 3, dtype=torch.long, device=dev)
        self.pad = torch.zeros(T, N, n_tokens, dtype=torch.bool, device=dev)
        self.turn = torch.zeros(T, N, dtype=torch.long, device=dev)
        self.mask = torch.zeros(T, N, n_words, dtype=torch.bool, device=dev)
        self.actions = torch.zeros(T, N, dtype=torch.long, device=dev)
        self.log_probs = torch.zeros(T, N, device=dev)
        self.values = torch.zeros(T, N, device=dev)
        self.rewards = torch.zeros(T, N, device=dev)
        self.dones = torch.zeros(T, N, dtype=torch.bool, device=dev)
        self.advantages = torch.zeros(T, N, device=dev)
        self.returns = torch.zeros(T, N, device=dev)

    def store(self, t, obs, action, log_prob, value, reward, done):
        self.tokens[t] = obs["tokens"]
        self.pad[t] = obs["pad"]
        self.turn[t] = obs["turn"]
        self.mask[t] = obs["mask"]
        self.actions[t] = action
        self.log_probs[t] = log_prob
        self.values[t] = value
        self.rewards[t] = reward
        self.dones[t] = done

    def compute_gae(self, last_value, gamma, lam):
        """dones[t] means obs[t+1] starts a new game: no bootstrap across it."""
        adv = torch.zeros(self.N, device=self.values.device)
        for t in reversed(range(self.T)):
            next_value = last_value if t == self.T - 1 else self.values[t + 1]
            nonterminal = (~self.dones[t]).float()
            delta = self.rewards[t] + gamma * next_value * nonterminal - self.values[t]
            adv = delta + gamma * lam * nonterminal * adv
            self.advantages[t] = adv
        self.returns = self.advantages + self.values

    def flat_obs(self, idx):
        """Observation dict for flat indices into the (T*N) rollout."""
        return {
            "tokens": self.tokens.reshape(-1, *self.tokens.shape[2:])[idx],
            "pad": self.pad.reshape(-1, self.pad.shape[2])[idx],
            "turn": self.turn.reshape(-1)[idx],
            "mask": self.mask.reshape(-1, self.mask.shape[2])[idx],
        }


def ppo_update(policy, optimizer, buf, cfg):
    """Clipped-surrogate PPO over the buffer. Returns mean diagnostics."""
    n = buf.T * buf.N
    dev = buf.values.device
    adv = buf.advantages.reshape(n)
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    ret = buf.returns.reshape(n)
    old_lp = buf.log_probs.reshape(n)
    old_v = buf.values.reshape(n)
    actions = buf.actions.reshape(n)

    sums = {k: 0.0 for k in ["policy_loss", "value_loss", "entropy", "approx_kl", "clip_frac"]}
    n_batches = 0
    epochs_run = 0
    policy.train()
    for _ in range(cfg.epochs):
        perm = torch.randperm(n, device=dev)
        epoch_kl, epoch_batches = 0.0, 0
        for start in range(0, n, cfg.minibatch):
            idx = perm[start : start + cfg.minibatch]
            lp, ent, v = policy.evaluate_actions(buf.flat_obs(idx), actions[idx])
            ratio = torch.exp(lp - old_lp[idx])
            a = adv[idx]
            pg_loss = torch.max(-a * ratio, -a * torch.clamp(ratio, 1 - cfg.clip, 1 + cfg.clip)).mean()
            v_clipped = old_v[idx] + torch.clamp(v - old_v[idx], -cfg.clip, cfg.clip)
            v_loss = 0.5 * torch.max((v - ret[idx]) ** 2, (v_clipped - ret[idx]) ** 2).mean()
            entropy = ent.mean()
            loss = pg_loss - cfg.ent_coef * entropy + cfg.vf_coef * v_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), cfg.max_grad_norm)
            optimizer.step()

            with torch.no_grad():
                log_ratio = lp - old_lp[idx]
                approx_kl = ((ratio - 1) - log_ratio).mean().item()
                clip_frac = ((ratio - 1).abs() > cfg.clip).float().mean().item()
            sums["policy_loss"] += pg_loss.item()
            sums["value_loss"] += v_loss.item()
            sums["entropy"] += entropy.item()
            sums["approx_kl"] += approx_kl
            sums["clip_frac"] += clip_frac
            n_batches += 1
            epoch_kl += approx_kl
            epoch_batches += 1
        epochs_run += 1
        if cfg.target_kl is not None and epoch_kl / max(epoch_batches, 1) > cfg.target_kl:
            break

    with torch.no_grad():
        var_ret = ret.var()
        ev = float("nan") if var_ret == 0 else (1 - (ret - old_v).var() / var_ret).item()
    out = {k: v / max(n_batches, 1) for k, v in sums.items()}
    out["explained_variance"] = ev
    out["epochs_run"] = epochs_run
    return out
```

- [ ] **Step 4: Run tests**

Run: `cd gym-wordle && uv run pytest tests/test_ppo.py -q`
Expected: 7 passed

- [ ] **Step 5: Commit**

```bash
git add gym-wordle/gym_wordle/agents/ppo/ppo.py gym-wordle/tests/test_ppo.py
git commit -m "feat(ppo): rollout buffer, GAE and clipped update with KL early stop"
```
(append the two trailers)

---

### Task 7: training loop, evaluation, CLI

**Files:**
- Create: `gym-wordle/gym_wordle/agents/ppo/train.py`
- Modify: `gym-wordle/gym_wordle/agents/ppo/__init__.py`
- Modify: `gym-wordle/pyproject.toml` (add `slow` marker)
- Test: `gym-wordle/tests/test_ppo.py`

**Interfaces:**
- Consumes: everything above; `save_checkpoint`, `load_checkpoint`, `word_feature_matrix`; `DEFAULT_VALIDS`, `DEFAULT_SOLUTIONS`.
- Produces: `collect(policy, env, buf, obs, ep_ret) -> (obs, last_value, stats)`; `train(args) -> (policy, run_dir)`; `evaluate(policy, words, solutions, *, hard_mode, device, L=4.0, max_attempts=6) -> dict(solve_rate, mean_guesses, histogram, fails)`; `build_parser()`; `main(argv=None)`.

- [ ] **Step 1: Write the failing tests** (append to `tests/test_ppo.py`)

```python
import sys

from gym_wordle.agents.ppo import train as train_mod


def test_eval_only_requires_checkpoint():
    with pytest.raises(SystemExit):
        train_mod.main(["--eval-only"])


def test_collect_reports_finished_episodes(small):
    words, env, policy = small
    buf = RolloutBuffer(12, env.n_games, 30, len(words), "cpu")
    obs = env.reset()
    ep_ret = torch.zeros(env.n_games)
    obs, last_value, stats = train_mod.collect(policy, env, buf, obs, ep_ret)
    assert last_value.shape == (env.n_games,)
    assert stats["episodes"] >= env.n_games          # 12 steps, games last <= 6
    assert 0.0 <= stats["solve_rate"] <= 1.0
    assert set(obs) == {"tokens", "pad", "turn", "mask"}


def test_evaluate_plays_every_solution_once(small):
    words, env, policy = small
    out = train_mod.evaluate(policy, words, env.solutions, hard_mode=True, device="cpu")
    assert out["fails"] + sum(out["histogram"]) == len(env.solutions)
    assert len(out["histogram"]) == 6
    assert 0.0 <= out["solve_rate"] <= 1.0


@pytest.mark.slow
def test_ppo_learns_the_fixture_game(env_paths, tmp_path):
    argv = [
        "--valids", str(env_paths["valid_words_path"]),
        "--solutions", str(env_paths["solution_words_path"]),
        "--iterations", "60", "--n-games", "64", "--rollout-len", "8",
        "--d-model", "32", "--n-layers", "1", "--n-heads", "4", "--d-ff", "64",
        "--minibatch", "128", "--lr", "3e-3", "--ent-coef", "0.001",
        "--device", "cpu", "--seed", "0", "--run-dir", str(tmp_path / "run"),
        "--checkpoint-every", "1000",
    ]
    policy, run_dir, final = train_mod.main(argv)
    assert final["solve_rate"] == 1.0
    assert (run_dir / "log.csv").exists()
    assert (run_dir / "policy_final.pt").exists()
```

Add to `pyproject.toml` under `[tool.pytest.ini_options]`:

```toml
markers = ["slow: takes tens of seconds; run with -m slow or by default"]
```

- [ ] **Step 2: Run to verify failure**

Run: `cd gym-wordle && uv run pytest tests/test_ppo.py -q`
Expected: FAIL with `ImportError: cannot import name 'train'`

- [ ] **Step 3: Implement**

```python
# gym_wordle/agents/ppo/train.py
"""Train or evaluate the PPO transformer agent on batched Wordle."""
import argparse
import csv
import datetime as dt
import logging
import time
from pathlib import Path

import torch

from gym_wordle.agents.common import load_checkpoint, save_checkpoint, word_feature_matrix
from gym_wordle.agents.ppo.model import WordlePolicy
from gym_wordle.agents.ppo.ppo import PPOConfig, RolloutBuffer, ppo_update
from gym_wordle.envs.batched import BatchedWordle
from gym_wordle.envs.wordle_env import DEFAULT_SOLUTIONS, DEFAULT_VALIDS, load_word_lists

log = logging.getLogger(__name__)

CSV_FIELDS = [
    "iteration", "steps", "episodes", "solve_rate", "mean_guesses", "mean_reward",
    "entropy", "approx_kl", "clip_frac", "value_loss", "explained_variance",
    "epochs_run", "steps_per_s", "lr",
]


def collect(policy, env, buf, obs, ep_ret):
    """Roll the policy for buf.T steps. Returns (obs, last_value, episode stats).

    ep_ret accumulates per-game return across calls and is zeroed on done.
    """
    policy.eval()
    solved_n, finished_n, guesses_sum, ret_sum = 0, 0, 0.0, 0.0
    with torch.no_grad():
        for t in range(buf.T):
            action, log_prob, _, value = policy.act(obs)
            next_obs, reward, done, info = env.step(action)
            buf.store(t, obs, action, log_prob, value, reward, done)
            ep_ret += reward
            if bool(done.any()):
                fin = done
                solved = info["solved"][fin]
                finished_n += int(fin.sum())
                solved_n += int(solved.sum())
                guesses_sum += float(info["n_guesses"][fin][solved].sum())
                ret_sum += float(ep_ret[fin].sum())
                ep_ret[fin] = 0.0
            obs = next_obs
        _, last_value = policy(obs)
    stats = {
        "episodes": finished_n,
        "solve_rate": solved_n / finished_n if finished_n else float("nan"),
        "mean_guesses": guesses_sum / solved_n if solved_n else float("nan"),
        "mean_reward": ret_sum / finished_n if finished_n else float("nan"),
    }
    return obs, last_value, stats


def evaluate(policy, words, solutions, *, hard_mode, device, L=4.0, max_attempts=6):
    """Greedy play of every solution exactly once. Never touches training state."""
    env = BatchedWordle(words, solutions, len(solutions), hard_mode=hard_mode,
                        max_attempts=max_attempts, L=L, device=device, seed=0)
    obs = env.set_secrets(env.solution_idx)
    N = env.n_games
    solved = torch.zeros(N, dtype=torch.bool, device=env.device)
    n_guesses = torch.zeros(N, dtype=torch.long, device=env.device)
    active = torch.ones(N, dtype=torch.bool, device=env.device)
    policy.eval()
    with torch.no_grad():
        for _ in range(max_attempts):
            action, *_ = policy.act(obs, greedy=True)
            obs, _, done, info = env.step(action)
            fin = done & active
            solved = torch.where(fin, info["solved"], solved)
            n_guesses = torch.where(fin, info["n_guesses"], n_guesses)
            active &= ~done
            if not bool(active.any()):
                break
    hist = torch.bincount(n_guesses[solved], minlength=max_attempts + 1)[1:].tolist()
    n_solved = int(solved.sum())
    return {
        "solve_rate": n_solved / N,
        "mean_guesses": float(n_guesses[solved].float().mean()) if n_solved else float("nan"),
        "histogram": hist,
        "fails": N - n_solved,
    }


def build_parser():
    p = argparse.ArgumentParser(description="PPO transformer agent for Wordle")
    p.add_argument("--valids", default=str(DEFAULT_VALIDS))
    p.add_argument("--solutions", default=str(DEFAULT_SOLUTIONS))
    p.add_argument("--iterations", type=int, default=300)
    p.add_argument("--n-games", type=int, default=2048)
    p.add_argument("--rollout-len", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--clip", type=float, default=0.2)
    p.add_argument("--target-kl", type=float, default=0.02)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--minibatch", type=int, default=4096)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--n-layers", type=int, default=3)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--d-ff", type=int, default=512)
    p.add_argument("--hard-mode", dest="hard_mode", action="store_true", default=True)
    p.add_argument("--no-hard-mode", dest="hard_mode", action="store_false")
    p.add_argument("--shaping-coef", type=float, default=0.0)
    p.add_argument("--L", type=float, default=4.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--run-dir", default=None, help="default runs/<timestamp>")
    p.add_argument("--checkpoint-every", type=int, default=25)
    p.add_argument("--checkpoint", default=None, help="resume from / evaluate this .pt file")
    p.add_argument("--eval-only", action="store_true")
    return p


def train(args):
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    words, solutions = load_word_lists(args.valids, args.solutions)
    n_letters = len(words[0])
    env = BatchedWordle(words, solutions, args.n_games, hard_mode=args.hard_mode, L=args.L,
                        shaping_coef=args.shaping_coef, device=device, seed=args.seed)
    if args.checkpoint:
        policy = load_checkpoint(WordlePolicy, args.checkpoint, device)
    else:
        policy = WordlePolicy(
            len(words), word_feature_matrix(words, n_letters), d_model=args.d_model,
            n_heads=args.n_heads, d_ff=args.d_ff, n_layers=args.n_layers,
            max_turns=env.max_attempts, n_letters=n_letters,
        ).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr, eps=1e-5)
    cfg = PPOConfig(clip=args.clip, vf_coef=args.vf_coef, ent_coef=args.ent_coef, epochs=args.epochs,
                    minibatch=args.minibatch, target_kl=args.target_kl, gamma=1.0, lam=args.gae_lambda)
    buf = RolloutBuffer(args.rollout_len, args.n_games, env.max_attempts * n_letters, len(words), device)

    run_dir = Path(args.run_dir or Path("runs") / dt.datetime.now().strftime("%Y%m%d.%H.%M.%S"))
    run_dir.mkdir(parents=True, exist_ok=True)
    log.info("run dir %s  params %d", run_dir, sum(p.numel() for p in policy.parameters()))

    obs = env.reset()
    ep_ret = torch.zeros(args.n_games, device=device)
    steps = 0
    with open(run_dir / "log.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for it in range(1, args.iterations + 1):
            lr = args.lr * (1 - (it - 1) / args.iterations)
            for g in optimizer.param_groups:
                g["lr"] = lr
            t0 = time.perf_counter()
            obs, last_value, ep = collect(policy, env, buf, obs, ep_ret)
            buf.compute_gae(last_value, cfg.gamma, cfg.lam)
            upd = ppo_update(policy, optimizer, buf, cfg)
            steps += buf.T * buf.N
            row = {
                "iteration": it, "steps": steps, **ep,
                "entropy": upd["entropy"], "approx_kl": upd["approx_kl"], "clip_frac": upd["clip_frac"],
                "value_loss": upd["value_loss"], "explained_variance": upd["explained_variance"],
                "epochs_run": upd["epochs_run"],
                "steps_per_s": buf.T * buf.N / (time.perf_counter() - t0), "lr": lr,
            }
            writer.writerow(row)
            f.flush()
            log.info(
                "it %d  steps %d  solve %.3f  guesses %.2f  reward %.3f  ent %.2f  kl %.4f  clip %.2f  ev %.2f  %.0f steps/s",
                it, steps, row["solve_rate"], row["mean_guesses"], row["mean_reward"], row["entropy"],
                row["approx_kl"], row["clip_frac"], row["explained_variance"], row["steps_per_s"],
            )
            if it % args.checkpoint_every == 0:
                save_checkpoint(policy, str(run_dir / f"policy_{it}.pt"))
    save_checkpoint(policy, str(run_dir / "policy_final.pt"))
    return policy, run_dir


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.eval_only and not args.checkpoint:
        parser.error("--eval-only requires --checkpoint")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    words, solutions = load_word_lists(args.valids, args.solutions)
    if args.eval_only:
        policy = load_checkpoint(WordlePolicy, args.checkpoint, args.device)
        run_dir = None
    else:
        policy, run_dir = train(args)
    final = evaluate(policy, words, solutions, hard_mode=args.hard_mode, device=args.device, L=args.L)
    log.info("eval: solve rate %.4f  mean guesses %.3f  histogram %s  fails %d",
             final["solve_rate"], final["mean_guesses"], final["histogram"], final["fails"])
    return policy, run_dir, final


if __name__ == "__main__":
    main()
```

Update `gym_wordle/agents/ppo/__init__.py`:

```python
from gym_wordle.agents.ppo.model import WordlePolicy
from gym_wordle.agents.ppo.ppo import PPOConfig, RolloutBuffer, ppo_update

__all__ = ["PPOConfig", "RolloutBuffer", "WordlePolicy", "ppo_update"]
```

- [ ] **Step 4: Run tests**

Run: `cd gym-wordle && uv run pytest tests/test_ppo.py -q --durations=3`
Expected: 11 passed; the slow test finishes in under about 30 s. If the learning test does not reach 1.0, raise `--iterations` to at most 120 or `--lr` to 5e-3 before changing anything else, and record what you changed in the report.

- [ ] **Step 5: Commit**

```bash
git add gym-wordle/gym_wordle/agents/ppo/train.py gym-wordle/gym_wordle/agents/ppo/__init__.py gym-wordle/pyproject.toml gym-wordle/tests/test_ppo.py
git commit -m "feat(ppo): training loop, greedy evaluation over all solutions, CLI"
```
(append the two trailers)

---

### Task 8: README, ignore rules, GPU smoke run

**Files:**
- Modify: `README.md`
- Modify: `.gitignore` (add `runs/`)
- Modify: `gym-wordle/pyproject.toml` (description)

- [ ] **Step 1: `.gitignore`** — append the line `**/runs/`.

- [ ] **Step 2: `pyproject.toml`** — change `description` to `"Wordle as a batched RL environment, with a PPO transformer agent"`.

- [ ] **Step 3: README** — replace the opening paragraph, the Layout block and the Training section with the text below; keep Setup, Environment and History sections, but retitle "Environment" to "Legacy gymnasium environment" and "Training" content for the DQNs moves under a new "Legacy agents" heading.

Opening paragraph:

```markdown
A batched Wordle environment and a PPO agent whose transformer policy reads
only the coloured feedback to its own guesses. The agent knows the valid
guess dictionary but never the solution list. A single-game gymnasium
environment and two DQN trainers from the earlier version of the project
remain as legacy code.
```

Layout block:

```
    words.json                    2309 solution words (flat JSON list)
    valids.json                   10638 additional allowed guesses
    gym-wordle/                   the installable package (uv project)
      gym_wordle/envs/feedback.py      colouring, hard-mode mask, consistency mask (pure, batched)
      gym_wordle/envs/batched.py       BatchedWordle: N games per step, auto-reset
      gym_wordle/agents/ppo/model.py   WordlePolicy: transformer + factored lexical head
      gym_wordle/agents/ppo/ppo.py     rollout buffer, GAE, clipped PPO update
      gym_wordle/agents/ppo/train.py   training loop, evaluation, CLI
      gym_wordle/envs/wordle_env.py    legacy: WordleEnv, registered as Wordle-v0
      gym_wordle/agents/{common,dqn,transformer}.py  legacy DQN trainers
      tests/                           pytest suite on a small fixture list
      ppo_agent.py, restore_policy.py  STALE: Ray 1.x RLlib, not ported
    docs/superpowers/             design specs and implementation plans
```

New section, placed before the legacy sections:

```markdown
## PPO agent

    cd gym-wordle
    uv run python -m gym_wordle.agents.ppo.train
    uv run python -m gym_wordle.agents.ppo.train --eval-only --checkpoint runs/<stamp>/policy_final.pt

The env plays 2048 games in lockstep on the GPU. The policy sees one token
per board cell (letter, colour, slot), prepends a learned readout token,
runs a small pre-norm transformer, and scores every valid word as the dot
product between the readout and a word tower over fixed lexical features
(one-hot letter per position plus letter counts). Illegal words are masked
out before sampling. Hard mode follows the official rule (greens fixed,
known letters kept) and is on by default; `--no-hard-mode` lifts it.

Reward is paid on the solving guess only: P(X ≥ k) / P(X ≥ 1) for
X ~ Poisson(4) on guess k, so 1.0, 0.93, 0.78, 0.58, 0.38, 0.22, and 0 for
a fail. `--shaping-coef c` adds c times the drop in log2 of the number of
valid words still consistent with the feedback, a dense signal that uses
only the dictionary.

Each iteration logs solve rate, mean guesses, entropy, approximate KL, clip
fraction, value loss, explained variance and steps per second to stdout and
`runs/<stamp>/log.csv`. Training ends with a greedy pass over every
solution and prints the guess-count histogram.
```

- [ ] **Step 4: GPU smoke run** (verifies the real-scale path; not a test)

Run from `gym-wordle/`:
```bash
uv run python -m gym_wordle.agents.ppo.train --iterations 5 --run-dir /tmp/claude-1000/-home-jamestaylor-wordle/b78e5c99-0f79-4b0b-8fb3-2e8c9cb4942f/scratchpad/ppo-smoke 2>&1 | tail -8
```
Expected: five `it N ...` lines with finite numbers and a `steps/s` figure, then an `eval:` line with a histogram of six integers summing with `fails` to 2309. Record the steps/s figure and peak GPU memory (`nvidia-smi --query-gpu=memory.used --format=csv` during the run, or `torch.cuda.max_memory_allocated` if easier) in the task report. If the run fails with out-of-memory, halve `--n-games` and report it; do not change defaults in code.

- [ ] **Step 5: Full suite**

Run: `cd gym-wordle && uv run pytest -q`
Expected: all pass

- [ ] **Step 6: Commit**

```bash
git add README.md .gitignore gym-wordle/pyproject.toml
git commit -m "docs: PPO agent is the main line; legacy labels; ignore runs/"
```
(append the two trailers)

---

## Self-review notes

- **Spec coverage.** §5.1 → Tasks 1–2; §5.2–5.4 → Tasks 3–4; §6 → Task 5; §7.1–7.3 → Task 6; §7.4–7.6 → Task 7; §8 tests are distributed to the task that builds each unit; README/.gitignore → Task 8. The `slow` learning test is in Task 7.
- **Type consistency.** `observation()` keys `tokens/pad/turn/mask` are used identically in Tasks 4, 5, 6, 7. `info` keys `solved/n_guesses/secret` are used identically in Tasks 3 and 7. `set_secrets` takes word indices (into `words`), and `evaluate` passes `env.solution_idx`, which are word indices.
- **Memory note for the implementer.** `hard_mode_mask` materialises a `(B,V,26)` bool tensor, about 0.7 GB at B=2048 on the real list; it is transient and fine on a 24 GB GPU. `consistent_mask` materialises `(B,n,V)` int8, about 0.13 GB. Neither is called on the critical path unless shaping is on, except `hard_mode_mask` once per step.
- **Learning test tuning latitude** is stated in Task 7 Step 4 and nowhere else.
