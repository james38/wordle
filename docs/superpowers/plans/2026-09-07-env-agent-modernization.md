# Wordle RL Modernization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** One gymnasium `WordleEnv` in an installable `gym_wordle` package, the two PyTorch DQN trainers moved in beside it with their confirmed bugs fixed, a regression test suite, and an optional factored prediction head for the SE-ResNet DQN.

**Architecture:** `gym_wordle/envs/wordle_env.py` holds the environment; hard-mode legality is computed as a boolean mask over the whole word list with vectorized numpy, and `info["action_mask"]` is the single source of truth agents use. `gym_wordle/agents/common.py` holds the pieces both trainers share (masked argmax, replay buffer, checkpoint IO, word features). `dqn.py` and `transformer.py` are the two trainers. Old checkpoints are not loadable; new ones are `state_dict` + a JSON of constructor kwargs.

**Tech Stack:** Python >= 3.10, uv, gymnasium >= 1.0, numpy, torch >= 2.2, pytest. No matplotlib: the per-episode reward plot is dropped; trainers keep the reward history on `trainer.rewards` and log a rolling mean.

**Spec:** `docs/superpowers/specs/2026-09-07-env-agent-modernization-design.md`

## Global Constraints

- All commands run from `gym-wordle/` (the package root, which holds `pyproject.toml`). Use `uv run ...`; never call `pip` or bare `python`.
- Data files live at the repository root: `words.json` (2309 solutions, flat list) and `valids.json` (10638 non-solution guesses, flat list). The action space is the **sorted union** of both. Tests never touch these; they use `tests/fixtures/`.
- Observation is a flat `np.int8` vector of length `1 + 26*n_letters + 2*26` (183 for 5 letters). Slot 0 is guesses used, `POS + 5*letter + pos` is positional status (0 other, 1 possible yellow, 2 green), `EXCEEDED + letter` is the exceeded bit, `YELLOWS + letter` is max yellows. No negative indexing anywhere.
- Hard-mode rejection: state, turn counter and reward unchanged, `info["valid"] == False`. Re-guessing an already accepted word is also a rejection.
- The default DQN head is `"flat"`. Every test before Task 8 runs against it.
- `ppo_agent.py` and `restore_policy.py` keep their code unchanged; the only edit to them is the two-line stale-marker comment in Task 9.
- Reward schedule, epsilon schedule, and hyperparameter values are copied from the old code, not tuned.
- Checkpoints: `torch.save(model.state_dict(), path)` plus `path + ".json"` holding `{"class": ..., "kwargs": model.ctor_kwargs}`. Loading uses `weights_only=True`.
- Commit after each task with the message given in the task. Do not push.

---

### Task 1: Package scaffold with uv

**Files:**
- Create: `gym-wordle/pyproject.toml`
- Create: `gym-wordle/gym_wordle/agents/__init__.py` (empty)
- Create: `gym-wordle/tests/__init__.py` (empty)
- Create: `gym-wordle/tests/fixtures/words_small.json`
- Create: `gym-wordle/tests/fixtures/valids_small.json`
- Create: `gym-wordle/tests/conftest.py`
- Create: `gym-wordle/tests/test_scaffold.py`
- Modify: `.gitignore` (repo root)
- Delete: `gym-wordle/setup.py`

**Interfaces:**
- Produces: pytest fixtures `fixture_dir: Path`, `env_paths: dict` with keys `valid_words_path` and `solution_words_path`, used by every later test file.

- [ ] **Step 1: Write pyproject.toml**

```toml
[project]
name = "gym-wordle"
version = "2.0.0"
description = "Wordle as a gymnasium environment, with DQN agents"
requires-python = ">=3.10"
dependencies = [
    "gymnasium>=1.0",
    "numpy>=1.26",
    "torch>=2.2",
]

[dependency-groups]
dev = ["pytest>=8"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["gym_wordle"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

- [ ] **Step 2: Remove setup.py and create empty package files**

```bash
cd gym-wordle
git rm setup.py
mkdir -p gym_wordle/agents tests/fixtures
touch gym_wordle/agents/__init__.py tests/__init__.py
```

- [ ] **Step 3: Write the fixture word lists**

`tests/fixtures/words_small.json`:

```json
["apple", "crane", "stale", "maple", "delta"]
```

`tests/fixtures/valids_small.json`:

```json
["puppy", "paper", "dealt", "ample", "pupae", "apply", "happy", "least", "stale", "plate", "stone", "tales", "slate", "cable", "table"]
```

Note `stale` is a solution and `slate`, `least`, `tales` are anagrams of it; `dealt` and `delta` are anagrams; `cable` and `table` differ only in a letter that `crane` reveals as grey against `apple`. Tests rely on these words.

- [ ] **Step 4: Write conftest.py**

```python
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
```

- [ ] **Step 5: Write a scaffold test**

`tests/test_scaffold.py`:

```python
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
```

- [ ] **Step 6: Install and run**

```bash
uv sync
uv run pytest tests/test_scaffold.py -v
```

Expected: 2 passed. The first `uv sync` downloads torch, which is large; this is expected.

- [ ] **Step 7: Update .gitignore at the repo root**

Replace the file contents with:

```
**/__pycache__/
*.png
**/checkpoints/
**/models/
.venv/
```

Then confirm the data files are now visible to git:

```bash
cd ..
git status --short words.json valids.json
```

Expected: both listed as untracked (`??`).

- [ ] **Step 8: Commit**

```bash
git add .gitignore words.json valids.json gym-wordle/pyproject.toml gym-wordle/uv.lock gym-wordle/gym_wordle/agents/__init__.py gym-wordle/tests
git commit -m "build: uv-managed gym-wordle package, test fixtures, commit word lists"
```

---

### Task 2: Environment construction, reset, and state encoding

This task writes the environment without hard mode. Every guess is accepted, but the observation encoding, reward, termination, and copy semantics are complete. Task 3 adds hard mode.

**Files:**
- Create: `gym-wordle/gym_wordle/envs/wordle_env.py`
- Modify: `gym-wordle/gym_wordle/envs/__init__.py`
- Create: `gym-wordle/tests/test_env.py`

**Interfaces:**
- Produces: `WordleEnv(valid_words_path, solution_words_path, n_letters=5, max_attempts=6, L=4.0, min_guesses=True)` with attributes `words: list[str]`, `solutions: list[str]`, `word_to_action: dict[str,int]`, `word_letters: np.ndarray (N,5) int8`, `word_counts: np.ndarray (N,26) int8`, `POS`, `EXCEEDED`, `YELLOWS`, `obs_size`, `rewards: list[float]`, `n_letters`, `max_attempts`, `secret_word`, `n_guesses`, `terminated`, `overused_letters: set[str]`. Methods `reset(*, seed=None, options=None) -> (obs, info)`, `step(action) -> (obs, reward, terminated, False, info)`, `_hard_mode_mask() -> np.ndarray[bool]` (stub here), `_action_mask()`, `_pos_slice(letter) -> view`.
- `info` keys: `valid`, `valid_words`, `action_mask`, `secret_word`, `n_guesses`.
- Module constants: `ALPHABET`, `N_ALPHABET`, `LETTER_INDEX`, `REPO_ROOT`, `DEFAULT_VALIDS`, `DEFAULT_SOLUTIONS`. Functions `load_word_lists(valid_words_path, solution_words_path) -> (words, solutions)`, `letter_matrix(words, n_letters) -> (letters, counts)`.

- [ ] **Step 1: Write the failing tests**

`tests/test_env.py`:

```python
import numpy as np
import pytest

from gym_wordle.envs.wordle_env import WordleEnv, LETTER_INDEX, load_word_lists


def make_env(env_paths, **kw):
    return WordleEnv(**env_paths, **kw)


def action(env, word):
    return env.word_to_action[word]


# --- construction -----------------------------------------------------------

def test_action_space_is_sorted_union_of_both_files(env_paths):
    env = make_env(env_paths)
    assert env.words == sorted(set(env.words))
    assert set(env.solutions) <= set(env.words)
    assert len(env.words) == 19  # 15 valids + 5 solutions, 'stale' shared
    assert env.action_space.n == 18
    assert all(env.word_to_action[w] == i for i, w in enumerate(env.words))


def test_letter_matrix_shapes_and_counts(env_paths):
    env = make_env(env_paths)
    assert env.word_letters.shape == (18, 5)
    assert env.word_counts.shape == (18, 26)
    a = action(env, "apple")
    assert env.word_counts[a, LETTER_INDEX["p"]] == 2
    assert env.word_counts[a].sum() == 5
    assert list(env.word_letters[a]) == [LETTER_INDEX[c] for c in "apple"]


def test_observation_layout_constants(env_paths):
    env = make_env(env_paths)
    assert env.POS == 1
    assert env.EXCEEDED == 1 + 26 * 5
    assert env.YELLOWS == env.EXCEEDED + 26
    assert env.obs_size == 183
    nvec = env.observation_space.nvec
    assert nvec[0] == 7  # 0..6 guesses
    assert set(nvec[env.POS:env.EXCEEDED]) == {3}
    assert set(nvec[env.EXCEEDED:env.YELLOWS]) == {2}
    assert set(nvec[env.YELLOWS:]) == {3}  # 0..2 yellows
    assert env.observation_space.dtype == np.int8


def test_reward_schedule_matches_poisson_formula(env_paths):
    env = make_env(env_paths, L=4.0)
    assert len(env.rewards) == 6
    assert env.rewards[0] == pytest.approx(1 - 1 / 5)  # 5 solutions
    assert all(a > b for a, b in zip(env.rewards, env.rewards[1:]))
    binary = make_env(env_paths, min_guesses=False)
    assert binary.rewards == [1.0] * 6


# --- reset ------------------------------------------------------------------

def test_reset_returns_zero_obs_copy_and_info(env_paths):
    env = make_env(env_paths)
    obs, info = env.reset(seed=1)
    assert obs.shape == (183,) and obs.dtype == np.int8
    assert not obs.any()
    assert obs is not env._state
    assert info["valid"] is True
    assert info["n_guesses"] == 0
    assert info["secret_word"] in env.solutions
    assert info["action_mask"].dtype == bool and info["action_mask"].all()
    assert set(info["valid_words"]) == set(range(18))


def test_reset_with_seed_is_reproducible(env_paths):
    env = make_env(env_paths)
    a = [env.reset(seed=7)[1]["secret_word"] for _ in range(5)]
    b = [env.reset(seed=7)[1]["secret_word"] for _ in range(5)]
    assert a == b


def test_reset_with_secret_word_option(env_paths):
    env = make_env(env_paths)
    _, info = env.reset(options={"secret_word": "crane"})
    assert env.secret_word == "crane" and info["secret_word"] == "crane"
    with pytest.raises(AssertionError):
        env.reset(options={"secret_word": "toolong"})


# --- step: encoding, reward, termination --------------------------------------

def test_green_pass_encoding(env_paths):
    env = make_env(env_paths)
    env.reset(options={"secret_word": "apple"})
    obs, r, term, trunc, info = env.step(action(env, "ample"))
    assert obs[0] == 1 and r == 0.0 and term is False and trunc is False
    for pos, c in [(0, "a"), (2, "p"), (3, "l"), (4, "e")]:
        assert obs[env.POS + 5 * LETTER_INDEX[c] + pos] == 2
    assert obs[env.EXCEEDED + LETTER_INDEX["m"]] == 1
    assert obs[env.EXCEEDED + LETTER_INDEX["p"]] == 0
    assert "m" in env.overused_letters


def test_yellow_pass_encoding(env_paths):
    env = make_env(env_paths)
    env.reset(options={"secret_word": "apple"})
    obs, *_ = env.step(action(env, "crane"))  # e green at 4; a yellow at 2
    a, e = LETTER_INDEX["a"], LETTER_INDEX["e"]
    # 'a' guessed at 2, secret has it at 0: possible yellow everywhere but 2.
    # Slot 4 is also marked 1 even though e is green there; that is the
    # existing (slightly loose) semantics and hard mode still handles it.
    assert list(obs[env.POS + 5 * a: env.POS + 5 * a + 5]) == [1, 1, 0, 1, 1]
    assert obs[env.YELLOWS + a] == 1
    assert list(obs[env.POS + 5 * e: env.POS + 5 * e + 5]) == [0, 0, 0, 0, 2]
    assert obs[env.YELLOWS + e] == 0
    for c in "crn":
        assert obs[env.EXCEEDED + LETTER_INDEX[c]] == 1
    assert obs[env.EXCEEDED + e] == 0


def test_exceeded_bit_set_when_letter_over_guessed_with_yellows(env_paths):
    env = make_env(env_paths)
    env.reset(options={"secret_word": "apple"})
    obs, *_ = env.step(action(env, "puppy"))  # three p's, secret has two
    p = LETTER_INDEX["p"]
    assert obs[env.EXCEEDED + p] == 1
    assert obs[env.YELLOWS + p] == 1
    assert list(obs[env.POS + 5 * p: env.POS + 5 * p + 5]) == [0, 1, 2, 0, 1]
    assert "p" not in env.overused_letters  # still has an unplaced copy
    for c in "uy":
        assert obs[env.EXCEEDED + LETTER_INDEX[c]] == 1


def test_win_reward_is_per_step_and_terminates(env_paths):
    env = make_env(env_paths)
    env.reset(options={"secret_word": "apple"})
    _, r1, t1, *_ = env.step(action(env, "crane"))
    _, r2, t2, *_ = env.step(action(env, "apple"))
    assert r1 == 0.0 and t1 is False
    assert r2 == pytest.approx(env.rewards[1]) and t2 is True


def test_final_turn_terminates_and_stays_inside_space(env_paths):
    env = make_env(env_paths, max_attempts=1)
    env.reset(options={"secret_word": "apple"})
    obs, r, term, *_ = env.step(action(env, "crane"))
    assert term is True and r == 0.0
    assert obs[0] == 1
    assert env.observation_space.contains(obs)


def test_observations_are_not_aliased_to_env_state(env_paths):
    env = make_env(env_paths)
    obs0, _ = env.reset(options={"secret_word": "apple"})
    obs1, *_ = env.step(action(env, "crane"))
    assert not obs0.any()
    assert obs1[0] == 1
    obs2, *_ = env.step(action(env, "slate"))
    assert obs1[0] == 1 and obs2[0] == 2
    assert obs1 is not obs2


def test_step_after_termination_raises(env_paths):
    env = make_env(env_paths)
    env.reset(options={"secret_word": "apple"})
    env.step(action(env, "apple"))
    with pytest.raises(AssertionError):
        env.step(action(env, "crane"))
```

- [ ] **Step 2: Run to verify failure**

```bash
uv run pytest tests/test_env.py -v
```

Expected: ImportError, `No module named 'gym_wordle.envs.wordle_env'`.

- [ ] **Step 3: Write the environment**

`gym_wordle/envs/wordle_env.py`:

```python
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
        """Which words satisfy hard mode given the current state. Task 3."""
        return np.ones(len(self.words), dtype=bool)

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
```

`gym_wordle/envs/__init__.py`:

```python
from gym_wordle.envs.wordle_env import WordleEnv

__all__ = ["WordleEnv"]
```

- [ ] **Step 4: Run tests**

```bash
uv run pytest tests/test_env.py -v
```

Expected: all 15 tests in the file pass.

- [ ] **Step 5: Commit**

```bash
git add gym_wordle/envs tests/test_env.py
git commit -m "feat(env): gymnasium WordleEnv with explicit observation layout and copy semantics"
```

---

### Task 3: Hard mode, action mask, registration, and env_checker

**Files:**
- Modify: `gym-wordle/gym_wordle/envs/wordle_env.py` (replace `_hard_mode_mask`)
- Modify: `gym-wordle/gym_wordle/__init__.py`
- Modify: `gym-wordle/tests/test_env.py` (append)
- Delete: `wordle_rl.py` (repo root)

**Interfaces:**
- Consumes: `WordleEnv` from Task 2.
- Produces: `_hard_mode_mask()` real implementation; gymnasium id `"Wordle-v0"`; `import gym_wordle` registers it.

- [ ] **Step 1: Append the failing tests**

Append to `tests/test_env.py`:

```python
# --- hard mode ----------------------------------------------------------------

def test_first_guess_is_never_rejected(env_paths):
    env = make_env(env_paths)
    env.reset(options={"secret_word": "apple"})
    assert env._action_mask().all()


def test_hard_mode_rejection_leaves_state_and_turn_unchanged(env_paths):
    env = make_env(env_paths)
    env.reset(options={"secret_word": "apple"})
    obs1, *_ = env.step(action(env, "crane"))  # a yellow, e green at 4
    obs2, r, term, trunc, info = env.step(action(env, "puppy"))  # no a, no e
    assert info["valid"] is False
    assert r == 0.0 and term is False
    assert env.n_guesses == 1 and obs2[0] == 1
    assert np.array_equal(obs1, obs2)
    assert action(env, "puppy") not in info["valid_words"]
    assert not info["action_mask"][action(env, "puppy")]


def test_greens_must_be_kept_in_place(env_paths):
    env = make_env(env_paths)
    env.reset(options={"secret_word": "apple"})
    _, _, _, _, info = env.step(action(env, "ample"))  # a?ple
    assert set(info["valid_words"].values()) == {"apple"}


def test_grey_letter_may_not_be_reused(env_paths):
    env = make_env(env_paths)
    env.reset(options={"secret_word": "apple"})
    _, _, _, _, info = env.step(action(env, "crane"))  # c, r, n grey
    legal = set(info["valid_words"].values())
    # cable and table both keep e at 4 and contain a; only c is grey
    assert "table" in legal
    assert "cable" not in legal


def test_yellow_letters_must_be_reused(env_paths):
    env = make_env(env_paths)
    env.reset(options={"secret_word": "apple"})
    _, _, _, _, info = env.step(action(env, "crane"))  # a yellow, e green at 4
    legal = set(info["valid_words"].values())
    assert {"apple", "ample", "maple", "stale", "plate", "table"} <= legal
    assert "happy" not in legal  # has a, but no e at 4
    assert "least" not in legal  # has a and e, but e is not at 4
    assert "stone" not in legal  # no a


def test_accepted_guess_is_removed_from_valid_words(env_paths):
    env = make_env(env_paths)
    env.reset(options={"secret_word": "delta"})
    _, _, _, _, info = env.step(action(env, "dealt"))  # d,e green; a,l,t yellow
    assert env._hard_mode_mask()[action(env, "dealt")]  # still hard-mode legal
    assert action(env, "dealt") not in info["valid_words"]
    assert not info["action_mask"][action(env, "dealt")]
    _, _, _, _, info2 = env.step(action(env, "dealt"))
    assert info2["valid"] is False and env.n_guesses == 1


def test_valid_words_matches_acceptance_rule(env_paths):
    env = make_env(env_paths)
    for secret in env.solutions:
        for opener in ["crane", "puppy", "paper", "dealt", "stale"]:
            env.reset(options={"secret_word": secret})
            if opener == secret:
                continue
            _, _, _, _, info = env.step(action(env, opener))
            mask = info["action_mask"]
            for a in range(env.action_space.n):
                probe = make_env(env_paths)
                probe.reset(options={"secret_word": secret})
                probe.step(action(probe, opener))
                _, _, _, _, probe_info = probe.step(a)
                assert probe_info["valid"] == bool(mask[a]), (secret, opener, env.words[a])


def test_secret_word_is_always_legal(env_paths):
    env = make_env(env_paths)
    for secret in env.solutions:
        for opener in ["crane", "puppy", "paper", "dealt"]:
            env.reset(options={"secret_word": secret})
            if opener == secret:
                continue
            _, _, _, _, info = env.step(action(env, opener))
            assert info["action_mask"][action(env, secret)]


# --- registration and checker -------------------------------------------------

def test_registered_with_gymnasium(env_paths):
    import gymnasium
    import gym_wordle  # noqa: F401  registers Wordle-v0

    env = gymnasium.make("Wordle-v0", **env_paths)
    obs, info = env.reset(seed=0)
    assert obs.shape == (183,)
    assert isinstance(env.unwrapped, WordleEnv)


def test_passes_gymnasium_env_checker(env_paths):
    from gymnasium.utils.env_checker import check_env

    check_env(make_env(env_paths), skip_render_check=True)
```

- [ ] **Step 2: Run to verify failure**

```bash
uv run pytest tests/test_env.py -v -k "hard or greens_must or grey or yellow_letters or accepted or matches_acceptance or always_legal or registered or checker"
```

Expected: the hard-mode tests fail on assertions (everything is currently legal); `test_registered_with_gymnasium` fails with `NameNotFound: Environment Wordle doesn't exist`.

- [ ] **Step 3: Implement `_hard_mode_mask`**

Replace the stub in `gym_wordle/envs/wordle_env.py`:

```python
    def _hard_mode_mask(self):
        """Boolean mask over all words: which satisfy hard mode right now.

        1. A green letter must be kept at its position.
        2. An overused letter (grey, or every copy already green) may only
           appear at positions where it is green.
        3. A letter with a known yellow must appear at least
           1 + (its greens) times.
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
        for letter in np.flatnonzero(has_yellow):
            ok &= self.word_counts[:, letter] >= 1 + n_green[letter]
        return ok
```

- [ ] **Step 4: Register with gymnasium**

`gym_wordle/__init__.py`:

```python
from gymnasium.envs.registration import register

from gym_wordle.envs.wordle_env import WordleEnv

register(id="Wordle-v0", entry_point="gym_wordle.envs.wordle_env:WordleEnv")

__all__ = ["WordleEnv"]
```

- [ ] **Step 5: Run the whole env suite**

```bash
uv run pytest tests/test_env.py -v
```

Expected: all pass. If `test_passes_gymnasium_env_checker` fails inside `data_equivalence` on the bool `action_mask`, change `_action_mask` to return `(self._legal & ~self._guessed)` unchanged but cast the copy placed in `info` with `.astype(np.int8)` and update `test_reset_returns_zero_obs_copy_and_info` to expect `np.int8`. Do not change anything else.

- [ ] **Step 6: Delete the top-level environment copy and run everything**

```bash
cd ..
git rm wordle_rl.py
cd gym-wordle
uv run pytest -v
```

Expected: all pass. `agent.py` and `transformer_agent.py` at the repo root still import `wordle_rl`; they are replaced in Tasks 6 and 7.

- [ ] **Step 7: Commit**

```bash
cd ..
git add gym-wordle/gym_wordle gym-wordle/tests/test_env.py
git commit -m "feat(env): vectorized hard-mode mask, Wordle-v0 registration, env_checker clean"
```

---

### Task 4: Shared agent utilities

**Files:**
- Create: `gym-wordle/gym_wordle/agents/common.py`
- Create: `gym-wordle/tests/test_common.py`

**Interfaces:**
- Produces:
  - `invalid_from_mask(action_mask: np.ndarray[bool]) -> np.ndarray[int64]` indices where the mask is False.
  - `masked_argmax(q: torch.Tensor (B,N), invalid) -> (actions LongTensor (B,1), values Tensor (B,1))`. `invalid` is either one 1-D int array applied to every row, or a `list`/`tuple` of B 1-D int arrays, one per row. Dispatch is on type, never on B.
  - `class ReplayBuffer(capacity, obs_size, alpha=0.6, eps=1e-6, rng=None)` with arrays `state`, `next_state` `(capacity, obs_size) int8`, `action int64`, `reward float32`, `done bool`, `priority float64`, list `invalid_next`; methods `__len__`, `add(state, action, reward, next_state, done, invalid_next) -> int`, `sample_prioritized(batch_size, beta) -> (idx, weights float32)`, `sample_with_probs(batch_size, probs) -> idx`, `update_priority(idx, td_error)`, `keep_random_subset(k)`.
  - `save_checkpoint(model, path)` and `load_checkpoint(model_cls, path, device) -> model`. `model.ctor_kwargs` must be a JSON-serializable dict.
  - `word_feature_matrix(words, n_letters) -> torch.FloatTensor (N, 26*n_letters + 26)`.

- [ ] **Step 1: Write the failing tests**

`tests/test_common.py`:

```python
import json

import numpy as np
import pytest
import torch
import torch.nn as nn

from gym_wordle.agents.common import (
    ReplayBuffer,
    invalid_from_mask,
    load_checkpoint,
    masked_argmax,
    save_checkpoint,
    word_feature_matrix,
)


def test_invalid_from_mask():
    mask = np.array([True, False, True, False])
    assert invalid_from_mask(mask).tolist() == [1, 3]
    assert invalid_from_mask(np.ones(3, dtype=bool)).shape == (0,)


def test_masked_argmax_single_mask_applies_to_all_rows():
    q = torch.tensor([[1.0, 5.0, 3.0], [9.0, 0.0, 2.0]])
    a, v = masked_argmax(q, np.array([1]))
    assert a.tolist() == [[2], [0]] and v.tolist() == [[3.0], [9.0]]
    assert a.dtype == torch.int64


def test_masked_argmax_list_of_masks_one_per_row():
    q = torch.tensor([[1.0, 5.0, 3.0], [9.0, 0.0, 2.0]])
    a, _ = masked_argmax(q, [np.array([1]), np.array([0, 2])])
    assert a.tolist() == [[2], [1]]


def test_masked_argmax_list_with_one_row_does_not_crash():
    q = torch.tensor([[1.0, 5.0, 3.0]])
    a, v = masked_argmax(q, [np.array([1])])
    assert a.tolist() == [[2]] and v.tolist() == [[3.0]]


def test_masked_argmax_does_not_mutate_input():
    q = torch.tensor([[1.0, 5.0]])
    masked_argmax(q, np.array([1]))
    assert q.tolist() == [[1.0, 5.0]]


@pytest.fixture
def buf():
    return ReplayBuffer(capacity=4, obs_size=3, rng=np.random.default_rng(0))


def _add(buf, reward=0.0, done=False):
    return buf.add(np.zeros(3, np.int8), 1, reward, np.ones(3, np.int8), done, np.array([0]))


def test_new_rows_get_current_max_priority(buf):
    _add(buf)
    assert buf.priority[0] == 1.0
    buf.update_priority(np.array([0]), np.array([0.25]))
    _add(buf)
    assert buf.priority[1] == 0.25
    buf.update_priority(np.array([1]), np.array([2.0]))
    _add(buf)
    assert buf.priority[2] == 2.0


def test_ring_wraps_and_len_caps(buf):
    for _ in range(6):
        _add(buf)
    assert len(buf) == 4 and buf.n_seen == 6


def test_update_priority_floors_at_eps(buf):
    _add(buf)
    buf.update_priority(np.array([0]), np.array([0.0]))
    assert buf.priority[0] == buf.eps


def test_sample_prioritized_shapes_and_weight_range(buf):
    for r in [0.0, 1.0, 0.0]:
        _add(buf, r)
    buf.update_priority(np.array([0, 1, 2]), np.array([0.1, 1.0, 0.5]))
    idx, w = buf.sample_prioritized(8, beta=0.4)
    assert idx.shape == (8,) and w.shape == (8,) and w.dtype == np.float32
    assert idx.max() < 3
    assert 0 < w.min() and w.max() <= 1.0 + 1e-6


def test_sample_with_probs(buf):
    for _ in range(3):
        _add(buf)
    idx = buf.sample_with_probs(5, np.array([0.0, 1.0, 0.0]))
    assert idx.tolist() == [1] * 5


def test_keep_random_subset(buf):
    for r in [0.0, 1.0, 2.0, 3.0]:
        _add(buf, r)
    buf.keep_random_subset(2)
    assert len(buf) == 2 and buf.n_seen == 2
    assert set(buf.reward[:2].tolist()) <= {0.0, 1.0, 2.0, 3.0}
    assert len(buf.reward[:2].tolist()) == len(set(buf.reward[:2].tolist()))
    assert all(isinstance(x, np.ndarray) for x in buf.invalid_next[:2])


class Tiny(nn.Module):
    def __init__(self, n_in, n_out):
        super().__init__()
        self.lin = nn.Linear(n_in, n_out)
        self.register_buffer("table", torch.zeros(n_out))
        self.ctor_kwargs = {"n_in": n_in, "n_out": n_out}

    def forward(self, x):
        return self.lin(x) + self.table


def test_checkpoint_roundtrip(tmp_path):
    m = Tiny(3, 2)
    m.table[:] = torch.tensor([1.0, 2.0])
    path = tmp_path / "models" / "m.pt"  # directory does not exist yet
    save_checkpoint(m, str(path))
    assert path.exists() and (tmp_path / "models" / "m.pt.json").exists()
    meta = json.load(open(str(path) + ".json"))
    assert meta == {"class": "Tiny", "kwargs": {"n_in": 3, "n_out": 2}}
    m2 = load_checkpoint(Tiny, str(path), torch.device("cpu"))
    x = torch.randn(4, 3)
    assert torch.allclose(m(x), m2(x))


def test_word_feature_matrix():
    phi = word_feature_matrix(["apple", "stale", "slate"], 5)
    assert phi.shape == (3, 156) and phi.dtype == torch.float32
    row = phi[0]
    assert row[:130].sum() == 5  # one letter per position
    assert row[130:].sum() == 5  # counts sum to word length
    assert row[130 + ord("p") - ord("a")] == 2
    assert torch.equal(phi[1, 130:], phi[2, 130:])  # anagrams share counts
    assert not torch.equal(phi[1, :130], phi[2, :130])  # but not positions
```

- [ ] **Step 2: Run to verify failure**

```bash
uv run pytest tests/test_common.py -v
```

Expected: ImportError on `gym_wordle.agents.common`.

- [ ] **Step 3: Write common.py**

`gym_wordle/agents/common.py`:

```python
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
    invalid: one 1-D integer array applied to every row, or a list/tuple of
    B such arrays (one per row). Dispatch is on the type of `invalid`, so a
    batch of size one with a one-element list works.
    Returns (actions (B,1) int64, values (B,1)).
    """
    q = q.clone()
    if isinstance(invalid, (list, tuple)):
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

    `invalid_next[i]` holds the actions that are invalid in `next_state[i]`
    (already guessed or hard-mode illegal). It is used only to mask the
    bootstrap argmax over `next_state`.
    """

    def __init__(self, capacity, obs_size, alpha=0.6, eps=1e-6, rng=None):
        self.capacity = capacity
        self.alpha = alpha
        self.eps = eps
        self.rng = rng if rng is not None else np.random.default_rng()
        self.state = np.zeros((capacity, obs_size), dtype=np.int8)
        self.next_state = np.zeros((capacity, obs_size), dtype=np.int8)
        self.action = np.zeros(capacity, dtype=np.int64)
        self.reward = np.zeros(capacity, dtype=np.float32)
        self.done = np.zeros(capacity, dtype=bool)
        self.priority = np.zeros(capacity, dtype=np.float64)
        self.invalid_next = [None] * capacity
        self.n_seen = 0

    def __len__(self):
        return min(self.n_seen, self.capacity)

    def add(self, state, action, reward, next_state, done, invalid_next):
        n = len(self)
        i = self.n_seen % self.capacity
        self.state[i] = state
        self.next_state[i] = next_state
        self.action[i] = int(action)
        self.reward[i] = reward
        self.done[i] = bool(done)
        self.invalid_next[i] = np.asarray(invalid_next, dtype=np.int64)
        # Standard PER: a new transition gets the current max priority so it
        # is replayed at least once soon.
        self.priority[i] = self.priority[:n].max() if n > 0 else 1.0
        self.n_seen += 1
        return i

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
        for name in ("state", "next_state", "action", "reward", "done", "priority"):
            arr = getattr(self, name)
            arr[: keep.size] = arr[keep]
        self.invalid_next[: keep.size] = [self.invalid_next[i] for i in keep]
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
```

- [ ] **Step 4: Run tests**

```bash
uv run pytest tests/test_common.py -v
```

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add gym_wordle/agents/common.py tests/test_common.py
git commit -m "feat(agents): shared masking, replay buffer, checkpoint IO, word features"
```

---

### Task 5: SE-ResNet model and observation-to-input mapping

**Files:**
- Create: `gym-wordle/gym_wordle/agents/dqn.py` (model half; trainer added in Task 6)
- Create: `gym-wordle/tests/test_dqn.py`

**Interfaces:**
- Consumes: `save_checkpoint`, `load_checkpoint` from Task 4; `WordleEnv` layout constants from Task 2.
- Produces: `ResNN(n_actions, n_letters=5, in_channels=4, channels=12, kernel_size=3, head="flat", d_head=128, word_features=None)` with `.ctor_kwargs` and `forward(x: (B,4,26,n_letters)) -> (B, n_actions)`; `obs_to_input(obs, env, device) -> torch.FloatTensor (B,4,26,n_letters)` accepting `(obs_size,)` or `(B, obs_size)`.

- [ ] **Step 1: Write the failing tests**

`tests/test_dqn.py`:

```python
import numpy as np
import pytest
import torch

from gym_wordle.agents.common import load_checkpoint, save_checkpoint
from gym_wordle.agents.dqn import ResNN, obs_to_input
from gym_wordle.envs.wordle_env import WordleEnv, LETTER_INDEX

CPU = torch.device("cpu")


@pytest.fixture
def env(env_paths):
    return WordleEnv(**env_paths)


def test_obs_to_input_layout(env):
    env.reset(options={"secret_word": "apple"})
    obs, *_ = env.step(env.word_to_action["puppy"])
    x = obs_to_input(obs, env, CPU)
    assert x.shape == (1, 4, 26, 5) and x.dtype == torch.float32
    p = LETTER_INDEX["p"]
    assert x[0, 0].unique().tolist() == [(1 - 1) / 5]  # turn channel is constant
    assert x[0, 1, p].tolist() == [0.0, 1.0, 2.0, 0.0, 1.0]
    assert x[0, 2, p].tolist() == [1.0] * 5  # exceeded bit broadcast over positions
    assert x[0, 3, p].tolist() == [1.0] * 5  # yellow count broadcast
    batch = obs_to_input(np.stack([obs, obs]), env, CPU)
    assert batch.shape == (2, 4, 26, 5)


def test_resnn_flat_output_shape_and_kwargs(env):
    m = ResNN(n_actions=env.action_space.n, channels=4)
    x = torch.zeros(3, 4, 26, 5)
    assert m(x).shape == (3, env.action_space.n)
    assert m.ctor_kwargs["head"] == "flat" and m.ctor_kwargs["n_actions"] == 18
    m.eval()
    assert m(torch.zeros(1, 4, 26, 5)).shape == (1, 18)  # batch of one in eval


def test_resnn_batch_of_one_in_train_mode(env):
    m = ResNN(n_actions=18, channels=4)
    m.train()
    m(torch.randn(1, 4, 26, 5)).sum().backward()


def test_resnn_checkpoint_roundtrip(tmp_path, env):
    m = ResNN(n_actions=18, channels=4).eval()
    save_checkpoint(m, str(tmp_path / "models" / "m.pt"))
    m2 = load_checkpoint(ResNN, str(tmp_path / "models" / "m.pt"), CPU).eval()
    x = torch.randn(2, 4, 26, 5)
    assert torch.allclose(m(x), m2(x))
```

- [ ] **Step 2: Run to verify failure**

```bash
uv run pytest tests/test_dqn.py -v
```

Expected: ImportError on `gym_wordle.agents.dqn`.

- [ ] **Step 3: Write the model half of dqn.py**

`gym_wordle/agents/dqn.py`:

```python
"""SE-ResNet Double DQN with prioritized replay for Wordle."""
import argparse
import datetime as dt
import logging
import os
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn

from gym_wordle.agents.common import (
    ReplayBuffer,
    invalid_from_mask,
    load_checkpoint,
    masked_argmax,
    save_checkpoint,
    word_feature_matrix,
)
from gym_wordle.envs.wordle_env import N_ALPHABET, WordleEnv

log = logging.getLogger(__name__)


# ----------------------------------------------------------------- input map

def obs_to_input(obs, env, device):
    """Flat observation(s) -> (B, 4, 26, n_letters) float tensor.

    Channel 0: scaled turn counter, constant over the grid.
    Channel 1: positional status grid (letter x position).
    Channel 2: exceeded bit per letter, broadcast over positions.
    Channel 3: max-yellows per letter, broadcast over positions.
    """
    obs = np.atleast_2d(np.asarray(obs)).astype(np.float32)
    b, n = obs.shape[0], env.n_letters
    x = np.zeros((b, 4, N_ALPHABET, n), dtype=np.float32)
    x[:, 0] = ((obs[:, 0] - 1) / 5).reshape(b, 1, 1)
    x[:, 1] = obs[:, env.POS : env.EXCEEDED].reshape(b, N_ALPHABET, n)
    x[:, 2] = obs[:, env.EXCEEDED : env.YELLOWS].reshape(b, N_ALPHABET, 1)
    x[:, 3] = obs[:, env.YELLOWS : env.obs_size].reshape(b, N_ALPHABET, 1)
    return torch.as_tensor(x, device=device)


# --------------------------------------------------------------------- model

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, 1, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return self.bn(self.conv(x))


class SEBlock(nn.Module):
    """Squeeze-and-excitation channel gate."""

    def __init__(self, channels, reduction=4):
        super().__init__()
        squeezed = max(1, channels // reduction)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, squeezed, bias=False),
            nn.Mish(inplace=True),
            nn.Linear(squeezed, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        n, c, _, _ = x.shape
        y = self.fc(self.pool(x).view(n, c)).view(n, c, 1, 1)
        return x * y


class ResidualBlock(nn.Module):
    def __init__(self, channels, kernel_size, padding):
        super().__init__()
        self.conv1 = ConvBlock(channels, channels, kernel_size, padding)
        self.conv2 = ConvBlock(channels, channels, kernel_size, padding)
        self.se = SEBlock(channels)
        self.mish = nn.Mish()

    def forward(self, x):
        a = self.mish(self.conv1(x))
        a = self.se(self.conv2(a))
        return self.mish(a + x)


class ResNN(nn.Module):
    """Conv trunk over the (letter x position) grid, then a Q head over words."""

    def __init__(
        self,
        n_actions,
        n_letters=5,
        in_channels=4,
        channels=12,
        kernel_size=3,
        head="flat",
        d_head=128,
        word_features=None,
    ):
        super().__init__()
        pad = kernel_size // 2
        self.conv_block = ConvBlock(in_channels, channels, kernel_size, pad)
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(channels, kernel_size, pad) for _ in range(3)]
        )
        self.bn = nn.BatchNorm2d(channels)
        self.dropout = nn.Dropout2d(0.1)
        self.mish = nn.Mish()
        flat_dim = channels * N_ALPHABET * n_letters
        if head == "flat":
            self.head = nn.Linear(flat_dim, n_actions)
        else:
            raise ValueError(f"unknown head {head!r}")
        self.ctor_kwargs = dict(
            n_actions=n_actions,
            n_letters=n_letters,
            in_channels=in_channels,
            channels=channels,
            kernel_size=kernel_size,
            head=head,
            d_head=d_head,
        )

    def forward(self, x):
        a = self.dropout(self.mish(self.conv_block(x)))
        a = self.bn(self.res_blocks(a))
        return self.head(a.flatten(1))
```

The imports `argparse`, `datetime`, `logging`, `os`, `deepcopy`, `ReplayBuffer`, `invalid_from_mask`, `masked_argmax`, `word_feature_matrix`, `WordleEnv` are used by Task 6 and Task 8; leave them in place.

- [ ] **Step 4: Run tests**

```bash
uv run pytest tests/test_dqn.py -v
```

Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add gym_wordle/agents/dqn.py tests/test_dqn.py
git commit -m "feat(dqn): SE-ResNet model with ctor kwargs and grid input mapping"
```

---

### Task 6: DQN trainer

**Files:**
- Modify: `gym-wordle/gym_wordle/agents/dqn.py` (append trainer, evaluation, CLI)
- Modify: `gym-wordle/tests/test_dqn.py` (append)
- Delete: `agent.py` (repo root)

**Interfaces:**
- Consumes: `ResNN`, `obs_to_input` (Task 5); `ReplayBuffer`, `invalid_from_mask`, `masked_argmax`, `save_checkpoint`, `load_checkpoint` (Task 4); `WordleEnv` (Tasks 2-3).
- Produces: `DQNTrainer(env, device="cpu", fixed_start=None, seed=19, head="flat", channels=12, model_dir="models", alpha=3e-4, epsilon=0.99, min_epsilon=0.02, gamma=0.9999, p_alpha=0.6, p_beta=0.4)` with methods `build_model()`, `q_values(model, obs)`, `choose_action(obs, info) -> int`, `setup(max_episodes, checkpoint=None, teacher_checkpoint=None, batch_size=64, max_exp=30000, clip_val=2.0, T_max=None)`, `learn(teacher_influence=0.0) -> float`, `solve(max_episodes, ...) -> (model, path)`, `shuffle_solutions(n_sols, max_episodes) -> list[int]`, `save(prefix) -> path`; attributes `model`, `target_model`, `teacher_model`, `buffer`, `beta`, `scheduler`, `rewards`. Module functions `run_episode(trainer, model, secret_word=None, fixed_start=None) -> float`, `evaluate(trainer, model, secret_words=None, episodes=100) -> list[float]`, `main(argv=None)`.

- [ ] **Step 1: Append the failing tests**

Append to `tests/test_dqn.py`:

```python
import os

from gym_wordle.agents.common import invalid_from_mask
from gym_wordle.agents.dqn import DQNTrainer, run_episode


def make_trainer(env, tmp_path, **kw):
    return DQNTrainer(env, "cpu", channels=4, model_dir=str(tmp_path / "models"), **kw)


def test_solve_smoke_creates_checkpoint_in_new_models_dir(env, tmp_path):
    t = make_trainer(env, tmp_path)
    model, path = t.solve(max_episodes=3, batch_size=2, max_exp=32, C=2)
    assert os.path.exists(path) and os.path.exists(path + ".json")
    assert path.startswith(str(tmp_path / "models"))
    assert len(t.rewards) == 3
    assert t.beta <= 1.0


def test_learn_with_single_nonterminal_transition(env, tmp_path):
    t = make_trainer(env, tmp_path)
    t.setup(max_episodes=1, batch_size=1, max_exp=8)
    obs, info = env.reset(options={"secret_word": "apple"})
    a = env.word_to_action["crane"]
    next_obs, r, term, _, info = env.step(a)
    assert term is False
    t.buffer.add(obs, a, r, next_obs, term, invalid_from_mask(info["action_mask"]))
    loss = t.learn()
    assert np.isfinite(loss)


def test_learn_leaves_target_in_eval_with_no_grads(env, tmp_path):
    t = make_trainer(env, tmp_path)
    t.setup(max_episodes=1, batch_size=2, max_exp=8)
    obs, info = env.reset(options={"secret_word": "apple"})
    for w in ["crane", "stale"]:
        a = env.word_to_action[w]
        next_obs, r, term, _, info = env.step(a)
        t.buffer.add(obs, a, r, next_obs, term, invalid_from_mask(info["action_mask"]))
        obs = next_obs
    t.learn()
    assert not t.target_model.training
    assert t.model.training
    assert all(p.grad is None for p in t.target_model.parameters())


def test_bootstrap_mask_is_taken_from_post_step_state(env, tmp_path):
    t = make_trainer(env, tmp_path)
    t.solve(max_episodes=1, batch_size=1, max_exp=8)
    first = t.buffer.invalid_next[0]
    # In s' the guess just played is invalid, so the stored mask must contain it.
    assert first.size >= 1
    assert t.buffer.action[0] in first


def test_beta_advances_per_episode_and_clamps(env, tmp_path):
    t = make_trainer(env, tmp_path, p_beta=0.9)
    t.solve(max_episodes=4, batch_size=2, max_exp=32)
    assert t.beta == pytest.approx(1.0)


def test_shuffle_solutions_exact_length_in_whole_passes(env, tmp_path):
    t = make_trainer(env, tmp_path)
    inds = t.shuffle_solutions(5, 12)
    assert len(inds) == 12 and set(inds) <= set(range(5))
    assert sorted(inds[:5]) == list(range(5))
    assert sorted(inds[5:10]) == list(range(5))


def test_scheduler_period_defaults_to_max_episodes(env, tmp_path):
    t = make_trainer(env, tmp_path)
    t.setup(max_episodes=17, batch_size=1, max_exp=8)
    assert t.scheduler.T_max == 17


def test_fixed_start_used_once_then_masked(env, tmp_path):
    t = make_trainer(env, tmp_path, fixed_start="dealt")
    t.setup(max_episodes=1, batch_size=1, max_exp=8)
    t.epsilon = 0.0  # greedy
    obs, info = env.reset(options={"secret_word": "delta"})
    a1 = t.choose_action(obs, info)
    assert env.words[a1] == "dealt"
    obs, _, _, _, info = env.step(a1)
    a2 = t.choose_action(obs, info)
    assert env.words[a2] != "dealt"


def test_run_episode_terminates_with_valid_guesses(env, tmp_path):
    t = make_trainer(env, tmp_path)
    t.setup(max_episodes=1, batch_size=1, max_exp=8)
    total = run_episode(t, t.model, secret_word="apple")
    assert 0.0 <= total <= 1.0
    assert env.terminated
```

- [ ] **Step 2: Run to verify failure**

```bash
uv run pytest tests/test_dqn.py -v
```

Expected: ImportError, `cannot import name 'DQNTrainer'`.

- [ ] **Step 3: Append the trainer to dqn.py**

Append to `gym_wordle/agents/dqn.py`:

```python
# ------------------------------------------------------------------- trainer

class DQNTrainer:
    """Double DQN with proportional prioritized replay on WordleEnv."""

    def __init__(
        self,
        env,
        device="cpu",
        fixed_start=None,
        seed=19,
        head="flat",
        channels=12,
        model_dir="models",
        alpha=3e-4,
        epsilon=0.99,
        min_epsilon=0.02,
        gamma=0.9999,
        p_alpha=0.6,
        p_beta=0.4,
    ):
        self.env = env.unwrapped
        self.device = torch.device(device)
        self.np_rng = np.random.default_rng(seed)
        self.env.reset(seed=seed)
        if fixed_start is None:
            self.fixed_start = None
        elif isinstance(fixed_start, str):
            self.fixed_start = self.env.word_to_action[fixed_start]
        else:
            self.fixed_start = int(fixed_start)
        self.head = head
        self.channels = channels
        self.model_dir = model_dir
        self.alpha = alpha
        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.gamma = gamma
        self.p_alpha = p_alpha
        self.p_beta = p_beta
        self.n_actions = int(self.env.action_space.n)
        self.model = self.target_model = self.teacher_model = None
        self.n_episodes = 0
        self.rewards = []

    # -- model -------------------------------------------------------------

    def build_model(self):
        return ResNN(
            n_actions=self.n_actions,
            n_letters=self.env.n_letters,
            channels=self.channels,
            head=self.head,
        ).to(self.device)

    def q_values(self, model, obs):
        """Q(s, .) in eval mode under no_grad. Restores the model's prior mode."""
        was_training = model.training
        model.eval()
        with torch.no_grad():
            q = model(obs_to_input(obs, self.env, self.device))
        if was_training:
            model.train()
        return q

    def sync_target(self):
        self.target_model = deepcopy(self.model).eval()

    # -- acting ------------------------------------------------------------

    def choose_action(self, obs, info):
        mask = info["action_mask"]
        if self.fixed_start is not None and mask[self.fixed_start]:
            return self.fixed_start
        if self.np_rng.random() > self.epsilon:
            a, _ = masked_argmax(self.q_values(self.model, obs), invalid_from_mask(mask))
            return int(a.item())
        return int(self.np_rng.choice(np.flatnonzero(mask)))

    # -- training ----------------------------------------------------------

    def setup(
        self,
        max_episodes,
        checkpoint=None,
        teacher_checkpoint=None,
        batch_size=64,
        max_exp=30000,
        clip_val=2.0,
        T_max=None,
    ):
        if checkpoint:
            self.model = load_checkpoint(ResNN, checkpoint, self.device)
        else:
            self.model = self.build_model()
        self.model.train()
        self.teacher_model = (
            load_checkpoint(ResNN, teacher_checkpoint, self.device).eval()
            if teacher_checkpoint
            else None
        )
        for p in self.model.parameters():
            p.register_hook(lambda g: torch.clamp(g, -clip_val, clip_val))
        self.sync_target()
        self.loss_fx = nn.MSELoss(reduction="none")
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.alpha, betas=(0.9, 0.999), weight_decay=1e-2
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=T_max or max_episodes
        )
        self.buffer = ReplayBuffer(
            max_exp, self.env.obs_size, alpha=self.p_alpha, rng=self.np_rng
        )
        self.batch_size = batch_size
        # Decay so epsilon reaches about half of min_epsilon by the end,
        # assuming about five guesses per episode.
        self.epsilon_decay = float(
            np.exp(np.log((self.min_epsilon / 2) / self.epsilon) / (max_episodes * 5))
        )
        self.beta = self.p_beta
        self.beta_inc = (1 - self.p_beta) / max_episodes

    def learn(self, teacher_influence=0.0):
        buf = self.buffer
        idx, is_w = buf.sample_prioritized(self.batch_size, self.beta)
        states = obs_to_input(buf.state[idx], self.env, self.device)
        actions = torch.as_tensor(buf.action[idx], device=self.device).view(-1, 1)
        y = torch.as_tensor(
            buf.reward[idx], device=self.device, dtype=torch.float32
        ).view(-1, 1)

        cont = np.flatnonzero(~buf.done[idx])
        if cont.size:
            next_obs = buf.next_state[idx[cont]]
            invalid_next = [buf.invalid_next[i] for i in idx[cont]]
            # Double DQN: the online net picks a*, the target net values it.
            a_star, _ = masked_argmax(self.q_values(self.model, next_obs), invalid_next)
            next_in = obs_to_input(next_obs, self.env, self.device)
            with torch.no_grad():
                q_next = torch.gather(self.target_model(next_in), 1, a_star)
                if teacher_influence > 0 and self.teacher_model is not None:
                    q_teacher = torch.gather(self.teacher_model(next_in), 1, a_star)
                    q_next = (1 - teacher_influence) * q_next + teacher_influence * q_teacher
            y[torch.as_tensor(cont, device=self.device)] += self.gamma * q_next

        q = torch.gather(self.model(states), 1, actions)
        buf.update_priority(idx, (y - q).detach().cpu().numpy().reshape(-1))
        weights = torch.as_tensor(is_w, device=self.device).view(-1, 1)
        loss = (self.loss_fx(q, y) * weights).mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return float(loss.item())

    def solve(
        self,
        max_episodes,
        checkpoint=None,
        teacher_checkpoint=None,
        C=8,
        batch_size=64,
        max_exp=30000,
        clip_val=2.0,
        warmup=2,
        C_factor=2,
        T_max=None,
    ):
        self.setup(max_episodes, checkpoint, teacher_checkpoint, batch_size, max_exp, clip_val, T_max)
        env = self.env
        n_sols = len(env.solutions)
        sol_inds = self.shuffle_solutions(n_sols, max_episodes)
        C0 = C
        history = []
        for ep in range(max_episodes):
            self.n_episodes = ep
            if ep <= n_sols:
                C = self.modulate_C(C, C0, warmup, C_factor, n_sols)
            influence = (
                self.teacher_influence(max_episodes) if self.teacher_model is not None else 0.0
            )
            obs, info = env.reset(options={"secret_word": env.solutions[sol_inds[ep]]})
            episode_reward, terminated = 0.0, False
            while not terminated:
                action = self.choose_action(obs, info)
                next_obs, reward, terminated, _, info = env.step(action)
                while not info["valid"]:
                    # choose_action only proposes masked-valid actions, so this
                    # loop is defensive; the env has already dropped `action`
                    # from the mask carried in `info`.
                    action = self.choose_action(obs, info)
                    next_obs, reward, terminated, _, info = env.step(action)
                self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
                episode_reward += reward
                # The stored mask describes next_obs: it is used only to mask
                # the bootstrap argmax over next_obs.
                self.buffer.add(
                    obs, action, reward, next_obs, terminated,
                    invalid_from_mask(info["action_mask"]),
                )
                self.learn(influence)
                obs = next_obs
            self.beta = min(1.0, self.beta + self.beta_inc)
            self.scheduler.step()
            history.append(episode_reward)
            if (ep + 1) % C == 0:
                self.sync_target()
            if (ep + 1) % 100 == 0:
                log.info(
                    "episode %d  mean reward (last 100) %.3f  epsilon %.3f  lr %.2e",
                    ep + 1, float(np.mean(history[-100:])), self.epsilon,
                    self.optimizer.param_groups[0]["lr"],
                )
            if max_episodes >= 2 and ep + 1 == max_episodes // 2:
                self.save("model_halftrain")
        path = self.save("model")
        self.rewards = history
        return self.model, path

    # -- schedules ---------------------------------------------------------

    def shuffle_solutions(self, n_sols, max_episodes):
        """Exactly max_episodes indices: whole shuffled passes, then a partial one."""
        out = []
        while len(out) < max_episodes:
            out.extend(self.np_rng.permutation(n_sols).tolist())
        return out[:max_episodes]

    def modulate_C(self, C, C0, warmup, C_factor, n_sols):
        if self.n_episodes < warmup * C0:
            return max(2, int((1 + self.n_episodes) / warmup))
        if self.n_episodes == n_sols:
            return C * C_factor
        return C

    def teacher_influence(self, max_episodes):
        """Exponential decay plus a linear term that reaches 0 near halfway."""
        return max(
            0.0,
            float(
                np.exp(-5 * (1 + self.n_episodes) / max_episodes)
                - 0.164 * (self.n_episodes / max_episodes)
            ),
        )

    def save(self, prefix):
        stamp = dt.datetime.now().strftime("%Y%m%d.%H.%M.%S")
        path = os.path.join(self.model_dir, f"{prefix}_{stamp}.pt")
        save_checkpoint(self.model, path)
        log.info("saved %s", path)
        return path


# ---------------------------------------------------------------- evaluation

def run_episode(trainer, model, secret_word=None, fixed_start=None):
    env = trainer.env
    options = {"secret_word": secret_word} if secret_word else None
    obs, info = env.reset(options=options)
    total, terminated, guesses = 0.0, False, []
    while not terminated:
        if fixed_start is not None and info["action_mask"][fixed_start]:
            action = fixed_start
        else:
            a, _ = masked_argmax(
                trainer.q_values(model, obs), invalid_from_mask(info["action_mask"])
            )
            action = int(a.item())
        obs, reward, terminated, _, info = env.step(action)
        assert info["valid"], "a masked greedy action must be valid"
        guesses.append(env.words[action])
        total += reward
    log.info("secret %s  guesses %s  reward %.3f", env.secret_word, guesses, total)
    return total


def evaluate(trainer, model, secret_words=None, episodes=100):
    if secret_words is None:
        history = [
            run_episode(trainer, model, fixed_start=trainer.fixed_start)
            for _ in range(episodes)
        ]
    else:
        history = [
            run_episode(trainer, model, w, fixed_start=trainer.fixed_start)
            for w in secret_words
        ]
    fail_rate = sum(r == 0.0 for r in history) / len(history)
    log.info("fail rate %.3f  mean reward %.3f", fail_rate, float(np.mean(history)))
    return history


# ----------------------------------------------------------------------- cli

def main(argv=None):
    parser = argparse.ArgumentParser(description="Train or evaluate the SE-ResNet DQN on Wordle")
    parser.add_argument("--episodes", type=int, default=4618)
    parser.add_argument("--head", choices=["flat"], default="flat")
    parser.add_argument("--channels", type=int, default=12)
    parser.add_argument("--fixed-start", default=None, help="opening word, e.g. dealt")
    parser.add_argument("--checkpoint", default=None, help="resume from this .pt file")
    parser.add_argument("--teacher", default=None, help="teacher .pt file for Q_prime blending")
    parser.add_argument("--eval-only", action="store_true", help="evaluate --checkpoint on every solution")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--model-dir", default="models")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    env = WordleEnv()
    trainer = DQNTrainer(
        env, args.device, fixed_start=args.fixed_start, head=args.head,
        channels=args.channels, model_dir=args.model_dir,
    )
    if args.eval_only:
        model = load_checkpoint(ResNN, args.checkpoint, trainer.device)
        evaluate(trainer, model, secret_words=env.solutions)
        return
    model, _ = trainer.solve(
        args.episodes, checkpoint=args.checkpoint,
        teacher_checkpoint=args.teacher, batch_size=args.batch_size,
    )
    evaluate(trainer, model, secret_words=env.solutions)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests**

```bash
uv run pytest tests/test_dqn.py -v
```

Expected: all pass.

- [ ] **Step 5: Smoke run against the real word lists**

```bash
SMOKE=$(mktemp -d)
uv run python -m gym_wordle.agents.dqn --episodes 2 --channels 2 --device cpu --model-dir "$SMOKE" --batch-size 4
ls "$SMOKE"
```

Expected: two episodes train, `ls` shows a `model_*.pt` and its `.json`, and evaluation over all 2309 solutions completes with a logged fail rate. The temp directory is disposable.

- [ ] **Step 6: Delete the old agent and commit**

```bash
cd ..
git rm agent.py
git add gym-wordle/gym_wordle/agents/dqn.py gym-wordle/tests/test_dqn.py
git commit -m "feat(dqn): trainer with eval-mode targets, post-step masks, per-episode beta, state_dict checkpoints"
cd gym-wordle
```

---

### Task 7: Transformer DQN with corrected positional encoding

**Files:**
- Create: `gym-wordle/gym_wordle/agents/transformer.py`
- Create: `gym-wordle/tests/test_transformer.py`
- Delete: `transformer_agent.py` (repo root)

**Interfaces:**
- Consumes: `ReplayBuffer`, `invalid_from_mask`, `masked_argmax`, `save_checkpoint`, `load_checkpoint` (Task 4); `WordleEnv`.
- Produces: `PositionalEncoding(d_model, dropout, max_seq_len=183)`; `TransformerQNet(n_actions, n_tokens=14, d_model=128, nhead=8, d_hid=512, nlayers=2, dropout=0.1, max_seq_len=183)` with `.ctor_kwargs`, `forward(x: (B,S) ints) -> (B, n_actions)`; `TransformerTrainer(env, device="cpu", seed=19, model_dir="models", alpha=1e-4, epsilon=0.9, min_epsilon=0.02, gamma=0.9999, d_model=128, nhead=8, d_hid=512, nlayers=2)` with `setup(max_episodes, checkpoint=None, teacher_checkpoint=None, batch_size=64, max_experience=100000, clip_val=2.0)`, `learn(teacher_influence=0.0) -> float`, `solve(max_episodes, ...) -> (model, path)`, `choose_action`, `q_values`, `to_input`, `replay_probs`, `save`; `run_episode(trainer, model, secret_word=None)`; `main(argv=None)`.

The old `invalid_batch_size` option (pushing Q of sampled invalid actions toward 0) defaulted to 0 and is dropped.

- [ ] **Step 1: Write the failing tests**

`tests/test_transformer.py`:

```python
import math
import os

import numpy as np
import pytest
import torch

from gym_wordle.agents.common import invalid_from_mask, load_checkpoint, save_checkpoint
from gym_wordle.agents.transformer import (
    PositionalEncoding,
    TransformerQNet,
    TransformerTrainer,
)
from gym_wordle.envs.wordle_env import WordleEnv

CPU = torch.device("cpu")
SMALL = dict(d_model=16, nhead=2, d_hid=32, nlayers=1)


@pytest.fixture
def env(env_paths):
    return WordleEnv(**env_paths)


def make_trainer(env, tmp_path, **kw):
    return TransformerTrainer(env, "cpu", model_dir=str(tmp_path / "models"), **SMALL, **kw)


def test_positional_encoding_indexes_sequence_not_batch():
    pe = PositionalEncoding(d_model=16, dropout=0.0, max_seq_len=10).eval()
    x = torch.zeros(200, 10, 16)  # batch larger than max_seq_len must work
    out = pe(x)
    assert out.shape == x.shape
    assert torch.allclose(out[0], out[199])  # every batch row gets the same encoding
    assert not torch.allclose(out[0, 0], out[0, 1])  # which varies along the sequence
    div = torch.exp(torch.arange(0, 16, 2) * (-math.log(10000.0) / 16))
    assert torch.allclose(out[0, 1, 0::2], torch.sin(1.0 * div))
    assert torch.allclose(out[0, 1, 1::2], torch.cos(1.0 * div))


def test_qnet_output_shape_and_kwargs(env):
    m = TransformerQNet(n_actions=18, **SMALL).eval()
    obs, _ = env.reset(options={"secret_word": "apple"})
    x = torch.as_tensor(np.stack([obs, obs]).astype(np.int64))
    assert m(x).shape == (2, 18)
    assert m.ctor_kwargs["n_actions"] == 18 and m.ctor_kwargs["d_model"] == 16


def test_qnet_checkpoint_roundtrip(tmp_path):
    m = TransformerQNet(n_actions=18, **SMALL).eval()
    save_checkpoint(m, str(tmp_path / "m.pt"))
    m2 = load_checkpoint(TransformerQNet, str(tmp_path / "m.pt"), CPU).eval()
    x = torch.randint(0, 7, (2, 183))
    assert torch.allclose(m(x), m2(x))


def test_solve_smoke(env, tmp_path):
    t = make_trainer(env, tmp_path)
    model, path = t.solve(max_episodes=3, batch_size=2, max_experience=32, C=2)
    assert os.path.exists(path) and os.path.exists(path + ".json")
    assert len(t.rewards) == 3


def test_learn_with_terminal_row_whose_mask_is_empty_is_finite(env, tmp_path):
    t = make_trainer(env, tmp_path)
    t.setup(max_episodes=1, batch_size=2, max_experience=8)
    obs, info = env.reset(options={"secret_word": "apple"})
    a = env.word_to_action["apple"]
    next_obs, r, term, _, info = env.step(a)  # win: every remaining word is now masked
    assert term is True and not info["action_mask"].any()
    t.buffer.add(obs, a, r, next_obs, term, invalid_from_mask(info["action_mask"]))
    assert np.isfinite(t.learn())


def test_learn_leaves_target_in_eval_with_no_grads(env, tmp_path):
    t = make_trainer(env, tmp_path)
    t.setup(max_episodes=1, batch_size=2, max_experience=8)
    obs, info = env.reset(options={"secret_word": "apple"})
    for w in ["crane", "stale"]:
        a = env.word_to_action[w]
        next_obs, r, term, _, info = env.step(a)
        t.buffer.add(obs, a, r, next_obs, term, invalid_from_mask(info["action_mask"]))
        obs = next_obs
    t.learn()
    assert not t.target_model.training and t.model.training
    assert all(p.grad is None for p in t.target_model.parameters())


def test_buffer_is_thinned_every_two_passes(env, tmp_path):
    t = make_trainer(env, tmp_path)
    n_sols = len(env.solutions)
    t.solve(max_episodes=2 * n_sols, batch_size=2, max_experience=200, C=2)
    assert len(t.buffer) <= int(math.sqrt(2 * n_sols * env.max_attempts))


def test_stored_observations_are_copies(env, tmp_path):
    t = make_trainer(env, tmp_path)
    t.solve(max_episodes=1, batch_size=1, max_experience=8)
    assert not t.buffer.state[0].any()  # first stored state is the empty board
    assert t.buffer.next_state[0][0] == 1
```

- [ ] **Step 2: Run to verify failure**

```bash
uv run pytest tests/test_transformer.py -v
```

Expected: ImportError on `gym_wordle.agents.transformer`.

- [ ] **Step 3: Write transformer.py**

`gym_wordle/agents/transformer.py`:

```python
"""Transformer DQN that reads the flat observation as a token sequence."""
import argparse
import datetime as dt
import logging
import math
import os
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer

from gym_wordle.agents.common import (
    ReplayBuffer,
    invalid_from_mask,
    load_checkpoint,
    masked_argmax,
    save_checkpoint,
)
from gym_wordle.envs.wordle_env import WordleEnv

log = logging.getLogger(__name__)


class PositionalEncoding(nn.Module):
    """Sinusoidal encoding indexed by sequence position, batch-first."""

    def __init__(self, d_model, dropout, max_seq_len=183):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_seq_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x: (batch, seq, d_model). Slice along the sequence axis, not the batch.
        return self.dropout(x + self.pe[:, : x.size(1)])


class TransformerQNet(nn.Module):
    """Embed each observation slot as a token; read Q-values from the last token."""

    def __init__(
        self,
        n_actions,
        n_tokens=14,
        d_model=128,
        nhead=8,
        d_hid=512,
        nlayers=2,
        dropout=0.1,
        max_seq_len=183,
    ):
        super().__init__()
        self.embedding = nn.Embedding(n_tokens, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_seq_len)
        layer = TransformerEncoderLayer(d_model, nhead, d_hid, dropout, batch_first=True)
        self.encoder = TransformerEncoder(layer, nlayers)
        self.decoder = nn.Linear(d_model, n_actions)
        self.d_model = d_model
        self.ctor_kwargs = dict(
            n_actions=n_actions, n_tokens=n_tokens, d_model=d_model, nhead=nhead,
            d_hid=d_hid, nlayers=nlayers, dropout=dropout, max_seq_len=max_seq_len,
        )
        self.init_weights()

    def init_weights(self):
        r = 0.1
        self.embedding.weight.data.uniform_(-r, r)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-r, r)

    def forward(self, x):
        x = self.embedding(x.long()) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)
        return self.decoder(self.encoder(x)[:, -1, :])


class TransformerTrainer:
    """Plain DQN (target-net max) with reward-weighted replay sampling."""

    def __init__(
        self,
        env,
        device="cpu",
        seed=19,
        model_dir="models",
        alpha=1e-4,
        epsilon=0.9,
        min_epsilon=0.02,
        gamma=0.9999,
        d_model=128,
        nhead=8,
        d_hid=512,
        nlayers=2,
    ):
        self.env = env.unwrapped
        self.device = torch.device(device)
        self.np_rng = np.random.default_rng(seed)
        self.env.reset(seed=seed)
        self.model_dir = model_dir
        self.alpha = alpha
        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.gamma = gamma
        self.arch = dict(d_model=d_model, nhead=nhead, d_hid=d_hid, nlayers=nlayers)
        self.n_actions = int(self.env.action_space.n)
        self.model = self.target_model = self.teacher_model = None
        self.n_episodes = 0
        self.rewards = []

    # -- model -------------------------------------------------------------

    def build_model(self):
        return TransformerQNet(
            n_actions=self.n_actions, max_seq_len=self.env.obs_size, **self.arch
        ).to(self.device)

    def to_input(self, obs):
        return torch.as_tensor(
            np.atleast_2d(np.asarray(obs)).astype(np.int64), device=self.device
        )

    def q_values(self, model, obs):
        was_training = model.training
        model.eval()
        with torch.no_grad():
            q = model(self.to_input(obs))
        if was_training:
            model.train()
        return q

    def sync_target(self):
        self.target_model = deepcopy(self.model).eval()

    def set_lr(self, lr):
        for group in self.optimizer.param_groups:
            group["lr"] = lr

    def scale_lr(self, factor):
        for group in self.optimizer.param_groups:
            group["lr"] *= factor

    # -- acting ------------------------------------------------------------

    def choose_action(self, obs, info):
        mask = info["action_mask"]
        if self.np_rng.random() > self.epsilon:
            a, _ = masked_argmax(self.q_values(self.model, obs), invalid_from_mask(mask))
            return int(a.item())
        return int(self.np_rng.choice(np.flatnonzero(mask)))

    # -- training ----------------------------------------------------------

    def setup(
        self,
        max_episodes,
        checkpoint=None,
        teacher_checkpoint=None,
        batch_size=64,
        max_experience=100000,
        clip_val=2.0,
    ):
        if checkpoint:
            self.model = load_checkpoint(TransformerQNet, checkpoint, self.device)
        else:
            self.model = self.build_model()
        self.model.train()
        self.teacher_model = (
            load_checkpoint(TransformerQNet, teacher_checkpoint, self.device).eval()
            if teacher_checkpoint
            else None
        )
        for p in self.model.parameters():
            p.register_hook(lambda g: torch.clamp(g, -clip_val, clip_val))
        self.sync_target()
        self.loss_fx = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.alpha)
        self.buffer = ReplayBuffer(max_experience, self.env.obs_size, rng=self.np_rng)
        self.batch_size = batch_size
        self.epsilon_decay = float(
            np.exp(np.log((self.min_epsilon / 2) / self.epsilon) / (max_episodes * 5))
        )

    def replay_probs(self):
        """Favor rewarded and later-turn transitions, as the original did."""
        n = len(self.buffer)
        return np.minimum(0.9, self.buffer.reward[:n]) + (
            np.log(2 + n) + self.buffer.state[:n, 0]
        ) / 100

    def learn(self, teacher_influence=0.0):
        buf = self.buffer
        idx = buf.sample_with_probs(self.batch_size, self.replay_probs())
        states = self.to_input(buf.state[idx])
        actions = torch.as_tensor(buf.action[idx], device=self.device).view(-1, 1)
        rewards = torch.as_tensor(
            buf.reward[idx], device=self.device, dtype=torch.float32
        ).view(-1, 1)
        done = torch.as_tensor(buf.done[idx], device=self.device).view(-1, 1)
        next_in = self.to_input(buf.next_state[idx])
        invalid_next = [buf.invalid_next[i] for i in idx]
        with torch.no_grad():
            _, q_next = masked_argmax(self.target_model(next_in), invalid_next)
            if teacher_influence > 0 and self.teacher_model is not None:
                _, q_teacher = masked_argmax(self.teacher_model(next_in), invalid_next)
                q_next = (1 - teacher_influence) * q_next + teacher_influence * q_teacher
        # A terminal next-state may have every action masked (q_next = -inf);
        # torch.where selects the plain reward for those rows.
        y = torch.where(done, rewards, rewards + self.gamma * q_next)
        q = torch.gather(self.model(states), 1, actions)
        loss = self.loss_fx(q, y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return float(loss.item())

    def solve(
        self,
        max_episodes,
        checkpoint=None,
        teacher_checkpoint=None,
        C=8,
        batch_size=64,
        max_experience=100000,
        clip_val=2.0,
        warmup=2,
        C_factor=4,
        settle_rate=0.3,
        lr_drop_at=0.9,
        lr_drop_rate=0.1,
    ):
        self.setup(max_episodes, checkpoint, teacher_checkpoint, batch_size, max_experience, clip_val)
        env = self.env
        n_sols = len(env.solutions)
        C0 = C
        history = []
        for ep in range(max_episodes):
            self.n_episodes = ep
            if ep > 0 and ep == int(lr_drop_at * max_episodes):
                self.scale_lr(lr_drop_rate)
                self.min_epsilon *= lr_drop_rate
                C *= C_factor
            elif ep == n_sols:
                self.scale_lr(settle_rate)
                C *= C_factor
            if ep < warmup * C0:
                self.set_lr(self.alpha * (1 + ep) / (warmup * C0))
                C = max(2, int((1 + ep) / warmup))
            influence = (
                self.teacher_influence(max_episodes) if self.teacher_model is not None else 0.0
            )
            obs, info = env.reset(options={"secret_word": env.solutions[ep % n_sols]})
            episode_reward, terminated = 0.0, False
            while not terminated:
                action = self.choose_action(obs, info)
                next_obs, reward, terminated, _, info = env.step(action)
                while not info["valid"]:
                    action = self.choose_action(obs, info)
                    next_obs, reward, terminated, _, info = env.step(action)
                self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
                episode_reward += reward
                self.buffer.add(
                    obs, action, reward, next_obs, terminated,
                    invalid_from_mask(info["action_mask"]),
                )
                self.learn(influence)
                obs = next_obs
            history.append(episode_reward)
            if (ep + 1) % C == 0:
                self.sync_target()
            if (ep + 1) % (2 * n_sols) == 0:
                self.buffer.keep_random_subset(int(math.sqrt(len(self.buffer))))
            if (ep + 1) % 100 == 0:
                log.info(
                    "episode %d  mean reward (last 100) %.3f  epsilon %.3f",
                    ep + 1, float(np.mean(history[-100:])), self.epsilon,
                )
            if max_episodes >= 2 and ep + 1 == max_episodes // 2:
                self.save("transformer_halftrain")
        path = self.save("transformer")
        self.rewards = history
        return self.model, path

    def teacher_influence(self, max_episodes):
        return max(
            0.0,
            float(
                np.exp(-5 * (1 + self.n_episodes) / max_episodes)
                - 0.164 * (self.n_episodes / max_episodes)
            ),
        )

    def save(self, prefix):
        stamp = dt.datetime.now().strftime("%Y%m%d.%H.%M.%S")
        path = os.path.join(self.model_dir, f"{prefix}_{stamp}.pt")
        save_checkpoint(self.model, path)
        log.info("saved %s", path)
        return path


def run_episode(trainer, model, secret_word=None):
    env = trainer.env
    options = {"secret_word": secret_word} if secret_word else None
    obs, info = env.reset(options=options)
    total, terminated, guesses = 0.0, False, []
    while not terminated:
        a, _ = masked_argmax(trainer.q_values(model, obs), invalid_from_mask(info["action_mask"]))
        action = int(a.item())
        obs, reward, terminated, _, info = env.step(action)
        assert info["valid"]
        guesses.append(env.words[action])
        total += reward
    log.info("secret %s  guesses %s  reward %.3f", env.secret_word, guesses, total)
    return total


def main(argv=None):
    parser = argparse.ArgumentParser(description="Train or evaluate the transformer DQN on Wordle")
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--teacher", default=None)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--model-dir", default="models")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    env = WordleEnv()
    trainer = TransformerTrainer(env, args.device, model_dir=args.model_dir)
    if args.eval_only:
        model = load_checkpoint(TransformerQNet, args.checkpoint, trainer.device)
    else:
        model, _ = trainer.solve(
            args.episodes, checkpoint=args.checkpoint,
            teacher_checkpoint=args.teacher, batch_size=args.batch_size,
        )
    history = [run_episode(trainer, model, w) for w in env.solutions]
    log.info(
        "fail rate %.3f  mean reward %.3f",
        sum(r == 0.0 for r in history) / len(history), float(np.mean(history)),
    )


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests**

```bash
uv run pytest tests/test_transformer.py -v
```

Expected: all pass.

- [ ] **Step 5: Delete the old agent and commit**

```bash
cd ..
git rm transformer_agent.py
git add gym-wordle/gym_wordle/agents/transformer.py gym-wordle/tests/test_transformer.py
git commit -m "feat(transformer): sequence-indexed positional encoding, eval-mode targets, state_dict checkpoints"
cd gym-wordle
```

---

### Task 8: Factored two-tower head for the DQN

**Files:**
- Modify: `gym-wordle/gym_wordle/agents/dqn.py` (add `FactoredHead`, extend `ResNN`, `DQNTrainer.build_model`, CLI choices)
- Modify: `gym-wordle/tests/test_dqn.py` (append)

**Interfaces:**
- Consumes: `word_feature_matrix` (Task 4), `ResNN`, `DQNTrainer` (Tasks 5-6).
- Produces: `FactoredHead(in_dim, n_actions, n_letters, d, word_features=None)` with buffer `word_features (N, 26*n_letters+26)` and `forward(z: (B,in_dim)) -> (B,N)`; `ResNN(..., head="factored", word_features=phi)`.

- [ ] **Step 1: Append the failing tests**

Append to `tests/test_dqn.py`:

```python
from gym_wordle.agents.common import word_feature_matrix
from gym_wordle.agents.dqn import FactoredHead


def test_factored_head_output_matches_flat_shape(env):
    phi = word_feature_matrix(env.words, 5)
    m = ResNN(n_actions=18, channels=4, head="factored", word_features=phi).eval()
    assert m(torch.zeros(3, 4, 26, 5)).shape == (3, 18)
    assert m.ctor_kwargs["head"] == "factored"
    assert "word_features" not in m.ctor_kwargs  # lives in the state_dict instead


def test_factored_head_distinguishes_anagrams(env):
    phi = word_feature_matrix(env.words, 5)
    torch.manual_seed(0)
    m = ResNN(n_actions=18, channels=4, head="factored", word_features=phi).eval()
    q = m(torch.randn(1, 4, 26, 5))
    i, j = env.word_to_action["stale"], env.word_to_action["slate"]
    assert q[0, i] != q[0, j]


def test_factored_head_parameter_budget():
    n_actions = 12947
    factored = ResNN(n_actions=n_actions, head="factored", word_features=torch.zeros(n_actions, 156))
    flat = ResNN(n_actions=n_actions)
    n_factored = sum(p.numel() for p in factored.head.parameters())
    n_flat = sum(p.numel() for p in flat.head.parameters())
    assert n_factored < 500_000
    assert n_flat > 20_000_000


def test_factored_checkpoint_restores_word_features(tmp_path, env):
    phi = word_feature_matrix(env.words, 5)
    m = ResNN(n_actions=18, channels=4, head="factored", word_features=phi).eval()
    save_checkpoint(m, str(tmp_path / "f.pt"))
    m2 = load_checkpoint(ResNN, str(tmp_path / "f.pt"), CPU).eval()
    assert torch.equal(m2.head.word_features, phi)
    x = torch.randn(2, 4, 26, 5)
    assert torch.allclose(m(x), m2(x))


def test_trainer_builds_factored_model_with_env_words(env, tmp_path):
    t = make_trainer(env, tmp_path, head="factored")
    t.setup(max_episodes=1, batch_size=1, max_exp=8)
    assert isinstance(t.model.head, FactoredHead)
    assert torch.equal(t.model.head.word_features, word_feature_matrix(env.words, 5))
    model, path = t.solve(max_episodes=2, batch_size=2, max_exp=16)
    assert os.path.exists(path)


def test_unknown_head_rejected():
    with pytest.raises(ValueError):
        ResNN(n_actions=18, head="bilinear")
```

- [ ] **Step 2: Run to verify failure**

```bash
uv run pytest tests/test_dqn.py -v -k "factored or unknown_head"
```

Expected: ImportError, `cannot import name 'FactoredHead'`.

- [ ] **Step 3: Add the head**

In `gym_wordle/agents/dqn.py`, insert directly above `class ResNN`:

```python
class FactoredHead(nn.Module):
    """Two-tower bilinear Q head: Q(s, w) = state_proj(z) . word_tower(phi(w)).

    phi(w) is a fixed per-word feature vector (one-hot letter per position,
    then letter counts). It is a buffer, so a checkpoint carries it and a
    freshly constructed model can start from zeros and be filled by
    load_state_dict.
    """

    def __init__(self, in_dim, n_actions, n_letters, d, word_features=None):
        super().__init__()
        n_feat = N_ALPHABET * n_letters + N_ALPHABET
        if word_features is None:
            word_features = torch.zeros(n_actions, n_feat)
        assert tuple(word_features.shape) == (n_actions, n_feat), word_features.shape
        self.register_buffer("word_features", word_features.float())
        self.state_proj = nn.Linear(in_dim, d)
        self.word_tower = nn.Sequential(
            nn.Linear(n_feat, d), nn.Mish(), nn.Linear(d, d)
        )

    def forward(self, z):
        h = self.state_proj(z)                    # (B, d)
        u = self.word_tower(self.word_features)   # (N, d), shared across words
        return h @ u.T                            # (B, N)
```

Then in `ResNN.__init__` replace

```python
        if head == "flat":
            self.head = nn.Linear(flat_dim, n_actions)
        else:
            raise ValueError(f"unknown head {head!r}")
```

with

```python
        if head == "flat":
            self.head = nn.Linear(flat_dim, n_actions)
        elif head == "factored":
            self.head = FactoredHead(flat_dim, n_actions, n_letters, d_head, word_features)
        else:
            raise ValueError(f"unknown head {head!r}")
```

In `DQNTrainer.build_model` replace the body with:

```python
    def build_model(self):
        kwargs = dict(
            n_actions=self.n_actions,
            n_letters=self.env.n_letters,
            channels=self.channels,
            head=self.head,
        )
        if self.head == "factored":
            kwargs["word_features"] = word_feature_matrix(self.env.words, self.env.n_letters)
        return ResNN(**kwargs).to(self.device)
```

In `main`, change the head argument to:

```python
    parser.add_argument("--head", choices=["flat", "factored"], default="flat")
```

- [ ] **Step 4: Run the DQN tests**

```bash
uv run pytest tests/test_dqn.py -v
```

Expected: all pass.

- [ ] **Step 5: Smoke run the factored head on real data**

```bash
SMOKE=$(mktemp -d)
uv run python -m gym_wordle.agents.dqn --episodes 2 --channels 2 --head factored --device cpu --model-dir "$SMOKE" --batch-size 4
ls "$SMOKE"
```

Expected: trains and evaluates without error; `ls` shows the checkpoint pair.

- [ ] **Step 6: Commit**

```bash
git add gym_wordle/agents/dqn.py tests/test_dqn.py
git commit -m "feat(dqn): optional factored two-tower Q head sharing weights across words"
```

---

### Task 9: README, stale-file notes, and final verification

**Files:**
- Modify: `README.md` (repo root, rewrite)
- Modify: `gym-wordle/ppo_agent.py` (top-of-file comment only)
- Modify: `gym-wordle/restore_policy.py` (top-of-file comment only)

- [ ] **Step 1: Rewrite README.md**

```markdown
# Wordle Environment and Agents

A gymnasium environment for Wordle with a strict hard mode, and two
PyTorch Double-DQN trainers.

## Layout

    words.json                    2309 solution words (flat JSON list)
    valids.json                   10638 additional allowed guesses
    gym-wordle/                   the installable package (uv project)
      gym_wordle/envs/wordle_env.py    WordleEnv, registered as Wordle-v0
      gym_wordle/agents/common.py      masking, replay buffer, checkpoints
      gym_wordle/agents/dqn.py         SE-ResNet DQN (flat or factored head)
      gym_wordle/agents/transformer.py transformer DQN
      tests/                           pytest suite on a small fixture list
      ppo_agent.py, restore_policy.py  STALE: Ray 1.x RLlib, not ported
    docs/superpowers/             design spec and implementation plan

The action space is the sorted union of both word lists (12947 words).

## Setup

    cd gym-wordle
    uv sync
    uv run pytest

## Environment

    import gymnasium, gym_wordle
    env = gymnasium.make("Wordle-v0")
    obs, info = env.reset(seed=0, options={"secret_word": "crane"})
    obs, reward, terminated, truncated, info = env.step(action)

`obs` is a flat int8 vector of length 183: slot 0 is the number of guesses,
then for each letter five positional slots (0 unknown, 1 possible yellow,
2 green), then 26 "guessed more times than present" bits, then 26
max-yellow counts. `info["action_mask"]` marks the guesses that hard mode
still allows and that have not been played; a guess outside the mask is
rejected without using a turn and `info["valid"]` is False. Reward is
paid only on a winning guess and decreases with the number of guesses.

## Training

    uv run python -m gym_wordle.agents.dqn --episodes 4618 --fixed-start dealt
    uv run python -m gym_wordle.agents.dqn --episodes 4618 --head factored
    uv run python -m gym_wordle.agents.transformer --episodes 1000

Checkpoints are written to `models/` as a `state_dict` `.pt` plus a
`.pt.json` holding the constructor arguments. Evaluate one with
`--eval-only --checkpoint models/<file>.pt`.

The `factored` head scores a word as a dot product between a state
embedding and a small shared network over the word's letter features,
instead of one weight row per word. It has about 240k head parameters
versus about 20M for the flat head.

## History

The original 2022-2023 code used OpenAI `gym`, pickled whole models, and
had several training bugs; see `docs/superpowers/specs/` for what changed.
```

- [ ] **Step 2: Mark the RLlib files stale**

Add as the first lines of both `gym-wordle/ppo_agent.py` and `gym-wordle/restore_policy.py`:

```python
# STALE: written against Ray 1.x RLlib and OpenAI gym. Not ported to the
# gymnasium WordleEnv; kept for reference. See README.md.
```

- [ ] **Step 3: Full verification**

```bash
cd gym-wordle
uv run pytest -v
uv run python -c "import gymnasium, gym_wordle; e = gymnasium.make('Wordle-v0'); o, i = e.reset(seed=0); print(o.shape, e.action_space.n, i['secret_word'])"
cd ..
git status --short
```

Expected: every test passes; the one-liner prints `(183,) 12947 <a word>`; `git status` shows only the README and the two stale-marked files as modified, nothing untracked.

- [ ] **Step 4: Commit**

```bash
git add README.md gym-wordle/ppo_agent.py gym-wordle/restore_policy.py
git commit -m "docs: README for the gymnasium package; mark RLlib scripts stale"
```
