# Wordle RL: environment consolidation and agent fixes

Date: 2026-09-07
Status: approved in discussion, pending spec review

## Goal

Make the repository runnable again on current libraries and fix the
confirmed correctness bugs in the environment and the two PyTorch
agents, without changing the reward function or reproducing old
results.

## Scope

In scope:

- One `WordleEnv` on the gymnasium API, installable as `gym_wordle`.
- The SE-ResNet DQN trainer (`agent.py`) and the transformer DQN
  trainer (`transformer_agent.py`), moved into the package.
- A pytest suite that pins each fixed bug as a regression test.
- Packaging with `pyproject.toml` managed by uv.

Out of scope:

- The RLlib PPO agent (`ppo_agent.py`, `restore_policy.py`). These
  stay in place, untouched, and the README notes they target Ray 1.x.
- Any change to the Poisson-based reward schedule.
- Re-training or benchmarking.

## Repository layout after the change

```
gym-wordle/
  pyproject.toml
  gym_wordle/
    __init__.py            # registers Wordle-v0 with gymnasium
    envs/
      __init__.py
      wordle_env.py        # the single canonical environment
    agents/
      __init__.py
      common.py            # replay buffer, PER, action masking, checkpoint IO
      dqn.py               # SE-ResNet DQN trainer (from agent.py)
      transformer.py       # transformer DQN trainer (from transformer_agent.py)
  ppo_agent.py             # unchanged, stale
  restore_policy.py        # unchanged, stale
  tests/
    fixtures/
      valids_small.json
      words_small.json
    test_env.py
    test_agents.py
words.json                 # solutions, flat list (committed)
valids.json                # allowed guesses, flat list (committed once procured)
README.md
```

Deleted: top-level `wordle_rl.py`, `agent.py`, `transformer_agent.py`
(moved), and `gym-wordle/setup.py` (replaced by `pyproject.toml`).

`.gitignore` stops ignoring `*.json`. It ignores `models/`,
`checkpoints/`, `*.png`, and `__pycache__/`.

## Data files

- `words.json`: flat JSON list of solution words. The user supplied
  2309 words.
- `valids.json`: flat JSON list of allowed non-solution guesses. The
  user supplied 10638 words.
- The action space is the sorted union of both lists (12947 words with
  the supplied files). Action `i` is `words[i]`. Sorting makes indices
  deterministic and independent of file order; duplicates across the
  two files are harmless.
- The env takes both paths as constructor arguments with defaults that
  resolve relative to the repository root, not the working directory.
- The network output width is `len(valids)`, read from the env, never
  hard-coded.

## Environment

`WordleEnv(gymnasium.Env)` in `gym_wordle/envs/wordle_env.py`,
registered as `Wordle-v0`.

### API

- `reset(seed=None, options=None) -> (obs, info)`. `options` may
  contain `secret_word`. Randomness uses `self.np_random`.
- `step(action) -> (obs, reward, terminated, truncated, info)`.
  `truncated` is always `False`; running out of attempts is
  `terminated`.
- `obs` is a fresh copy of the internal state on every call. Callers
  never hold a reference to env memory.
- `info` contains:
  - `valid`: bool, whether the guess was accepted.
  - `valid_words`: dict `{action_id: word}` of guesses still legal under
    hard mode, a copy.
  - `action_mask`: `np.ndarray[bool]` of length `len(valids)`, true
    where the action is in `valid_words`.
  - `secret_word` and `n_guesses`.

### Observation layout

A single `MultiDiscrete` of length `1 + 26*n_letters + 2*26`:

| slot | meaning | range |
|---|---|---|
| 0 | guesses used so far | `0..max_attempts` (size `1 + max_attempts`) |
| `1 + 5*letter + pos` | positional status | 0 other, 1 possible yellow, 2 green |
| `EXCEEDED + letter` | letter guessed more times than it appears | 0/1 |
| `YELLOWS + letter` | max yellows seen for letter, less later greens | `0..n_letters//2` |

`EXCEEDED = 1 + 26*n_letters` and `YELLOWS = EXCEEDED + 26` are named
constants. No negative indexing.

### Behavior fixes

1. Hard-mode rejection: state, turn counter, and reward unchanged,
   `info["valid"] = False`, and the rejected action is removed from
   `valid_words` by action id.
2. An accepted guess is removed from `valid_words`, so it is never
   offered again in the same episode.
3. The exceeded bit is set whenever `guess.count(letter) >
   secret.count(letter)`, including when some occurrences are yellow.
4. The declared observation space admits every reachable state,
   including the final turn.
5. `reward` is per step: the Poisson-schedule value on a winning guess,
   otherwise 0. It is not carried over from a previous step.

Everything else about scoring, yellow tracking, and pruning keeps the
current top-level semantics, which a brute-force check showed agree
with the hard-mode acceptance rule.

## Agents

Both trainers move into `gym_wordle/agents/` and share
`common.py`. Shared fixes:

- `device` is a constructor argument; no module globals.
- Target and teacher forwards run under `model.eval()` and
  `torch.no_grad()`; TD targets are detached.
- The invalid-action mask stored with a transition is taken from
  `info["action_mask"]` after the step, plus the guessed action, so it
  masks the argmax over `s_prime`, which is its only use.
- Action selection dispatches on whether the mask argument is one
  mask or a list of masks, not on the batch row count.
- Checkpoints are `state_dict` files plus a small JSON of constructor
  args. Loading rebuilds the architecture from code. `models/` is
  created before saving.
- Unused imports (`pickle`, `einops`, `gym`) removed.

DQN-specific:

- PER beta advances once per episode by `(1 - beta0) / max_episodes`
  and is clamped at 1.0.
- New replay rows get the current maximum priority (1.0 when the
  buffer is empty). One named epsilon floors priorities.
- `T_max` for cosine annealing defaults to `max_episodes`.
- `shuffle_solutions` yields exactly `max_episodes` indices.
- The stale mode-test checkpoint path is removed.

Transformer-specific:

- `PositionalEncoding` stores `pe` as `(1, max_len, d_model)` and adds
  `pe[:, :seq_len]`, matching `batch_first=True`.
- `self.state` is a copy of the observation, never the env array.

Reward, epsilon schedule, and hyperparameters are unchanged. The
network trunks are unchanged; the output head gains an option, below.

## Factored prediction head (DQN)

The current DQN head is `Linear(1560, len(valids))`, one weight row per
word, about 20M parameters. A guess is a tuple of five letters at five
positions, so the label space factorizes and the head can share
weights across words.

`ResNN` takes `head="flat" | "factored"`, default `"flat"`, so every
earlier task and test runs against the existing architecture.

`"factored"` is a two-tower bilinear head:

- Word features `phi(w)`: a fixed vector of length `26*n_letters + 26`,
  one-hot letter-per-position followed by letter counts. Computed once
  for the whole guess list into a constant buffer `Phi` of shape
  `(len(valids), 156)`.
- Word tower: `u = MLP(phi)`, two `Linear` layers with Mish, output
  dim `d=128`, shared across all words. Applied to `Phi` on every
  forward pass (cheap; it is a 12947 x 156 matmul).
- State tower: flatten the trunk grid `(12, 26, 5)` to 1560 and project
  with `Linear(1560, d)`.
- `Q(s, w) = h . u(w)` for every word at once: `H @ U.T`, shape
  `(batch, len(valids))`. No per-word bias.

Output shape and dtype match the flat head, so the trainer, masking,
and checkpoint code are unchanged. The head choice is recorded in the
checkpoint's constructor-args JSON.

Adding the same option to the transformer agent is a follow-on and is
not part of this pass.

## Testing

`pytest` against the fixture lists (about 20 guesses, 4 solutions).

Environment tests:

- `gymnasium.utils.env_checker.check_env` passes.
- One regression test per behavior fix above.
- Observations returned by `reset` and `step` are not the same object
  as env state and do not change after a later `step`.
- Brute force over fixture words: for several secret/opener pairs,
  `valid_words` after the opener equals the set of actions the env
  accepts on the next turn.
- The action list is the sorted union of both files, with no duplicates,
  and every solution has an action id.

Agent tests (CPU, tiny model):

- `PositionalEncoding` output equals input plus the same position
  vector for every batch row; batch size larger than `max_len` works.
- One optimizer step of each trainer runs on the fixture env with
  `batch_size=1` and with a batch containing exactly one non-terminal
  transition.
- Save then load a checkpoint yields identical outputs.
- Factored head: output shape equals the flat head's; `phi` of a known
  word has exactly five position one-hots set and letter counts that
  sum to five; two anagrams get different Q-values; parameter count of
  the head is below 500k for the real guess list size.

Tests are written before the code they cover and observed failing.

## Migration notes

- Old pickled whole-module checkpoints are not loadable. There is no
  shim.
- `gym.make("wordle-v0")` becomes `gymnasium.make("Wordle-v0")`.
