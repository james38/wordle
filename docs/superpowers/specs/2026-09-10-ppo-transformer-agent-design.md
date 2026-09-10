# PPO Transformer Agent for Wordle — Design

Date: 2026-09-10
Status: approved in brainstorming, awaiting written review
Supersedes as the main line: the DQN and transformer-DQN trainers under
`gym_wordle/agents/` (kept in place, labelled legacy in the README).

## 1. Goal

Replace the value-learning agents with a policy-gradient agent that learns
Wordle the way a player experiences it: it sees only the valid-guess
dictionary and the coloured feedback to its own guesses, and it is trained
with PPO on thousands of games in parallel so that a full run finishes in
minutes on one RTX 3090.

## 2. Hard rules

1. **The agent never sees the solution list.** No is-solution feature, no
   per-word parameter learned over the solution list, no candidate set or
   reward term computed over the solutions. Everything the policy consumes
   must be derivable from the valid-guess list plus observed feedback. The
   environment may sample secrets from the solution list; that is the
   game's business.
2. **Hand-rolled PPO.** No RL library. The algorithm lives in this repo in
   plain PyTorch.
3. **Nothing existing is modified** except `README.md`, `.gitignore`
   (add `runs/`) and package registration. `WordleEnv`, `dqn.py`, `transformer.py`, `common.py` and
   their tests are untouched.
4. Package tooling stays as it is: uv + hatchling, `gymnasium`, `torch`,
   `numpy`, `pytest`. No new runtime dependencies.

## 3. Decisions made in brainstorming

| Topic | Decision |
|---|---|
| Agent input | Raw feedback history as tokens (letter, colour, turn per board cell). No hand-built summary state. |
| Hard mode | Configurable flag, default on. Real Wordle semantics (see §5.3), which are laxer than the old env's rule 2. |
| Action head | Factored: logits = readout · u(w), u(w) from a small tower over fixed lexical features (one-hot letter per position + letter counts, 156 dims). |
| Algorithm | PPO with masked categorical policy, GAE, clipped surrogate. |
| Environment | New natively batched torch env, N games per step, auto-reset. |
| Reward | Poisson tail schedule normalised so guess 1 pays 1.0; fail pays 0. Optional information-gain shaping flag, default off. |
| Compute | CUDA GPU available; target minutes per run. |

## 4. File layout

```
gym-wordle/gym_wordle/envs/feedback.py        pure batched functions, no state
gym-wordle/gym_wordle/envs/batched.py         BatchedWordle training env
gym-wordle/gym_wordle/agents/ppo/__init__.py
gym-wordle/gym_wordle/agents/ppo/model.py     WordlePolicy (transformer + heads)
gym-wordle/gym_wordle/agents/ppo/ppo.py       RolloutBuffer, GAE, ppo_update
gym-wordle/gym_wordle/agents/ppo/train.py     CLI, loop, logging, checkpoints, evaluation
gym-wordle/tests/test_feedback.py
gym-wordle/tests/test_batched_env.py
gym-wordle/tests/test_ppo_model.py
gym-wordle/tests/test_ppo.py
README.md                                     PPO becomes the main line; old agents labelled legacy
```

`gym_wordle.envs.wordle_env.load_word_lists`, `letter_matrix`, `ALPHABET`,
`LETTER_INDEX`, `N_ALPHABET` are reused by import. `save_checkpoint` /
`load_checkpoint` / `word_feature_matrix` from `gym_wordle.agents.common`
are reused by import.

## 5. Environment

### 5.1 `feedback.py` — pure functions

All functions take and return torch tensors on whatever device the inputs
are on. `words` below means the sorted union of valid and solution lists
(12947 words for the real data); `W_letters` is its `(V, 5)` int8 letter
matrix and `W_counts` its `(V, 26)` count matrix, both produced by
`letter_matrix`.

- `colour(guess_letters (B,5), secret_letters (B,5)) -> (B,5) int8`
  Standard Wordle colouring: 0 grey, 1 yellow, 2 green. Greens first; then
  scanning left to right, a non-green cell is yellow iff the secret still
  has an unmatched copy of that letter, decrementing the remaining count.
  Vectorised over B (a loop over the 5 positions is fine).
- `hard_mode_mask(green_pos (B,5) int8 [-1 = none], min_count (B,26) int8,
  W_letters, W_counts) -> (B,V) bool`
  Word w is legal iff for every position with a green, `W_letters[w,p] ==
  green_pos[b,p]`, and for every letter, `W_counts[w,l] >= min_count[b,l]`.
- `consistent_mask(guess_letters (B,T,5), colours (B,T,5), n_guesses (B,),
  W_letters) -> (B,V) bool`
  Word w is consistent iff colouring every past guess against w reproduces
  the recorded colours. Implemented by colouring each past guess against
  all V words (loop over T ≤ 6, vectorised over B×V) and comparing. Used
  for the shaping reward and for the evaluation metric "candidates
  remaining"; never fed to the policy.

### 5.2 `BatchedWordle`

```python
BatchedWordle(
    words, solutions,           # lists of str, from load_word_lists
    n_games, *, hard_mode=True, max_attempts=6, L=4.0,
    shaping_coef=0.0, device="cpu", seed=0,
)
```

State (all tensors on `device`, first dim `n_games`):

| field | shape | meaning |
|---|---|---|
| `secret` | (N,) long | index into `words` of the current secret |
| `turn` | (N,) long | guesses made so far, 0..6 |
| `guess_letters` | (N,6,5) int8 | letters of past guesses, 0 where unused |
| `colours` | (N,6,5) int8 | colours of past guesses |
| `guessed` | (N,V) bool | words already played this game |
| `green_pos` | (N,5) int8 | fixed green letter per position, -1 if none |
| `min_count` | (N,26) int8 | max over past guesses of greens+yellows per letter |

API:

- `reset() -> obs` resets every game with fresh secrets.
- `step(actions (N,) long) -> (obs, reward (N,) float32, done (N,) bool, info)`
  Every action must be legal under the current mask; an illegal action
  raises `ValueError` (the policy masks, so this is a bug guard, not a
  game rule). After stepping, any finished game is reset in place; `obs`
  already describes the fresh game for those rows. `info` is a dict with
  `solved (N,) bool`, `n_guesses (N,) long` (of the game that just ended,
  meaningful where `done`), and `secret (N,) long` (likewise).
- `legal_mask() -> (N,V) bool` = `hard_mode_mask(...) & ~guessed` in hard
  mode, `~guessed` otherwise.
- `observation()` — see §6.1.
- `candidates() -> (N,) long` count of consistent words (for logging).
- `set_secrets(indices (N,) long)` resets every game with the given
  secrets. Evaluation only; training uses `reset()`.

Secrets are drawn uniformly from `solutions` using a `torch.Generator`
seeded from `seed`; the same seed gives the same secrets.

### 5.3 Hard mode (real Wordle rule)

After each guess, for every position that came up green,
`green_pos[p] = letter`; for every letter, `min_count[l] = max(min_count[l],
greens+yellows of l in this guess)`. Greys are *not* forbidden. This is
laxer than the old `WordleEnv` rule 2 (which excluded exhausted letters)
and matches the official game.

### 5.4 Reward

Solving on guess k (1-based) pays

    r_k = P(X >= k) / P(X >= 1),   X ~ Poisson(L),  L = 4.0 default

so r_1 = 1.0 exactly and, for L=4, r = 1.000, 0.925, 0.776, 0.577, 0.378,
0.219 for k = 1..6. Failing to solve within `max_attempts` pays 0. All
other steps pay 0.

With `shaping_coef = c > 0`, every step additionally pays
`c * (log2(candidates_before) - log2(candidates_after))`, computed with
`consistent_mask` over the *valid* list. Default `c = 0`.

## 6. Model

### 6.1 Observation

`observation()` returns a dict of tensors:

| key | shape | meaning |
|---|---|---|
| `tokens` | (N,30,3) long | per board cell in guess order: letter 0..25, colour 0..2, turn 0..5 |
| `pad` | (N,30) bool | True for cells of guesses not yet made (key padding mask) |
| `turn` | (N,) long | current turn 0..6, attached to the readout token |
| `mask` | (N,V) bool | legal actions |

Slot index s = 5·turn + position, so position-in-word is implicit in s.

### 6.2 `WordlePolicy`

```python
WordlePolicy(n_words, word_features (V,156) float,
             d_model=128, n_heads=4, d_ff=512, n_layers=3, max_turns=6, n_letters=5)
```

- Cell embedding = `letter_emb[letter] + colour_emb[colour] + slot_emb[s]`
  (tables 26×d, 3×d, 30×d).
- Readout token = `readout (d,) + turn_emb[turn]` (table 7×d), prepended
  at index 0. Sequence length 31. Padding positions are excluded via
  `src_key_padding_mask`; the readout is never padded.
- Encoder: `nn.TransformerEncoder` of `n_layers` `TransformerEncoderLayer`
  with `batch_first=True, norm_first=True, dropout=0.0, activation="gelu"`,
  followed by a final `LayerNorm`.
- Policy head: `word_tower = Linear(156,d) → GELU → Linear(d,d)` applied to
  the `word_features` buffer once per forward giving `u (V,d)`;
  `logits = (z @ u.T) / sqrt(d)` where `z` is the readout output.
  `logits[~mask] = -inf`.
- Value head: `Linear(d,d) → GELU → Linear(d,1)`.
- `forward(obs) -> (logits (N,V), value (N,))`.
- `act(obs, greedy=False) -> (action, log_prob, entropy, value)` and
  `evaluate_actions(obs, actions) -> (log_prob, entropy, value)`, both via
  `torch.distributions.Categorical(logits=logits)`.
- `word_features` is a registered buffer; `ctor_kwargs` is saved so
  `save_checkpoint` / `load_checkpoint` from `common.py` work unchanged.

Parameter count is well under one million.

## 7. PPO

### 7.1 Collection

Defaults `n_games N = 2048`, `rollout_len T = 16` → 32768 transitions per
iteration. `RolloutBuffer` preallocates tensors on the training device for
`tokens, pad, turn, mask, actions, log_probs, values, rewards, dones`
with leading dims `(T, N, ...)`. Collection runs the policy in eval mode
under `torch.no_grad()`; nothing crosses to the host during a rollout.

### 7.2 Advantages

GAE with `gamma = 1.0`, `lam = 0.95`. `done[t]` stops the bootstrap at t
(the env auto-reset means `obs[t+1]` is a new game where `done[t]`). The
bootstrap for the last step is one extra value forward on the observation
after the final step. Returns = advantages + values.

### 7.3 Update

Per iteration: normalise advantages over the whole batch; `epochs = 4`;
`minibatch = 4096`; clipped surrogate with `clip = 0.2`; value loss
clipped with the same `clip`, weight `vf_coef = 0.5`; entropy bonus
`ent_coef = 0.01`; Adam `lr = 3e-4` with linear decay to 0 over
`iterations`; `max_grad_norm = 0.5`; stop the epoch loop early when
approximate KL (`(ratio - 1) - log(ratio)`, mean) exceeds
`target_kl = 0.02`. Log-probs and entropy always come from masked logits
using the mask stored at collection time.

`ppo_update(policy, optimizer, buffer, cfg) -> dict` returns the diagnostic
means (policy loss, value loss, entropy, approx KL, clip fraction,
explained variance).

### 7.4 Training loop (`train.py`)

`iterations = 300` default (~10M steps). Each iteration: collect, compute
GAE, update, log. Logged per iteration to stdout and to
`<run_dir>/log.csv`: iteration, steps, solve rate (finished episodes this
iteration), mean guesses among solves, mean episode reward, entropy,
approx KL, clip fraction, value loss, explained variance, steps/s,
learning rate. Checkpoint every 25 iterations and at the end to
`<run_dir>/policy_<iter>.pt` via `save_checkpoint`.

### 7.5 Evaluation

`evaluate(policy, env_kwargs, device)` plays every solution exactly once
in a `BatchedWordle` built with `n_games = len(solutions)` and secrets set
directly (an explicit `set_secrets(indices)` hook on the env, for
evaluation only), greedily, and reports solve rate, mean guesses among
solves, and the histogram of guess counts 1..6 plus fails.

### 7.6 CLI

```
python -m gym_wordle.agents.ppo.train
    [--iterations 300] [--n-games 2048] [--rollout-len 16]
    [--lr 3e-4] [--ent-coef 0.01] [--clip 0.2] [--target-kl 0.02]
    [--d-model 128] [--n-layers 3] [--n-heads 4] [--d-ff 512]
    [--hard-mode/--no-hard-mode] [--shaping-coef 0.0] [--L 4.0]
    [--seed 0] [--device cuda|cpu] [--run-dir runs/<timestamp>]
    [--checkpoint PATH] [--eval-only]
```

`--eval-only` requires `--checkpoint` (parser error otherwise). Training
ends with an evaluation over all solutions. `runs/` is added to
`.gitignore`.

## 8. Testing

CPU only, small fixtures (`tests/fixtures/words_small.json`,
`valids_small.json`) unless stated.

**feedback.py**
- Independent plain-Python reference `colour_ref(guess, secret)` compared
  to `colour` on ≥ 2000 random pairs from the real lists and on hand
  cases: `paper`/`apple`, `label`/`llama`, `eerie`/`there`, `speed`/`abide`.
- `hard_mode_mask` vs a brute-force per-word Python check on ≥ 50 random
  partial games from the real lists.
- `consistent_mask`: the secret is always consistent; a word whose
  colouring against a past guess differs from the record is not; the
  count is non-increasing along a game.

**batched.py**
- Same seed → same secrets; reset clears state.
- Illegal action raises; a played word is never legal again; in hard mode
  the mask equals `hard_mode_mask & ~guessed`, in normal mode `~guessed`.
- Reward table for L=4 matches §5.4 to 1e-6; fail pays 0; intermediate
  steps pay 0 with shaping off; with shaping on the step reward equals
  `c·Δlog2(candidates)` (checked against `candidates()`).
- Auto-reset: after a solve the row has `turn == 0`, all-pad tokens, a
  cleared `guessed` row and a fresh secret, and unfinished rows are
  unchanged.
- Observation: tokens for made guesses match `guess_letters`/`colours`,
  pad is True exactly for unmade cells.

**model.py**
- Output shapes; `logits[~mask] == -inf` and their probability is 0;
  entropy equals that of the legal subset.
- Changing token content at padded positions does not change the output.
- One backward pass gives a non-None grad on every parameter.
- `save_checkpoint` then `load_checkpoint` reproduces the outputs.

**ppo.py**
- GAE on a hand-computed 3-step trajectory with a done in the middle,
  gamma 1, lam 0.95.
- One full collect + update on the fixtures with N=8, T=4 runs and
  returns finite diagnostics.
- Learning test (marked `slow`, budget ≈ 30 s CPU): on the five-word
  fixture with a fixed seed, greedy solve rate reaches 1.0 within a
  bounded number of iterations.

**train.py**
- `--eval-only` without `--checkpoint` → `SystemExit` from `parser.error`.

## 9. Out of scope

- Behaviour-cloning warm start from a heuristic solver (approach 2 from
  brainstorming; kept in reserve if PPO learns too slowly).
- Deleting the legacy DQN / transformer-DQN code.
- Normal-mode curriculum scheduling; the flag exists, scheduling does not.
- Multi-GPU.
