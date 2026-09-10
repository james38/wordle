# Wordle Environment and Agents

A batched Wordle environment and a PPO agent whose transformer policy reads
only the coloured feedback to its own guesses. The agent knows the valid
guess dictionary but never the solution list. A single-game gymnasium
environment and two DQN trainers from the earlier version of the project
remain as legacy code.

## Layout

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

The action space is the sorted union of both word lists (12947 words).

## Setup

    cd gym-wordle
    uv sync
    uv run pytest

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

## Legacy gymnasium environment

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

## Legacy agents

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
Checkpoints saved by the 2023 code (pickled whole models) cannot be loaded
by the new trainers; retrain from scratch.
