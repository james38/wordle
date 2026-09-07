# Wordle Environment and Agents

A gymnasium environment for Wordle with a strict hard mode, a Double-DQN
trainer on an SE-ResNet, and a plain DQN trainer on a transformer.

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
