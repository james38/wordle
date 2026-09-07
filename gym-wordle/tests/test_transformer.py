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
    main,
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


def test_main_eval_only_requires_checkpoint(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--eval-only"])
    assert exc.value.code == 2
    assert "--eval-only requires --checkpoint" in capsys.readouterr().err
