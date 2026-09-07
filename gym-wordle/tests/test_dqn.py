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
    assert m.ctor_kwargs["head"] == "flat" and m.ctor_kwargs["n_actions"] == 19
    m.eval()
    assert m(torch.zeros(1, 4, 26, 5)).shape == (1, 19)  # batch of one in eval


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
