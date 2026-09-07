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


import os

from gym_wordle.agents.common import invalid_from_mask
from gym_wordle.agents.dqn import DQNTrainer, main, run_episode


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


def test_main_eval_only_requires_checkpoint(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--eval-only"])
    assert exc.value.code == 2
    assert "--eval-only requires --checkpoint" in capsys.readouterr().err


def test_learn_handles_mixed_terminal_and_nonterminal_batch(env):
    trainer = DQNTrainer(env, device="cpu", channels=2)
    trainer.setup(max_episodes=2, batch_size=4, max_exp=64)
    env.reset(options={"secret_word": "apple"})
    obs, info = env.reset(options={"secret_word": "apple"})
    # one non-terminal transition
    a = env.word_to_action["crane"]
    obs2, r, term, _, info2 = env.step(a)
    trainer.buffer.add(obs, a, r, obs2, term, invalid_from_mask(info2["action_mask"]))
    # one terminal (winning) transition
    a_win = env.word_to_action["apple"]
    obs3, r3, term3, _, info3 = env.step(a_win)
    assert term3 and r3 > 0
    trainer.buffer.add(obs2, a_win, r3, obs3, term3, invalid_from_mask(info3["action_mask"]))
    # pad so the batch has both kinds
    for _ in range(2):
        trainer.buffer.add(obs, a, r, obs2, term, invalid_from_mask(info2["action_mask"]))
    loss = trainer.learn()
    assert np.isfinite(loss)
    assert np.isfinite(trainer.buffer.priority[: len(trainer.buffer)]).all()
