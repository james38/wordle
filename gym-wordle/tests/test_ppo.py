import pytest
import torch

from gym_wordle.agents.common import word_feature_matrix
from gym_wordle.agents.ppo import train as train_mod
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


def _collect_buf(env, policy, T):
    """Run the real trainer collect() for T steps and GAE it, for use as test fixture data."""
    buf = RolloutBuffer(T, env.n_games, env.max_attempts * env.n_letters, len(env.words), "cpu")
    obs, last_value, stats = train_mod.collect(policy, env, buf, env.reset(), torch.zeros(env.n_games))
    buf.compute_gae(last_value, 1.0, 0.95)
    return buf


def test_store_and_flat_obs_roundtrip(small):
    words, env, policy = small
    buf = _collect_buf(env, policy, T=4)
    idx = torch.tensor([0, 9, 31])
    ob = buf.flat_obs(idx)
    assert ob["tokens"].shape == (3, 30, 3) and ob["mask"].shape == (3, len(words))
    assert torch.equal(ob["tokens"][1], buf.tokens[1, 1])         # flat 9 -> (t=1, n=1)
    assert torch.equal(ob["turn"][2], buf.turn[3, 7])
    assert buf.mask[torch.arange(4).unsqueeze(1), torch.arange(8).unsqueeze(0), buf.actions].all()

    # Pre-step-obs / post-step-reward alignment: the value stored at row t is the
    # policy's value estimate on the pre-step observation stored at row t, so
    # re-running the policy on flat_obs for step t's flat indices must reproduce it.
    t = 2
    idx_t = torch.arange(t * env.n_games, (t + 1) * env.n_games)
    with torch.no_grad():
        _, v = policy(buf.flat_obs(idx_t))
    assert torch.allclose(buf.values[t], v)


def test_one_update_runs_and_is_finite(small):
    words, env, policy = small
    buf = _collect_buf(env, policy, T=4)
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
    buf = _collect_buf(env, policy, T=4)
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
    buf = _collect_buf(env, policy, T=4)
    opt = torch.optim.Adam(policy.parameters(), lr=1e-1)   # huge lr -> big KL
    stats = ppo_update(policy, opt, buf, PPOConfig(minibatch=32, epochs=8, target_kl=1e-6))
    assert stats["epochs_run"] < 8


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
