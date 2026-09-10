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
