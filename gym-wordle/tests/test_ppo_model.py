import pytest
import torch

from gym_wordle.agents.common import load_checkpoint, save_checkpoint, word_feature_matrix
from gym_wordle.agents.ppo.model import WordlePolicy
from gym_wordle.envs.batched import BatchedWordle
from gym_wordle.envs.wordle_env import load_word_lists


@pytest.fixture
def setup(env_paths):
    words, solutions = load_word_lists(env_paths["valid_words_path"], env_paths["solution_words_path"])
    env = BatchedWordle(words, solutions, 4, seed=0)
    torch.manual_seed(0)
    policy = WordlePolicy(len(words), word_feature_matrix(words, 5), d_model=32, n_heads=4, d_ff=64, n_layers=2)
    env.reset()
    env.step(torch.tensor([env.word_to_action[w] for w in ["slate", "crane", "paper", "least"]]))
    return words, env, policy


def test_shapes_and_masking(setup):
    words, env, policy = setup
    obs = env.observation()
    logits, value = policy(obs)
    assert logits.shape == (4, len(words)) and value.shape == (4,)
    assert torch.isinf(logits[~obs["mask"]]).all() and (logits[~obs["mask"]] < 0).all()
    assert torch.isfinite(logits[obs["mask"]]).all()
    probs = torch.softmax(logits, -1)
    assert (probs[~obs["mask"]] == 0).all()
    assert torch.allclose(probs.sum(-1), torch.ones(4))


def test_entropy_is_over_legal_subset(setup):
    words, env, policy = setup
    obs = env.observation()
    _, _, ent, _ = policy.act(obs)
    n_legal = obs["mask"].sum(-1).float()
    assert (ent <= torch.log(n_legal) + 1e-5).all()
    assert (ent >= 0).all()


def test_act_and_evaluate_actions_agree(setup):
    words, env, policy = setup
    obs = env.observation()
    a, lp, ent, v = policy.act(obs)
    assert obs["mask"][torch.arange(4), a].all(), "sampled actions must be legal"
    lp2, ent2, v2 = policy.evaluate_actions(obs, a)
    assert torch.allclose(lp, lp2) and torch.allclose(ent, ent2) and torch.allclose(v, v2)
    g, *_ = policy.act(obs, greedy=True)
    logits, _ = policy(obs)
    assert torch.equal(g, logits.argmax(-1))


def test_padding_tokens_do_not_affect_output(setup):
    words, env, policy = setup
    policy.eval()
    obs = env.observation()
    logits, value = policy(obs)
    noisy = dict(obs)
    tokens = obs["tokens"].clone()
    tokens[:, 5:, 0] = 25            # scribble on padded cells
    tokens[:, 5:, 1] = 2
    noisy["tokens"] = tokens
    logits2, value2 = policy(noisy)
    assert torch.allclose(logits[obs["mask"]], logits2[obs["mask"]], atol=1e-5)
    assert torch.allclose(value, value2, atol=1e-5)


def test_every_parameter_gets_a_gradient(setup):
    words, env, policy = setup
    obs = env.observation()
    a, lp, ent, v = policy.act(obs)
    (lp.mean() + ent.mean() + v.mean()).backward()
    for name, p in policy.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), name


def test_checkpoint_roundtrip(setup, tmp_path):
    words, env, policy = setup
    policy.eval()
    path = str(tmp_path / "policy.pt")
    save_checkpoint(policy, path)
    loaded = load_checkpoint(WordlePolicy, path, "cpu").eval()
    assert torch.equal(loaded.word_features, policy.word_features)
    obs = env.observation()
    l1, v1 = policy(obs)
    l2, v2 = loaded(obs)
    assert torch.allclose(l1[obs["mask"]], l2[obs["mask"]]) and torch.allclose(v1, v2)


def test_parameter_count_is_small(setup):
    words, env, policy = setup
    full = WordlePolicy(12947, torch.zeros(12947, 156))
    assert sum(p.numel() for p in full.parameters()) < 1_000_000
