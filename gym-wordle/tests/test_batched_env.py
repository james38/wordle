import pytest
import torch

from gym_wordle.envs.batched import BatchedWordle, poisson_rewards
from gym_wordle.envs.feedback import hard_mode_mask
from gym_wordle.envs.wordle_env import load_word_lists


@pytest.fixture
def lists(env_paths):
    return load_word_lists(env_paths["valid_words_path"], env_paths["solution_words_path"])


def make(lists, n=4, **kw):
    words, solutions = lists
    return BatchedWordle(words, solutions, n, seed=0, **kw)


def test_poisson_rewards_table():
    r = poisson_rewards(4.0, 6)
    assert r[0] == pytest.approx(1.0)
    assert r == pytest.approx([1.0, 0.9254, 0.7761, 0.5771, 0.3781, 0.2189], abs=1e-4)
    assert r == sorted(r, reverse=True)


def test_construction_and_reset(lists):
    env = make(lists)
    words, solutions = lists
    assert env.W_letters.shape == (len(words), 5)
    env.reset()
    assert env.turn.tolist() == [0, 0, 0, 0]
    assert not env.guessed.any()
    assert (env.green_pos == -1).all() and not env.min_count.any()
    sol_set = set(env.solution_idx.tolist())
    assert all(s in sol_set for s in env.secret.tolist())


def test_same_seed_same_secrets(lists):
    a = make(lists, n=32); b = make(lists, n=32)
    a.reset(); b.reset()
    assert torch.equal(a.secret, b.secret)


def test_set_secrets(lists):
    env = make(lists, n=2)
    idx = torch.tensor([env.word_to_action["crane"], env.word_to_action["delta"]])
    env.set_secrets(idx)
    assert torch.equal(env.secret, idx)


def test_solve_on_first_guess_pays_one_and_resets(lists):
    env = make(lists, n=2)
    env.set_secrets(torch.tensor([env.word_to_action["crane"], env.word_to_action["delta"]]))
    a = torch.tensor([env.word_to_action["crane"], env.word_to_action["slate"]])
    obs, reward, done, info = env.step(a)
    assert done.tolist() == [True, False]
    assert info["solved"].tolist() == [True, False]
    assert reward[0].item() == pytest.approx(1.0) and reward[1].item() == 0.0
    assert info["n_guesses"][0].item() == 1
    assert info["secret"][0].item() == env.word_to_action["crane"]
    # row 0 auto-reset, row 1 untouched
    assert env.turn.tolist() == [0, 1]
    assert not env.guessed[0].any()
    assert env.guessed[1, env.word_to_action["slate"]]
    assert (env.green_pos[0] == -1).all()


def test_reward_schedule_by_guess_number(lists):
    env = make(lists, n=1, hard_mode=False)
    words, _ = lists
    secret = env.word_to_action["crane"]
    fillers = [w for w in words if w != "crane"]
    for k in range(1, 7):
        env.set_secrets(torch.tensor([secret]))
        for w in fillers[: k - 1]:
            _, r, d, _ = env.step(torch.tensor([env.word_to_action[w]]))
            assert r.item() == 0.0 and not d.item()
        _, r, d, info = env.step(torch.tensor([secret]))
        assert d.item() and info["solved"].item()
        assert r.item() == pytest.approx(env.rewards[k - 1].item())


def test_fail_pays_zero_and_resets(lists):
    env = make(lists, n=1, hard_mode=False)
    words, _ = lists
    env.set_secrets(torch.tensor([env.word_to_action["crane"]]))
    fillers = [w for w in words if w != "crane"][:6]
    for i, w in enumerate(fillers):
        _, r, d, info = env.step(torch.tensor([env.word_to_action[w]]))
        assert r.item() == 0.0
        assert d.item() == (i == 5)
    assert not info["solved"].item() and info["n_guesses"].item() == 6
    assert env.turn.item() == 0


def test_played_word_never_legal_again_and_illegal_raises(lists):
    env = make(lists, n=1, hard_mode=False)
    env.reset()
    a = torch.tensor([env.word_to_action["puppy"]])
    env.step(a)
    assert not env.legal_mask()[0, a.item()]
    with pytest.raises(ValueError):
        env.step(a)


def test_mask_cache_matches_fresh_legal_mask(lists):
    """The pre-step guard mask cached from the previous observation() must agree
    exactly with a freshly computed legal_mask(), and the illegal-action guard
    must still fire off the cache."""
    env = make(lists, n=2, hard_mode=True)
    env.set_secrets(torch.tensor([env.word_to_action["apple"], env.word_to_action["delta"]]))
    assert torch.equal(env._mask, env.legal_mask())

    a = torch.tensor([env.word_to_action["crane"], env.word_to_action["slate"]])
    obs, _, done, _ = env.step(a)
    assert not done.any()          # neither guess matches its secret: no auto-reset
    assert torch.equal(env._mask, env.legal_mask())
    assert torch.equal(obs["mask"], env.legal_mask())

    with pytest.raises(ValueError):
        env.step(a)   # crane/slate already guessed by both games


def test_legal_mask_hard_vs_normal(lists):
    hard = make(lists, n=1, hard_mode=True)
    soft = make(lists, n=1, hard_mode=False)
    for env in (hard, soft):
        env.set_secrets(torch.tensor([env.word_to_action["crane"]]))
        env.step(torch.tensor([env.word_to_action["slate"]]))   # a, e green
    expected_hard = hard_mode_mask(hard.green_pos, hard.min_count, hard.W_letters, hard.W_counts) & ~hard.guessed
    assert torch.equal(hard.legal_mask(), expected_hard)
    assert torch.equal(soft.legal_mask(), ~soft.guessed)
    assert hard.legal_mask().sum() < soft.legal_mask().sum()
    assert hard.legal_mask()[0, hard.word_to_action["crane"]]


def test_hard_mode_bookkeeping(lists):
    env = make(lists, n=1)
    env.set_secrets(torch.tensor([env.word_to_action["apple"]]))
    env.step(torch.tensor([env.word_to_action["paper"]]))        # colours 1,1,2,1,0
    from gym_wordle.envs.wordle_env import LETTER_INDEX as L
    assert env.green_pos[0].tolist() == [-1, -1, L["p"], -1, -1]
    mc = env.min_count[0]
    assert mc[L["p"]] == 2 and mc[L["a"]] == 1 and mc[L["e"]] == 1 and mc[L["r"]] == 0


def test_observation_layout(lists):
    env = make(lists, n=2)
    env.set_secrets(torch.tensor([env.word_to_action["apple"], env.word_to_action["crane"]]))
    obs = env.observation()
    assert obs["tokens"].shape == (2, 30, 3) and obs["tokens"].dtype == torch.long
    assert obs["pad"].shape == (2, 30) and obs["pad"].all()
    assert obs["turn"].tolist() == [0, 0]
    assert obs["mask"].shape == (2, len(lists[0]))

    obs, *_ = env.step(torch.tensor([env.word_to_action["paper"], env.word_to_action["slate"]]))
    from gym_wordle.envs.wordle_env import LETTER_INDEX as L
    row = obs["tokens"][0]
    assert row[:5, 0].tolist() == [L[c] for c in "paper"]
    assert row[:5, 1].tolist() == [1, 1, 2, 1, 0]
    assert row[:5, 2].tolist() == [0] * 5
    assert row[5:10, 2].tolist() == [1] * 5
    assert obs["pad"][0].tolist() == [False] * 5 + [True] * 25
    assert obs["turn"].tolist() == [1, 1]
    assert torch.equal(obs["mask"], env.legal_mask())


def test_observation_after_auto_reset_is_fresh(lists):
    env = make(lists, n=1)
    env.set_secrets(torch.tensor([env.word_to_action["crane"]]))
    obs, _, done, _ = env.step(torch.tensor([env.word_to_action["crane"]]))
    assert done.item()
    assert obs["pad"].all() and obs["turn"].item() == 0 and obs["mask"].all()


def test_candidates_and_shaping(lists):
    words, _ = lists
    env = make(lists, n=1, hard_mode=False, shaping_coef=0.5)
    env.set_secrets(torch.tensor([env.word_to_action["crane"]]))
    assert env.candidates().item() == len(words)
    _, r, _, _ = env.step(torch.tensor([env.word_to_action["slate"]]))
    after = env.candidates().item()
    assert 0 < after < len(words)
    import math
    assert r.item() == pytest.approx(0.5 * (math.log2(len(words)) - math.log2(after)))
    _, r2, d, _ = env.step(torch.tensor([env.word_to_action["crane"]]))
    assert d.item()
    assert r2.item() == pytest.approx(env.rewards[1].item() + 0.5 * (math.log2(after) - 0.0))


def test_shaping_off_gives_zero_intermediate_reward(lists):
    env = make(lists, n=1, hard_mode=False)
    env.set_secrets(torch.tensor([env.word_to_action["crane"]]))
    _, r, _, _ = env.step(torch.tensor([env.word_to_action["slate"]]))
    assert r.item() == 0.0


def test_shaping_cand_resets_to_full_dict_after_auto_reset(lists):
    """A game that solves (and auto-resets) mid-rollout must not carry its stale,
    narrowed candidate count into the freshly reset game's next shaping term."""
    words, _ = lists
    env = make(lists, n=1, hard_mode=False, shaping_coef=0.5)
    env.set_secrets(torch.tensor([env.word_to_action["crane"]]))
    _, _, d, _ = env.step(torch.tensor([env.word_to_action["crane"]]))   # solves -> auto-resets
    assert d.item()
    assert env._cand.item() == len(words)

    new_secret_idx = env.secret.item()
    guess_word = next(w for w in words if env.word_to_action[w] != new_secret_idx)
    _, r2, d2, _ = env.step(torch.tensor([env.word_to_action[guess_word]]))
    assert not d2.item()
    after = env.candidates().item()
    import math
    assert r2.item() == pytest.approx(0.5 * (math.log2(len(words)) - math.log2(after)))


def test_batched_wordle_exported():
    from gym_wordle.envs import BatchedWordle as B
    assert B is BatchedWordle
