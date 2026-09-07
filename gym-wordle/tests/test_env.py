import numpy as np
import pytest

from gym_wordle.envs.wordle_env import WordleEnv, LETTER_INDEX, load_word_lists


def make_env(env_paths, **kw):
    return WordleEnv(**env_paths, **kw)


def action(env, word):
    return env.word_to_action[word]


# --- construction -----------------------------------------------------------

def test_action_space_is_sorted_union_of_both_files(env_paths):
    env = make_env(env_paths)
    assert env.words == sorted(set(env.words))
    assert set(env.solutions) <= set(env.words)
    assert len(env.words) == 19  # 15 valids + 5 solutions, 'stale' shared
    assert env.action_space.n == 19
    assert all(env.word_to_action[w] == i for i, w in enumerate(env.words))


def test_letter_matrix_shapes_and_counts(env_paths):
    env = make_env(env_paths)
    assert env.word_letters.shape == (19, 5)
    assert env.word_counts.shape == (19, 26)
    a = action(env, "apple")
    assert env.word_counts[a, LETTER_INDEX["p"]] == 2
    assert env.word_counts[a].sum() == 5
    assert list(env.word_letters[a]) == [LETTER_INDEX[c] for c in "apple"]


def test_observation_layout_constants(env_paths):
    env = make_env(env_paths)
    assert env.POS == 1
    assert env.EXCEEDED == 1 + 26 * 5
    assert env.YELLOWS == env.EXCEEDED + 26
    assert env.obs_size == 183
    nvec = env.observation_space.nvec
    assert nvec[0] == 7  # 0..6 guesses
    assert set(nvec[env.POS:env.EXCEEDED]) == {3}
    assert set(nvec[env.EXCEEDED:env.YELLOWS]) == {2}
    assert set(nvec[env.YELLOWS:]) == {3}  # 0..2 yellows
    assert env.observation_space.dtype == np.int8


def test_reward_schedule_matches_poisson_formula(env_paths):
    env = make_env(env_paths, L=4.0)
    assert len(env.rewards) == 6
    assert env.rewards[0] == pytest.approx(1 - 1 / 5)  # 5 solutions
    assert all(a > b for a, b in zip(env.rewards, env.rewards[1:]))
    binary = make_env(env_paths, min_guesses=False)
    assert binary.rewards == [1.0] * 6


# --- reset ------------------------------------------------------------------

def test_reset_returns_zero_obs_copy_and_info(env_paths):
    env = make_env(env_paths)
    obs, info = env.reset(seed=1)
    assert obs.shape == (183,) and obs.dtype == np.int8
    assert not obs.any()
    assert obs is not env._state
    assert info["valid"] is True
    assert info["n_guesses"] == 0
    assert info["secret_word"] in env.solutions
    assert info["action_mask"].dtype == bool and info["action_mask"].all()
    assert set(info["valid_words"]) == set(range(19))


def test_reset_with_seed_is_reproducible(env_paths):
    env = make_env(env_paths)
    a = [env.reset(seed=7)[1]["secret_word"] for _ in range(5)]
    b = [env.reset(seed=7)[1]["secret_word"] for _ in range(5)]
    assert a == b


def test_reset_with_secret_word_option(env_paths):
    env = make_env(env_paths)
    _, info = env.reset(options={"secret_word": "crane"})
    assert env.secret_word == "crane" and info["secret_word"] == "crane"
    with pytest.raises(AssertionError):
        env.reset(options={"secret_word": "toolong"})


# --- step: encoding, reward, termination --------------------------------------

def test_green_pass_encoding(env_paths):
    env = make_env(env_paths)
    env.reset(options={"secret_word": "apple"})
    obs, r, term, trunc, info = env.step(action(env, "ample"))
    assert obs[0] == 1 and r == 0.0 and term is False and trunc is False
    for pos, c in [(0, "a"), (2, "p"), (3, "l"), (4, "e")]:
        assert obs[env.POS + 5 * LETTER_INDEX[c] + pos] == 2
    assert obs[env.EXCEEDED + LETTER_INDEX["m"]] == 1
    assert obs[env.EXCEEDED + LETTER_INDEX["p"]] == 0
    assert "m" in env.overused_letters


def test_yellow_pass_encoding(env_paths):
    env = make_env(env_paths)
    env.reset(options={"secret_word": "apple"})
    obs, *_ = env.step(action(env, "crane"))  # e green at 4; a yellow at 2
    a, e = LETTER_INDEX["a"], LETTER_INDEX["e"]
    # 'a' guessed at 2, secret has it at 0: possible yellow everywhere but 2.
    # Slot 4 is also marked 1 even though e is green there; that is the
    # existing (slightly loose) semantics and hard mode still handles it.
    assert list(obs[env.POS + 5 * a: env.POS + 5 * a + 5]) == [1, 1, 0, 1, 1]
    assert obs[env.YELLOWS + a] == 1
    assert list(obs[env.POS + 5 * e: env.POS + 5 * e + 5]) == [0, 0, 0, 0, 2]
    assert obs[env.YELLOWS + e] == 0
    for c in "crn":
        assert obs[env.EXCEEDED + LETTER_INDEX[c]] == 1
    assert obs[env.EXCEEDED + e] == 0


def test_exceeded_bit_set_when_letter_over_guessed_with_yellows(env_paths):
    env = make_env(env_paths)
    env.reset(options={"secret_word": "apple"})
    obs, *_ = env.step(action(env, "puppy"))  # three p's, secret has two
    p = LETTER_INDEX["p"]
    assert obs[env.EXCEEDED + p] == 1
    assert obs[env.YELLOWS + p] == 1
    assert list(obs[env.POS + 5 * p: env.POS + 5 * p + 5]) == [0, 1, 2, 0, 1]
    assert "p" not in env.overused_letters  # still has an unplaced copy
    for c in "uy":
        assert obs[env.EXCEEDED + LETTER_INDEX[c]] == 1


def test_win_reward_is_per_step_and_terminates(env_paths):
    env = make_env(env_paths)
    env.reset(options={"secret_word": "apple"})
    _, r1, t1, *_ = env.step(action(env, "crane"))
    _, r2, t2, *_ = env.step(action(env, "apple"))
    assert r1 == 0.0 and t1 is False
    assert r2 == pytest.approx(env.rewards[1]) and t2 is True


def test_final_turn_terminates_and_stays_inside_space(env_paths):
    env = make_env(env_paths, max_attempts=1)
    env.reset(options={"secret_word": "apple"})
    obs, r, term, *_ = env.step(action(env, "crane"))
    assert term is True and r == 0.0
    assert obs[0] == 1
    assert env.observation_space.contains(obs)


def test_observations_are_not_aliased_to_env_state(env_paths):
    env = make_env(env_paths)
    obs0, _ = env.reset(options={"secret_word": "apple"})
    obs1, *_ = env.step(action(env, "crane"))
    assert not obs0.any()
    assert obs1[0] == 1
    obs2, *_ = env.step(action(env, "slate"))
    assert obs1[0] == 1 and obs2[0] == 2
    assert obs1 is not obs2


def test_step_after_termination_raises(env_paths):
    env = make_env(env_paths)
    env.reset(options={"secret_word": "apple"})
    env.step(action(env, "apple"))
    with pytest.raises(AssertionError):
        env.step(action(env, "crane"))
