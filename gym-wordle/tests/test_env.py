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


# --- hard mode ----------------------------------------------------------------

def test_first_guess_is_never_rejected(env_paths):
    env = make_env(env_paths)
    env.reset(options={"secret_word": "apple"})
    assert env._action_mask().all()


def test_hard_mode_rejection_leaves_state_and_turn_unchanged(env_paths):
    env = make_env(env_paths)
    env.reset(options={"secret_word": "apple"})
    obs1, *_ = env.step(action(env, "crane"))  # a yellow, e green at 4
    obs2, r, term, trunc, info = env.step(action(env, "puppy"))  # no a, no e
    assert info["valid"] is False
    assert r == 0.0 and term is False
    assert env.n_guesses == 1 and obs2[0] == 1
    assert np.array_equal(obs1, obs2)
    assert action(env, "puppy") not in info["valid_words"]
    assert not info["action_mask"][action(env, "puppy")]


def test_greens_must_be_kept_in_place(env_paths):
    env = make_env(env_paths)
    env.reset(options={"secret_word": "apple"})
    _, _, _, _, info = env.step(action(env, "ample"))  # a?ple
    assert set(info["valid_words"].values()) == {"apple"}


def test_grey_letter_may_not_be_reused(env_paths):
    env = make_env(env_paths)
    env.reset(options={"secret_word": "apple"})
    _, _, _, _, info = env.step(action(env, "crane"))  # c, r, n grey
    legal = set(info["valid_words"].values())
    # cable and table both keep e at 4 and contain a; only c is grey
    assert "table" in legal
    assert "cable" not in legal


def test_yellow_letters_must_be_reused(env_paths):
    env = make_env(env_paths)
    env.reset(options={"secret_word": "apple"})
    _, _, _, _, info = env.step(action(env, "crane"))  # a yellow, e green at 4
    legal = set(info["valid_words"].values())
    assert {"apple", "ample", "maple", "stale", "plate", "table"} <= legal
    assert "happy" not in legal  # has a, but no e at 4
    assert "least" not in legal  # has a and e, but e is not at 4
    assert "stone" not in legal  # no a


def test_accepted_guess_is_removed_from_valid_words(env_paths):
    env = make_env(env_paths)
    env.reset(options={"secret_word": "delta"})
    _, _, _, _, info = env.step(action(env, "dealt"))  # d,e green; a,l,t yellow
    assert env._hard_mode_mask()[action(env, "dealt")]  # still hard-mode legal
    assert action(env, "dealt") not in info["valid_words"]
    assert not info["action_mask"][action(env, "dealt")]
    _, _, _, _, info2 = env.step(action(env, "dealt"))
    assert info2["valid"] is False and env.n_guesses == 1


def wordle_feedback(secret, guess):
    """Standard Wordle colouring: 'g' green, 'y' yellow, '.' grey, duplicates handled."""
    fb = ["."] * len(guess)
    remaining = list(secret)
    for p, (g, s) in enumerate(zip(guess, secret)):
        if g == s:
            fb[p] = "g"
            remaining[p] = None
    for p, g in enumerate(guess):
        if fb[p] == "." and g in remaining:
            fb[p] = "y"
            remaining[remaining.index(g)] = None
    return fb


def reference_legal(secret, guesses, word):
    """Hard-mode rule written independently of the env's state encoding.

    1. Every green position is fixed.
    2. A letter is exhausted once a guess shows a grey copy of it and no
       yellow copy (all of its copies in the secret are then green); an
       exhausted letter may not appear at a non-green position.
    3. The word must contain each letter at least max over guesses of
       (greens + yellows for that letter in that guess) times.
    """
    greens = {}
    exhausted = set()
    required = {}
    for g in guesses:
        fb = wordle_feedback(secret, g)
        for p, (c, f) in enumerate(zip(g, fb)):
            if f == "g":
                greens[p] = c
        letters = set(g)
        for c in letters:
            marks = [f for gc, f in zip(g, fb) if gc == c]
            if "." in marks and "y" not in marks:
                exhausted.add(c)
            n = sum(1 for f in marks if f != ".")
            if n:
                required[c] = max(required.get(c, 0), n)
    if any(word[p] != c for p, c in greens.items()):
        return False
    if any(word[p] in exhausted for p in range(len(word)) if p not in greens):
        return False
    return all(word.count(c) >= n for c, n in required.items())


def test_valid_words_matches_acceptance_rule(env_paths):
    env = make_env(env_paths)
    openers = ["crane", "puppy", "paper", "dealt", "stale"]
    checked = 0
    for secret in env.solutions:
        for opener in openers:
            if opener == secret:
                continue
            env.reset(options={"secret_word": secret})
            _, _, _, _, info = env.step(action(env, opener))
            for a in range(env.action_space.n):
                expected = reference_legal(secret, [opener], env.words[a]) and env.words[a] != opener
                assert bool(info["action_mask"][a]) == expected, (secret, [opener], env.words[a])
            second_guesses = [env.words[a] for a in np.flatnonzero(info["action_mask"]) if env.words[a] != secret]
            for second in second_guesses:
                env.reset(options={"secret_word": secret})
                env.step(action(env, opener))
                _, _, _, _, info2 = env.step(action(env, second))
                for a in range(env.action_space.n):
                    expected = reference_legal(secret, [opener, second], env.words[a]) and env.words[a] not in (opener, second)
                    assert bool(info2["action_mask"][a]) == expected, (secret, [opener, second], env.words[a])
                    checked += 1
    assert checked > 0


def test_secret_word_is_always_legal(env_paths):
    env = make_env(env_paths)
    for secret in env.solutions:
        for opener in ["crane", "puppy", "paper", "dealt"]:
            env.reset(options={"secret_word": secret})
            if opener == secret:
                continue
            _, _, _, _, info = env.step(action(env, opener))
            assert info["action_mask"][action(env, secret)]


# --- registration and checker -------------------------------------------------

def test_registered_with_gymnasium(env_paths):
    import gymnasium
    import gym_wordle  # noqa: F401  registers Wordle-v0

    env = gymnasium.make("Wordle-v0", **env_paths)
    obs, info = env.reset(seed=0)
    assert obs.shape == (183,)
    assert isinstance(env.unwrapped, WordleEnv)


def test_passes_gymnasium_env_checker(env_paths):
    from gymnasium.utils.env_checker import check_env

    check_env(make_env(env_paths), skip_render_check=True)
