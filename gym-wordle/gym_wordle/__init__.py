from gymnasium.envs.registration import register

from gym_wordle.envs.wordle_env import WordleEnv

register(id="Wordle-v0", entry_point="gym_wordle.envs.wordle_env:WordleEnv")

__all__ = ["WordleEnv"]
