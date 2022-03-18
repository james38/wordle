from gym.envs.registration import register

register(
    id="wordle-v0",
    entry_point="gym_wordle.envs:WordleEnv",
)
