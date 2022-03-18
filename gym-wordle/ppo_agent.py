import ray
from ray.tune.registry import register_env
from gym_wordle.envs.wordle_rl import WordleEnv

from ray.rllib.agents import ppo

ray.init(ignore_reinit_error=True)

register_env("wordle-v0", lambda config: WordleEnv())

config = ppo.DEFAULT_CONFIG.copy()
# config = {
#     'use_critic': True,
#     'use_gae': True,
#     'lambda': 1.0,
#     'kl_coeff': 0.2,
#     'sgd_minibatch_size': 128,
#     'shuffle_sequences': True,
#     'num_sgd_iter': 30,
#     'lr': 5e-5,
#     'vf_loss_coeff': 1.0,
#     'entropy_coeff': 0.0,
#     'clip_param': 0.3,
#     'vf_clip_param': 10.0,
#     'kl_target': 0.01,
#     'batch_mode': "complete_episodes",
#     'log_sys_usage': True,
#     'output_max_file_size': 67108864,
#     'output_compress_columns': ['obs', 'new_obs'],
#     'input_evaluation': ['is', 'wis'],
#     'num_gpus': 1,
#     'log_level': "WARN",
#     'framework': "torch",
#     'explore': True,
#     'exploration_config': {'type': 'StochasticSampling'},
#     'callbacks': ray.rllib.agents.callbacks.DefaultCallbacks
# }
config["log_level"] = "WARN"
config["framework"] = "torch"
config["num_gpus"] = 1
# config["batch_mode"] = "complete_episodes"
config["num_gpus_per_worker"] = 0.25
config["sgd_minibatch_size"] = 32
# config["ignore_worker_failures"] = True

agent = ppo.PPOTrainer(config, env="wordle-v0")

n_episodes = 100

for n in range(n_episodes):
    result = agent.train()
    chkpt_file = agent.save("checkpoints")
    print(
        f"Episode {n}: reward {result['episode_reward_mean']} in {result['episode_len_mean']} steps, saved {chkpt_file}"
    )
