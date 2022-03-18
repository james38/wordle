import os

import gym
import ray
from ray.tune.registry import register_env
from gym_wordle.envs.wordle_rl import WordleEnv

from ray.rllib.agents import ppo


def restore_evaluate(policy_checkpoint):
    homedir = os.getenv("HOME")

    config = ppo.DEFAULT_CONFIG.copy()
    config["log_level"] = "WARN"
    config["framework"] = "torch"
    config["num_gpus"] = 1
    # config["batch_mode"] = "complete_episodes"
    config["num_gpus_per_worker"] = 0.25
    config["sgd_minibatch_size"] = 32

    agent = ppo.PPOTrainer(config, env="wordle-v0")
    agent.restore(
        f"{homedir}/rl/wordle/gym-wordle/checkpoints/{policy_checkpoint}"
    )

    env = gym.make("wordle-v0")
    state = env.reset()
    total_reward = 0

    done = False
    while not done:
        action = agent.compute_action(state)
        state, reward, done, info = env.step(action)
        total_reward += reward

    print(total_reward)
    print(env.secret_word)

    return None


def main(policy_checkpoint, n_episodes):
    homedir = os.getenv("HOME")
    # n_cpu = os.cpu_count()
    config = ppo.DEFAULT_CONFIG.copy()
    config["log_level"] = "WARN"
    config["framework"] = "torch"
    config["num_gpus"] = 1
    # config["batch_mode"] = "complete_episodes"
    config["num_gpus_per_worker"] = 0.25
    config["sgd_minibatch_size"] = 32
    # config["ignore_worker_failures"] = True
    # config['num_workers'] = max(1, n_cpu - 1)
    # config['num_envs_per_worker'] = 1

    agent = ppo.PPOTrainer(config, env="wordle-v0")
    agent.restore(
        f"{homedir}/rl/wordle/gym-wordle/checkpoints/{policy_checkpoint}"
    )

    for n in range(n_episodes):
        result = agent.train()
        chkpt_file = agent.save("checkpoints")
        print(
            f"Episode {n}: reward {result['episode_reward_mean']} in {result['episode_len_mean']} steps, saved {chkpt_file}"
        )

    return None


if __name__ == "__main__":

    ray.init(ignore_reinit_error=True)
    register_env("wordle-v0", lambda config: WordleEnv())

    policy_checkpoint = "checkpoint_000122/checkpoint-122"
    main(policy_checkpoint, n_episodes=100)
