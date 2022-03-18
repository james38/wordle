import gym
import gym_wordle


def run_episode(env):
    state = env.reset()
    reward = 0

    done = False
    while not done:
        action = env.action_space.sample()
        state, R, done, info = env.step(action)
        reward += R

    print(f"reward: {round(reward, 2)}")
    return reward


def run_baseline(env, episodes=1000):

    history = [run_episode(env) for _ in range(episodes)]

    print(f"Baseline average reward: {round(sum(history) / len(history), 3)}")

    return history


if __name__ == "__main__":
    env = gym.make("wordle-v0")

    history = run_baseline(env)
