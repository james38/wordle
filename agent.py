import numpy as np
import gym

from wordle_rl import WordleEnv


class Wordler(object):
    def __init__(self):
        self.env = gym.make(WordleEnv).env
        self.env.seed(19)
        self.np_rng = np.random.default_rng(19)


    def solve(self, max_episodes):

        self.n_episodes = 0
        is_solved = False
        while not is_solved:
            self.state = self.env.reset()
            is_terminal = False
            while not is_terminal:
                action = self.np_rng.choice(self.env.action_space.n)
                # action = self.epsilon_greedy_action_selection(
                #     self.model,
                #     self.input_layer,
                # )
                s_prime, R, is_terminal, info = self.env.step(action)

                self.state = s_prime

            self.n_episodes += 1
            q_val_converged = False
            if (q_val_converged):
                is_solved = True
            elif self.n_episodes == max_episodes:
                break

def main():
    agent = Wordler()
    agent.solve(1)
    return None


if __name__ == "__main__":
    main()
