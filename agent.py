from copy import deepcopy
import pickle
import numpy as np
import matplotlib.pyplot as plt
import gym

import torch
import torch.nn as nn
import torch.nn.functional as F

from wordle_rl import WordleEnv


class Wordler(object):
    def __init__(self, device, verbose=False):
        # self.env = gym.make(WordleEnv).env
        self.env = WordleEnv()
        self.env.seed(19)
        self.np_rng = np.random.default_rng(19)

        self.n_inputs = 183

        self.alpha = 0.03
        self.epsilon = 0.99
        self.epsilon_decay = 0.999
        self.min_epsilon = 0.02
        self.gamma = 0.999

        self.device = device
        self.verbose = verbose


    def solve(
        self,
        max_episodes=1,
        C=8,
        batch_size=128,
        max_experience=1000000,
    ):

        self.model = Model().to(self.device).float()
        self.target_model = deepcopy(self.model)
        self.loss_fx = nn.MSELoss() # nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.alpha,
        )

        self.initialize_replay_memory(n=max_experience)

        self.n_t = 0 # replay memory index
        self.n_episodes = 0
        overall_rewards = list()
        is_solved = False
        while not is_solved:
            self.state = self.env.reset()
            episode_R = 0
            is_terminal = False
            while not is_terminal:
                # action = self.np_rng.choice(self.env.action_space.n)
                info = {'valid': False}
                while not info['valid']:
                    action = self.epsilon_greedy_action_selection(
                        self.model,
                        self.state,
                    )
                    if type(action) is torch.Tensor:
                        action = action.item()
                    s_prime, R, is_terminal, info = self.env.step(action)

                self.epsilon *= self.epsilon_decay
                self.epsilon = max(self.min_epsilon, self.epsilon)

                episode_R += R

                self.replay_memory[self.n_t,:self.n_inputs] = self.state
                self.replay_memory[self.n_t,self.n_inputs:-3] = s_prime
                self.replay_memory[self.n_t,-3] = action
                self.replay_memory[self.n_t,-2] = R
                self.replay_memory[self.n_t,-1] = 1 if is_terminal else 0
                self.n_t += 1

                batch_inds = np.random.choice(
                    self.n_t,
                    size=batch_size,
                    # p=(probs / np.sum(probs)),
                )
                batch = self.replay_memory[batch_inds,:]

                y_targets = torch.zeros(batch_size, 1, device=self.device)
                for i,j in enumerate(batch_inds):
                    if self.replay_memory[j,-1]:
                        y_targets[i,0] = self.replay_memory[j,-2]
                    else:
                        y_targets[i,0] = (
                            (
                                self.replay_memory[j,-2]
                                + self.gamma *
                                self.select_action(
                                    self.target_model,
                                    self.replay_memory[j,self.n_inputs:-3],
                                )[1]
                            )
                        )

                input = torch.zeros(
                    batch_size, self.n_inputs, device=self.device
                )
                input[:,:] = torch.tensor(
                    batch[:,:self.n_inputs], device=self.device
                )

                # y_preds = self.model(input).sum(dim=1, keepdims=True)
                y_preds = torch.gather(
                    self.model(input),
                    1,
                    torch.tensor(batch[:,-3].astype(int)).view(-1,1).to(device)
                )
                loss = self.loss_fx(y_preds, y_targets)

                self.optimizer.zero_grad()
                loss.backward()
                total_norm = torch.norm(
                    torch.stack(
                        [
                            torch.norm(p.grad.detach(), 2).to(device)
                            for p in self.model.parameters()
                        ]
                    ),
                    2
                )
                print("total norm", round(total_norm.item(), 1))
                nn.utils.clip_grad_norm_(self.model.parameters(), 1000000000)
                self.optimizer.step()

                if (self.n_episodes % C) == 0:
                    self.target_model = deepcopy(self.model)

                self.state = s_prime

            if self.verbose:
                print("guesses", self.env.n_guesses)
                print("reward", round(episode_R, 2))
            # print(self.env.secret_word)
            # print(self.state)
            # print(self.env.reward)
            if self.verbose and (self.n_episodes % 10 == 0): print(self.n_t)
            overall_rewards.append(episode_R)
            if self.verbose: print(
                f"{self.n_episodes}, mean of last 100 episode rewards",
                np.mean(overall_rewards[-100:]).round(3)
            )

            self.n_episodes += 1
            q_val_converged = False
            if (q_val_converged):
                is_solved = True
            elif self.n_episodes == max_episodes:
                break

        self.rewards = overall_rewards

        return self.model


    def initialize_replay_memory(self, n):
        # create as array of maximum experience rows
        self.replay_memory = np.zeros(
            (
                n, # experiences to track in replay memory
                2 * self.n_inputs + 3 # two states, one action, reward, done
            )
        )
        return None


    def epsilon_greedy_action_selection(self, model, input):
        if np.random.random() > self.epsilon:
            action, _ = self.select_action(model, input)
        else:
            action = np.random.randint(self.env.action_space.n)

        return action


    def select_action(self, model, input):
        if type(input) is np.ndarray:
            input = torch.tensor(input.astype(np.float32)).to(self.device)

        out = model(input)
        action = torch.argmax(out)
        best_action_value = out[action]

        return action, best_action_value


    def create_r_per_ep_fig(
        self,
        out_path='./training_reward_per_episode.png',
        title_suffix='',
    ):
        fig, ax = plt.subplots()

        ax.set_xlabel('Episode Number')
        ax.set_ylabel('Reward')
        ax.set_title(f'Agent training reward per episode{title_suffix}')

        ax.plot(np.arange(self.n_episodes), self.rewards)

        plt.xticks(
            np.arange(0, self.n_episodes, int(round(self.n_episodes / 10)))
        )
        plt.savefig(out_path)
        return None


class Model(nn.Module):
    def __init__(
        self, n_inputs=183,
        hidden_layer_1=256,
        hidden_layer_2=512,
        hidden_layer_3=256,
        n_outputs=12972,
    ):
        super(Model, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(n_inputs, hidden_layer_1),
            nn.ReLU(),
            nn.Linear(hidden_layer_1, hidden_layer_2),
            nn.ReLU(),
            nn.Linear(hidden_layer_2, hidden_layer_3),
            nn.ReLU(),
            nn.Linear(hidden_layer_3, n_outputs),
        )


    def forward(self, x):
        return self.net(x)


def main(device, max_episodes=1000, C=8, batch_size=128, verbose=False):
    agent = Wordler(device, verbose)
    agent.solve(
        max_episodes=max_episodes,
        C=C,
        batch_size=batch_size,
    )

    agent.create_r_per_ep_fig()

    with open('./model_main_20220316', 'wb') as f:
        pickle.dump(agent, f)

    return None


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    main(device=device, verbose=True)
