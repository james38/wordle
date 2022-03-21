from copy import deepcopy
import pickle
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
import gym

import torch
import torch.nn as nn

from wordle_rl import WordleEnv


class Wordler(object):
    def __init__(self, device, verbose=False):
        self.env = WordleEnv()
        self.env.seed(19)
        self.np_rng = np.random.default_rng(19)

        self.n_inputs = 183

        self.alpha = 0.001
        self.epsilon = 0.999
        self.epsilon_decay = 0.99998
        self.min_epsilon = 0.02
        self.gamma = 0.9999

        self.device = device
        self.verbose = verbose


    def solve(
        self,
        model_loc=None,
        max_episodes=1,
        C=8,
        batch_size=64,
        max_experience=1000000,
        clip_val=2,
        warmup=2,
        settle_at=0.01,
        lr_drop_at=0.9,
        lr_drop_rate=0.1,
    ):

        if model_loc is not None:
            self.model = torch.load(model_loc)
        else:
            self.model = Model().to(self.device).float()
        for p in self.model.parameters():
            p.register_hook(lambda x: torch.clamp(x, -clip_val, clip_val))
        self.target_model = deepcopy(self.model)
        C_init = C
        self.loss_fx = nn.MSELoss()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.alpha,
        )

        self.initialize_replay_memory(n=max_experience)

        self.n_episodes = 0
        overall_rewards = list()
        is_solved = False
        while not is_solved:
            if self.n_episodes == int(lr_drop_at * max_episodes):
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] *= lr_drop_rate
                self.min_epsilon *= lr_drop_rate
                C *= 4
            elif self.n_episodes == int(settle_at * max_episodes):
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] *= 0.3
                C *= 4
            if self.n_episodes < (warmup * C_init):
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = (
                        self.alpha * (1 + self.n_episodes) / (warmup * C_init)
                    )
                C = max(2, int((1 + self.n_episodes) / warmup))
            self.state = self.env.reset()
            self.guessed = set()
            episode_R = 0
            is_terminal = False
            while not is_terminal:
                info = {'valid': False}
                while not info['valid']:
                    action = self.epsilon_greedy_action_selection(
                        self.model,
                        self.state,
                    )
                    if type(action) is torch.Tensor:
                        action = action.item()
                    s_prime, R, is_terminal, info = self.env.step(action)

                self.guessed.add(action)
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

                y_targets = torch.full(
                    (2 * batch_size, 1),
                    -1e-4,
                    device=self.device
                )
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

                y_preds = torch.gather(
                    self.model(self.input_scaling(input)),
                    1,
                    torch.tensor(batch[:,-3].astype(int)).view(-1,1).to(device)
                )

                valid_actions = self.env.info['valid_words'].keys()
                invalid_actions = {
                    w for w in self.env.action_words.keys()
                    if w not in valid_actions
                }
                invalid_actions.update(self.guessed)
                invalid_action_inds = np.random.choice(
                    list(invalid_actions), size=batch_size
                )

                state_input = torch.tensor(
                    self.state,
                    device=self.device
                ).float()
                state_logits = self.model(self.input_scaling(state_input))
                y_preds_invalid = state_logits[invalid_action_inds]

                y_preds = torch.cat((y_preds, y_preds_invalid.view(-1,1)))
                loss = self.loss_fx(y_preds, y_targets)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                self.state = s_prime

            if self.verbose:
                print("guesses", self.env.n_guesses)
                print("reward", round(episode_R, 2))

            overall_rewards.append(episode_R)
            if self.verbose and (self.n_episodes % 10 == 0):
                print("total guesses: ", self.n_t)
                print(
                    f"episode {self.n_episodes}, last 100 episode mean reward",
                    np.mean(overall_rewards[-100:]).round(3)
                )

            if (self.n_episodes % C) == 0:
                self.target_model = deepcopy(self.model)
            if self.n_episodes == int(max_episodes / 2):
                fsufx = dt.datetime.now().strftime("%Y%m%d.%H.%M.%S")
                torch.save(self.model, f'./model_halftrain_{fsufx}')
            self.n_episodes += 1
            # q_val_converged = False
            # if (q_val_converged):
            #     is_solved = True
            if self.n_episodes == max_episodes:
                break

        fsufx = dt.datetime.now().strftime("%Y%m%d.%H.%M.%S")
        torch.save(self.model, f'./model_{fsufx}')

        self.rewards = overall_rewards

        return self.model, fsufx


    def initialize_replay_memory(self, n):
        # create as array of maximum experience rows
        self.replay_memory = np.zeros(
            (
                n, # experiences to track in replay memory
                2 * self.n_inputs + 3 # two states, one action, reward, done
            )
        )
        self.n_t = 0
        return None


    def epsilon_greedy_action_selection(self, model, input):
        if np.random.random() > self.epsilon:
            action, _ = self.select_action(model, input)
        else:
            # action = np.random.randint(self.env.action_space.n)
            valid_actions = list(self.env.info['valid_words'].keys())
            action = np.random.choice(valid_actions)

        return action


    def select_action(self, model, input, exploit=False):
        if type(input) is np.ndarray:
            input = torch.tensor(input.astype(np.float32)).to(self.device)

        out = model(self.input_scaling(input))
        valid_actions = self.env.info['valid_words'].keys()
        if exploit:
            invalid_actions = [
                w for w in self.env.action_words.keys()
                if (w not in valid_actions) or (w in self.guessed)
            ]
        else:
            invalid_actions = [
                w for w in self.env.action_words.keys()
                if (w not in valid_actions)
            ]
        out[invalid_actions] = float("-inf")
        action = torch.argmax(out)
        best_action_value = out[action]

        return action, best_action_value


    def input_scaling(self, x):
        return (x - 1) / 5


    def create_r_per_ep_fig(
        self,
        out_path='./training_reward_per_episode.png',
        title_suffix='',
        window=100,
    ):
        fig, ax = plt.subplots()

        ax.set_xlabel('Episode Number')
        ax.set_ylabel('Reward')
        ax.set_title(
            f'Agent training mean last {window} rewards\n{title_suffix}'
        )

        from numpy.lib.stride_tricks import sliding_window_view
        window = min(window, len(self.rewards))
        rolling = sliding_window_view(
            np.array(self.rewards), window_shape=window
        ).mean(axis=1)
        n_rolling = rolling.shape[0]
        n_episodes = n_rolling + window
        ax.plot(np.arange(window, n_episodes), rolling)

        plt.xticks(
            np.arange(
                window, n_episodes, max(1, int(round(n_episodes / 10)))
            )
        )
        plt.savefig(out_path)
        return None


class Model(nn.Module):
    def __init__(
        self, n_inputs=183,
        # hidden_layer_1=256,
        # hidden_layer_2=512,
        # hidden_layer_3=256,
        hidden_layer_1=128,
        hidden_layer_2=64,
        hidden_layer_3=32,
        hidden_layer_4=32,
        hidden_layer_5=64,
        hidden_layer_6=128,
        n_outputs=12947,
    ):
        super(Model, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(n_inputs, hidden_layer_1),
            nn.ReLU(),
            nn.Linear(hidden_layer_1, hidden_layer_2),
            nn.ReLU(),
            nn.Linear(hidden_layer_2, hidden_layer_3),
            nn.ReLU(),
            nn.Linear(hidden_layer_3, hidden_layer_4),
            nn.ReLU(),
            nn.Linear(hidden_layer_4, hidden_layer_5),
            nn.ReLU(),
            nn.Linear(hidden_layer_5, hidden_layer_6),
            nn.ReLU(),
            nn.Linear(hidden_layer_6, n_outputs),
        )


    def forward(self, x):
        return self.net(x)


def test_model(model_dir, episodes=100, verbose=True):
    model = torch.load(model_dir)

    agent = Wordler(device, verbose)

    agent.env = WordleEnv()

    def run_episode(env, agent, model):
        state = env.reset()
        print("secret word: ", env.secret_word)
        agent.guessed = set()
        reward = 0
        done = False
        while not done:
            action, a_val = agent.select_action(model, state, exploit=True)
            action = action.item()
            agent.guessed.add(action)
            print(env.action_words[action], round(a_val.item(), 3))
            state, R, done, info = env.step(action)
            print("guess: ", env.action_words[action])
            reward += R

        print(f"reward: {round(reward, 2)}")
        return reward

    history = [run_episode(agent.env, agent, model) for _ in range(episodes)]

    print(f"Average reward: {round(sum(history) / len(history), 3)}")

    return history


def main(
    device,
    model_loc=None,
    max_episodes=100,
    C=8,
    batch_size=64,
    verbose=False,
):
    agent = Wordler(device, verbose)
    model, fsufx = agent.solve(
        model_loc=model_loc,
        max_episodes=max_episodes,
        C=C,
        batch_size=batch_size,
    )

    agent.create_r_per_ep_fig(title_suffix=fsufx)

    return agent, fsufx


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mode = "train-test"
    if "train" in mode and "test" in mode:
        agent, fsufx = main(
            model_loc=None,
            max_episodes=50000,
            device=device,
            verbose=True,
        )
        history = test_model(
            model_dir=f"./model_{fsufx}",
            episodes=200,
        )
    elif "test" in mode:
        history = test_model(
            model_dir='./model_20220321.13.51.29',
            episodes=100,
        )
    else:
        agent, fsufx = main(
            model_loc=None,
            max_episodes=1000,
            device=device,
            verbose=True,
        )
