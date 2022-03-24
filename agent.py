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
        self.min_epsilon = 0.02
        self.gamma = 0.9999

        self.device = device
        self.verbose = verbose


    def solve(
        self,
        model_dir=None,
        teacher_model_dir=None,
        max_episodes=1,
        C=8,
        batch_size=64,
        invalid_batch_size=128,
        max_experience=1000000,
        clip_val=2,
        warmup=2,
        C_factor=4,
        settle_at=0.01,
        settle_rate=0.3,
        lr_drop_at=0.9,
        lr_drop_rate=0.1,
    ):

        if model_dir is not None:
            self.model = torch.load(model_dir)
        else:
            self.model = Model().to(self.device).float()
        if teacher_model_dir is not None:
            self.teacher_model = torch.load(teacher_model_dir)
        for p in self.model.parameters():
            p.register_hook(lambda x: torch.clamp(x, -clip_val, clip_val))
        self.target_model = deepcopy(self.model)
        self.C = C
        self.loss_fx = nn.MSELoss()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.alpha,
        )

        self.initialize_replay_memory(n=max_experience)
        # set epsilon decay dynamically based on max episodes
        #  such that it would reach roughly half of the min by
        #  the end of the max episodes (with estimate of guess/episode)
        self.epsilon_decay = np.exp(
            np.log((self.min_epsilon / 2) / self.epsilon)
            / (max_episodes * 5)
        )

        self.n_episodes = 0
        n_sols = len(self.env.poss_solutions)
        overall_rewards = list()
        is_solved = False
        while not is_solved:
            if self.n_episodes == int(lr_drop_at * max_episodes):
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] *= lr_drop_rate
                self.min_epsilon *= lr_drop_rate
                C *= C_factor
            elif self.n_episodes == int(settle_at * max_episodes):
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] *= settle_rate
                C *= C_factor
            if self.n_episodes < (warmup * self.C):
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = (
                        self.alpha * (1 + self.n_episodes) / (warmup * self.C)
                    )
                C = max(2, int((1 + self.n_episodes) / warmup))

            self.state = self.env.reset(
                self.env.poss_solutions[self.n_episodes % n_sols]
            )
            self.guessed = set()
            episode_R = 0
            is_terminal = False
            while not is_terminal:
                state_prior = self.state.copy()
                if state_prior[0] > 0:
                    valid_actions = self.env.info['valid_words'].keys()
                    invalid_actions = {
                        w for w in self.env.action_words.keys()
                        if w not in valid_actions
                    }
                    invalid_actions.update(self.guessed)
                    invalid_action_inds = np.random.choice(
                        list(invalid_actions), size=invalid_batch_size
                    )
                else:
                    invalid_actions = list()
                self.replay_invalid_words.append(invalid_actions)

                info = {'valid': False}
                while not info['valid']:
                    action = self.epsilon_greedy_action_selection(
                        self.model,
                        self.state,
                        invalid_actions,
                    )
                    s_prime, R, is_terminal, info = self.env.step(action)

                self.guessed.add(action)
                self.epsilon *= self.epsilon_decay
                self.epsilon = max(self.min_epsilon, self.epsilon)

                episode_R += R

                self.replay_memory[self.n_t,:self.n_inputs] = state_prior
                self.replay_memory[self.n_t,self.n_inputs:-3] = s_prime
                self.replay_memory[self.n_t,-3] = action
                self.replay_memory[self.n_t,-2] = R
                self.replay_memory[self.n_t,-1] = 1 if is_terminal else 0
                self.n_t += 1

                probs = (
                    np.minimum(0.8, self.replay_memory[:self.n_t,-2])
                    + (
                        np.log(2 + self.n_t)
                        + self.replay_memory[:self.n_t,0]
                    ) / 100
                )
                batch_inds = np.random.choice(
                    self.n_t,
                    size=batch_size,
                    p=(probs / np.sum(probs)),
                )
                batch = self.replay_memory[batch_inds,:]
                replay_batch_invalid_words = [
                    self.replay_invalid_words[i] for i in batch_inds
                ]

                y_targets = torch.zeros(
                    (batch_size + invalid_batch_size, 1),
                    device=self.device
                )

                # numbers chosen so that influence will exponentially decay
                #  plus a bit of linear decay to hit 0 at roughly halfway
                if teacher_model_dir is not None:
                    teacher_influence = max(0, np.exp(
                        -5 * (1 + self.n_episodes) / max_episodes
                    ) - 0.164 * (self.n_episodes / max_episodes))

                Q_prime = self.gamma * self.select_action(
                    self.target_model,
                    batch[:,self.n_inputs:-3],
                    replay_batch_invalid_words,
                )[1].cpu().detach().numpy().reshape(-1)

                Q_vals = np.where(
                    batch[:,-1] == 1,
                    batch[:,-2],
                    batch[:,-2] + Q_prime
                )

                y_targets[:batch_size, 0] = torch.tensor(
                    Q_vals, device=self.device
                ).float()

                input = torch.tensor(
                    batch[:,:self.n_inputs], device=self.device
                ).float()

                y_preds = torch.gather(
                    self.model(self.input_scaling(input)),
                    1,
                    torch.tensor(batch[:,-3].astype(int)).view(-1,1).to(device)
                )

                if state_prior[0] > 0:
                    state_input = torch.tensor(
                        state_prior,
                        device=self.device
                    ).float()
                    state_logits = self.model(self.input_scaling(state_input))
                    y_preds_invalid = state_logits[invalid_action_inds]

                    y_preds = torch.cat((y_preds, y_preds_invalid.view(-1,1)))
                else:
                    y_targets = y_targets[:batch_size,:]

                loss = self.loss_fx(y_preds, y_targets)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                # for some reason, the environment's step method is already
                #  setting the agent's self.state to the output state of step
                # self.state = s_prime

            self.n_episodes += 1

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
        self.replay_invalid_words = list()
        return None


    def epsilon_greedy_action_selection(self, model, input, invalid_actions):
        if np.random.random() > self.epsilon:
            action, _ = self.select_action(model, input, invalid_actions)
        else:
            valid_actions = list(self.env.info['valid_words'].keys())
            action = np.random.choice(valid_actions)

        return action


    def select_action(self, model, input, invalid_actions, debug=False):
        if type(input) is np.ndarray:
            input = torch.tensor(input.astype(np.float32)).to(self.device)

        out = model(self.input_scaling(input))

        if debug:
            top_words = [
                self.env.words[torch.argmin(torch.abs(out - x)).item()]
                for x in out[out > 0.42].cpu().detach().numpy()
            ]
            print(top_words)
            top_valid_words = [
                w for w in top_words
                if self.env.words.index(w) in self.env.valid_words
            ]
            print(top_valid_words)

        if input.dim() > 1:
            for j,ia in enumerate(invalid_actions):
                out[j,list(ia)] = float("-inf")
            action = torch.argmax(out, keepdims=True, dim=1)
            best_action_value = torch.gather(out, 1, action)
        else:
            out[list(invalid_actions)] = float("-inf")
            action = torch.argmax(out).item()
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
        hidden_layer_1=256,
        hidden_layer_2=192,
        hidden_layer_3=128,
        hidden_layer_4=96,
        hidden_layer_5=64,
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
            nn.Linear(hidden_layer_5, n_outputs),
        )


    def forward(self, x):
        return self.net(x)


def test_model(model_dir, episodes=100, verbose=True):
    model = torch.load(model_dir)

    agent = Wordler(device, verbose)

    agent.env = WordleEnv()

    def run_episode(env, agent, model, secret_word=None):
        state = env.reset(secret_word)
        invalid_actions = set()
        reward = 0
        done = False
        while not done:
            action, a_val = agent.select_action(model, state, invalid_actions)
            print("guess: ", env.action_words[action], round(a_val.item(), 3))
            state, R, done, info = env.step(action)
            invalid_actions = {
                w for w in env.action_words.keys()
                if w not in env.info["valid_words"].keys()
            }
            invalid_actions.add(action)
            reward += R

        print("secret word: ", env.secret_word, f"reward: {round(reward, 2)}")
        return reward

    if type(episodes) is int:
        history = [
            run_episode(agent.env, agent, model) for _ in range(episodes)
        ]
    else:
        history = [
            run_episode(agent.env, agent, model, secret_word)
            for secret_word in agent.env.poss_solutions
        ]

    print(f"Average reward: {round(sum(history) / len(history), 3)}")

    return history


def main(
    device,
    model_dir=None,
    teacher_model_dir=None,
    max_episodes=100,
    C=8,
    batch_size=64,
    verbose=False,
):
    agent = Wordler(device, verbose)
    model, fsufx = agent.solve(
        model_dir=model_dir,
        teacher_model_dir=teacher_model_dir,
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
            model_dir=None,
            teacher_model_dir=None,
            max_episodes=129304,
            device=device,
            verbose=True,
        )
        history = test_model(
            model_dir=f"./model_{fsufx}",
            episodes="full",
        )
    elif "test" in mode:
        history = test_model(
            model_dir='./model_20220323.08.16.00',
            episodes="full",
        )
    else:
        agent, fsufx = main(
            model_dir=None,
            max_episodes=1000,
            device=device,
            verbose=True,
        )
