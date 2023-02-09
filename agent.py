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
    def __init__(self, device, fixed_start=None, verbose=False):
        self.env = WordleEnv()
        self.env.seed(19)
        self.np_rng = np.random.default_rng(19)

        self.fixed_start = fixed_start
        self.n_inputs = 183

        self.alpha = 0.0001
        self.epsilon = 0.4
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
        invalid_batch_size=0,
        max_experience=100000,
        clip_val=2,
        warmup=2,
        C_factor=2,
        T_max=23090,
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
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.alpha,
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=T_max,
        )

        self.initialize_replay_memory(n=max_experience)
        self.initialize_epsilon_decay(max_episodes)

        self.n_episodes = 0
        n_sols = len(self.env.poss_solutions)
        sol_inds = self.shuffle_solutions(n_sols, max_episodes)
        overall_rewards = list()
        is_solved = False
        while not is_solved:
            if self.n_episodes <= n_sols:
                C = self.modulate_C(C, warmup, C_factor, n_sols)

            self.state = self.env.reset(
                self.env.poss_solutions[sol_inds[self.n_episodes]]
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
                    if invalid_batch_size > 0:
                        invalid_action_inds = self.np_rng.choice(
                            list(invalid_actions), size=invalid_batch_size
                        )
                else:
                    invalid_actions = list()

                self.replay_invalid_words.append(invalid_actions)

                info = {'valid': False}
                while not info['valid']:
                    if self.fixed_start in self.env.info["valid_words"].keys():
                        action = self.fixed_start
                    else:
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

                if state_prior[0] > 0:
                    n_batch_inds = batch_size
                else:
                    n_batch_inds = batch_size + invalid_batch_size

                batch_inds = self.np_rng.choice(
                    self.n_t,
                    size=n_batch_inds,
                    p=self.gen_replay_probs(),
                )
                batch = self.replay_memory[batch_inds,:]
                replay_batch_invalid_words = [
                    self.replay_invalid_words[i] for i in batch_inds
                ]

                Q_prime = self.gamma * self.select_action(
                    self.target_model,
                    batch[:,self.n_inputs:-3],
                    replay_batch_invalid_words,
                )[1].cpu().detach().numpy().reshape(-1)

                if teacher_model_dir is not None:
                    # numbers chosen so that influence will exponentially decay
                    #  plus a bit of linear decay to hit 0 at roughly halfway
                    teacher_influence = np.exp(
                        -5 * (1 + self.n_episodes) / max_episodes
                    ) - 0.164 * (self.n_episodes / max_episodes)
                    if teacher_influence > 0:
                        Q_prime = (
                            (1 - teacher_influence) * Q_prime
                            + teacher_influence * self.gamma * (
                                self.select_action(
                                    self.teacher_model,
                                    batch[:,self.n_inputs:-3],
                                    replay_batch_invalid_words,
                                )[1].cpu().detach().numpy().reshape(-1)
                            )
                        )

                Q_vals = np.where(
                    batch[:,-1] == 1,
                    batch[:,-2],
                    batch[:,-2] + Q_prime
                )

                if state_prior[0] > 0:
                    y_targets = torch.zeros(
                        (batch_size + invalid_batch_size, 1),
                        device=self.device
                    )
                    y_targets[:batch_size,0] = torch.tensor(
                        Q_vals, device=self.device
                    ).float()
                else:
                    y_targets = torch.tensor(
                        Q_vals, device=self.device
                    ).float().view(-1,1)

                input = torch.tensor(
                    batch[:,:self.n_inputs], device=self.device
                ).float()

                y_preds = torch.gather(
                    self.model(input),
                    1,
                    torch.tensor(batch[:,-3].astype(int)).view(-1,1).to(device)
                )

                if (invalid_batch_size > 0) and (state_prior[0] > 0):
                    state_input = torch.tensor(
                        state_prior,
                        device=self.device
                    ).float().view(1,-1)
                    self.model.eval()
                    state_logits = self.model(state_input)
                    self.model.train()
                    y_preds_invalid = torch.gather(
                        state_logits,
                        1,
                        torch.tensor(
                            invalid_action_inds.astype(int)
                        ).view(1,-1).to(device)
                    ).view(-1,1)
                    y_preds = torch.cat((y_preds, y_preds_invalid.view(-1,1)))
                else:
                    pass

                loss = self.loss_fx(y_preds, y_targets)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                # for some reason, the environment's step method is already
                #  setting the agent's self.state to the output state of step
                # self.state = s_prime

            self.n_episodes += 1
            scheduler.step()

            if self.verbose:
                print("guesses", self.env.n_guesses)
                print("reward", round(episode_R, 2))

            overall_rewards.append(episode_R)
            if self.verbose and (self.n_episodes % 10 == 0):
                print("replay memory: ", self.n_t)
                print(
                    f"episode {self.n_episodes}, last 100 episode mean reward",
                    np.mean(overall_rewards[-100:]).round(3)
                )

            if (self.n_episodes % C) == 0:
                self.target_model = deepcopy(self.model)
            if (self.n_episodes % (2 * n_sols)) == 0:
                self.initialize_replay_memory(n=max_experience, continued=True)
            if self.n_episodes == int(max_episodes / 2):
                fsufx = dt.datetime.now().strftime("%Y%m%d.%H.%M.%S")
                torch.save(self.model, f'./models/model_halftrain_{fsufx}')


            if self.q_val_converged():
                is_solved = True
            if self.n_episodes == max_episodes:
                break

        fsufx = dt.datetime.now().strftime("%Y%m%d.%H.%M.%S")
        torch.save(self.model, f'./models/model_{fsufx}')

        self.rewards = overall_rewards

        return self.model, fsufx


    def q_val_converged(self):
        """Not implemented"""
        return False


    def shuffle_solutions(self, n_sols, max_episodes):
        times_thru = int(max_episodes / n_sols)
        all_sol_inds = list()
        for _ in range(times_thru):
            sol_inds = list(range(n_sols))
            self.np_rng.shuffle(sol_inds)
            all_sol_inds.extend(sol_inds)
        return all_sol_inds


    def initialize_replay_memory(self, n, continued=False):
        if continued:
            inds = self.np_rng.choice(self.n_t, size=int(np.sqrt(self.n_t)))
            memory_continued = self.replay_memory[inds].copy()

        # create as array of maximum experience rows
        self.replay_memory = np.zeros(
            (
                n, # experiences to track in replay memory
                2 * self.n_inputs + 3 # two states, one action, reward, done
            )
        )

        if continued:
            new_n_t = inds.shape[0]
            self.replay_memory[:new_n_t,:] = memory_continued
            self.n_t = new_n_t
            self.replay_invalid_words = [
                self.replay_invalid_words[i] for i in inds
            ]
        else:
            self.n_t = 0
            self.replay_invalid_words = list()

        return None


    def epsilon_greedy_action_selection(self, model, input, invalid_actions):
        if self.np_rng.random() > self.epsilon:
            action, _ = self.select_action(model, input, invalid_actions)
        else:
            valid_actions = list(self.env.info['valid_words'].keys())
            action = self.np_rng.choice(valid_actions)

        return action


    def select_action(self, model, input, invalid_actions, debug=False):
        if type(input) is np.ndarray:
            input = torch.tensor(input.astype(np.float32)).to(self.device)
            if input.dim() == 1:
                input = input.view(1,-1)

        model.eval()
        with torch.no_grad():
            out = model(input)
        model.train()

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

        if input.size(0) > 1:
            for j,ia in enumerate(invalid_actions):
                out[j,list(ia)] = float("-inf")
            action = torch.argmax(out, keepdims=True, dim=1)
            best_action_value = torch.gather(out, 1, action)
        else:
            out[0,list(invalid_actions)] = float("-inf")
            action = torch.argmax(out).item()
            best_action_value = out[0,action]

        return action, best_action_value


    def modulate_C(self, C, warmup, C_factor, n_sols):
        if self.n_episodes < (warmup * self.C):
            C = max(2, int((1 + self.n_episodes) / warmup))
        elif self.n_episodes == n_sols:
            C *= C_factor
        return C


    def input_scaling(self, x):
        return (x - 1) / 5


    def initialize_epsilon_decay(self, max_episodes):
        """
        set epsilon decay dynamically based on max episodes
         such that it would reach roughly half of the min by
         the end of the max episodes (with estimate of guess/episode)
        """
        self.epsilon_decay = np.exp(
            np.log((self.min_epsilon / 2) / self.epsilon)
            / (max_episodes * 5)
        )
        return None


    def gen_replay_probs(self):
        probs = (
            np.minimum(0.8, self.replay_memory[:self.n_t,-2])
            + (
                np.log(2 + self.n_t)
                + self.replay_memory[:self.n_t,0]
            ) / 100
        )
        return probs / np.sum(probs)


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
        split_out_path = out_path.split('.')
        out_path = (
            '.' + split_out_path[1] + title_suffix + '.' + split_out_path[-1]
        )
        plt.savefig(out_path)
        return None


class Model(nn.Module):
    def __init__(
        self, n_inputs=183,
        hidden_layer_1=1024,
        hidden_layer_2=512,
        hidden_layer_3=384,
        hidden_layer_4=256,
        hidden_layer_5=192,
        hidden_layer_6=128,
        hidden_layer_7=96,
        hidden_layer_8=64,
        n_outputs=12947,
    ):
        super(Model, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(n_inputs, hidden_layer_1),
            nn.BatchNorm1d(hidden_layer_1),
            nn.ReLU(),
            nn.Linear(hidden_layer_1, hidden_layer_2),
            nn.BatchNorm1d(hidden_layer_2),
            nn.ReLU(),
            nn.Linear(hidden_layer_2, hidden_layer_3),
            nn.BatchNorm1d(hidden_layer_3),
            nn.ReLU(),
            nn.Linear(hidden_layer_3, hidden_layer_4),
            nn.BatchNorm1d(hidden_layer_4),
            nn.ReLU(),
            nn.Linear(hidden_layer_4, hidden_layer_5),
            nn.BatchNorm1d(hidden_layer_5),
            nn.ReLU(),
            nn.Linear(hidden_layer_5, hidden_layer_6),
            nn.BatchNorm1d(hidden_layer_6),
            nn.ReLU(),
            nn.Linear(hidden_layer_6, hidden_layer_7),
            nn.BatchNorm1d(hidden_layer_7),
            nn.ReLU(),
            nn.Linear(hidden_layer_7, hidden_layer_8),
            nn.BatchNorm1d(hidden_layer_8),
            nn.ReLU(),
            nn.Linear(hidden_layer_8, n_outputs),
        )


    def forward(self, x):
        return self.net(x)


def test_model(model_dir, episodes=100, fixed_start=None, verbose=True):
    model = torch.load(model_dir)

    agent = Wordler(device, fixed_start, verbose)

    agent.env = WordleEnv()

    def run_episode(env, agent, model, secret_word=None):
        state = env.reset(secret_word)
        invalid_actions = set()
        reward = 0
        done = False
        while not done:
            action, a_val = agent.select_action(model, state, invalid_actions)
            if fixed_start in env.info["valid_words"].keys():
                action = fixed_start
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

    if isinstance(episodes, int):
        history = [
            run_episode(agent.env, agent, model) for _ in range(episodes)
        ]
    else:
        history = [
            run_episode(agent.env, agent, model, secret_word)
            for secret_word in agent.env.poss_solutions
        ]

    print(f"Fail rate: {round(history.count(0) / len(history), 3)}")
    print(f"Average reward: {round(sum(history) / len(history), 3)}")

    return history


def main(
    device,
    fixed_start=None,
    model_dir=None,
    teacher_model_dir=None,
    max_episodes=100,
    C=8,
    batch_size=64,
    verbose=False,
):
    agent = Wordler(device, fixed_start, verbose)
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
            model_dir='./models/model_20221118.08.02.40',
            teacher_model_dir=None,
            max_episodes=115450,
            device=device,
            verbose=True,
        )
        history = test_model(
            model_dir=f"./models/model_{fsufx}",
            episodes="full",
        )
    elif "test" in mode:
        history = test_model(
            model_dir='./models/model_20220403.07.13.49',
            # fixed_start=2709, # DEALT
            episodes="full",
        )
    else:
        agent, fsufx = main(
            model_dir=None,
            max_episodes=1000,
            device=device,
            verbose=True,
        )
