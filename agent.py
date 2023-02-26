from copy import deepcopy
import logging
import pickle
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
import gym

import torch
import torch.nn as nn

from wordle_rl import WordleEnv


class Wordler(object):
    def __init__(self, device, fixed_start=None, seed=19):
        self.env = WordleEnv()
        self.env.seed(seed)
        self.np_rng = np.random.default_rng(seed)

        self.fixed = fixed_start
        self.n_inputs = 183

        self.alpha = 0.001
        self.epsilon = 0.99
        self.min_epsilon = 0.02
        self.gamma = 0.9999

        self.device = device


    def solve(
        self,
        model_dir=None,
        teacher_model_dir=None,
        max_episodes=1,
        C=8,
        batch_size=64,
        max_experience=100000,
        clip_val=2,
        warmup=2,
        C_factor=2,
        T_max=23090,
    ):

        if model_dir is not None:
            self.model = torch.load(model_dir)
        else:
            self.model = ResNN(
                in_channels=4,
                out_channels=12,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
                pool_kernel_size=1,
                pool_stride=1,
            ).to(self.device).float()
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
            betas=(0.9, 0.999),
            weight_decay=1e-2,
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

            self.env.reset(self.env.poss_solutions[sol_inds[self.n_episodes]])
            self.guessed = set()
            episode_R = 0
            is_terminal = False
            while not is_terminal:
                state_prior = self.env.state.copy()
                invalid_actions = self.gen_invalid_actions(state_prior)
                self.replay_invalid_words.append(invalid_actions)

                info = {'valid': False}
                while not info['valid']:
                    if (
                        self.fixed is not None
                        and self.fixed in self.env.info["valid_words"].keys()
                    ):
                        action = self.fixed
                    else:
                        action = self.epsilon_greedy_action_selection(
                            self.model,
                            self.env.state,
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

                batch_inds = self.np_rng.choice(
                    self.n_t,
                    size=batch_size,
                )
                batch = self.replay_memory[batch_inds,:]

                # the invalid words are only used for generating Q_prime,
                #  which is only used in non-terminal resultant states
                cont_eps = np.argwhere(batch[:,-1] == 0)[:,0]
                cont_inds = batch_inds[cont_eps]
                replay_batch_invalid_words = [
                    self.replay_invalid_words[i] for i in cont_inds
                ]

                if cont_inds.shape[0] > 0:
                    Q_prime = self.gamma * self.select_action(
                        self.target_model,
                        self.replay_memory[cont_inds,self.n_inputs:-3],
                        replay_batch_invalid_words,
                    )[1].cpu().detach().numpy().reshape(-1)

                if teacher_model_dir is not None:
                    teacher_influence = self.gen_teacher_influence(max_episodes)
                    if teacher_influence > 0 and cont_inds.shape[0] > 0:
                        Q_prime = (
                            (1 - teacher_influence) * Q_prime
                            + teacher_influence * self.gamma * (
                                self.select_action(
                                    self.teacher_model,
                                    self.replay_memory[
                                        cont_inds, self.n_inputs:-3
                                    ],
                                    replay_batch_invalid_words,
                                )[1].cpu().detach().numpy().reshape(-1)
                            )
                        )

                Q_vals = batch[:,-2]
                if cont_eps.shape[0] > 0:
                    Q_vals[cont_eps] += Q_prime

                y_targets = torch.tensor(
                    Q_vals, device=self.device
                ).float().view(-1,1)

                input = self.batch_to_input(batch[:,:self.n_inputs])

                y_preds = torch.gather(
                    self.model(input),
                    1,
                    torch.tensor(batch[:,-3].astype(int)).view(-1,1).to(device)
                )

                loss = self.loss_fx(y_preds, y_targets)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            self.n_episodes += 1
            self.scheduler.step()

            logging.debug(f"guesses {self.env.n_guesses}")
            logging.debug(f"reward {round(episode_R, 2)}")

            overall_rewards.append(episode_R)
            if (self.n_episodes % 100 == 0):
                logging.info(f"replay memory: {self.n_t}")
                logging.info(f"episode {self.n_episodes}")
                mean_reward = np.mean(overall_rewards[-100:]).round(3)
                logging.info(f"last 100 episode rewards {mean_reward}")

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
        logging.info(f"model saved {fsufx}")

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


    def gen_invalid_actions(self, state_prior):
        if state_prior[0] > 0:
            invalid_actions = (
                self.env.action_words.keys()
                - self.env.info['valid_words'].keys()
            )
            # with all greens and yellows, last guess is not invalid,
            #  so inserting guesses prevents agent being stuck
            invalid_actions.update(self.guessed)
        else:
            invalid_actions = set()

        return invalid_actions


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
            self.n_t = inds.shape[0]
            self.replay_memory[:self.n_t,:] = memory_continued
            self.replay_invalid_words = [
                self.replay_invalid_words[i] for i in inds
            ]
        else:
            self.n_t = 0
            self.replay_invalid_words = list()

        return None


    def epsilon_greedy_action_selection(self, model, input, invalid_actions):
        if self.np_rng.random() > self.epsilon:
            action, val = self.select_action(model, input, invalid_actions)
            if self.env.state[0] == 0:
                logging.info(f"start word: {self.env.action_words[action]}")
                logging.info(f"value: {val}")
        else:
            valid_actions = list(self.env.info['valid_words'].keys())
            action = self.np_rng.choice(valid_actions)

        return action


    def select_action(self, model, input, invalid_actions, debug=False):
        if isinstance(input, np.ndarray):
            if input.ndim > 1:
                input = self.batch_to_input(input)
            else:
                input = self.state_to_input(input)

        model.eval()
        with torch.no_grad():
            out = model(input)
        model.train()

        if debug: self.get_model_top_words(out)

        if input.size(0) > 1:
            action, val = self.get_valid_batch_action_val(out, invalid_actions)
        else:
            action, val = self.get_valid_state_action_val(out, invalid_actions)

        return action, val


    def get_valid_batch_action_val(self, out, invalid_actions):
        for j,ia in enumerate(invalid_actions):
            out[j,list(ia)] = float("-inf")
        action = torch.argmax(out, keepdims=True, dim=1)
        best_action_value = torch.gather(out, 1, action)

        return action, best_action_value


    def get_valid_state_action_val(self, out, invalid_actions):
        """
        not sure why invalid_actions can show up as
         a list containing a single element set
         when using gen_secret_word_regret() method
        """
        # if isinstance(invalid_actions, list):
        #     invalid_actions = invalid_actions[0]
        out[0,list(invalid_actions)] = float("-inf")
        action = torch.argmax(out).item()
        best_action_value = out[0,action]

        return action, best_action_value


    def batch_to_input(self, batch):
        input = np.zeros(
            (batch.shape[0],4,self.env.len_alphabet,self.env.n_letters)
        )
        input[:,0,:,:] = ((batch[:,0] - 1) / 5).reshape(input.shape[0],1,1)
        input[:,1,:,:] = batch[
            :,
            1:(self.n_inputs - 2 * 26)
        ].reshape(
            input.shape[0],input.shape[2],input.shape[3]
        )
        input[:,2,:,:] = batch[
            :,
            (self.n_inputs - 2 * 26):(self.n_inputs - 26)
        ].reshape(
            input.shape[0],input.shape[2],-1
        )
        input[:,3,:,:] = batch[
            :,
            (self.n_inputs - 26):self.n_inputs
        ].reshape(
            input.shape[0],input.shape[2],-1
        )
        return torch.tensor(input, device=self.device).float()


    def state_to_input(self, state):
        input = np.zeros((1,4,self.env.len_alphabet,self.env.n_letters))
        input[0,0,:,:] = self.input_scaling(state[0])
        input[0,1,:,:] = state[1:(self.n_inputs - 2 * 26)].reshape(
            input.shape[2],input.shape[3]
        )
        input[0,2,:,:] = state[
            (self.n_inputs - 2 * 26):(self.n_inputs - 26)
        ].reshape(
            input.shape[2],-1
        )
        input[0,3,:,:] = state[(self.n_inputs - 26):self.n_inputs].reshape(
            input.shape[2],-1
        )
        return torch.tensor(input, device=self.device).float()


    def get_model_top_words(self, out, threshold=0.42):
        top_words = [
            self.env.words[torch.argmin(torch.abs(out - x)).item()]
            for x in out[out > threshold].cpu().detach().numpy()
        ]
        logging.debug(top_words)
        top_valid_words = [
            w for w in top_words
            if self.env.words.index(w) in self.env.valid_words
        ]
        logging.debug(top_valid_words)

        return None


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


    def gen_teacher_influence(self, max_episodes):
        """
        numbers chosen so that influence will exponentially decay
         plus a bit of linear decay to hit 0 at roughly halfway
        """
        return np.exp(
            -5 * (1 + self.n_episodes) / max_episodes
        ) - 0.164 * (self.n_episodes / max_episodes)


    def gen_secret_word_regret(self):
        term = self.n_t - 1 # np.argwhere(self.replay_memory[:,-1])[-1,0]
        state_turns = self.replay_memory[term,0] # range of 0-5
        regrets = self.replay_memory[int(term-state_turns):self.n_t,:]

        regrets[:,-3] = self.env.words.index(self.env.secret_word)
        for i in range(len(self.env.rewards)):
            regrets[regrets[:,0] == i,-2] = self.env.rewards[i]
        regrets[:,-1] = 1

        # the state_prime will not be accurate, but should be no issue
        eps_len = regrets.shape[0]
        self.replay_memory[self.n_t:(self.n_t+eps_len),:] = regrets
        self.n_t += eps_len
        self.replay_invalid_words += self.replay_invalid_words[-eps_len:]

        return None


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


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        bias,
    ):
        super(ConvBlock, self).__init__()

        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        )
        self.bn = nn.BatchNorm2d(out_channels)


    def forward(self, x):
        a = self.bn(self.conv(x))
        return a


class SEBlock(nn.Module):
    """
    https://github.com/moskomule/senet.pytorch/blob/master/senet/se_module.py
    """
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.squeezed = int(channels / reduction)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, self.squeezed, bias=False),
            nn.Mish(inplace=True),
            nn.Linear(self.squeezed, channels, bias=False),
            nn.Sigmoid()
        )


    def forward(self, x):
        n, c, _, _ = x.size()
        y = self.pool(x).view(n, c)
        y = self.fc(y).view(n, c, 1, 1)
        return x * y.expand_as(x)


class ResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        bias,
    ):
        super(ResidualBlock, self).__init__()

        self.conv1 = ConvBlock(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            bias,
        )
        self.conv2 = ConvBlock(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            bias,
        )
        self.se = SEBlock(out_channels, reduction=4)
        self.mish = nn.Mish()


    def forward(self, x):
        a = self.mish(self.conv1(x))
        a = self.conv2(a)
        a = self.se(a)
        a += x
        return self.mish(a)


class ResNN(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        bias,
        pool_kernel_size,
        pool_stride
    ):
        super(ResNN, self).__init__()
        self.conv_block = ConvBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        )
        self.res_block1 = ResidualBlock(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        )
        self.res_block2 = ResidualBlock(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        )
        self.res_block3 = ResidualBlock(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        )
        self.pool1 = nn.MaxPool2d(
            kernel_size=pool_kernel_size,
            stride=pool_stride,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout2d(1e-1)
        self.linear1 = nn.Sequential(
            nn.Linear(1560, 12947),
        )
        self.mish = nn.Mish()

    def forward(self, x):
        a = self.dropout(self.pool1(self.mish(self.conv_block(x))))
        a = self.res_block1(a)
        a = self.res_block2(a)
        a = self.res_block3(a)
        a = self.bn(a)
        z = a.view(a.size(0), -1)
        y = self.linear1(z)
        return y


def run_episode(env, agent, model, fixed_start=None, secret_word=None):
    state = env.reset(secret_word)
    invalid_actions = set()
    reward = 0
    done = False
    while not done:
        action, val = agent.select_action(model, state, invalid_actions)
        if fixed_start in env.info["valid_words"].keys():
            action = fixed_start
        logging.info(f"guess: {env.action_words[action]} {round(val.item(),3)}")
        state, R, done, info = env.step(action)
        invalid_actions = (
            env.action_words.keys() - env.info['valid_words'].keys()
        )
        invalid_actions.add(action)
        reward += R

    logging.info(f"secret word: {env.secret_word} reward: {round(reward, 2)}")
    return reward


def test_model(model_dir, episodes=100, fixed_start=None):
    model = torch.load(model_dir)

    agent = Wordler(device, fixed_start)

    agent.env = WordleEnv()

    if isinstance(episodes, int):
        history = [
            run_episode(agent.env, agent, model, fixed_start)
            for _ in range(episodes)
        ]
    else:
        history = [
            run_episode(agent.env, agent, model, fixed_start, secret_word)
            for secret_word in agent.env.poss_solutions
        ]

    logging.info(f"Fail rate: {round(history.count(0) / len(history), 3)}")
    logging.info(f"Average reward: {round(sum(history) / len(history), 3)}")

    return history


def main(
    max_episodes,
    device,
    fixed_start=None,
    model_dir=None,
    teacher_model_dir=None,
    C=8,
    batch_size=64,
):
    agent = Wordler(device, fixed_start)
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
    logging.basicConfig(level=logging.INFO)
    mode = "train-test"
    if "train" in mode and "test" in mode:
        agent, fsufx = main(
            max_episodes=577250,
            model_dir='models/model_20230223.22.15.43',
            teacher_model_dir=None,
            device=device,
        )
        history = test_model(
            model_dir=f"./models/model_{fsufx}",
            episodes="full",
        )
    elif "test" in mode:
        history = test_model(
            # model_dir='./models/model_20220403.07.13.49',
            model_dir='./models/model_20221118.08.02.40',
            fixed_start=2709, # DEALT
            episodes="full",
        )
    else:
        agent, fsufx = main(
            max_episodes=4618,
            model_dir=None,
            device=device,
        )
