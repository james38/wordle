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
    def __init__(self, device, fixed_start=None, verbose=False, seed=19):
        self.env = WordleEnv()
        self.env.seed(seed)
        self.np_rng = np.random.default_rng(seed)

        self.fixed_start = fixed_start
        self.n_inputs = 183

        self.alpha = 0.0001
        self.epsilon = 0.99
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
            self.model = ResNN(
                in_channels=4,
                out_channels=8,
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

                input = self.batch_to_input(batch[:,:self.n_inputs])

                y_preds = torch.gather(
                    self.model(input),
                    1,
                    torch.tensor(batch[:,-3].astype(int)).view(-1,1).to(device)
                )

                if (invalid_batch_size > 0) and (state_prior[0] > 0):
                    state_input = self.state_to_input(state_prior)
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
            self.scheduler.step()

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
        if isinstance(input, np.ndarray):
            if input.ndim > 1:
                input = self.batch_to_input(input)
            else:
                input = self.state_to_input(input)

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
            nn.ReLU(inplace=True),
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


    def forward(self, x):
        a = torch.relu(self.conv1(x))
        a = self.conv2(a)
        a = self.se(a)
        a += x
        return torch.relu(a)


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
        self.pool1 = nn.MaxPool2d(
            kernel_size=pool_kernel_size,
            stride=pool_stride,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout2d(1e-1)
        self.linear1 = nn.Linear(1040, 12947)


    def forward(self, x):
        a = self.dropout(self.pool1(torch.relu(self.conv_block(x))))
        a = self.res_block1(a)
        a = self.res_block2(a)
        a = self.bn(a)
        z = a.view(a.size(0), -1)
        y = self.linear1(z)
        return y


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
            # model_dir='./models/model_20221118.08.02.40',
            model_dir=None,
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
            max_episodes=4618,
            device=device,
            verbose=True,
        )
