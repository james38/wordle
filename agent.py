from copy import deepcopy
import pickle
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
import gym

import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import einops

from wordle_rl import WordleEnv


class Wordler(object):
    def __init__(self, device, verbose=False):
        self.env = WordleEnv()
        self.env.seed(19)
        self.np_rng = np.random.default_rng(19)

        self.n_inputs = 183

        self.alpha = 0.0001
        self.epsilon = 0.9
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
        self.initialize_epsilon_decay(max_episodes)

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
            elif self.n_episodes == len(self.env.poss_solutions):
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
                    if invalid_batch_size > 0:
                        invalid_action_inds = self.np_rng.choice(
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

                # numbers chosen so that influence will exponentially decay
                #  plus a bit of linear decay to hit 0 at roughly halfway
                if teacher_model_dir is not None:
                    teacher_influence = max(0, np.exp(
                        -5 * (1 + self.n_episodes) / max_episodes
                    ) - 0.164 * (self.n_episodes / max_episodes))
                else:
                    teacher_influence = 0

                if teacher_influence > 0:
                    Q_prime = self.gamma * (
                        teacher_influence * self.select_action(
                            self.teacher_model,
                            batch[:,self.n_inputs:-3],
                            replay_batch_invalid_words,
                        )[1].cpu().detach().numpy().reshape(-1)
                        + (1 - teacher_influence) * self.select_action(
                            self.target_model,
                            batch[:,self.n_inputs:-3],
                            replay_batch_invalid_words,
                        )[1].cpu().detach().numpy().reshape(-1)
                    )
                else:
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
                    self.model(input)[:,-1,:],
                    1,
                    torch.tensor(batch[:,-3].astype(int)).view(-1,1).to(device)
                )

                if (invalid_batch_size > 0) and (state_prior[0] > 0):
                    state_input = torch.tensor(
                        state_prior,
                        device=self.device
                    ).float().view(1,-1)
                    self.model.eval()
                    state_logits = self.model(state_input)[:,-1,:]
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
        out = out[:,-1,:]

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
            np.minimum(0.9, self.replay_memory[:self.n_t,-2])
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
    """
    https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    """
    # def __init__(
    #     self, n_inputs=183,
    #     hidden_layer_1=1024,
    #     hidden_layer_2=512,
    #     hidden_layer_3=384,
    #     hidden_layer_4=256,
    #     hidden_layer_5=192,
    #     hidden_layer_6=128,
    #     hidden_layer_7=96,
    #     hidden_layer_8=64,
    #     n_outputs=12947,
    # ):
    def __init__(
        self,
        ntokens=14, # size of vocabulary
        n_outputs=12947,
        d_model=128, # embedding dimension
        nhead=8, # number of heads in nn.MultiheadAttention
        d_hid=512, # feedforward dimension in nn.TransformerEncoder
        nlayers=2, # nn.TransformerEncoderLayer in nn.TransformerEncoder
        dropout=0.1, # dropout probability
    ):
        super(Model, self).__init__()

        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = TransformerEncoderLayer(
            d_model, nhead, d_hid, dropout, batch_first=True
        )
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.encoder = nn.Embedding(ntokens, d_model)
        self.d_model = d_model
        self.decoder = nn.Linear(d_model, n_outputs)

        self.init_weights()

        # self.net = nn.Sequential(
        #     nn.Linear(n_inputs, hidden_layer_1),
        #     nn.BatchNorm1d(hidden_layer_1),
        #     nn.ReLU(),
        #     nn.Linear(hidden_layer_1, hidden_layer_2),
        #     nn.BatchNorm1d(hidden_layer_2),
        #     nn.ReLU(),
        #     nn.Linear(hidden_layer_2, hidden_layer_3),
        #     nn.BatchNorm1d(hidden_layer_3),
        #     nn.ReLU(),
        #     nn.Linear(hidden_layer_3, hidden_layer_4),
        #     nn.BatchNorm1d(hidden_layer_4),
        #     nn.ReLU(),
        #     nn.Linear(hidden_layer_4, hidden_layer_5),
        #     nn.BatchNorm1d(hidden_layer_5),
        #     nn.ReLU(),
        #     nn.Linear(hidden_layer_5, hidden_layer_6),
        #     nn.BatchNorm1d(hidden_layer_6),
        #     nn.ReLU(),
        #     nn.Linear(hidden_layer_6, hidden_layer_7),
        #     nn.BatchNorm1d(hidden_layer_7),
        #     nn.ReLU(),
        #     nn.Linear(hidden_layer_7, hidden_layer_8),
        #     nn.BatchNorm1d(hidden_layer_8),
        #     nn.ReLU(),
        #     nn.Linear(hidden_layer_8, n_outputs),
        # )


    def init_weights(self):
        init_range = 0.1
        self.encoder.weight.data.uniform_(-init_range, init_range)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-init_range, init_range)
        return None


    def forward(self, x):
        # return self.net(x)
        x = x.long()
        x = self.encoder(x) * np.sqrt(self.d_model)
        x = self.pos_encoder(x)
        output = self.transformer_encoder(x)
        return self.decoder(output)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_seq_len=183):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_seq_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_seq_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)


    def forward(self, x):
        x = torch.add(x, self.pe[:x.size(0)])
        return self.dropout(x)


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

    print(f"Fail rate: {round(history.count(0) / len(history), 3)}")
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
            model_dir='./model_20220408.16.21.45',
            teacher_model_dir=None,
            max_episodes=69270,
            device=device,
            verbose=True,
        )
        history = test_model(
            model_dir=f"./model_{fsufx}",
            episodes="full",
        )
    elif "test" in mode:
        history = test_model(
            model_dir='./model_20220408.16.21.45',
            episodes="full",
        )
    else:
        agent, fsufx = main(
            model_dir=None,
            max_episodes=1000,
            device=device,
            verbose=True,
        )
