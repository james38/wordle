"""Transformer DQN that reads the flat observation as a token sequence."""
import argparse
import datetime as dt
import logging
import math
import os
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer

from gym_wordle.agents.common import (
    ReplayBuffer,
    invalid_from_mask,
    load_checkpoint,
    masked_argmax,
    save_checkpoint,
)
from gym_wordle.envs.wordle_env import WordleEnv

log = logging.getLogger(__name__)


class PositionalEncoding(nn.Module):
    """Sinusoidal encoding indexed by sequence position, batch-first."""

    def __init__(self, d_model, dropout, max_seq_len=183):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_seq_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x: (batch, seq, d_model). Slice along the sequence axis, not the batch.
        return self.dropout(x + self.pe[:, : x.size(1)])


class TransformerQNet(nn.Module):
    """Embed each observation slot as a token; read Q-values from the last token."""

    def __init__(
        self,
        n_actions,
        n_tokens=14,
        d_model=128,
        nhead=8,
        d_hid=512,
        nlayers=2,
        dropout=0.1,
        max_seq_len=183,
    ):
        super().__init__()
        self.embedding = nn.Embedding(n_tokens, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_seq_len)
        layer = TransformerEncoderLayer(d_model, nhead, d_hid, dropout, batch_first=True)
        self.encoder = TransformerEncoder(layer, nlayers)
        self.decoder = nn.Linear(d_model, n_actions)
        self.d_model = d_model
        self.ctor_kwargs = dict(
            n_actions=n_actions, n_tokens=n_tokens, d_model=d_model, nhead=nhead,
            d_hid=d_hid, nlayers=nlayers, dropout=dropout, max_seq_len=max_seq_len,
        )
        self.init_weights()

    def init_weights(self):
        r = 0.1
        self.embedding.weight.data.uniform_(-r, r)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-r, r)

    def forward(self, x):
        x = self.embedding(x.long()) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)
        return self.decoder(self.encoder(x)[:, -1, :])


class TransformerTrainer:
    """Plain DQN (target-net max) with reward-weighted replay sampling."""

    def __init__(
        self,
        env,
        device="cpu",
        seed=19,
        model_dir="models",
        alpha=1e-4,
        epsilon=0.9,
        min_epsilon=0.02,
        gamma=0.9999,
        d_model=128,
        nhead=8,
        d_hid=512,
        nlayers=2,
    ):
        self.env = env.unwrapped
        self.device = torch.device(device)
        self.np_rng = np.random.default_rng(seed)
        self.env.reset(seed=seed)
        self.model_dir = model_dir
        self.alpha = alpha
        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.gamma = gamma
        self.arch = dict(d_model=d_model, nhead=nhead, d_hid=d_hid, nlayers=nlayers)
        self.n_actions = int(self.env.action_space.n)
        self.model = self.target_model = self.teacher_model = None
        self.n_episodes = 0
        self.rewards = []

    # -- model -------------------------------------------------------------

    def build_model(self):
        return TransformerQNet(
            n_actions=self.n_actions, max_seq_len=self.env.obs_size, **self.arch
        ).to(self.device)

    def to_input(self, obs):
        return torch.as_tensor(
            np.atleast_2d(np.asarray(obs)).astype(np.int64), device=self.device
        )

    def q_values(self, model, obs):
        was_training = model.training
        model.eval()
        with torch.no_grad():
            q = model(self.to_input(obs))
        if was_training:
            model.train()
        return q

    def sync_target(self):
        self.target_model = deepcopy(self.model).eval()

    def set_lr(self, lr):
        for group in self.optimizer.param_groups:
            group["lr"] = lr

    def scale_lr(self, factor):
        for group in self.optimizer.param_groups:
            group["lr"] *= factor

    # -- acting ------------------------------------------------------------

    def choose_action(self, obs, info):
        mask = info["action_mask"]
        if self.np_rng.random() > self.epsilon:
            a, _ = masked_argmax(self.q_values(self.model, obs), invalid_from_mask(mask))
            return int(a.item())
        return int(self.np_rng.choice(np.flatnonzero(mask)))

    # -- training ----------------------------------------------------------

    def setup(
        self,
        max_episodes,
        checkpoint=None,
        teacher_checkpoint=None,
        batch_size=64,
        max_experience=100000,
        clip_val=2.0,
    ):
        if checkpoint:
            self.model = load_checkpoint(TransformerQNet, checkpoint, self.device)
        else:
            self.model = self.build_model()
        self.model.train()
        self.teacher_model = (
            load_checkpoint(TransformerQNet, teacher_checkpoint, self.device).eval()
            if teacher_checkpoint
            else None
        )
        for p in self.model.parameters():
            p.register_hook(lambda g: torch.clamp(g, -clip_val, clip_val))
        self.sync_target()
        self.loss_fx = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.alpha)
        self.buffer = ReplayBuffer(max_experience, self.env.obs_size, rng=self.np_rng)
        self.batch_size = batch_size
        self.epsilon_decay = float(
            np.exp(np.log((self.min_epsilon / 2) / self.epsilon) / (max_episodes * 5))
        )

    def replay_probs(self):
        """Favor rewarded and later-turn transitions, as the original did."""
        n = len(self.buffer)
        return np.minimum(0.9, self.buffer.reward[:n]) + (
            np.log(2 + n) + self.buffer.state[:n, 0]
        ) / 100

    def learn(self, teacher_influence=0.0):
        buf = self.buffer
        idx = buf.sample_with_probs(self.batch_size, self.replay_probs())
        states = self.to_input(buf.state[idx])
        actions = torch.as_tensor(buf.action[idx], device=self.device).view(-1, 1)
        rewards = torch.as_tensor(
            buf.reward[idx], device=self.device, dtype=torch.float32
        ).view(-1, 1)
        done = torch.as_tensor(buf.done[idx], device=self.device).view(-1, 1)
        next_in = self.to_input(buf.next_state[idx])
        invalid_next = [buf.invalid_next[i] for i in idx]
        with torch.no_grad():
            _, q_next = masked_argmax(self.target_model(next_in), invalid_next)
            if teacher_influence > 0 and self.teacher_model is not None:
                _, q_teacher = masked_argmax(self.teacher_model(next_in), invalid_next)
                q_next = (1 - teacher_influence) * q_next + teacher_influence * q_teacher
        # A terminal next-state may have every action masked (q_next = -inf);
        # torch.where selects the plain reward for those rows.
        y = torch.where(done, rewards, rewards + self.gamma * q_next)
        q = torch.gather(self.model(states), 1, actions)
        loss = self.loss_fx(q, y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return float(loss.item())

    def solve(
        self,
        max_episodes,
        checkpoint=None,
        teacher_checkpoint=None,
        C=8,
        batch_size=64,
        max_experience=100000,
        clip_val=2.0,
        warmup=2,
        C_factor=4,
        settle_rate=0.3,
        lr_drop_at=0.9,
        lr_drop_rate=0.1,
    ):
        self.setup(max_episodes, checkpoint, teacher_checkpoint, batch_size, max_experience, clip_val)
        env = self.env
        n_sols = len(env.solutions)
        C0 = C
        history = []
        for ep in range(max_episodes):
            self.n_episodes = ep
            if ep > 0 and ep == int(lr_drop_at * max_episodes):
                self.scale_lr(lr_drop_rate)
                self.min_epsilon *= lr_drop_rate
                C *= C_factor
            elif ep == n_sols:
                self.scale_lr(settle_rate)
                C *= C_factor
            if ep < warmup * C0:
                self.set_lr(self.alpha * (1 + ep) / (warmup * C0))
                C = max(2, int((1 + ep) / warmup))
            influence = (
                self.teacher_influence(max_episodes) if self.teacher_model is not None else 0.0
            )
            obs, info = env.reset(options={"secret_word": env.solutions[ep % n_sols]})
            episode_reward, terminated = 0.0, False
            while not terminated:
                action = self.choose_action(obs, info)
                next_obs, reward, terminated, _, info = env.step(action)
                while not info["valid"]:
                    action = self.choose_action(obs, info)
                    next_obs, reward, terminated, _, info = env.step(action)
                self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
                episode_reward += reward
                self.buffer.add(
                    obs, action, reward, next_obs, terminated,
                    invalid_from_mask(info["action_mask"]),
                )
                self.learn(influence)
                obs = next_obs
            history.append(episode_reward)
            if (ep + 1) % C == 0:
                self.sync_target()
            if (ep + 1) % (2 * n_sols) == 0:
                self.buffer.keep_random_subset(int(math.sqrt(len(self.buffer))))
            if (ep + 1) % 100 == 0:
                log.info(
                    "episode %d  mean reward (last 100) %.3f  epsilon %.3f",
                    ep + 1, float(np.mean(history[-100:])), self.epsilon,
                )
            if max_episodes >= 2 and ep + 1 == max_episodes // 2:
                self.save("transformer_halftrain")
        path = self.save("transformer")
        self.rewards = history
        return self.model, path

    def teacher_influence(self, max_episodes):
        return max(
            0.0,
            float(
                np.exp(-5 * (1 + self.n_episodes) / max_episodes)
                - 0.164 * (self.n_episodes / max_episodes)
            ),
        )

    def save(self, prefix):
        stamp = dt.datetime.now().strftime("%Y%m%d.%H.%M.%S")
        path = os.path.join(self.model_dir, f"{prefix}_{stamp}.pt")
        save_checkpoint(self.model, path)
        log.info("saved %s", path)
        return path


def run_episode(trainer, model, secret_word=None):
    env = trainer.env
    options = {"secret_word": secret_word} if secret_word else None
    obs, info = env.reset(options=options)
    total, terminated, guesses = 0.0, False, []
    while not terminated:
        a, _ = masked_argmax(trainer.q_values(model, obs), invalid_from_mask(info["action_mask"]))
        action = int(a.item())
        obs, reward, terminated, _, info = env.step(action)
        assert info["valid"]
        guesses.append(env.words[action])
        total += reward
    log.info("secret %s  guesses %s  reward %.3f", env.secret_word, guesses, total)
    return total


def main(argv=None):
    parser = argparse.ArgumentParser(description="Train or evaluate the transformer DQN on Wordle")
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--teacher", default=None)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--model-dir", default="models")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args(argv)
    if args.eval_only and not args.checkpoint:
        parser.error("--eval-only requires --checkpoint")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    env = WordleEnv()
    trainer = TransformerTrainer(env, args.device, model_dir=args.model_dir)
    if args.eval_only:
        model = load_checkpoint(TransformerQNet, args.checkpoint, trainer.device)
    else:
        model, _ = trainer.solve(
            args.episodes, checkpoint=args.checkpoint,
            teacher_checkpoint=args.teacher, batch_size=args.batch_size,
        )
    history = [run_episode(trainer, model, w) for w in env.solutions]
    log.info(
        "fail rate %.3f  mean reward %.3f",
        sum(r == 0.0 for r in history) / len(history), float(np.mean(history)),
    )


if __name__ == "__main__":
    main()
