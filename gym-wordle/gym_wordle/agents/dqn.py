"""SE-ResNet Double DQN with prioritized replay for Wordle."""
import argparse
import datetime as dt
import logging
import os
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn

from gym_wordle.agents.common import (
    ReplayBuffer,
    invalid_from_mask,
    load_checkpoint,
    masked_argmax,
    save_checkpoint,
    word_feature_matrix,
)
from gym_wordle.envs.wordle_env import N_ALPHABET, WordleEnv

log = logging.getLogger(__name__)


# ----------------------------------------------------------------- input map

def obs_to_input(obs, env, device):
    """Flat observation(s) -> (B, 4, 26, n_letters) float tensor.

    Channel 0: scaled turn counter, constant over the grid.
    Channel 1: positional status grid (letter x position).
    Channel 2: exceeded bit per letter, broadcast over positions.
    Channel 3: max-yellows per letter, broadcast over positions.
    """
    obs = np.atleast_2d(np.asarray(obs)).astype(np.float32)
    b, n = obs.shape[0], env.n_letters
    x = np.zeros((b, 4, N_ALPHABET, n), dtype=np.float32)
    x[:, 0] = ((obs[:, 0] - 1) / 5).reshape(b, 1, 1)
    x[:, 1] = obs[:, env.POS : env.EXCEEDED].reshape(b, N_ALPHABET, n)
    x[:, 2] = obs[:, env.EXCEEDED : env.YELLOWS].reshape(b, N_ALPHABET, 1)
    x[:, 3] = obs[:, env.YELLOWS : env.obs_size].reshape(b, N_ALPHABET, 1)
    return torch.as_tensor(x, device=device)


# --------------------------------------------------------------------- model

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, 1, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return self.bn(self.conv(x))


class SEBlock(nn.Module):
    """Squeeze-and-excitation channel gate."""

    def __init__(self, channels, reduction=4):
        super().__init__()
        squeezed = max(1, channels // reduction)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, squeezed, bias=False),
            nn.Mish(inplace=True),
            nn.Linear(squeezed, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        n, c, _, _ = x.shape
        y = self.fc(self.pool(x).view(n, c)).view(n, c, 1, 1)
        return x * y


class ResidualBlock(nn.Module):
    def __init__(self, channels, kernel_size, padding):
        super().__init__()
        self.conv1 = ConvBlock(channels, channels, kernel_size, padding)
        self.conv2 = ConvBlock(channels, channels, kernel_size, padding)
        self.se = SEBlock(channels)
        self.mish = nn.Mish()

    def forward(self, x):
        a = self.mish(self.conv1(x))
        a = self.se(self.conv2(a))
        return self.mish(a + x)


class ResNN(nn.Module):
    """Conv trunk over the (letter x position) grid, then a Q head over words."""

    def __init__(
        self,
        n_actions,
        n_letters=5,
        in_channels=4,
        channels=12,
        kernel_size=3,
        head="flat",
        d_head=128,
        word_features=None,
    ):
        super().__init__()
        pad = kernel_size // 2
        self.conv_block = ConvBlock(in_channels, channels, kernel_size, pad)
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(channels, kernel_size, pad) for _ in range(3)]
        )
        self.bn = nn.BatchNorm2d(channels)
        self.dropout = nn.Dropout2d(0.1)
        self.mish = nn.Mish()
        flat_dim = channels * N_ALPHABET * n_letters
        if head == "flat":
            self.head = nn.Linear(flat_dim, n_actions)
        else:
            raise ValueError(f"unknown head {head!r}")
        self.ctor_kwargs = dict(
            n_actions=n_actions,
            n_letters=n_letters,
            in_channels=in_channels,
            channels=channels,
            kernel_size=kernel_size,
            head=head,
            d_head=d_head,
        )

    def forward(self, x):
        a = self.dropout(self.mish(self.conv_block(x)))
        a = self.bn(self.res_blocks(a))
        return self.head(a.flatten(1))


# ------------------------------------------------------------------- trainer

class DQNTrainer:
    """Double DQN with proportional prioritized replay on WordleEnv."""

    def __init__(
        self,
        env,
        device="cpu",
        fixed_start=None,
        seed=19,
        head="flat",
        channels=12,
        model_dir="models",
        alpha=3e-4,
        epsilon=0.99,
        min_epsilon=0.02,
        gamma=0.9999,
        p_alpha=0.6,
        p_beta=0.4,
    ):
        self.env = env.unwrapped
        self.device = torch.device(device)
        self.np_rng = np.random.default_rng(seed)
        self.env.reset(seed=seed)
        if fixed_start is None:
            self.fixed_start = None
        elif isinstance(fixed_start, str):
            self.fixed_start = self.env.word_to_action[fixed_start]
        else:
            self.fixed_start = int(fixed_start)
        self.head = head
        self.channels = channels
        self.model_dir = model_dir
        self.alpha = alpha
        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.gamma = gamma
        self.p_alpha = p_alpha
        self.p_beta = p_beta
        self.n_actions = int(self.env.action_space.n)
        self.model = self.target_model = self.teacher_model = None
        self.n_episodes = 0
        self.rewards = []

    # -- model -------------------------------------------------------------

    def build_model(self):
        return ResNN(
            n_actions=self.n_actions,
            n_letters=self.env.n_letters,
            channels=self.channels,
            head=self.head,
        ).to(self.device)

    def q_values(self, model, obs):
        """Q(s, .) in eval mode under no_grad. Restores the model's prior mode."""
        was_training = model.training
        model.eval()
        with torch.no_grad():
            q = model(obs_to_input(obs, self.env, self.device))
        if was_training:
            model.train()
        return q

    def sync_target(self):
        self.target_model = deepcopy(self.model).eval()

    # -- acting ------------------------------------------------------------

    def choose_action(self, obs, info):
        mask = info["action_mask"]
        if self.fixed_start is not None and mask[self.fixed_start]:
            return self.fixed_start
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
        max_exp=30000,
        clip_val=2.0,
        T_max=None,
    ):
        if checkpoint:
            self.model = load_checkpoint(ResNN, checkpoint, self.device)
        else:
            self.model = self.build_model()
        self.model.train()
        self.teacher_model = (
            load_checkpoint(ResNN, teacher_checkpoint, self.device).eval()
            if teacher_checkpoint
            else None
        )
        for p in self.model.parameters():
            p.register_hook(lambda g: torch.clamp(g, -clip_val, clip_val))
        self.sync_target()
        self.loss_fx = nn.MSELoss(reduction="none")
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.alpha, betas=(0.9, 0.999), weight_decay=1e-2
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=T_max or max_episodes
        )
        self.buffer = ReplayBuffer(
            max_exp, self.env.obs_size, alpha=self.p_alpha, rng=self.np_rng
        )
        self.batch_size = batch_size
        # Decay so epsilon reaches about half of min_epsilon by the end,
        # assuming about five guesses per episode.
        self.epsilon_decay = float(
            np.exp(np.log((self.min_epsilon / 2) / self.epsilon) / (max_episodes * 5))
        )
        self.beta = self.p_beta
        self.beta_inc = (1 - self.p_beta) / max_episodes

    def learn(self, teacher_influence=0.0):
        buf = self.buffer
        idx, is_w = buf.sample_prioritized(self.batch_size, self.beta)
        states = obs_to_input(buf.state[idx], self.env, self.device)
        actions = torch.as_tensor(buf.action[idx], device=self.device).view(-1, 1)
        y = torch.as_tensor(
            buf.reward[idx], device=self.device, dtype=torch.float32
        ).view(-1, 1)

        cont = np.flatnonzero(~buf.done[idx])
        if cont.size:
            next_obs = buf.next_state[idx[cont]]
            invalid_next = [buf.invalid_next[i] for i in idx[cont]]
            # Double DQN: the online net picks a*, the target net values it.
            a_star, _ = masked_argmax(self.q_values(self.model, next_obs), invalid_next)
            next_in = obs_to_input(next_obs, self.env, self.device)
            with torch.no_grad():
                q_next = torch.gather(self.target_model(next_in), 1, a_star)
                if teacher_influence > 0 and self.teacher_model is not None:
                    q_teacher = torch.gather(self.teacher_model(next_in), 1, a_star)
                    q_next = (1 - teacher_influence) * q_next + teacher_influence * q_teacher
            y[torch.as_tensor(cont, device=self.device)] += self.gamma * q_next

        q = torch.gather(self.model(states), 1, actions)
        buf.update_priority(idx, (y - q).detach().cpu().numpy().reshape(-1))
        weights = torch.as_tensor(is_w, device=self.device).view(-1, 1)
        loss = (self.loss_fx(q, y) * weights).mean()
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
        max_exp=30000,
        clip_val=2.0,
        warmup=2,
        C_factor=2,
        T_max=None,
    ):
        self.setup(max_episodes, checkpoint, teacher_checkpoint, batch_size, max_exp, clip_val, T_max)
        env = self.env
        n_sols = len(env.solutions)
        sol_inds = self.shuffle_solutions(n_sols, max_episodes)
        C0 = C
        history = []
        for ep in range(max_episodes):
            self.n_episodes = ep
            if ep <= n_sols:
                C = self.modulate_C(C, C0, warmup, C_factor, n_sols)
            influence = (
                self.teacher_influence(max_episodes) if self.teacher_model is not None else 0.0
            )
            obs, info = env.reset(options={"secret_word": env.solutions[sol_inds[ep]]})
            episode_reward, terminated = 0.0, False
            while not terminated:
                action = self.choose_action(obs, info)
                next_obs, reward, terminated, _, info = env.step(action)
                while not info["valid"]:
                    # choose_action only proposes masked-valid actions, so this
                    # loop is defensive; the env has already dropped `action`
                    # from the mask carried in `info`.
                    action = self.choose_action(obs, info)
                    next_obs, reward, terminated, _, info = env.step(action)
                self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
                episode_reward += reward
                # The stored mask describes next_obs: it is used only to mask
                # the bootstrap argmax over next_obs.
                self.buffer.add(
                    obs, action, reward, next_obs, terminated,
                    invalid_from_mask(info["action_mask"]),
                )
                self.learn(influence)
                obs = next_obs
            self.beta = min(1.0, self.beta + self.beta_inc)
            self.scheduler.step()
            history.append(episode_reward)
            if (ep + 1) % C == 0:
                self.sync_target()
            if (ep + 1) % 100 == 0:
                log.info(
                    "episode %d  mean reward (last 100) %.3f  epsilon %.3f  lr %.2e",
                    ep + 1, float(np.mean(history[-100:])), self.epsilon,
                    self.optimizer.param_groups[0]["lr"],
                )
            if max_episodes >= 2 and ep + 1 == max_episodes // 2:
                self.save("model_halftrain")
        path = self.save("model")
        self.rewards = history
        return self.model, path

    # -- schedules ---------------------------------------------------------

    def shuffle_solutions(self, n_sols, max_episodes):
        """Exactly max_episodes indices: whole shuffled passes, then a partial one."""
        out = []
        while len(out) < max_episodes:
            out.extend(self.np_rng.permutation(n_sols).tolist())
        return out[:max_episodes]

    def modulate_C(self, C, C0, warmup, C_factor, n_sols):
        if self.n_episodes < warmup * C0:
            return max(2, int((1 + self.n_episodes) / warmup))
        if self.n_episodes == n_sols:
            return C * C_factor
        return C

    def teacher_influence(self, max_episodes):
        """Exponential decay plus a linear term that reaches 0 near halfway."""
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


# ---------------------------------------------------------------- evaluation

def run_episode(trainer, model, secret_word=None, fixed_start=None):
    env = trainer.env
    options = {"secret_word": secret_word} if secret_word else None
    obs, info = env.reset(options=options)
    total, terminated, guesses = 0.0, False, []
    while not terminated:
        if fixed_start is not None and info["action_mask"][fixed_start]:
            action = fixed_start
        else:
            a, _ = masked_argmax(
                trainer.q_values(model, obs), invalid_from_mask(info["action_mask"])
            )
            action = int(a.item())
        obs, reward, terminated, _, info = env.step(action)
        assert info["valid"], "a masked greedy action must be valid"
        guesses.append(env.words[action])
        total += reward
    log.info("secret %s  guesses %s  reward %.3f", env.secret_word, guesses, total)
    return total


def evaluate(trainer, model, secret_words=None, episodes=100):
    if secret_words is None:
        history = [
            run_episode(trainer, model, fixed_start=trainer.fixed_start)
            for _ in range(episodes)
        ]
    else:
        history = [
            run_episode(trainer, model, w, fixed_start=trainer.fixed_start)
            for w in secret_words
        ]
    fail_rate = sum(r == 0.0 for r in history) / len(history)
    log.info("fail rate %.3f  mean reward %.3f", fail_rate, float(np.mean(history)))
    return history


# ----------------------------------------------------------------------- cli

def main(argv=None):
    parser = argparse.ArgumentParser(description="Train or evaluate the SE-ResNet DQN on Wordle")
    parser.add_argument("--episodes", type=int, default=4618)
    parser.add_argument("--head", choices=["flat"], default="flat")
    parser.add_argument("--channels", type=int, default=12)
    parser.add_argument("--fixed-start", default=None, help="opening word, e.g. dealt")
    parser.add_argument("--checkpoint", default=None, help="resume from this .pt file")
    parser.add_argument("--teacher", default=None, help="teacher .pt file for Q_prime blending")
    parser.add_argument("--eval-only", action="store_true", help="evaluate --checkpoint on every solution")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--model-dir", default="models")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    env = WordleEnv()
    trainer = DQNTrainer(
        env, args.device, fixed_start=args.fixed_start, head=args.head,
        channels=args.channels, model_dir=args.model_dir,
    )
    if args.eval_only:
        model = load_checkpoint(ResNN, args.checkpoint, trainer.device)
        evaluate(trainer, model, secret_words=env.solutions)
        return
    model, _ = trainer.solve(
        args.episodes, checkpoint=args.checkpoint,
        teacher_checkpoint=args.teacher, batch_size=args.batch_size,
    )
    evaluate(trainer, model, secret_words=env.solutions)


if __name__ == "__main__":
    main()
