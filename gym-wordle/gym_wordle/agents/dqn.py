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
