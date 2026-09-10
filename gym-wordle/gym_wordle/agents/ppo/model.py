"""Transformer policy over raw feedback tokens with a factored lexical head.

The policy sees letters, colours and slot positions of its own guesses, and
a fixed lexical feature vector per valid word. It never sees which words
are solutions.
"""
import math

import torch
import torch.nn as nn
from torch.distributions import Categorical

from gym_wordle.envs.feedback import N_ALPHABET

N_COLOURS = 3


class WordlePolicy(nn.Module):
    def __init__(
        self,
        n_words,
        word_features=None,
        d_model=128,
        n_heads=4,
        d_ff=512,
        n_layers=3,
        max_turns=6,
        n_letters=5,
    ):
        super().__init__()
        n_feat = N_ALPHABET * n_letters + N_ALPHABET
        if word_features is None:
            word_features = torch.zeros(n_words, n_feat)
        assert tuple(word_features.shape) == (n_words, n_feat), word_features.shape
        self.register_buffer("word_features", word_features.float())

        self.letter_emb = nn.Embedding(N_ALPHABET, d_model)
        self.colour_emb = nn.Embedding(N_COLOURS, d_model)
        self.slot_emb = nn.Embedding(max_turns * n_letters, d_model)
        self.turn_emb = nn.Embedding(max_turns + 1, d_model)
        self.readout = nn.Parameter(torch.randn(d_model) * 0.02)

        layer = nn.TransformerEncoderLayer(
            d_model, n_heads, d_ff, dropout=0.0, activation="gelu",
            batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.word_tower = nn.Sequential(nn.Linear(n_feat, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        self.value_head = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, 1))
        self.d_model = d_model
        self.ctor_kwargs = dict(
            n_words=n_words, d_model=d_model, n_heads=n_heads, d_ff=d_ff,
            n_layers=n_layers, max_turns=max_turns, n_letters=n_letters,
        )

    def forward(self, obs):
        tokens, pad, turn, mask = obs["tokens"], obs["pad"], obs["turn"], obs["mask"]
        B, S, _ = tokens.shape
        slots = torch.arange(S, device=tokens.device).unsqueeze(0).expand(B, S)
        x = self.letter_emb(tokens[..., 0]) + self.colour_emb(tokens[..., 1]) + self.slot_emb(slots)
        r = (self.readout + self.turn_emb(turn)).unsqueeze(1)                   # (B,1,d)
        x = torch.cat([r, x], dim=1)                                             # (B,1+S,d)
        kpm = torch.cat([torch.zeros(B, 1, dtype=torch.bool, device=pad.device), pad], dim=1)
        h = self.encoder(x, src_key_padding_mask=kpm)
        z = self.norm(h[:, 0])                                                   # (B,d)
        u = self.word_tower(self.word_features)                                  # (V,d)
        logits = (z @ u.t()) / math.sqrt(self.d_model)
        logits = logits.masked_fill(~mask, float("-inf"))
        value = self.value_head(z).squeeze(-1)
        return logits, value

    def act(self, obs, greedy=False):
        logits, value = self(obs)
        dist = Categorical(logits=logits)
        action = logits.argmax(-1) if greedy else dist.sample()
        return action, dist.log_prob(action), dist.entropy(), value

    def evaluate_actions(self, obs, actions):
        logits, value = self(obs)
        dist = Categorical(logits=logits)
        return dist.log_prob(actions), dist.entropy(), value
