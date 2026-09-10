"""Rollout storage, generalised advantage estimation and the clipped PPO update."""
from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class PPOConfig:
    clip: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    epochs: int = 4
    minibatch: int = 4096
    max_grad_norm: float = 0.5
    target_kl: float | None = 0.02
    gamma: float = 1.0
    lam: float = 0.95


class RolloutBuffer:
    """Preallocated (T, N, ...) tensors on the training device."""

    def __init__(self, T, N, n_tokens, n_words, device):
        self.T, self.N = int(T), int(N)
        dev = torch.device(device)
        self.tokens = torch.zeros(T, N, n_tokens, 3, dtype=torch.long, device=dev)
        self.pad = torch.zeros(T, N, n_tokens, dtype=torch.bool, device=dev)
        self.turn = torch.zeros(T, N, dtype=torch.long, device=dev)
        self.mask = torch.zeros(T, N, n_words, dtype=torch.bool, device=dev)
        self.actions = torch.zeros(T, N, dtype=torch.long, device=dev)
        self.log_probs = torch.zeros(T, N, device=dev)
        self.values = torch.zeros(T, N, device=dev)
        self.rewards = torch.zeros(T, N, device=dev)
        self.dones = torch.zeros(T, N, dtype=torch.bool, device=dev)
        self.advantages = torch.zeros(T, N, device=dev)
        self.returns = torch.zeros(T, N, device=dev)

    def store(self, t, obs, action, log_prob, value, reward, done):
        self.tokens[t] = obs["tokens"]
        self.pad[t] = obs["pad"]
        self.turn[t] = obs["turn"]
        self.mask[t] = obs["mask"]
        self.actions[t] = action
        self.log_probs[t] = log_prob
        self.values[t] = value
        self.rewards[t] = reward
        self.dones[t] = done

    def compute_gae(self, last_value, gamma, lam):
        """dones[t] means obs[t+1] starts a new game: no bootstrap across it."""
        adv = torch.zeros(self.N, device=self.values.device)
        for t in reversed(range(self.T)):
            next_value = last_value if t == self.T - 1 else self.values[t + 1]
            nonterminal = (~self.dones[t]).float()
            delta = self.rewards[t] + gamma * next_value * nonterminal - self.values[t]
            adv = delta + gamma * lam * nonterminal * adv
            self.advantages[t] = adv
        self.returns = self.advantages + self.values

    def flat_obs(self, idx):
        """Observation dict for flat indices into the (T*N) rollout."""
        return {
            "tokens": self.tokens.reshape(-1, *self.tokens.shape[2:])[idx],
            "pad": self.pad.reshape(-1, self.pad.shape[2])[idx],
            "turn": self.turn.reshape(-1)[idx],
            "mask": self.mask.reshape(-1, self.mask.shape[2])[idx],
        }


def ppo_update(policy, optimizer, buf, cfg):
    """Clipped-surrogate PPO over the buffer. Returns mean diagnostics."""
    n = buf.T * buf.N
    dev = buf.values.device
    adv = buf.advantages.reshape(n)
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    ret = buf.returns.reshape(n)
    old_lp = buf.log_probs.reshape(n)
    old_v = buf.values.reshape(n)
    actions = buf.actions.reshape(n)

    sums = {k: 0.0 for k in ["policy_loss", "value_loss", "entropy", "approx_kl", "clip_frac"]}
    n_batches = 0
    epochs_run = 0
    policy.train()
    for _ in range(cfg.epochs):
        perm = torch.randperm(n, device=dev)
        epoch_kl, epoch_batches = 0.0, 0
        for start in range(0, n, cfg.minibatch):
            idx = perm[start : start + cfg.minibatch]
            lp, ent, v = policy.evaluate_actions(buf.flat_obs(idx), actions[idx])
            ratio = torch.exp(lp - old_lp[idx])
            a = adv[idx]
            pg_loss = torch.max(-a * ratio, -a * torch.clamp(ratio, 1 - cfg.clip, 1 + cfg.clip)).mean()
            v_clipped = old_v[idx] + torch.clamp(v - old_v[idx], -cfg.clip, cfg.clip)
            v_loss = 0.5 * torch.max((v - ret[idx]) ** 2, (v_clipped - ret[idx]) ** 2).mean()
            entropy = ent.mean()
            loss = pg_loss - cfg.ent_coef * entropy + cfg.vf_coef * v_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), cfg.max_grad_norm)
            optimizer.step()

            with torch.no_grad():
                log_ratio = lp - old_lp[idx]
                approx_kl = ((ratio - 1) - log_ratio).mean().item()
                clip_frac = ((ratio - 1).abs() > cfg.clip).float().mean().item()
            sums["policy_loss"] += pg_loss.item()
            sums["value_loss"] += v_loss.item()
            sums["entropy"] += entropy.item()
            sums["approx_kl"] += approx_kl
            sums["clip_frac"] += clip_frac
            n_batches += 1
            epoch_kl += approx_kl
            epoch_batches += 1
        epochs_run += 1
        if cfg.target_kl is not None and epoch_kl / max(epoch_batches, 1) > cfg.target_kl:
            break

    with torch.no_grad():
        var_ret = ret.var()
        ev = float("nan") if var_ret == 0 else (1 - (ret - old_v).var() / var_ret).item()
    out = {k: v / max(n_batches, 1) for k, v in sums.items()}
    out["explained_variance"] = ev
    out["epochs_run"] = epochs_run
    return out
