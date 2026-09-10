"""Train or evaluate the PPO transformer agent on batched Wordle."""
import argparse
import csv
import datetime as dt
import logging
import time
from pathlib import Path

import torch

from gym_wordle.agents.common import load_checkpoint, save_checkpoint, word_feature_matrix
from gym_wordle.agents.ppo.model import WordlePolicy
from gym_wordle.agents.ppo.ppo import PPOConfig, RolloutBuffer, ppo_update
from gym_wordle.envs.batched import BatchedWordle
from gym_wordle.envs.wordle_env import DEFAULT_SOLUTIONS, DEFAULT_VALIDS, load_word_lists

log = logging.getLogger(__name__)

CSV_FIELDS = [
    "iteration", "steps", "episodes", "solve_rate", "mean_guesses", "mean_reward",
    "entropy", "approx_kl", "clip_frac", "value_loss", "explained_variance",
    "epochs_run", "steps_per_s", "lr",
]


def collect(policy, env, buf, obs, ep_ret):
    """Roll the policy for buf.T steps. Returns (obs, last_value, episode stats).

    ep_ret accumulates per-game return across calls and is zeroed on done.
    """
    policy.eval()
    solved_n, finished_n, guesses_sum, ret_sum = 0, 0, 0.0, 0.0
    with torch.no_grad():
        for t in range(buf.T):
            action, log_prob, _, value = policy.act(obs)
            next_obs, reward, done, info = env.step(action)
            buf.store(t, obs, action, log_prob, value, reward, done)
            ep_ret += reward
            if bool(done.any()):
                fin = done
                solved = info["solved"][fin]
                finished_n += int(fin.sum())
                solved_n += int(solved.sum())
                guesses_sum += float(info["n_guesses"][fin][solved].sum())
                ret_sum += float(ep_ret[fin].sum())
                ep_ret[fin] = 0.0
            obs = next_obs
        _, last_value = policy(obs)
    stats = {
        "episodes": finished_n,
        "solve_rate": solved_n / finished_n if finished_n else float("nan"),
        "mean_guesses": guesses_sum / solved_n if solved_n else float("nan"),
        "mean_reward": ret_sum / finished_n if finished_n else float("nan"),
    }
    return obs, last_value, stats


def evaluate(policy, words, solutions, *, hard_mode, device, L=4.0, max_attempts=6):
    """Greedy play of every solution exactly once. Never touches training state."""
    env = BatchedWordle(words, solutions, len(solutions), hard_mode=hard_mode,
                        max_attempts=max_attempts, L=L, device=device, seed=0)
    obs = env.set_secrets(env.solution_idx)
    N = env.n_games
    solved = torch.zeros(N, dtype=torch.bool, device=env.device)
    n_guesses = torch.zeros(N, dtype=torch.long, device=env.device)
    active = torch.ones(N, dtype=torch.bool, device=env.device)
    policy.eval()
    with torch.no_grad():
        for _ in range(max_attempts):
            action, *_ = policy.act(obs, greedy=True)
            obs, _, done, info = env.step(action)
            fin = done & active
            solved = torch.where(fin, info["solved"], solved)
            n_guesses = torch.where(fin, info["n_guesses"], n_guesses)
            active &= ~done
            if not bool(active.any()):
                break
    hist = torch.bincount(n_guesses[solved], minlength=max_attempts + 1)[1:].tolist()
    n_solved = int(solved.sum())
    return {
        "solve_rate": n_solved / N,
        "mean_guesses": float(n_guesses[solved].float().mean()) if n_solved else float("nan"),
        "histogram": hist,
        "fails": N - n_solved,
    }


def build_parser():
    p = argparse.ArgumentParser(description="PPO transformer agent for Wordle")
    p.add_argument("--valids", default=str(DEFAULT_VALIDS))
    p.add_argument("--solutions", default=str(DEFAULT_SOLUTIONS))
    p.add_argument("--iterations", type=int, default=300)
    p.add_argument("--n-games", type=int, default=2048)
    p.add_argument("--rollout-len", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--clip", type=float, default=0.2)
    p.add_argument("--target-kl", type=float, default=0.02)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--minibatch", type=int, default=4096)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--n-layers", type=int, default=3)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--d-ff", type=int, default=512)
    p.add_argument("--hard-mode", dest="hard_mode", action="store_true", default=True)
    p.add_argument("--no-hard-mode", dest="hard_mode", action="store_false")
    p.add_argument("--shaping-coef", type=float, default=0.0)
    p.add_argument("--L", type=float, default=4.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--run-dir", default=None, help="default runs/<timestamp>")
    p.add_argument("--checkpoint-every", type=int, default=25)
    p.add_argument("--checkpoint", default=None, help="resume from / evaluate this .pt file")
    p.add_argument("--eval-only", action="store_true")
    return p


def train(args):
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    words, solutions = load_word_lists(args.valids, args.solutions)
    n_letters = len(words[0])
    env = BatchedWordle(words, solutions, args.n_games, hard_mode=args.hard_mode, L=args.L,
                        shaping_coef=args.shaping_coef, device=device, seed=args.seed)
    if args.checkpoint:
        policy = load_checkpoint(WordlePolicy, args.checkpoint, device)
    else:
        policy = WordlePolicy(
            len(words), word_feature_matrix(words, n_letters), d_model=args.d_model,
            n_heads=args.n_heads, d_ff=args.d_ff, n_layers=args.n_layers,
            max_turns=env.max_attempts, n_letters=n_letters,
        ).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr, eps=1e-5)
    cfg = PPOConfig(clip=args.clip, vf_coef=args.vf_coef, ent_coef=args.ent_coef, epochs=args.epochs,
                    minibatch=args.minibatch, target_kl=args.target_kl, gamma=1.0, lam=args.gae_lambda)
    buf = RolloutBuffer(args.rollout_len, args.n_games, env.max_attempts * n_letters, len(words), device)

    run_dir = Path(args.run_dir or Path("runs") / dt.datetime.now().strftime("%Y%m%d.%H.%M.%S"))
    run_dir.mkdir(parents=True, exist_ok=True)
    log.info("run dir %s  params %d", run_dir, sum(p.numel() for p in policy.parameters()))

    obs = env.reset()
    ep_ret = torch.zeros(args.n_games, device=device)
    steps = 0
    with open(run_dir / "log.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for it in range(1, args.iterations + 1):
            lr = args.lr * (1 - (it - 1) / args.iterations)
            for g in optimizer.param_groups:
                g["lr"] = lr
            t0 = time.perf_counter()
            obs, last_value, ep = collect(policy, env, buf, obs, ep_ret)
            buf.compute_gae(last_value, cfg.gamma, cfg.lam)
            upd = ppo_update(policy, optimizer, buf, cfg)
            steps += buf.T * buf.N
            row = {
                "iteration": it, "steps": steps, **ep,
                "entropy": upd["entropy"], "approx_kl": upd["approx_kl"], "clip_frac": upd["clip_frac"],
                "value_loss": upd["value_loss"], "explained_variance": upd["explained_variance"],
                "epochs_run": upd["epochs_run"],
                "steps_per_s": buf.T * buf.N / (time.perf_counter() - t0), "lr": lr,
            }
            writer.writerow(row)
            f.flush()
            log.info(
                "it %d  steps %d  solve %.3f  guesses %.2f  reward %.3f  ent %.2f  kl %.4f  clip %.2f  ev %.2f  %.0f steps/s",
                it, steps, row["solve_rate"], row["mean_guesses"], row["mean_reward"], row["entropy"],
                row["approx_kl"], row["clip_frac"], row["explained_variance"], row["steps_per_s"],
            )
            if it % args.checkpoint_every == 0:
                save_checkpoint(policy, str(run_dir / f"policy_{it}.pt"))
    save_checkpoint(policy, str(run_dir / "policy_final.pt"))
    return policy, run_dir


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.eval_only and not args.checkpoint:
        parser.error("--eval-only requires --checkpoint")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    words, solutions = load_word_lists(args.valids, args.solutions)
    if args.eval_only:
        policy = load_checkpoint(WordlePolicy, args.checkpoint, args.device)
        run_dir = None
    else:
        policy, run_dir = train(args)
    final = evaluate(policy, words, solutions, hard_mode=args.hard_mode, device=args.device, L=args.L)
    log.info("eval: solve rate %.4f  mean guesses %.3f  histogram %s  fails %d",
             final["solve_rate"], final["mean_guesses"], final["histogram"], final["fails"])
    return policy, run_dir, final


if __name__ == "__main__":
    main()
