"""
agents/train.py
Train a PPO agent on TradingEnv (SPY train split) using Stable-Baselines3.
Saves the trained model to agents/models/ppo_spy.zip.

Usage:
    python agents/train.py
    python agents/train.py --timesteps 1_000_000 --seed 42
    python agents/train.py --max-position-delta 0.3 --slippage-base 0.0002

    # Re-plot an existing run without training:
    python agents/train.py --plot-only agents/logs/20260421_120000
"""

import argparse
import csv
import json
import os
import random
import sys
from datetime import datetime

# Make project root importable regardless of cwd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# from sb3_contrib import RecurrentPPO as PPO
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import configure

from env.train_env import TradingEnv

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
LOGS_DIR   = os.path.join(os.path.dirname(__file__), "logs")

# ── Default hyperparameters (tune via CLI or edit here) ───────────────────────
DEFAULTS = dict(
    timesteps  = 2000_000,
    n_envs     = 8,           # parallel envs for faster data collection
    seed       = 42,
    device     = "cpu",       # "cpu" or "cuda" (GPU training can be faster but is optional)
    # PPO-specific
    n_steps    = 2048,        # steps per env before update
    batch_size = 512,
    n_epochs   = 20,
    gamma      = 0.995,
    gae_lambda = 0.95,
    ent_coef   = 0.005,
    learning_rate = 3e-4,
)


# ── Logging callback ──────────────────────────────────────────────────────────

class EpisodeStatsCallback(BaseCallback):
    """
    Writes per-episode stats (reward, length, final portfolio value) to a CSV.
    Flushes after every episode so the file is readable during training.
    """

    def __init__(self, log_path: str, verbose: int = 0):
        super().__init__(verbose)
        self.log_path = log_path
        self._ep_count = 0
        self._f = None
        self._writer = None

    def _on_training_start(self) -> None:
        self._f = open(self.log_path, "w", newline="")
        self._writer = csv.DictWriter(
            self._f,
            fieldnames=["timestep", "episode", "reward", "length", "portfolio_value"],
        )
        self._writer.writeheader()

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "episode" not in info:
                continue
            self._ep_count += 1
            self._writer.writerow({
                "timestep":       self.num_timesteps,
                "episode":        self._ep_count,
                "reward":         info["episode"]["r"],
                "length":         info["episode"]["l"],
                "portfolio_value": info.get("portfolio_value", float("nan")),
            })
            self._f.flush()
        return True

    def _on_training_end(self) -> None:
        self.close()

    def close(self) -> None:
        if self._f and not self._f.closed:
            self._f.close()


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_history(log_dir: str) -> str | None:
    """Read progress.csv + episodes.csv from log_dir and save a training_history.png."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import pandas as pd
    except ImportError:
        print("matplotlib / pandas not available — skipping plot.")
        return None

    progress_csv = os.path.join(log_dir, "progress.csv")
    episodes_csv = os.path.join(log_dir, "episodes.csv")

    if not os.path.exists(progress_csv) and not os.path.exists(episodes_csv):
        print(f"No log files found in {log_dir}")
        return None

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle(f"Training History — {os.path.basename(log_dir)}", fontsize=13)

    # ── Progress metrics from SB3's built-in logger ──
    if os.path.exists(progress_csv):
        df = pd.read_csv(progress_csv)

        ts_col = "time/total_timesteps"
        if ts_col not in df.columns:
            ts_col = df.columns[0]

        def _plot(ax, col, title, ylabel=""):
            if col not in df.columns:
                ax.set_visible(False)
                return
            valid = df[col].notna()
            ax.plot(df.loc[valid, ts_col], df.loc[valid, col], linewidth=1)
            ax.set_title(title)
            ax.set_xlabel("Timesteps")
            if ylabel:
                ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.3)

        _plot(axes[0, 0], "rollout/ep_rew_mean",          "Mean Episode Reward",       "Reward")
        _plot(axes[0, 1], "train/value_loss",              "Value Loss")
        _plot(axes[0, 2], "train/policy_gradient_loss",    "Policy Gradient Loss")
        _plot(axes[1, 0], "train/entropy_loss",            "Entropy Loss")
        _plot(axes[1, 1], "train/approx_kl",               "Approx KL Divergence")

    # ── Per-episode portfolio value ──
    ax = axes[1, 2]
    if os.path.exists(episodes_csv):
        df_ep = pd.read_csv(episodes_csv)
        if not df_ep.empty and "portfolio_value" in df_ep.columns:
            window = min(50, max(1, len(df_ep) // 10))
            smoothed = df_ep["portfolio_value"].rolling(window, min_periods=1).mean()
            ax.plot(df_ep["episode"], df_ep["portfolio_value"], alpha=0.25, color="steelblue", label="raw")
            ax.plot(df_ep["episode"], smoothed, color="steelblue", label=f"MA-{window}")
            ax.axhline(100_000, color="gray", linestyle="--", alpha=0.6, label="initial $100k")
            ax.set_title("Episode Final Portfolio Value")
            ax.set_xlabel("Episode")
            ax.set_ylabel("Value ($)")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
        else:
            ax.set_visible(False)
    else:
        ax.set_visible(False)

    plt.tight_layout()
    out_path = os.path.join(log_dir, "training_history.png")
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ── Environment factory ───────────────────────────────────────────────────────

def make_env(seed: int = 0):
    def _init():
        return TradingEnv(ticker="SPY", split="train", seed=seed)
    return _init


# ── Training ──────────────────────────────────────────────────────────────────

def train(args: argparse.Namespace) -> None:
    os.makedirs(MODELS_DIR, exist_ok=True)

    # Timestamped log directory for this run
    run_ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_log_dir = os.path.join(LOGS_DIR, run_ts)
    os.makedirs(run_log_dir, exist_ok=True)

    # Persist hyperparameters alongside logs
    with open(os.path.join(run_log_dir, "hparams.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    # Persist feature list + reward config so every run is self-documenting
    from data.features import FEATURE_COLS as _FCOLS
    from env.rewards import RISK_FREE_DAILY as _RF, TRANSACTION_COST as _TC, SLIPPAGE as _SLIP
    run_config = {
        "features": {
            "market": _FCOLS,
            "env_injected": ["position", "equity_return"],
            "obs_dim": len(_FCOLS) + 2,
        },
        "reward": {
            "function": "sharpe_step_reward",
            "risk_free_daily": _RF,
            "transaction_cost": _TC,
            "slippage": _SLIP,
            "formula": "(r_t - rf) / vol - (tc + slip) * |delta| / vol",
        },
    }
    with open(os.path.join(run_log_dir, "run_config.json"), "w") as f:
        json.dump(run_config, f, indent=2)

    # Vectorized environment
    vec_env = make_vec_env(
        make_env(args.seed),
        n_envs=args.n_envs,
        seed=args.seed,
    )

    # SB3 logger: stdout + CSV (writes progress.csv into run_log_dir)
    sb3_logger = configure(run_log_dir, ["stdout", "csv"])

    model = PPO(
        # policy        = "MlpLstmPolicy",
        policy        = "MlpPolicy",
        env           = vec_env,
        n_steps       = args.n_steps,
        batch_size    = args.batch_size,
        n_epochs      = args.n_epochs,
        gamma         = args.gamma,
        gae_lambda    = args.gae_lambda,
        ent_coef      = args.ent_coef,
        learning_rate = args.learning_rate,
        device        = args.device,
        verbose       = 1,
        seed          = args.seed,
    )
    model.set_logger(sb3_logger)

    ep_callback = EpisodeStatsCallback(os.path.join(run_log_dir, "episodes.csv"))

    save_path = os.path.join(MODELS_DIR, "ppo_spy")

    print(f"\nTraining PPO for {args.timesteps:,} timesteps across {args.n_envs} envs …")
    print(f"Logs  → {run_log_dir}")
    print(f"Model → {save_path}.zip\n")

    try:
        model.learn(total_timesteps=args.timesteps, callback=ep_callback)
        interrupted = False
    except KeyboardInterrupt:
        interrupted = True
        print("\n\nInterrupted — saving checkpoint …")
    finally:
        # Always save model and close log files
        ep_callback.close()
        model.save(save_path)
        print(f"Model saved → {save_path}.zip")

        plot_path = plot_history(run_log_dir)
        if plot_path:
            print(f"Plot  saved → {plot_path}")

        if not interrupted:
            print("\nTraining complete.")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train PPO on TradingEnv")
    p.add_argument("--timesteps",     type=int,   default=DEFAULTS["timesteps"])
    p.add_argument("--n-envs",        type=int,   default=DEFAULTS["n_envs"])
    p.add_argument("--seed",          type=int,   default=DEFAULTS["seed"])
    p.add_argument("--n-steps",       type=int,   default=DEFAULTS["n_steps"])
    p.add_argument("--batch-size",    type=int,   default=DEFAULTS["batch_size"])
    p.add_argument("--n-epochs",      type=int,   default=DEFAULTS["n_epochs"])
    p.add_argument("--gamma",         type=float, default=DEFAULTS["gamma"])
    p.add_argument("--gae-lambda",    type=float, default=DEFAULTS["gae_lambda"])
    p.add_argument("--ent-coef",      type=float, default=DEFAULTS["ent_coef"])
    p.add_argument("--learning-rate", type=float, default=DEFAULTS["learning_rate"])
    p.add_argument("--device",        type=str,   default=DEFAULTS["device"],
                   choices=["mps", "cpu", "cuda"])
    p.add_argument("--plot-only",     type=str,   default=None, metavar="LOG_DIR",
                   help="Re-plot an existing run directory without training")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.plot_only:
        path = plot_history(args.plot_only)
        if path:
            print(f"Plot saved → {path}")
    else:
        train(args)
