"""
agents/train.py
Train a Recurrent PPO agent (LSTM policy) on TradingEnv (SPY train split) using Stable-Baselines3.
Saves the trained model to agents/models/ppo_spy.zip.

Usage:
    python agents/train.py
    python agents/train.py --timesteps 1_000_000 --seed 42
"""

import argparse
import os
import sys

# Make project root importable regardless of cwd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sb3_contrib import RecurrentPPO
from stable_baselines3.common.env_util import make_vec_env

from env.train_env import TradingEnv

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")

# ── Default hyperparameters (tune via CLI or edit here) ───────────────────────
DEFAULTS = dict(
    timesteps  = 500_000,
    n_envs     = 2,           # parallel envs for faster data collection
    seed       = 0,
    device     = "mps",       # "mps" (Apple GPU), "cpu", or "cuda"
    # PPO-specific
    n_steps    = 2048,        # steps per env before update
    batch_size = 256,
    n_epochs   = 10,
    gamma      = 0.99,
    gae_lambda = 0.95,
    ent_coef   = 0.01,
    learning_rate = 3e-4,
)


def make_env(seed: int = 0):
    def _init():
        return TradingEnv(ticker="SPY", split="train", seed=seed)
    return _init


def train(args: argparse.Namespace) -> None:
    os.makedirs(MODELS_DIR, exist_ok=True)

    # Vectorized environment
    vec_env = make_vec_env(
        make_env(args.seed),
        n_envs=args.n_envs,
        seed=args.seed,
    )

    model = RecurrentPPO(
        policy        = "MlpLstmPolicy",
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

    print(f"\nTraining PPO for {args.timesteps:,} timesteps across {args.n_envs} envs …\n")
    model.learn(total_timesteps=args.timesteps)

    save_path = os.path.join(MODELS_DIR, "ppo_spy")
    model.save(save_path)
    print(f"\nModel saved → {save_path}.zip")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train Recurrent PPO on TradingEnv")
    p.add_argument("--timesteps",    type=int,   default=DEFAULTS["timesteps"])
    p.add_argument("--n-envs",       type=int,   default=DEFAULTS["n_envs"])
    p.add_argument("--seed",         type=int,   default=DEFAULTS["seed"])
    p.add_argument("--device",       type=str,   default=DEFAULTS["device"],
                   choices=["mps", "cpu", "cuda"])
    p.add_argument("--n-steps",      type=int,   default=DEFAULTS["n_steps"])
    p.add_argument("--batch-size",   type=int,   default=DEFAULTS["batch_size"])
    p.add_argument("--n-epochs",     type=int,   default=DEFAULTS["n_epochs"])
    p.add_argument("--gamma",        type=float, default=DEFAULTS["gamma"])
    p.add_argument("--gae-lambda",   type=float, default=DEFAULTS["gae_lambda"])
    p.add_argument("--ent-coef",     type=float, default=DEFAULTS["ent_coef"])
    p.add_argument("--learning-rate",type=float, default=DEFAULTS["learning_rate"])
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
