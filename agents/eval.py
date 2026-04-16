"""
agents/eval.py
Evaluate a trained PPO agent against buy-and-hold on any split / ticker.

Usage:
    python agents/eval.py                          # SPY test split (default)
    python agents/eval.py --split val
    python agents/eval.py --ticker QQQ --split test
    python agents/eval.py --split test --no-plot
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
from stable_baselines3 import PPO

from env.train_env import TradingEnv, INITIAL_CASH, FEATURE_COLS

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
PROC_DIR   = os.path.join(os.path.dirname(__file__), "..", "data", "processed")


# ── Metrics ───────────────────────────────────────────────────────────────────

def sharpe(returns: np.ndarray, risk_free: float = 0.0) -> float:
    excess = returns - risk_free
    return float(np.mean(excess) / (np.std(excess) + 1e-8) * np.sqrt(252))


def max_drawdown(portfolio_values: np.ndarray) -> float:
    peak = np.maximum.accumulate(portfolio_values)
    dd   = (portfolio_values - peak) / peak
    return float(dd.min())


def total_return(portfolio_values: np.ndarray) -> float:
    return float(portfolio_values[-1] / portfolio_values[0] - 1.0)


def win_rate(daily_returns: np.ndarray) -> float:
    trades = daily_returns[daily_returns != 0]
    return float(np.mean(trades > 0)) if len(trades) > 0 else 0.0


# ── Rollout ───────────────────────────────────────────────────────────────────

def run_episode(model: PPO, env: TradingEnv) -> pd.DataFrame:
    """
    Run one deterministic episode from index 0 (no random start for eval).
    Forces start at the beginning of the split.
    """
    # Temporarily override random start to always begin at index 0
    env._start_idx    = 0
    env._current_step = 0
    env._position     = 0.0
    env._entry_price  = float(env._close[0])
    env._portfolio_val= INITIAL_CASH
    env._ret_history  = []

    obs = env._get_obs()

    records = []
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        records.append({
            "step":            env._current_step,
            "price":           info["price"],
            "position":        info["position"],
            "portfolio_value": info["portfolio_value"],
            "reward":          reward,
        })

    return pd.DataFrame(records)


def build_bnh(df: pd.DataFrame, ticker: str, split: str) -> np.ndarray:
    """Buy-and-hold portfolio curve aligned to the eval episode."""
    csv = pd.read_csv(os.path.join(PROC_DIR, ticker, f"{split}.csv"))
    # Slice to same length as eval episode
    n = len(df)
    prices = csv["close"].iloc[:n + 1].to_numpy(dtype=float)
    bnh = INITIAL_CASH * prices / prices[0]
    return bnh[1:]  # align: bnh[i] is value at end of step i


# ── Reporting ─────────────────────────────────────────────────────────────────

def print_metrics(label: str, portfolio: np.ndarray) -> None:
    returns = np.diff(portfolio) / portfolio[:-1]
    print(f"\n{'─'*40}")
    print(f"  {label}")
    print(f"{'─'*40}")
    print(f"  Total Return  : {total_return(portfolio):+.2%}")
    print(f"  Sharpe Ratio  : {sharpe(returns):+.3f}")
    print(f"  Max Drawdown  : {max_drawdown(portfolio):.2%}")
    print(f"  Win Rate      : {win_rate(returns):.2%}")
    print(f"  Avg Position  : {None}")  # filled below


def report(df: pd.DataFrame, bnh: np.ndarray, ticker: str, split: str) -> None:
    agent_portfolio = df["portfolio_value"].to_numpy()
    agent_returns   = np.diff(agent_portfolio) / agent_portfolio[:-1]
    bnh_returns     = np.diff(bnh) / bnh[:-1]

    header = f"\n{'='*40}\n  Eval: {ticker} | {split.upper()} split\n{'='*40}"
    print(header)

    # Agent
    print(f"\n--- PPO Agent ---")
    print(f"  Total Return  : {total_return(agent_portfolio):+.2%}")
    print(f"  Sharpe Ratio  : {sharpe(agent_returns):+.3f}")
    print(f"  Max Drawdown  : {max_drawdown(agent_portfolio):.2%}")
    print(f"  Win Rate      : {win_rate(agent_returns):.2%}")
    print(f"  Avg Position  : {df['position'].mean():.2%}")

    # Buy-and-hold
    print(f"\n--- Buy & Hold ---")
    print(f"  Total Return  : {total_return(bnh):+.2%}")
    print(f"  Sharpe Ratio  : {sharpe(bnh_returns):+.3f}")
    print(f"  Max Drawdown  : {max_drawdown(bnh):.2%}")
    print(f"  Win Rate      : {win_rate(bnh_returns):.2%}")
    print(f"  Avg Position  : 100.00%")
    print()


def plot_results(df: pd.DataFrame, bnh: np.ndarray, ticker: str, split: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed — skipping plot.")
        return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True,
                                   gridspec_kw={"height_ratios": [3, 1]})

    ax1.plot(df["portfolio_value"].to_numpy(), label="PPO Agent", linewidth=1.5)
    ax1.plot(bnh, label="Buy & Hold", linewidth=1.5, linestyle="--")
    ax1.set_ylabel("Portfolio Value ($)")
    ax1.set_title(f"PPO vs Buy-and-Hold — {ticker} {split.upper()}")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.fill_between(range(len(df)), df["position"].to_numpy(), alpha=0.5, label="Position")
    ax2.set_ylabel("Allocation")
    ax2.set_xlabel("Step (trading days)")
    ax2.set_ylim(0, 1)
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()


# ── Main ──────────────────────────────────────────────────────────────────────

def evaluate(args: argparse.Namespace) -> None:
    model_path = os.path.join(MODELS_DIR, "ppo_spy.zip")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No model at {model_path}. Run agents/train.py first.")

    model = PPO.load(model_path)
    env   = TradingEnv(ticker=args.ticker, split=args.split)

    df  = run_episode(model, env)
    bnh = build_bnh(df, args.ticker, args.split)

    report(df, bnh, args.ticker, args.split)

    if not args.no_plot:
        plot_results(df, bnh, args.ticker, args.split)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate PPO trading agent")
    p.add_argument("--ticker",   default="SPY",  choices=["SPY", "QQQ", "IWM"])
    p.add_argument("--split",    default="test", choices=["train", "val", "test"])
    p.add_argument("--no-plot",  action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
