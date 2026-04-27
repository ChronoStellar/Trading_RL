"""
agents/eval.py
Evaluate a trained PPO agent against buy-and-hold on any split / ticker.

Usage:
    python agents/eval.py                          # SPY test split (default)
    python agents/eval.py --split val
    python agents/eval.py --ticker QQQ --split test
    python agents/eval.py --split test --no-plot
    python agents/eval.py --no-save                # skip writing files

Outputs (saved to agents/eval_results/ by default):
    <TICKER>_<SPLIT>_<TIMESTAMP>.json   — metrics for agent + buy-and-hold
    <TICKER>_<SPLIT>_<TIMESTAMP>.csv    — step-by-step episode trajectory
"""

import argparse
import json
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
from stable_baselines3 import PPO

from env.train_env import TradingEnv, INITIAL_CASH, FEATURE_COLS

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
PROC_DIR   = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
EVAL_DIR   = os.path.join(os.path.dirname(__file__), "eval_results")


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

def collect_metrics(
    df: pd.DataFrame,
    bnh: np.ndarray,
    ticker: str,
    split: str,
) -> dict:
    """Build a serialisable metrics dict for both agent and buy-and-hold."""
    agent_portfolio = df["portfolio_value"].to_numpy()
    agent_returns   = np.diff(agent_portfolio) / agent_portfolio[:-1]
    bnh_returns     = np.diff(bnh) / bnh[:-1]

    return {
        "run_at":  datetime.now().isoformat(timespec="seconds"),
        "ticker":  ticker,
        "split":   split,
        "n_steps": len(df),
        "agent": {
            "total_return":  round(total_return(agent_portfolio), 6),
            "sharpe_ratio":  round(sharpe(agent_returns), 6),
            "max_drawdown":  round(max_drawdown(agent_portfolio), 6),
            "win_rate":      round(win_rate(agent_returns), 6),
            "avg_position":  round(float(df["position"].mean()), 6),
            "final_value":   round(float(agent_portfolio[-1]), 2),
        },
        "buy_and_hold": {
            "total_return": round(total_return(bnh), 6),
            "sharpe_ratio": round(sharpe(bnh_returns), 6),
            "max_drawdown": round(max_drawdown(bnh), 6),
            "win_rate":     round(win_rate(bnh_returns), 6),
            "avg_position": 1.0,
            "final_value":  round(float(bnh[-1]), 2),
        },
        "outperformance": {
            "return": round(total_return(agent_portfolio) - total_return(bnh), 6),
            "sharpe": round(sharpe(agent_returns) - sharpe(bnh_returns), 6),
            "drawdown_improvement": round(
                abs(max_drawdown(bnh)) - abs(max_drawdown(agent_portfolio)), 6
            ),
        },
    }


def report(df: pd.DataFrame, bnh: np.ndarray, ticker: str, split: str) -> dict:
    """Print metrics to stdout and return the metrics dict."""
    m = collect_metrics(df, bnh, ticker, split)
    a = m["agent"]
    b = m["buy_and_hold"]
    o = m["outperformance"]

    print(f"\n{'='*44}")
    print(f"  Eval: {ticker} | {split.upper()} split  ({m['n_steps']} steps)")
    print(f"{'='*44}")

    print(f"\n{'Metric':<22} {'PPO Agent':>12} {'Buy & Hold':>12}")
    print(f"{'─'*46}")
    print(f"{'Total Return':<22} {a['total_return']:>+11.2%}  {b['total_return']:>+11.2%}")
    print(f"{'Sharpe Ratio':<22} {a['sharpe_ratio']:>+11.3f}  {b['sharpe_ratio']:>+11.3f}")
    print(f"{'Max Drawdown':<22} {a['max_drawdown']:>+11.2%}  {b['max_drawdown']:>+11.2%}")
    print(f"{'Win Rate':<22} {a['win_rate']:>+11.2%}  {b['win_rate']:>+11.2%}")
    print(f"{'Avg Position':<22} {a['avg_position']:>+11.2%}  {b['avg_position']:>+11.2%}")
    print(f"{'Final Value ($)':<22} {a['final_value']:>11,.2f}  {b['final_value']:>11,.2f}")

    print(f"\n{'─'*46}")
    print(f"  Outperformance — Return : {o['return']:>+.2%}")
    print(f"  Outperformance — Sharpe : {o['sharpe']:>+.3f}")
    print(f"  Drawdown Improvement    : {o['drawdown_improvement']:>+.2%}")
    print()

    return m


def save_results(
    metrics: dict,
    df: pd.DataFrame,
    ticker: str,
    split: str,
) -> None:
    """Save metrics JSON + episode trajectory CSV to agents/eval_results/."""
    os.makedirs(EVAL_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = f"{ticker}_{split}_{ts}"

    json_path = os.path.join(EVAL_DIR, f"{stem}.json")
    with open(json_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved  → {json_path}")

    csv_path = os.path.join(EVAL_DIR, f"{stem}.csv")
    df.to_csv(csv_path, index=False)
    print(f"Trajectory saved → {csv_path}")


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

    df      = run_episode(model, env)
    bnh     = build_bnh(df, args.ticker, args.split)
    metrics = report(df, bnh, args.ticker, args.split)

    if not args.no_save:
        save_results(metrics, df, args.ticker, args.split)

    if not args.no_plot:
        plot_results(df, bnh, args.ticker, args.split)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate PPO trading agent")
    p.add_argument("--ticker",   default="SPY",  choices=["SPY", "QQQ", "IWM"])
    p.add_argument("--split",    default="test", choices=["train", "val", "test"])
    p.add_argument("--no-plot",  action="store_true")
    p.add_argument("--no-save",  action="store_true", help="Skip writing JSON/CSV to eval_results/")
    return p.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
