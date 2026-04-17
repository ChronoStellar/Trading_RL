"""
agents/backtest.py
Comprehensive backtesting framework: run multiple episodes, aggregate stats, walk-forward testing.

Usage:
    python agents/backtest.py                              # full walk-forward
    python agents/backtest.py --split test --n-episodes 5
    python agents/backtest.py --ticker QQQ --split test
"""

import argparse
import json
import os
import sys
from dataclasses import dataclass, asdict
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
from sb3_contrib import RecurrentPPO

from env.train_env import TradingEnv, INITIAL_CASH, FEATURE_COLS

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
PROC_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
BACKTEST_DIR = os.path.join(os.path.dirname(__file__), "backtest_results")


# ── Data Classes ───────────────────────────────────────────────────────────────

@dataclass
class EpisodeStats:
    """Statistics from a single episode."""
    episode:         int
    total_return:    float
    sharpe_ratio:    float
    max_drawdown:    float
    win_rate:        float
    avg_position:    float
    final_value:     float
    num_steps:       int


@dataclass
class BacktestStats:
    """Aggregated statistics across multiple episodes."""
    split:           str
    ticker:          str
    n_episodes:      int

    # Agent stats
    agent_return_mean:    float
    agent_return_std:     float
    agent_sharpe_mean:    float
    agent_sharpe_std:     float
    agent_dd_mean:        float
    agent_dd_std:         float
    agent_wr_mean:        float
    agent_wr_std:         float
    agent_position_mean:  float

    # Buy-and-hold stats
    bnh_return:      float
    bnh_sharpe:      float
    bnh_dd:          float
    bnh_wr:          float

    # Comparison
    return_outperformance: float  # agent - bnh
    sharpe_outperformance: float  # agent - bnh
    dd_improvement:        float  # |agent_dd| - |bnh_dd| (lower is better)


# ── Metrics ────────────────────────────────────────────────────────────────────

def sharpe(returns: np.ndarray, risk_free: float = 0.0) -> float:
    excess = returns - risk_free
    return float(np.mean(excess) / (np.std(excess) + 1e-8) * np.sqrt(252))


def max_drawdown(portfolio_values: np.ndarray) -> float:
    peak = np.maximum.accumulate(portfolio_values)
    dd = (portfolio_values - peak) / peak
    return float(dd.min())


def total_return(portfolio_values: np.ndarray) -> float:
    return float(portfolio_values[-1] / portfolio_values[0] - 1.0)


def win_rate(daily_returns: np.ndarray) -> float:
    trades = daily_returns[daily_returns != 0]
    return float(np.mean(trades > 0)) if len(trades) > 0 else 0.0


# ── Rollout ────────────────────────────────────────────────────────────────────

def run_episode(
    model: RecurrentPPO,
    env: TradingEnv,
    episode: int,
    deterministic: bool = True,
) -> tuple[pd.DataFrame, EpisodeStats]:
    """Run one episode and return trajectory + stats."""
    obs, _ = env.reset()
    records = []
    done = False

    while not done:
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        records.append({
            "step":            env._current_step,
            "price":           info["price"],
            "position":        info["position"],
            "portfolio_value": info["portfolio_value"],
            "reward":          reward,
        })

    df = pd.DataFrame(records)
    portfolio = df["portfolio_value"].to_numpy()
    returns = np.diff(portfolio) / portfolio[:-1]

    stats = EpisodeStats(
        episode=episode,
        total_return=total_return(portfolio),
        sharpe_ratio=sharpe(returns),
        max_drawdown=max_drawdown(portfolio),
        win_rate=win_rate(returns),
        avg_position=df["position"].mean(),
        final_value=float(portfolio[-1]),
        num_steps=len(df),
    )
    return df, stats


def build_bnh(df: pd.DataFrame, ticker: str, split: str) -> np.ndarray:
    """Buy-and-hold portfolio aligned to eval episode."""
    csv = pd.read_csv(os.path.join(PROC_DIR, ticker, f"{split}.csv"))
    n = len(df)
    prices = csv["close"].iloc[:n + 1].to_numpy(dtype=float)
    bnh = INITIAL_CASH * prices / prices[0]
    return bnh[1:]


# ── Aggregation ────────────────────────────────────────────────────────────────

def aggregate_stats(
    episode_stats: list[EpisodeStats],
    bnh_portfolio: np.ndarray,
) -> BacktestStats:
    """Aggregate episode stats."""
    bnh_returns = np.diff(bnh_portfolio) / bnh_portfolio[:-1]

    agent_returns = np.array([s.total_return for s in episode_stats])
    agent_sharpes = np.array([s.sharpe_ratio for s in episode_stats])
    agent_dds = np.array([s.max_drawdown for s in episode_stats])
    agent_wrs = np.array([s.win_rate for s in episode_stats])

    stats = BacktestStats(
        split=episode_stats[0].split if hasattr(episode_stats[0], "split") else "unknown",
        ticker="SPY",  # filled by caller
        n_episodes=len(episode_stats),
        agent_return_mean=float(np.mean(agent_returns)),
        agent_return_std=float(np.std(agent_returns)),
        agent_sharpe_mean=float(np.mean(agent_sharpes)),
        agent_sharpe_std=float(np.std(agent_sharpes)),
        agent_dd_mean=float(np.mean(agent_dds)),
        agent_dd_std=float(np.std(agent_dds)),
        agent_wr_mean=float(np.mean(agent_wrs)),
        agent_wr_std=float(np.std(agent_wrs)),
        agent_position_mean=float(np.mean([s.avg_position for s in episode_stats])),
        bnh_return=total_return(bnh_portfolio),
        bnh_sharpe=sharpe(bnh_returns),
        bnh_dd=max_drawdown(bnh_portfolio),
        bnh_wr=win_rate(bnh_returns),
        return_outperformance=float(
            np.mean(agent_returns) - total_return(bnh_portfolio)
        ),
        sharpe_outperformance=float(
            np.mean(agent_sharpes) - sharpe(bnh_returns)
        ),
        dd_improvement=float(
            abs(np.mean(agent_dds)) - abs(max_drawdown(bnh_portfolio))
        ),
    )
    return stats


# ── Reporting ──────────────────────────────────────────────────────────────────

def print_stats(stats: BacktestStats) -> None:
    print(f"\n{'='*60}")
    print(f"  BACKTEST: {stats.ticker} | {stats.split.upper()} | {stats.n_episodes} episodes")
    print(f"{'='*60}\n")

    print(f"{'AGENT':<30} {'MEAN':<15} {'STD':<15}")
    print(f"{'-'*60}")
    print(f"{'Total Return':<30} {stats.agent_return_mean:>+13.2%}  {stats.agent_return_std:>13.2%}")
    print(f"{'Sharpe Ratio':<30} {stats.agent_sharpe_mean:>+13.3f}  {stats.agent_sharpe_std:>13.3f}")
    print(f"{'Max Drawdown':<30} {stats.agent_dd_mean:>+13.2%}  {stats.agent_dd_std:>13.2%}")
    print(f"{'Win Rate':<30} {stats.agent_wr_mean:>+13.2%}  {stats.agent_wr_std:>13.2%}")
    print(f"{'Avg Position Size':<30} {stats.agent_position_mean:>13.2%}")

    print(f"\n{'BUY & HOLD':<30} {'VALUE':<15}")
    print(f"{'-'*60}")
    print(f"{'Total Return':<30} {stats.bnh_return:>+13.2%}")
    print(f"{'Sharpe Ratio':<30} {stats.bnh_sharpe:>+13.3f}")
    print(f"{'Max Drawdown':<30} {stats.bnh_dd:>+13.2%}")
    print(f"{'Win Rate':<30} {stats.bnh_wr:>+13.2%}")
    print(f"{'Avg Position Size':<30} {'100.00%':>13}")

    print(f"\n{'OUTPERFORMANCE':<30}")
    print(f"{'-'*60}")
    print(f"{'Return':<30} {stats.return_outperformance:>+13.2%}")
    print(f"{'Sharpe':<30} {stats.sharpe_outperformance:>+13.3f}")
    print(f"{'Drawdown Improvement':<30} {stats.dd_improvement:>+13.2%}")
    print()


def save_results(stats: BacktestStats, results_dir: str) -> None:
    os.makedirs(results_dir, exist_ok=True)
    out_path = os.path.join(
        results_dir, f"{stats.ticker}_{stats.split}_backtest.json"
    )
    with open(out_path, "w") as f:
        json.dump(asdict(stats), f, indent=2)
    print(f"Results saved → {out_path}")


# ── Main ───────────────────────────────────────────────────────────────────────

def backtest_single(
    ticker: str,
    split: str,
    n_episodes: int,
    model_path: str,
    seed: int = 0,
) -> BacktestStats:
    """Run backtest on one split."""
    model = RecurrentPPO.load(model_path)
    env = TradingEnv(ticker=ticker, split=split, seed=seed)

    print(f"\n[{ticker} | {split.upper()}] Running {n_episodes} episodes …")

    episode_stats = []
    for ep in range(n_episodes):
        df, stats = run_episode(model, env, ep)
        stats.split = split
        stats.ticker = ticker
        episode_stats.append(stats)
        print(f"  Episode {ep+1}/{n_episodes}: return={stats.total_return:+.2%}, "
              f"sharpe={stats.sharpe_ratio:+.3f}")

    # Get BnH from first episode
    df, _ = run_episode(model, env, 0)
    bnh = build_bnh(df, ticker, split)

    agg_stats = aggregate_stats(episode_stats, bnh)
    agg_stats.ticker = ticker
    return agg_stats


def backtest_walk_forward(
    ticker: str = "SPY",
    n_episodes: int = 5,
    seed: int = 0,
) -> list[BacktestStats]:
    """Run walk-forward backtest: train → val → test."""
    model_path = os.path.join(MODELS_DIR, "ppo_spy.zip")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No model at {model_path}. Run agents/train.py first.")

    results = []
    for split in ["train", "val", "test"]:
        try:
            stats = backtest_single(ticker, split, n_episodes, model_path, seed)
            results.append(stats)
            print_stats(stats)
        except Exception as e:
            print(f"Error on {ticker} {split}: {e}")

    return results


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Backtest trading agent")
    p.add_argument(
        "--mode",
        choices=["walk-forward", "single"],
        default="walk-forward",
        help="'walk-forward' tests train/val/test; 'single' tests one split",
    )
    p.add_argument("--ticker", default="SPY", choices=["SPY", "QQQ", "IWM"])
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--n-episodes", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-save", action="store_true", help="Don't save JSON results")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    os.makedirs(BACKTEST_DIR, exist_ok=True)

    if args.mode == "walk-forward":
        results = backtest_walk_forward(args.ticker, args.n_episodes, args.seed)
        if not args.no_save:
            for stats in results:
                save_results(stats, BACKTEST_DIR)
    else:
        stats = backtest_single(
            args.ticker, args.split, args.n_episodes,
            os.path.join(MODELS_DIR, "ppo_spy.zip"),
            args.seed,
        )
        print_stats(stats)
        if not args.no_save:
            save_results(stats, BACKTEST_DIR)


if __name__ == "__main__":
    main()
