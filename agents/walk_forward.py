"""
agents/walk_forward.py
Walk-forward validation: train on 2010–2021, validate on 2022–2023, test on 2024–2025.
Ensures no hyperparameter tuning on test set (lookahead bias prevention).

Usage:
    python agents/walk_forward.py              # full analysis
    python agents/walk_forward.py --generalize # also test QQQ/IWM
"""

import argparse
import json
import os
import sys
from tabulate import tabulate

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agents.backtest import (
    backtest_single,
    BacktestStats,
    print_stats,
)

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
BACKTEST_DIR = os.path.join(os.path.dirname(__file__), "backtest_results")


def summarize_splits() -> None:
    """Print data split summary."""
    print("\n" + "="*70)
    print(" WALK-FORWARD VALIDATION STRUCTURE")
    print("="*70)
    print("""
┌──────────────────────────────────────────────────────────────────┐
│ TRAINING SET (2010–2021)                                         │
│   • 12 years of SPY daily bars                                   │
│   • Policy learned here; no hyperparameter tuning                │
├──────────────────────────────────────────────────────────────────┤
│ VALIDATION SET (2022–2023)                                       │
│   • 2 years: includes 2022 bear market + recovery                │
│   • Evaluate and tune hyperparameters if needed                  │
├──────────────────────────────────────────────────────────────────┤
│ TEST SET (2024–2025)                                             │
│   • Unseen data; touch once, report final results                │
│   • No further tuning allowed                                    │
└──────────────────────────────────────────────────────────────────┘
    """)


def run_walk_forward(n_episodes: int = 5, seed: int = 0) -> dict:
    """Run walk-forward backtest on SPY."""
    model_path = os.path.join(MODELS_DIR, "ppo_spy.zip")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")

    results = {}
    for split in ["train", "val", "test"]:
        print(f"\n{'─'*70}")
        print(f" {split.upper()} SET")
        print(f"{'─'*70}")
        stats = backtest_single("SPY", split, n_episodes, model_path, seed)
        results[split] = stats
        print_stats(stats)

    return results


def run_generalization(n_episodes: int = 3, seed: int = 0) -> dict:
    """Test on QQQ and IWM without retraining."""
    model_path = os.path.join(MODELS_DIR, "ppo_spy.zip")
    results = {}

    print("\n" + "="*70)
    print(" GENERALIZATION TEST (QQQ & IWM)")
    print("="*70)
    print("""
Policy trained only on SPY (2010–2021).
Now we test on unseen assets and regimes without any retraining.
    """)

    for ticker in ["QQQ", "IWM"]:
        print(f"\n{'─'*70}")
        print(f" {ticker} TEST SET (2024–2025)")
        print(f"{'─'*70}")
        stats = backtest_single(ticker, "test", n_episodes, model_path, seed)
        results[ticker] = stats
        print_stats(stats)

    return results


def print_comparison_table(splits_results: dict) -> None:
    """Print side-by-side comparison of train/val/test."""
    print("\n" + "="*70)
    print(" WALK-FORWARD COMPARISON TABLE")
    print("="*70 + "\n")

    rows = []
    for split in ["train", "val", "test"]:
        s = splits_results[split]
        rows.append([
            split.upper(),
            f"{s.agent_return_mean:+.2%}",
            f"{s.agent_sharpe_mean:+.3f}",
            f"{s.agent_dd_mean:.2%}",
            f"{s.bnh_return:+.2%}",
            f"{s.bnh_sharpe:+.3f}",
            f"{s.bnh_dd:.2%}",
            f"{s.sharpe_outperformance:+.3f}",
        ])

    headers = [
        "Split",
        "Agent Return",
        "Agent Sharpe",
        "Agent DD",
        "BnH Return",
        "BnH Sharpe",
        "BnH DD",
        "Sharpe Δ",
    ]
    print(tabulate(rows, headers=headers, tablefmt="grid"))
    print()


def print_generalization_table(gen_results: dict) -> None:
    """Print generalization test results."""
    print("\n" + "="*70)
    print(" GENERALIZATION TEST RESULTS")
    print("="*70 + "\n")

    rows = []
    for ticker in ["QQQ", "IWM"]:
        s = gen_results[ticker]
        rows.append([
            ticker,
            f"{s.agent_return_mean:+.2%}",
            f"{s.agent_sharpe_mean:+.3f}",
            f"{s.agent_dd_mean:.2%}",
            f"{s.sharpe_outperformance:+.3f}",
        ])

    headers = ["Ticker", "Return", "Sharpe", "Drawdown", "Sharpe Δ"]
    print(tabulate(rows, headers=headers, tablefmt="grid"))
    print()


def save_report(splits: dict, gen: dict | None = None) -> None:
    """Save results to JSON."""
    os.makedirs(BACKTEST_DIR, exist_ok=True)

    report = {
        "walk_forward": {
            split: {
                k: v for k, v in vars(stats).items()
                if not callable(v)
            }
            for split, stats in splits.items()
        }
    }

    if gen:
        report["generalization"] = {
            ticker: {
                k: v for k, v in vars(stats).items()
                if not callable(v)
            }
            for ticker, stats in gen.items()
        }

    out_path = os.path.join(BACKTEST_DIR, "walk_forward_report.json")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Report saved → {out_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Walk-forward validation")
    p.add_argument("--n-episodes", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--generalize", action="store_true",
                   help="Also test QQQ/IWM generalization")
    p.add_argument("--no-save", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    summarize_splits()
    splits_results = run_walk_forward(args.n_episodes, args.seed)
    print_comparison_table(splits_results)

    gen_results = None
    if args.generalize:
        gen_results = run_generalization(args.n_episodes, args.seed)
        print_generalization_table(gen_results)

    if not args.no_save:
        save_report(splits_results, gen_results)


if __name__ == "__main__":
    main()