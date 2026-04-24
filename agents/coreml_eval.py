"""
agents/coreml_eval.py
Two-part evaluation of the exported TradingActor.mlpackage:

  A. Out-of-sample backtest — run ppo_spy on SPY data the model has never seen
     (default: 2026-01-01 → 2026-04-22, after the 2021 train cut-off).

  B. Quantization drift — compare fp32 PyTorch vs fp16 CoreML output at every
     step to detect whether the mlpackage conversion degrades the policy.

Usage:
    conda run -n trading-ppo python agents/coreml_eval.py
    conda run -n trading-ppo python agents/coreml_eval.py --start 2026-01-01 --end 2026-04-22
    conda run -n trading-ppo python agents/coreml_eval.py --no-plot
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import yfinance as yf
from stable_baselines3 import PPO

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data.features import compute_features, FEATURE_COLS
from env.rewards import TRANSACTION_COST, SLIPPAGE
from env.train_env import INITIAL_CASH

EXPORT_DIR  = os.path.join(os.path.dirname(__file__), "..", "export")
MODELS_DIR  = os.path.join(os.path.dirname(__file__), "models")
SCALER_PATH = os.path.join(EXPORT_DIR, "scaler.json")
MLPACKAGE   = os.path.join(EXPORT_DIR, "TradingActor.mlpackage")

# Features need up to 61 bars of history; pull extra calendar days as buffer.
WARMUP_CAL_DAYS = 120


# ── Data ──────────────────────────────────────────────────────────────────────

def _load_scaler() -> dict:
    with open(SCALER_PATH) as f:
        return json.load(f)


def _apply_scaler(df: pd.DataFrame, scaler: dict) -> pd.DataFrame:
    df = df.copy()
    for col in FEATURE_COLS:
        s = scaler.get(col)
        if s:
            df[col] = (df[col] - s["mean"]) / (s["std"] if s["std"] > 0 else 1.0)
    return df


def fetch_test_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    """
    Download OHLCV, compute features, apply the training scaler, and return
    only bars in [start, end] — all warmup rows are stripped after feature
    computation so every returned row is fully valid.
    """
    fetch_start = (
        pd.Timestamp(start) - pd.Timedelta(days=WARMUP_CAL_DAYS)
    ).strftime("%Y-%m-%d")

    print(f"Downloading {ticker}  {fetch_start} → {end} …")
    raw = yf.download(ticker, start=fetch_start, end=end,
                      auto_adjust=True, progress=False)
    if raw.empty:
        sys.exit(f"yfinance returned no data for {ticker}")

    # yfinance sometimes returns MultiIndex columns
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    df = (
        raw.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]]
        .copy()
    )
    df.index.name = "date"
    df = df.reset_index()
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")

    df = compute_features(df)          # drops warmup NaN rows internally
    df = _apply_scaler(df, _load_scaler())

    test_df = df[df["date"] >= start].reset_index(drop=True)
    if test_df.empty:
        sys.exit(f"No rows remain after filtering to {start}+. Check warmup window.")

    print(
        f"Test window : {test_df['date'].iloc[0]} → {test_df['date'].iloc[-1]}"
        f"  ({len(test_df)} bars)"
    )
    return test_df


# ── Shared helpers ─────────────────────────────────────────────────────────────

def make_obs(feat_row: np.ndarray, position: float, equity_return: float) -> np.ndarray:
    return np.concatenate([feat_row, [position, equity_return]]).astype(np.float32)


def _sharpe(r: np.ndarray) -> float:
    return float(np.mean(r) / (np.std(r) + 1e-8) * np.sqrt(252))


def _mdd(vals: np.ndarray) -> float:
    peak = np.maximum.accumulate(vals)
    return float(((vals - peak) / peak).min())


# ── A: Out-of-sample backtest ─────────────────────────────────────────────────

def run_backtest(model: PPO, test_df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Deterministic episode starting at bar 0, stepping through every available
    bar.  Returns the trajectory DataFrame and the buy-and-hold value curve.
    """
    features = test_df[FEATURE_COLS].to_numpy(dtype=np.float32)
    closes   = test_df["close"].to_numpy(dtype=np.float64)

    portfolio_val = INITIAL_CASH
    position      = 0.0
    records       = []

    for t in range(len(test_df) - 1):
        equity_return = portfolio_val / INITIAL_CASH - 1.0
        obs           = make_obs(features[t], position, equity_return)

        action, _ = model.predict(obs, deterministic=True)
        new_pos    = float(np.clip(action.flatten()[0], 0.0, 1.0))

        price_ret    = closes[t + 1] / closes[t] - 1.0
        cost         = (TRANSACTION_COST + SLIPPAGE) * abs(new_pos - position)
        portfolio_val *= (1.0 + position * price_ret) * (1.0 - cost)

        records.append({
            "date":            test_df["date"].iloc[t + 1],
            "close":           closes[t + 1],
            "position":        new_pos,
            "portfolio_value": portfolio_val,
        })
        position = new_pos

    traj    = pd.DataFrame(records)
    bnh_pv  = INITIAL_CASH * closes[1:] / closes[0]   # aligned to trajectory steps
    return traj, bnh_pv


def print_backtest_metrics(traj: pd.DataFrame, bnh_pv: np.ndarray) -> None:
    pv         = traj["portfolio_value"].to_numpy()
    agent_rets = np.diff(pv) / pv[:-1]
    bnh_rets   = np.diff(bnh_pv) / bnh_pv[:-1]

    metrics = {
        "agent": {
            "total_return": pv[-1] / INITIAL_CASH - 1,
            "sharpe":       _sharpe(agent_rets),
            "max_drawdown": _mdd(pv),
            "avg_position": float(traj["position"].mean()),
            "final_value":  float(pv[-1]),
        },
        "buy_and_hold": {
            "total_return": bnh_pv[-1] / INITIAL_CASH - 1,
            "sharpe":       _sharpe(bnh_rets),
            "max_drawdown": _mdd(bnh_pv),
            "avg_position": 1.0,
            "final_value":  float(bnh_pv[-1]),
        },
    }

    a, b = metrics["agent"], metrics["buy_and_hold"]
    print(f"\n{'Metric':<22} {'PPO Agent':>12} {'Buy & Hold':>12}")
    print(f"{'─'*48}")
    print(f"{'Total Return':<22} {a['total_return']:>+11.2%}  {b['total_return']:>+11.2%}")
    print(f"{'Sharpe Ratio':<22} {a['sharpe']:>+11.3f}  {b['sharpe']:>+11.3f}")
    print(f"{'Max Drawdown':<22} {a['max_drawdown']:>+11.2%}  {b['max_drawdown']:>+11.2%}")
    print(f"{'Avg Position':<22} {a['avg_position']:>+11.2%}  {b['avg_position']:>+11.2%}")
    print(f"{'Final Value ($)':<22} {a['final_value']:>11,.2f}  {b['final_value']:>11,.2f}")
    print(f"\n  Return outperformance : {a['total_return'] - b['total_return']:>+.2%}")
    print(f"  Sharpe outperformance : {a['sharpe'] - b['sharpe']:>+.3f}")
    print(f"  Drawdown improvement  : {abs(b['max_drawdown']) - abs(a['max_drawdown']):>+.2%}")


# ── B: Quantization drift ─────────────────────────────────────────────────────

def compare_quantization(model: PPO, test_df: pd.DataFrame) -> dict | None:
    try:
        import coremltools as ct
    except ImportError:
        print("  coremltools not installed — skipping. Install with: pip install coremltools")
        return None

    if not os.path.exists(MLPACKAGE):
        print(f"  CoreML model not found at {MLPACKAGE}")
        return None

    ml_model = ct.models.MLModel(MLPACKAGE)
    features  = test_df[FEATURE_COLS].to_numpy(dtype=np.float32)
    n         = len(features)

    pt_outs, ml_outs = [], []

    for t in range(n):
        # position=0 / equity_return=0 so we isolate the model path only
        obs = make_obs(features[t], 0.0, 0.0)

        # PyTorch fp32
        action_pt, _ = model.predict(obs, deterministic=True)
        pt_outs.append(float(np.clip(action_pt.flatten()[0], 0.0, 1.0)))

        # CoreML fp16 weights
        ml_out = ml_model.predict({"observation": obs.reshape(1, -1)})
        ml_outs.append(float(np.clip(ml_out["allocation"].flatten()[0], 0.0, 1.0)))

    pt_arr = np.array(pt_outs)
    ml_arr = np.array(ml_outs)
    delta  = np.abs(pt_arr - ml_arr)
    corr   = float(np.corrcoef(pt_arr, ml_arr)[0, 1]) if pt_arr.std() > 0 else 1.0

    results = {
        "n_steps":       n,
        "max_delta":     round(float(delta.max()), 6),
        "mean_delta":    round(float(delta.mean()), 6),
        "std_delta":     round(float(delta.std()), 6),
        "correlation":   round(corr, 6),
        "pt_mean_alloc": round(float(pt_arr.mean()), 4),
        "ml_mean_alloc": round(float(ml_arr.mean()), 4),
    }

    print(f"\n  Steps compared   : {results['n_steps']}")
    print(f"  Max  |Δ|         : {results['max_delta']:.6f}")
    print(f"  Mean |Δ|         : {results['mean_delta']:.6f}")
    print(f"  Std  |Δ|         : {results['std_delta']:.6f}")
    print(f"  Correlation      : {results['correlation']:.6f}")
    print(f"  PT  mean alloc   : {results['pt_mean_alloc']:.4f}")
    print(f"  ML  mean alloc   : {results['ml_mean_alloc']:.4f}")

    THRESHOLD = 0.01
    if results["max_delta"] < THRESHOLD:
        print(f"\n  ✓  Quantization OK — max delta {results['max_delta']:.4f} < {THRESHOLD}")
    else:
        print(f"\n  ✗  Quantization DEGRADES model — max delta {results['max_delta']:.4f} >= {THRESHOLD}")

    return results


# ── Plot ──────────────────────────────────────────────────────────────────────

def plot(traj: pd.DataFrame, bnh_pv: np.ndarray, start: str, end: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed — skipping plot.")
        return

    steps = range(len(traj))

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(12, 7), sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )

    ax1.plot(steps, traj["portfolio_value"], label="PPO Agent", linewidth=1.5)
    ax1.plot(steps, bnh_pv, label="Buy & Hold", linestyle="--", linewidth=1.5)
    ax1.axhline(INITIAL_CASH, color="gray", linestyle=":", alpha=0.5, label="Initial $100k")
    ax1.set_ylabel("Portfolio Value ($)")
    ax1.set_title(f"ppo_spy on SPY {start} → {end}  (unseen data)")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.fill_between(steps, traj["position"], alpha=0.5, color="steelblue")
    ax2.set_ylabel("Allocation")
    ax2.set_xlabel("Step (trading days)")
    ax2.set_ylim(0, 1)
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()


# ── Entry point ───────────────────────────────────────────────────────────────

def run(args: argparse.Namespace) -> None:
    model_path = os.path.join(MODELS_DIR, "ppo_spy.zip")
    if not os.path.exists(model_path):
        sys.exit(f"Model not found: {model_path}\nRun agents/train.py first.")

    print(f"Loading PyTorch model: {model_path}")
    model = PPO.load(model_path)

    test_df = fetch_test_data("SPY", args.start, args.end)

    # ── A ────────────────────────────────────────────────────────────────────
    print(f"\n{'='*52}")
    print("  A — Out-of-Sample Backtest (unseen 2026 data)")
    print(f"{'='*52}")
    traj, bnh_pv = run_backtest(model, test_df)
    print_backtest_metrics(traj, bnh_pv)

    # ── B ────────────────────────────────────────────────────────────────────
    print(f"\n{'='*52}")
    print("  B — CoreML Quantization Drift  (fp32 PT vs fp16 ML)")
    print(f"{'='*52}")
    compare_quantization(model, test_df)

    if not args.no_plot:
        plot(traj, bnh_pv, args.start, args.end)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CoreML + out-of-sample evaluation")
    p.add_argument("--start",   default="2026-01-01", help="Test window start (inclusive)")
    p.add_argument("--end",     default="2026-04-22", help="Test window end (inclusive)")
    p.add_argument("--no-plot", action="store_true",  help="Skip the matplotlib chart")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
