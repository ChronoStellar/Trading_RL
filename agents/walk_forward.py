"""
agents/walk_forward.py
True walk-forward validation with expanding training windows.

For each fold the script trains a fresh PPO from scratch on all data
*before* the test year, then evaluates sequentially on the test year
(no random starts — one deterministic pass through the full year).

Fold structure (SPY data from 2010, features available from ~2010-10):
  Fold 1: Train 2010-10 → 2017-12,  Test 2018
  Fold 2: Train 2010-10 → 2018-12,  Test 2019
  …
  Fold N: Train 2010-10 → 2024-12,  Test 2025

Usage:
    python agents/walk_forward.py
    python agents/walk_forward.py --test-years 2022 2023 2024 2025
    python agents/walk_forward.py --timesteps 500000 --n-envs 4
"""

import argparse
import json
import os
import sys
from dataclasses import dataclass, asdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from data.features import compute_features, FEATURE_COLS, fit_scaler, apply_scaler
from env.rewards import sharpe_step_reward, TRANSACTION_COST, SLIPPAGE

RAW_DIR     = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "backtest_results")

INITIAL_CASH = 100_000.0
EPISODE_LEN  = 252


# ── Lightweight env for walk-forward ─────────────────────────────────────────

class _WFEnv(gym.Env):
    """
    Mirrors TradingEnv but accepts pre-normalised numpy arrays directly
    (no file I/O) and supports a *sequential* mode for evaluation where
    reset always starts at index 0 and the episode spans the full dataset.
    """

    metadata = {"render_modes": []}

    def __init__(self, features: np.ndarray, close: np.ndarray,
                 seed: int | None = None, sequential: bool = False):
        super().__init__()
        self._features = features.astype(np.float32)
        self._close    = close.astype(np.float32)
        self._n_rows   = len(features)
        self._sequential = sequential

        n_market = features.shape[1]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(n_market + 2,), dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=np.array([0.0], dtype=np.float32),
            high=np.array([1.0], dtype=np.float32),
            dtype=np.float32,
        )

        self._rng = np.random.default_rng(seed)
        self._start_idx     = 0
        self._current_step  = 0
        self._episode_len   = min(EPISODE_LEN, self._n_rows - 1)
        self._position      = 0.0
        self._portfolio_val = INITIAL_CASH
        self._ret_history: list[float] = []

    # ── Gymnasium API ─────────────────────────────────────────────────────────

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        if self._sequential:
            self._start_idx   = 0
            self._episode_len = self._n_rows - 1
        else:
            max_start = self._n_rows - EPISODE_LEN - 1
            if max_start <= 0:
                self._start_idx   = 0
                self._episode_len = self._n_rows - 1
            else:
                self._start_idx   = int(self._rng.integers(0, max_start + 1))
                self._episode_len = EPISODE_LEN

        self._current_step  = 0
        self._position      = 0.0
        self._portfolio_val = INITIAL_CASH

        lookback = min(20, self._start_idx)
        if lookback > 1:
            prices = self._close[self._start_idx - lookback : self._start_idx + 1]
            self._ret_history = list((prices[1:] / prices[:-1]) - 1.0)
        else:
            self._ret_history = []

        return self._get_obs(), {}

    def step(self, action: np.ndarray):
        action = np.clip(action, 0.0, 1.0)
        new_position   = float(action[0])
        position_delta = new_position - self._position

        idx           = self._start_idx + self._current_step
        current_price = float(self._close[idx])
        next_price    = float(self._close[idx + 1])

        price_return     = (next_price / current_price) - 1.0
        portfolio_return = self._position * price_return

        self._ret_history.append(price_return)
        if len(self._ret_history) > 20:
            self._ret_history.pop(0)
        rolling_vol = max(
            np.std(self._ret_history) if len(self._ret_history) > 1 else 0.01,
            0.003,
        )

        net_return = portfolio_return - (TRANSACTION_COST + SLIPPAGE) * abs(position_delta)
        reward = sharpe_step_reward(net_return, rolling_vol, position_delta)

        cost_frac = (TRANSACTION_COST + SLIPPAGE) * abs(position_delta)
        self._portfolio_val *= (1.0 + portfolio_return) * (1.0 - cost_frac)

        self._position      = new_position
        self._current_step += 1
        terminated = self._current_step >= self._episode_len

        obs  = self._get_obs()
        info = {
            "portfolio_value": self._portfolio_val,
            "position":        self._position,
            "price":           next_price,
        }
        return obs, reward, terminated, False, info

    def _get_obs(self) -> np.ndarray:
        idx     = self._start_idx + self._current_step
        market  = self._features[idx]
        eq_ret  = (self._portfolio_val / INITIAL_CASH) - 1.0
        return np.append(market, [self._position, eq_ret]).astype(np.float32)


# ── Metrics ──────────────────────────────────────────────────────────────────

def _sharpe(returns: np.ndarray) -> float:
    return float(np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252))

def _max_drawdown(values: np.ndarray) -> float:
    peak = np.maximum.accumulate(values)
    return float(((values - peak) / peak).min())

def _total_return(values: np.ndarray) -> float:
    return float(values[-1] / values[0] - 1.0)


# ── Fold result ──────────────────────────────────────────────────────────────

@dataclass
class FoldResult:
    test_year:          int
    train_rows:         int
    test_days:          int
    agent_return:       float
    agent_sharpe:       float
    agent_max_dd:       float
    agent_avg_position: float
    bnh_return:         float
    bnh_sharpe:         float
    bnh_max_dd:         float
    sharpe_delta:       float
    return_delta:       float
    dd_improvement:     float   # negative = agent had less drawdown


# ── Core logic ───────────────────────────────────────────────────────────────

def load_and_prepare(ticker: str = "SPY") -> pd.DataFrame:
    """Load raw OHLCV → compute all features (un-normalised)."""
    raw_path = os.path.join(RAW_DIR, f"{ticker}.csv")
    df = pd.read_csv(raw_path, parse_dates=["date"])
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")
    df = df.sort_values("date").reset_index(drop=True)
    df = compute_features(df)
    return df


def run_fold(
    df: pd.DataFrame,
    test_year: int,
    timesteps: int,
    n_envs: int,
    seed: int,
    device: str,
    ppo_kwargs: dict,
) -> FoldResult:
    """Train on data before *test_year*, evaluate on *test_year*."""

    train_end  = f"{test_year - 1}-12-31"
    test_start = f"{test_year}-01-01"
    test_end   = f"{test_year}-12-31"

    train_df = df[df["date"] <= train_end].copy()
    test_df  = df[(df["date"] >= test_start) & (df["date"] <= test_end)].copy()

    if len(test_df) < 20:
        raise ValueError(f"Test year {test_year} has only {len(test_df)} rows")
    if len(train_df) < EPISODE_LEN + 1:
        raise ValueError(
            f"Training data for fold {test_year} too small: {len(train_df)} rows"
        )

    # Per-fold scaler (fit on that fold's training data only)
    scaler     = fit_scaler(train_df)
    train_norm = apply_scaler(train_df, scaler)
    test_norm  = apply_scaler(test_df, scaler)

    train_features = train_norm[FEATURE_COLS].to_numpy(dtype=np.float32)
    train_close    = train_norm["close"].to_numpy(dtype=np.float32)
    test_features  = test_norm[FEATURE_COLS].to_numpy(dtype=np.float32)
    test_close     = test_norm["close"].to_numpy(dtype=np.float32)

    # ── Train ────────────────────────────────────────────────────────────────
    print(f"  Training on {len(train_df):,} rows (up to {train_end}) …")

    def make_env(s):
        def _init():
            return _WFEnv(train_features, train_close, seed=s)
        return _init

    vec_env = DummyVecEnv([make_env(seed + i) for i in range(n_envs)])

    model = PPO(
        "MlpPolicy",
        vec_env,
        seed=seed,
        device=device,
        verbose=0,
        **ppo_kwargs,
    )
    model.learn(total_timesteps=timesteps)
    vec_env.close()

    # ── Evaluate (single sequential pass through the test year) ──────────────
    eval_env = _WFEnv(test_features, test_close, seed=seed, sequential=True)
    obs, _ = eval_env.reset()

    portfolio_values = [INITIAL_CASH]
    positions: list[float] = []
    done = False

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, info = eval_env.step(action)
        done = terminated or truncated
        portfolio_values.append(info["portfolio_value"])
        positions.append(info["position"])

    portfolio_values = np.array(portfolio_values)
    daily_returns    = np.diff(portfolio_values) / portfolio_values[:-1]

    # ── Buy-and-hold benchmark (raw close, not z-scored) ─────────────────────
    bnh_prices = test_df["close"].to_numpy(dtype=np.float64)
    bnh_values = INITIAL_CASH * bnh_prices / bnh_prices[0]
    bnh_rets   = np.diff(bnh_values) / bnh_values[:-1]

    a_ret = _total_return(portfolio_values)
    a_sh  = _sharpe(daily_returns)
    a_dd  = _max_drawdown(portfolio_values)
    b_ret = _total_return(bnh_values)
    b_sh  = _sharpe(bnh_rets)
    b_dd  = _max_drawdown(bnh_values)

    result = FoldResult(
        test_year=test_year,
        train_rows=len(train_df),
        test_days=len(test_df),
        agent_return=a_ret,  agent_sharpe=a_sh,  agent_max_dd=a_dd,
        agent_avg_position=float(np.mean(positions)),
        bnh_return=b_ret,    bnh_sharpe=b_sh,    bnh_max_dd=b_dd,
        sharpe_delta=a_sh - b_sh,
        return_delta=a_ret - b_ret,
        dd_improvement=abs(a_dd) - abs(b_dd),
    )

    print(f"  Test {test_year}: agent {a_ret:+.2%}  sharpe {a_sh:+.3f}  "
          f"│ BnH {b_ret:+.2%}  sharpe {b_sh:+.3f}  "
          f"│ ΔSharpe {result.sharpe_delta:+.3f}")

    return result


# ── Reporting ────────────────────────────────────────────────────────────────

def print_report(folds: list[FoldResult]) -> None:
    try:
        from tabulate import tabulate
        has_tabulate = True
    except ImportError:
        has_tabulate = False

    print(f"\n{'='*80}")
    print("  WALK-FORWARD RESULTS  (expanding window, per-fold fresh PPO)")
    print(f"{'='*80}\n")

    headers = [
        "Year", "Train", "Test",
        "Agent Ret", "Agent Sharpe", "Agent DD", "Avg Pos",
        "BnH Ret", "BnH Sharpe", "BnH DD",
        "ΔSharpe",
    ]
    rows = []
    for f in folds:
        rows.append([
            f.test_year,
            f"{f.train_rows:,}",
            f.test_days,
            f"{f.agent_return:+.2%}",
            f"{f.agent_sharpe:+.3f}",
            f"{f.agent_max_dd:.2%}",
            f"{f.agent_avg_position:.2%}",
            f"{f.bnh_return:+.2%}",
            f"{f.bnh_sharpe:+.3f}",
            f"{f.bnh_max_dd:.2%}",
            f"{f.sharpe_delta:+.3f}",
        ])

    if has_tabulate:
        print(tabulate(rows, headers=headers, tablefmt="grid"))
    else:
        print("  ".join(f"{h:>12}" for h in headers))
        print("-" * (13 * len(headers)))
        for row in rows:
            print("  ".join(f"{c:>12}" for c in row))

    # Aggregate
    a_rets = np.array([f.agent_return for f in folds])
    a_shs  = np.array([f.agent_sharpe for f in folds])
    b_rets = np.array([f.bnh_return for f in folds])
    b_shs  = np.array([f.bnh_sharpe for f in folds])
    deltas = np.array([f.sharpe_delta for f in folds])

    print(f"\n{'─'*80}")
    print(f"  AGGREGATE ({len(folds)} folds)")
    print(f"{'─'*80}")
    print(f"  Agent return  {np.mean(a_rets):+.2%} ± {np.std(a_rets):.2%}")
    print(f"  Agent Sharpe  {np.mean(a_shs):+.3f} ± {np.std(a_shs):.3f}")
    print(f"  BnH return    {np.mean(b_rets):+.2%} ± {np.std(b_rets):.2%}")
    print(f"  BnH Sharpe    {np.mean(b_shs):+.3f} ± {np.std(b_shs):.3f}")
    print(f"  ΔSharpe       {np.mean(deltas):+.3f} ± {np.std(deltas):.3f}")
    print(f"  Folds where agent beats BnH Sharpe: "
          f"{int(np.sum(deltas > 0))}/{len(folds)}")
    print()


def save_report(folds: list[FoldResult], out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    report = {
        "folds": [asdict(f) for f in folds],
        "aggregate": {
            "n_folds":              len(folds),
            "agent_return_mean":    float(np.mean([f.agent_return for f in folds])),
            "agent_return_std":     float(np.std([f.agent_return for f in folds])),
            "agent_sharpe_mean":    float(np.mean([f.agent_sharpe for f in folds])),
            "agent_sharpe_std":     float(np.std([f.agent_sharpe for f in folds])),
            "bnh_return_mean":      float(np.mean([f.bnh_return for f in folds])),
            "bnh_sharpe_mean":      float(np.mean([f.bnh_sharpe for f in folds])),
            "sharpe_delta_mean":    float(np.mean([f.sharpe_delta for f in folds])),
            "sharpe_delta_std":     float(np.std([f.sharpe_delta for f in folds])),
            "folds_agent_wins":     int(np.sum([f.sharpe_delta > 0 for f in folds])),
        },
    }
    out_path = os.path.join(out_dir, "walk_forward_report.json")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Report saved → {out_path}")
    return out_path


# ── Plotting ─────────────────────────────────────────────────────────────────

def plot_results(folds: list[FoldResult], out_dir: str) -> str | None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    years  = [f.test_year for f in folds]
    a_rets = [f.agent_return * 100 for f in folds]
    b_rets = [f.bnh_return * 100 for f in folds]
    a_shs  = [f.agent_sharpe for f in folds]
    b_shs  = [f.bnh_sharpe for f in folds]
    a_dds  = [f.agent_max_dd * 100 for f in folds]
    b_dds  = [f.bnh_max_dd * 100 for f in folds]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Walk-Forward Validation (expanding window)", fontsize=13)

    x = np.arange(len(years))
    w = 0.35

    # Returns
    ax = axes[0]
    ax.bar(x - w/2, a_rets, w, label="Agent", color="steelblue")
    ax.bar(x + w/2, b_rets, w, label="Buy & Hold", color="gray", alpha=0.7)
    ax.set_xticks(x); ax.set_xticklabels(years, rotation=45)
    ax.set_ylabel("Return (%)"); ax.set_title("Annual Return")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3, axis="y")
    ax.axhline(0, color="black", linewidth=0.5)

    # Sharpe
    ax = axes[1]
    ax.bar(x - w/2, a_shs, w, label="Agent", color="steelblue")
    ax.bar(x + w/2, b_shs, w, label="Buy & Hold", color="gray", alpha=0.7)
    ax.set_xticks(x); ax.set_xticklabels(years, rotation=45)
    ax.set_ylabel("Sharpe"); ax.set_title("Annualised Sharpe Ratio")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3, axis="y")
    ax.axhline(0, color="black", linewidth=0.5)

    # Max Drawdown
    ax = axes[2]
    ax.bar(x - w/2, a_dds, w, label="Agent", color="steelblue")
    ax.bar(x + w/2, b_dds, w, label="Buy & Hold", color="gray", alpha=0.7)
    ax.set_xticks(x); ax.set_xticklabels(years, rotation=45)
    ax.set_ylabel("Max Drawdown (%)"); ax.set_title("Max Drawdown")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    out_path = os.path.join(out_dir, "walk_forward.png")
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ── CLI ──────────────────────────────────────────────────────────────────────

DEFAULT_TEST_YEARS = list(range(2018, 2026))   # 2018 … 2025

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Walk-forward validation (expanding window, fresh PPO per fold)",
    )
    p.add_argument("--ticker",       default="SPY")
    p.add_argument("--test-years",   type=int, nargs="+", default=DEFAULT_TEST_YEARS,
                   help="Which years to use as test folds (default: 2018–2025)")
    p.add_argument("--timesteps",    type=int, default=500_000,
                   help="PPO training timesteps per fold")
    p.add_argument("--n-envs",       type=int, default=4)
    p.add_argument("--seed",         type=int, default=42)
    p.add_argument("--device",       default="cpu", choices=["cpu", "cuda", "mps"])
    # PPO hyperparameters (same defaults as agents/train.py)
    p.add_argument("--n-steps",       type=int,   default=2048)
    p.add_argument("--batch-size",    type=int,   default=512)
    p.add_argument("--n-epochs",      type=int,   default=20)
    p.add_argument("--gamma",         type=float, default=0.995)
    p.add_argument("--gae-lambda",    type=float, default=0.95)
    p.add_argument("--ent-coef",      type=float, default=0.005)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--no-save",       action="store_true")
    p.add_argument("--no-plot",       action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    ppo_kwargs = dict(
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        ent_coef=args.ent_coef,
        learning_rate=args.learning_rate,
    )

    print(f"\n{'='*80}")
    print("  WALK-FORWARD VALIDATION")
    print(f"{'='*80}")
    print(f"  Ticker:       {args.ticker}")
    print(f"  Test years:   {args.test_years}")
    print(f"  Timesteps:    {args.timesteps:,} per fold")
    print(f"  Envs:         {args.n_envs}")
    print(f"  Device:       {args.device}")
    print()

    df = load_and_prepare(args.ticker)
    print(f"  Loaded {len(df):,} rows ({df['date'].iloc[0]} → {df['date'].iloc[-1]})")
    print()

    folds: list[FoldResult] = []
    for year in sorted(args.test_years):
        print(f"{'─'*80}")
        print(f"  FOLD: test year {year}")
        print(f"{'─'*80}")
        try:
            result = run_fold(
                df, year, args.timesteps, args.n_envs,
                args.seed, args.device, ppo_kwargs,
            )
            folds.append(result)
        except (ValueError, KeyboardInterrupt) as e:
            if isinstance(e, KeyboardInterrupt):
                print("\n  Interrupted — reporting results so far.")
                break
            print(f"  Skipping {year}: {e}")

    if not folds:
        print("No folds completed.")
        return

    print_report(folds)

    if not args.no_save:
        save_report(folds, RESULTS_DIR)

    if not args.no_plot:
        plot_path = plot_results(folds, RESULTS_DIR)
        if plot_path:
            print(f"Plot saved → {plot_path}")


if __name__ == "__main__":
    main()
