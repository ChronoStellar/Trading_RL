# Trading RL

A PPO reinforcement-learning agent that learns continuous, long-only position sizing on equity ETFs, exported to CoreML and deployed in a SwiftUI iOS app for paper trading.

## Goal

**Match SPY return, beat SPY Sharpe by going to cash during drawdowns.** Not "outperform on raw return" — that is the wrong target for a retail-scale RL agent. The realistic alpha is *avoiding the worst 20% of bear days*, not predicting tops. Long-only is auditable; long-short is not, given current scope.

## Architecture

```
yfinance OHLCV ─→ data/features.py ─→ data/processed/{train,val,test}.csv
                                              │
                                              ▼
                                       env/TradingEnv (Gymnasium)
                                              │
                                              ▼
                                  agents/train.py (PPO via SB3)
                                              │
                              ┌───────────────┴───────────────┐
                              ▼                               ▼
                    agents/eval.py + backtest.py      export/coreml_export.py
                                                              │
                                                              ▼
                                              paper_trade/PaperTrader (SwiftUI + CoreML)
```

## Directory Layout

```
data/                — pipeline (download, features, normalization, scaler.json)
env/                 — Gymnasium environment (TradingEnv) + reward function
agents/              — PPO training, eval, backtest, walk-forward, hyperparameter tuning
export/              — CoreML conversion (PPO policy → TradingActor.mlpackage)
paper_trade/         — iOS SwiftUI app (PaperTrader)
notebook/            — diagnostic notebooks (model probing, real-data backtest)
docs/                — PRD and design docs
script/              — shell wrappers for the pipeline
```

## Observation Vector (10-d)

Eight market features (z-scored using training-split statistics) plus two env-injected channels:

| # | Feature        | Description                                        |
|---|----------------|----------------------------------------------------|
| 1 | `sma_ratio`    | close / SMA-20                                     |
| 2 | `rsi_14`       | RSI (14-period, Wilder EWM)                        |
| 3 | `macd_hist`    | MACD histogram, normalized by close                |
| 4 | `obv_ret`      | On-Balance Volume 1-day % change                   |
| 5 | `adx`          | Average Directional Index (14)                     |
| 6 | `drawdown_60d` | close / rolling-60d-max − 1 (regime / drawdown)    |
| 7 | `vol_regime`   | vol_20d / vol_60d (>1 = expanding, <1 = contracting) |
| 8 | `ret_20d`      | 20-day return                                      |
| 9 | `position`     | current allocation [0, 1] (env-injected)           |
| 10| `equity_return`| (portfolio / initial_cash) − 1 (env-injected)      |

## Action Space

A single continuous value in **[0, 1]** — fraction allocated to the asset. `0` = all cash, `1` = fully long.

**Why long-only?** Shorting has asymmetric payoff (capped gain, uncapped loss), borrow recall risk that the simulator can't model, and fights SPY's ~9%/yr structural drift. Revisit only after a long-only agent reliably beats SPY Sharpe over a multi-year out-of-sample window.

## Reward Function

Per-step Sharpe contribution with internal turnover penalty (see [env/rewards.py](env/rewards.py)):

```
reward = (portfolio_return − rf) / rolling_vol  −  (cost + slippage) × |Δposition| / rolling_vol
```

The penalty is divided by `rolling_vol` so it lives in the same Sharpe units as the signal — a large rebalance during a calm market is penalized more.

A planned shift to a benchmark-relative reward (`(portfolio_return − spy_return) / tracking_error`) is on the roadmap; this directly rewards going to cash on a −3% SPY day.

## Environment Realism

- Episode length: 252 trading days, random start within the split.
- Vol estimate: rolling 20-day std of *position-scaled* returns, floored at 0.3% daily — closes the "hide in cash for 20 days, then strike" exploit.
- Transaction cost: 0.001 of trade size. Slippage: 5 bps. Both deducted from portfolio value as `(cost + slippage) × |Δposition|`.
- Train/val/test splits avoid lookahead — features computed only from data available at time `t`.

## Setup

```bash
# 1. Create env + install
conda create -n trading-ppo python=3.10 -y
conda activate trading-ppo
pip install -r requirements.txt

# 2. Build the dataset (downloads SPY/QQQ/IWM, computes features, fits scaler)
bash script/features.sh        # runs data/loader.py + data/features.py

# 3. Train, evaluate, export
bash script/run.sh             # train + eval + backtest + tune + walk-forward + coreml
```

Or run individual stages:

```bash
python data/loader.py                 # download raw OHLCV → data/raw/
python data/features.py               # compute features, fit scaler → data/processed/

python agents/train.py                # PPO training
python agents/eval.py --split test    # single-split evaluation
python agents/backtest.py             # train + val + test backtest
python agents/walk_forward.py         # walk-forward folds (no test-set leakage)
python agents/tune.py --trials 50 --timesteps 200000   # Optuna hyperparameter search

python export/coreml_export.py        # PPO policy → export/TradingActor.mlpackage + scaler.json
```

## Diagnostics

[notebook/diagnostic.ipynb](notebook/diagnostic.ipynb) probes the trained model two ways:

1. **Synthetic obs** — feeds zeroed market features with various `(position, equity_return)` pairs to compare CoreML vs PyTorch outputs and spot input-leak sensitivity.
2. **Real-data backtest** — fetches live SPY via yfinance up to today, runs the full pipeline (`compute_features` → scaler → sequential rollout), and plots portfolio value, allocation, and drawdown against buy-and-hold over the entire out-of-sample window.

## iOS App

The SwiftUI app ([paper_trade/PaperTrader](paper_trade/PaperTrader)) runs the exported policy on-device.

### Setup

1. Add **SwiftYFinance** as a Swift Package: `https://github.com/AlexRoar/SwiftYFinance`
2. Drag `export/TradingActor.mlpackage` into the Xcode target (replace existing after each retrain).
3. Add `export/scaler.json` to the Xcode target bundle (resource).
4. Build and run.

### Architecture

```
ContentView (TabView)
├── LiveSignalView          — fetches live bars, runs CoreML inference, shows allocation
│   ├── SearchBarRow
│   ├── BalanceCard
│   ├── QuickTickerRow
│   ├── StockChartCard
│   └── LiveSignalSheet     — detailed AI signal + apply allocation
└── SimulationView          — replay agent decisions over a historical window
    └── SimulationDetailView
```

Key Swift files:
- `Model/FeatureEngine.swift` — mirrors `data/features.py` for live feature computation. Requires ≥21 trading bars (~30 calendar days) to compute all rolling features.
- `View Model/TraderViewModel.swift` — fetches via `SwiftYFinance`, normalizes, runs CoreML, exposes `@Published` state.
- `View/*.swift` — modular cards composed in [ContentView.swift](paper_trade/PaperTrader/PaperTrader/ContentView.swift).

## Key Design Decisions

- **Long-only by design.** Do not re-enable shorts without an explicit revisit and an out-of-sample win.
- **Per-ticker normalization fit on training-split only** — done in `data/features.py`, not at runtime, so the Swift app and Python eval see identical scaled features.
- **Scaler must match training.** Re-running `data/features.py` regenerates `scaler.json`; copy it to the Xcode bundle and to `export/` after every retrain.
- **Diagnostics over complexity.** If the agent collapses to a constant policy, it's a reward / exploration problem. If it varies but underperforms, it's a feature / data problem. Check action entropy before adding network capacity.
- **No hyperparameter tuning on the test set.** Tune on val; touch test once at the end.

## Roadmap

1. **EDA aimed at the goal** — per-feature rank-IC vs next-day return, conditional on regime (bull / bear / sideways via 200-day MA). Reveals data ceiling before training.
2. **Extend data** — SPY back to 1993; sample 5–10 ETFs per episode (QQQ, IWM, sector SPDRs, TLT, GLD).
3. **Add macro features** — VIX level + 1-week change, term-structure slope (10Y − 2Y), credit spreads (HYG / LQD), 200-day MA distance. These are what call bear markets; asset-internal features alone won't.
4. **Switch to benchmark-relative reward** — `(portfolio_return − spy_return) / tracking_error`.
5. **Retrain**, then verify action entropy and per-regime performance.

## Status

Proof of concept. See [docs/prd.md](docs/prd.md) for the original 2-week scope and success criteria. Current state: training pipeline, eval suite, walk-forward, CoreML export, and iOS app all functional; roadmap items 1–4 outstanding.
