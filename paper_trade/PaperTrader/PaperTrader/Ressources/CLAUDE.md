THIS IS THE IOS PORTION OF THE APP

# Trading RL — Claude Guide

## Project Overview
PPO RL agent for paper trading equity ETFs. Trains on historical OHLCV data, exports to CoreML, and runs inference in an iOS SwiftUI app. **Long-only** continuous allocation with realistic execution modeling.

## Goal
**Match SPY return, beat SPY Sharpe by going to cash during drawdowns.** Not "outperform on raw return." The realistic alpha for a retail-scale RL agent is *avoiding the worst 20% of bear days*, not predicting tops. Long-only is auditable; long-short is not, given current scope.

## Directory Structure
```
data/               — data pipeline (download, features, normalization)
env/                — custom Gym environment (TradingEnv)
agents/             — PPO training + eval + backtest scripts
export/             — CoreML conversion
paper_trade/        — iOS SwiftUI app (PaperTrader)
  PaperTrader/
    FeatureEngine.swift    — mirrors data/features.py for live feature computation
    TraderViewModel.swift  — fetches SPY via SwiftYFinance, runs CoreML inference
    ContentView.swift      — SwiftUI UI
    TradingActor.mlpackage — exported CoreML model (drag from export/ after conversion)
    scaler.json            — z-score params (copy from data/processed/scaler.json)
```

## Data Strategy
- **History**: SPY from 1993 (yfinance supports it). 2010–2025 alone is too bull-heavy — only ~3 meaningful drawdown examples, insufficient for the agent to learn risk-off behavior.
- **Multi-ticker training**: sample per-episode from SPY, QQQ, IWM, and sector SPDRs (XLF, XLE, XLK), plus TLT/GLD for regime diversity. Same env, ticker chosen at `reset()`. Ticker identity is hidden from the policy — generalization comes from features, not from memorizing assets.
- **Per-ticker normalization**: z-score each ticker's features using *that ticker's train-split* statistics. Done in `data/features.py`, not at runtime.
- **Split**: hold out a contiguous bear regime as test (e.g. 2022 or 2008), not random days. Random splits leak because adjacent days are correlated.

## Observation Vector (17-d)
Features 1–15 are z-scored using `scaler.json`. Features 16–17 are injected by the env / app.

**Asset-derived (technical):**
1. `ret_1d`       — 1-day return
2. `ret_5d`       — 5-day return
3. `sma_ratio`    — close / SMA-20
4. `vol_20d`      — rolling 20-day return std-dev
5. `vol_ratio`    — volume / avg_volume_20d
6. `rsi_14`       — RSI (14-period, Wilder EWM)
7. `macd_hist`    — MACD histogram / close
8. `stoch_k`      — Stochastic %K (14)
9. `stoch_d`      — Stochastic %D signal
10. `bb_width`    — Bollinger Band width
11. `bb_pct`      — price position in Bollinger Bands
12. `obv_ret`     — OBV 1-day % change
13. `adx`         — ADX (14)
14. `adx_di_diff` — (+DI − −DI) / 100
15. `psar_bull`   — binary: 1 if Parabolic SAR uptrend

**Env-injected:**
16. `position`        — current allocation [0, 1]
17. `equity_return`   — (portfolio_value / INITIAL_CASH) − 1

### Macro features to add (priority for regime detection)
The current 15 are all asset-internal — they capture momentum but not macro regime. To learn drawdown avoidance, add:
- VIX level + 1-week change
- Term-structure slope (10Y − 2Y Treasury yield)
- Credit spreads (HYG / LQD ratio)
- 200-day MA distance (price / SMA-200 − 1)

These are what call bear markets. Asset-internal features alone won't.

## Action Space
Single continuous value in **[0, 1]** (long-only):
- `0` = all cash
- `1` = fully long

**Why not [-1, 1]?** Shorting has asymmetric payoff (capped gain, uncapped loss), borrow recall/squeeze risk that the sim can't model, and fights SPY's ~9%/yr structural drift. Revisit only after a long-only agent reliably beats SPY Sharpe over a multi-year out-of-sample window.

## Reward Function
Defined in `env/rewards.py`. Current: per-step Sharpe with internal turnover penalty (`sharpe_step_reward(gross_return, rolling_vol, position_delta)`).

**Planned shift to benchmark-relative reward:**
```
reward = (portfolio_return - spy_return) / tracking_error
```
- "Always long" yields ~0 gross reward (matches SPY) and negative net (costs). Forces the agent to find timing alpha.
- Going to cash during a -3% SPY day = +3% reward. Direct signal for drawdown avoidance.

Per-step Sharpe stays as a small auxiliary term, not the primary objective.

## Environment Realism (`env/train_env.py`)
- **Episode**: 252 trading days, random start within the split.
- **Vol estimate**: rolling 20-day std of *position-scaled* returns (`position × price_return`), floored at 0.3% daily. Position-scaling closes the "hide in cash for 20 days then strike" exploit by making the denominator scale with the agent's actual risk-taking.
- **Cost accounting**: `sharpe_step_reward` charges turnover internally (in Sharpe units); equity curve charges costs separately (in return units) for accurate P&L. Pass *gross* return to the reward — netting it would double-charge.
- **Transaction cost + slippage**: deducted from portfolio value as `(TRANSACTION_COST + SLIPPAGE) × |Δposition|`.

## Python Environment Setup
```bash
conda activate trading-ppo
pip install stable-baselines3 yfinance coremltools gymnasium

# Data pipeline
python data/loader.py
python data/features.py

# Train
python agents/train.py

# Evaluate
python agents/eval.py --split test

# Backtest
python agents/backtest.py

# Export to CoreML  (produces export/TradingActor.mlpackage + export/scaler.json)
conda run -n trading-ppo python export/coreml_export.py
```

## iOS App Setup (Xcode)
1. Add **SwiftYFinance** Swift Package: `https://github.com/AlexRoar/SwiftYFinance`
2. Drag `export/TradingActor.mlpackage` into the Xcode target (replace the existing one after retraining)
3. Add `export/scaler.json` to the Xcode target bundle
4. Build & run — the app fetches 40 days of SPY bars on launch, computes the observation, and shows the recommended allocation

## Key Design Decisions
- **Long-only by design.** See "Action Space" rationale above. Do not re-enable shorts without an explicit revisit.
- **Single inference per session**: the model outputs one allocation; the user taps "Apply" to record it as the active position.
- **Scaler must match training**: re-running `data/features.py` regenerates `scaler.json`; always copy it to the Xcode bundle and to `export/` after retraining.
- **Minimum bars**: `FeatureEngine` requires 21 trading bars (≈30 calendar days) to compute all rolling features.
- **Diagnostics over complexity**: if the agent collapses to a constant policy, it's a reward/exploration problem. If it varies but underperforms, it's a feature/data problem. Check action entropy before adding complexity.

## Roadmap (current)
1. **EDA aimed at the goal**: per-feature rank-IC vs next-day return, *conditional on regime* (bull/bear/sideways via 200-day MA). Reveals data ceiling before training.
2. **Extend data**: SPY back to 1993; add 5–10 ETFs sampled per episode.
3. **Add macro features**: VIX, term-structure, credit spreads, 200-day MA distance.
4. **Switch to benchmark-relative reward**.
5. **Retrain**, then check action entropy and per-regime performance.

