# Trading RL — Claude Guide

## Project Overview
PPO RL agent for SPY paper trading. Trains on historical OHLCV data, exports to CoreML, and runs inference in an iOS SwiftUI app.

## Directory Structure
```
Data/               — data pipeline (download, features, normalization)
Env/                — custom Gym environment (TradingEnv)
Agents/             — PPO training + eval + backtest scripts
Export/             — CoreML conversion
paper_trade/        — iOS SwiftUI app (PaperTrader)
  PaperTrader/
    FeatureEngine.swift    — mirrors data/features.py for live feature computation
    TraderViewModel.swift  — fetches SPY via SwiftYFinance, runs CoreML inference
    ContentView.swift      — SwiftUI UI
    TradingActor.mlpackage — exported CoreML model (drag from export/ after conversion)
    scaler.json            — z-score params (copy from data/processed/scaler.json)
```

## Observation Vector (8-d)
Features 1–6 are z-scored using `scaler.json`. Features 7–8 are injected by the env / app.
1. `ret_1d`    — 1-day return
2. `ret_5d`    — 5-day return
3. `rsi_14`    — RSI (14-period, Wilder EWM)
4. `sma_ratio` — close / SMA-20
5. `vol_20d`   — rolling 20-day return std-dev
6. `vol_ratio` — volume / avg_volume_20d
7. `position`  — current allocation [0, 1]
8. `unrealised_pnl` — (current_price / entry_price) - 1

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
- **Single inference per session**: the model outputs one allocation; the user taps "Apply" to record it as the active position
- **Scaler must match training**: re-running `data/features.py` regenerates `scaler.json`; always copy it to the Xcode bundle and to `export/` after retraining
- **Minimum bars**: `FeatureEngine` requires 21 trading bars (≈30 calendar days) to compute all rolling features
- **Action = allocation fraction**: 0 = all cash, 1 = fully invested in SPY — not a buy/sell signal
