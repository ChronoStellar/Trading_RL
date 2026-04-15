# Paper Trading RL — Product Requirements Document
**Project Type:** Proof of Concept  
**Timeline:** 2 weeks  
**Primary Showcase:** Reinforcement Learning (PPO), Tabular Data Pipeline  
**Deployment:** On-device iOS app (Swift + CoreML)

---

## 1. Project Summary

A single-asset reinforcement learning agent trained to manage continuous position sizing on SPY (S&P 500 ETF) using daily OHLCV bars. The agent learns a risk-adjusted trading policy evaluated against a buy-and-hold baseline. The trained policy is exported to CoreML and embedded in a Swift iOS app for paper trading simulation.

---

## 2. Goals & Non-Goals

### Goals
- Train a PPO agent with a defensible reward function (Sharpe-ratio based)
- Demonstrate walk-forward evaluation methodology
- Export policy network to CoreML for on-device inference
- Build a minimal Swift UI that visualizes agent decisions transparently
- Generalization test on QQQ and IWM without retraining

### Non-Goals
- Real brokerage integration or live trading
- Multi-asset simultaneous portfolio optimization
- Intraday / minute-bar trading
- Beating buy-and-hold on raw returns (not the objective)
- NLP / news-based features

---

## 3. Data Specification

### Primary Asset
| Field | Value |
|---|---|
| Ticker | SPY |
| Bar Type | Daily OHLCV |
| Source | yFinance |
| Start Date | January 2010 |

### Train / Validate / Test Split
| Split | Period | Purpose |
|---|---|---|
| Train | 2010 – 2021 | Core policy learning |
| Validation | 2022 – 2023 | Includes 2022 bear market + recovery |
| Test | 2024 – 2025 | Recent regime, unseen data |

### Generalization Test Assets (inference only, no retraining)
| Ticker | Profile |
|---|---|
| QQQ | Tech-heavy, higher volatility |
| IWM | Small-cap, different macro sensitivity |

### Data Pipeline Steps
1. Download raw OHLCV via yFinance
2. Compute derived features (see Section 5)
3. Save normalization parameters (mean/std) from training set only
4. Apply saved normalization to validation and test sets
5. Export feature scaler alongside CoreML model

---

## 4. RL Environment Design

### Framework
- **Library:** FinRL or custom Gym environment
- **Algorithm:** PPO (Proximal Policy Optimization) via Stable-Baselines3

### Action Space
- **Type:** Continuous
- **Range:** [0.0, 1.0] — fraction of portfolio allocated to SPY
- **Interpretation:** 0.0 = fully cash, 1.0 = fully invested

### State Space (8 features)
| Feature | Description | Justification |
|---|---|---|
| 1-day return | `(close_t / close_t-1) - 1` | Immediate momentum |
| 5-day return | `(close_t / close_t-5) - 1` | Short-term trend |
| RSI (14) | Relative Strength Index | Overbought/oversold signal |
| SMA ratio | `close / SMA_20` | Price relative to trend |
| 20-day volatility | Rolling std of daily returns | Risk context |
| Volume ratio | `volume / avg_volume_20d` | Confirms price moves |
| Current position | Agent's current allocation (0–1) | Memory of prior action |
| Unrealized PnL | `(current_price / entry_price) - 1` | Shapes exit behavior |

> **Note:** Features 7 and 8 are critical. Without them the agent has no memory of its current state and cannot learn trade management — only entry signals.

### Reward Function
```
reward = Sharpe contribution of the step
       = (portfolio_return_t - risk_free_rate) / rolling_volatility
```
- Penalizes excessive trading via a small transaction cost (0.001 per trade)
- Discourages holding large positions during high-volatility regimes

### Episode Structure
- **Episode length:** 252 trading days (1 calendar year)
- **Starting capital:** $100,000 (simulated)
- **Reset:** Random start date within training window per episode
- **Terminal condition:** End of episode window

---

## 5. Evaluation Methodology

### Walk-Forward Validation
1. Train on 2010–2021
2. Evaluate on 2022–2023 (validation) — tune hyperparameters here
3. Final evaluation on 2024–2025 (test) — touch once, report as final result

> **Critical:** No hyperparameter tuning on the test set. If an interviewer asks about lookahead bias, this structure is the answer.

### Metrics
| Metric | Description |
|---|---|
| Sharpe Ratio | Primary metric — risk-adjusted return |
| Max Drawdown | Largest peak-to-trough loss |
| Cumulative Return | Raw portfolio growth |
| Win Rate | % of trades closed in profit |
| Avg Position Size | Did the agent learn to size down in volatile periods? |

### Baseline Comparison
- **Buy-and-Hold SPY** — the honest benchmark
- Agent should outperform on Sharpe and Max Drawdown even if not on raw return

### Generalization Test
- Run trained policy (no retraining) on QQQ and IWM test periods
- Report same metrics
- Acceptable outcome: degraded but reasonable performance — document and explain why

---

## 6. Model Export (Python → CoreML)

### Steps
1. Train PPO in Stable-Baselines3, save policy network weights
2. Extract policy network as a standalone PyTorch model
3. Convert to CoreML using `coremltools`
4. Bundle CoreML model + normalization parameters (JSON) into the Xcode project
5. Swift calls the model synchronously — input: 8-feature vector, output: position size float

### Constraints
- Input: fixed 8-dimensional float vector (normalized)
- Output: single float in [0, 1]
- No training loop exported — policy inference only
- Normalization must happen in Swift before model call using saved parameters

---

## 7. iOS App (Swift) — Minimal PoC

### Scope
The app is a **visualization layer**, not a trading platform. Total development time budget: 2–3 days.

### Screens

#### Screen 1 — Simulation Setup
- Ticker selector: SPY / QQQ / IWM (hardcoded)
- Date range selector within available test period
- "Run Simulation" button

#### Screen 2 — Simulation Results
- **Portfolio curve** — line chart of portfolio value over time
- **Benchmark overlay** — buy-and-hold line on same chart
- **Trade log** — scrollable table: Date | Action | Position Size | Price | PnL
- **Summary metrics** — Sharpe, Max Drawdown, Total Return vs Baseline

### Explicitly Excluded
- Live data fetching
- Real brokerage API
- User-adjustable hyperparameters
- Push notifications or background processing
- Onboarding / authentication

---

## 8. Project Timeline

| Days | Milestone |
|---|---|
| 1–2 | Data pipeline: download, feature engineering, normalization, save scaler |
| 3–4 | RL environment: state/action/reward design, sanity-check with random agent |
| 5–8 | PPO training: iterate on reward shaping, validate agent behavior |
| 9–10 | Walk-forward evaluation: metrics, baseline comparison, generalization test |
| 11 | CoreML export: policy extraction, conversion, Swift integration test |
| 12–13 | Swift UI: simulation screen, chart, trade log |
| 14 | Buffer — debugging, documentation, README |

---

## 9. Risks & Mitigations

| Risk | Mitigation |
|---|---|
| Agent learns degenerate policy (always hold or always cash) | Check reward function and transaction cost weight early (Day 4) |
| CoreML conversion fails for PPO architecture | Test export on Day 9 with a dummy model before relying on trained weights |
| Normalization mismatch between Python and Swift | Save scaler as JSON during training, unit test Swift normalization against Python output |
| Lookahead bias in feature computation | Compute all features using only data available at time `t` — no future values |
| Overfitting to training regime | Walk-forward structure prevents this — do not tune on test set |

---

## 10. Success Criteria

| Criterion | Pass Condition |
|---|---|
| RL methodology | PPO with defensible state space, reward, and walk-forward evaluation |
| Performance | Agent Sharpe > Buy-and-Hold Sharpe on test set |
| Generalization | Agent runs on QQQ/IWM — results documented honestly |
| On-device inference | CoreML model runs in Swift with correct output |
| Explainability | App UI allows hand-tracing of agent decisions day by day |
| Documentation | README covers methodology, assumptions, and limitations clearly |

---

## 11. Out of Scope (Future Work)

- Multi-asset portfolio allocation (Dirichlet action space)
- Transformer-based policy network (attention over time series)
- Options or derivatives trading
- Fine-tuning on individual stock tickers
- Live paper trading with real-time data feed
