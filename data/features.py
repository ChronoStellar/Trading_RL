"""
data/features.py
Compute derived features for each ticker, fit a scaler on the training split,
apply it to all splits, and save the normalized CSVs + scaler JSON.

Market features (15 total — features 16 & 17 are injected by the Gym env):
  ── Price / return ──────────────────────────────────────────────────────────
  1.  ret_1d        — 1-day return:  (close_t / close_t-1) - 1
  2.  ret_5d        — 5-day return:  (close_t / close_t-5) - 1
  3.  sma_ratio     — close / SMA_20
  4.  vol_20d       — rolling 20-day std of daily returns
  5.  vol_ratio     — volume / avg_volume_20d
  ── 7 day-trading indicators (via `ta` library) ─────────────────────────────
  6.  rsi_14        — RSI (14-period, Wilder EWM)
  7.  macd_hist     — MACD histogram normalised by close  (scale-free)
  8.  stoch_k       — Stochastic %K (14-period)
  9.  stoch_d       — Stochastic %D (3-period signal of %K)
  10. bb_width      — Bollinger bandwidth: (upper - lower) / mid
  11. bb_pct        — Bollinger %B: (close - lower) / (upper - lower)
  12. obv_ret       — On-Balance Volume 1-day % change  (stationary proxy)
  13. adx           — Average Directional Index (14-period)
  14. adx_di_diff   — (+DI - -DI) / 100  — directional bias [-1, 1]
  15. psar_bull     — 1.0 if PSAR uptrend (price > SAR), 0.0 if downtrend
"""

import json
import os

import numpy as np
import pandas as pd
import ta

# ── Configuration ──────────────────────────────────────────────────────────────
RAW_DIR      = os.path.join(os.path.dirname(__file__), "raw")
PROC_DIR     = os.path.join(os.path.dirname(__file__), "processed")
SCALER_PATH  = os.path.join(PROC_DIR, "scaler.json")

TICKERS = ["SPY", "QQQ", "IWM"]

TRAIN_END = "2021-12-31"
VAL_END   = "2023-12-31"
# test is everything after VAL_END

FEATURE_COLS = [
    # original price/return features
    "ret_1d", "ret_5d", "sma_ratio", "vol_20d", "vol_ratio",
    # 7 day-trading indicators
    "rsi_14",
    "macd_hist",
    "stoch_k", "stoch_d",
    "bb_width", "bb_pct",
    "obv_ret",
    "adx", "adx_di_diff",
    "psar_bull",
]


# ── Feature computation ────────────────────────────────────────────────────────

def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add all 15 feature columns to a single-ticker OHLCV DataFrame."""
    df = df.copy()

    # ── 1–5: price / return features ──────────────────────────────────────────
    df["ret_1d"]     = df["close"].pct_change(1)
    df["ret_5d"]     = df["close"].pct_change(5)
    sma20            = df["close"].rolling(20).mean()
    df["sma_ratio"]  = df["close"] / sma20
    df["vol_20d"]    = df["ret_1d"].rolling(20).std()
    avg_vol20        = df["volume"].rolling(20).mean()
    df["vol_ratio"]  = df["volume"] / avg_vol20

    # ── 6: RSI (14) ───────────────────────────────────────────────────────────
    df["rsi_14"] = ta.momentum.RSIIndicator(
        close=df["close"], window=14
    ).rsi()

    # ── 7: MACD histogram (normalised by close to be scale-free) ─────────────
    macd = ta.trend.MACD(
        close=df["close"], window_slow=26, window_fast=12, window_sign=9
    )
    df["macd_hist"] = macd.macd_diff() / df["close"]

    # ── 8–9: Stochastic Oscillator %K / %D ───────────────────────────────────
    stoch = ta.momentum.StochasticOscillator(
        high=df["high"], low=df["low"], close=df["close"],
        window=14, smooth_window=3
    )
    df["stoch_k"] = stoch.stoch()
    df["stoch_d"] = stoch.stoch_signal()

    # ── 10–11: Bollinger Bands ────────────────────────────────────────────────
    bb = ta.volatility.BollingerBands(
        close=df["close"], window=20, window_dev=2
    )
    df["bb_width"] = bb.bollinger_wband()   # (upper - lower) / mid
    df["bb_pct"]   = bb.bollinger_pband()   # (close - lower) / (upper - lower)

    # ── 12: On-Balance Volume — 1-day % change (stationary) ──────────────────
    obv            = ta.volume.OnBalanceVolumeIndicator(
        close=df["close"], volume=df["volume"]
    ).on_balance_volume()
    df["obv_ret"]  = obv.pct_change(1).replace([np.inf, -np.inf], np.nan)

    # ── 13–14: ADX + directional bias ────────────────────────────────────────
    adx_ind         = ta.trend.ADXIndicator(
        high=df["high"], low=df["low"], close=df["close"], window=14
    )
    df["adx"]        = adx_ind.adx()
    df["adx_di_diff"]= (adx_ind.adx_pos() - adx_ind.adx_neg()) / 100.0

    # ── 15: Parabolic SAR — binary uptrend flag ───────────────────────────────
    psar            = ta.trend.PSARIndicator(
        high=df["high"], low=df["low"], close=df["close"],
        step=0.02, max_step=0.2
    )
    df["psar_bull"] = psar.psar_up().notna().astype(np.float32)

    # Drop helper intermediates
    df = df.dropna(subset=FEATURE_COLS).reset_index(drop=True)
    return df


# ── Normalization ──────────────────────────────────────────────────────────────

def fit_scaler(train_df: pd.DataFrame) -> dict:
    """Compute mean/std from the training split only."""
    scaler = {}
    for col in FEATURE_COLS:
        scaler[col] = {
            "mean": float(train_df[col].mean()),
            "std":  float(train_df[col].std()),
        }
    return scaler


def apply_scaler(df: pd.DataFrame, scaler: dict) -> pd.DataFrame:
    """Z-score normalize features using pre-computed scaler."""
    df = df.copy()
    for col in FEATURE_COLS:
        mean = scaler[col]["mean"]
        std  = scaler[col]["std"]
        df[col] = (df[col] - mean) / (std if std > 0 else 1.0)
    return df


# ── Split helpers ──────────────────────────────────────────────────────────────

def split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = df[df["date"] <= TRAIN_END]
    val   = df[(df["date"] > TRAIN_END) & (df["date"] <= VAL_END)]
    test  = df[df["date"] > VAL_END]
    return train, val, test


# ── Main pipeline ──────────────────────────────────────────────────────────────

def process_ticker(ticker: str, scaler: dict | None = None) -> dict | None:
    """
    Load raw CSV, compute features, optionally fit a scaler (SPY only),
    then save normalized train/val/test splits.

    Returns the fitted scaler when scaler=None (i.e., for SPY).
    """
    raw_path = os.path.join(RAW_DIR, f"{ticker}.csv")
    if not os.path.exists(raw_path):
        raise FileNotFoundError(
            f"{raw_path} not found. Run data/loader.py first."
        )

    df = pd.read_csv(raw_path, parse_dates=["date"])
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")
    df = df.sort_values("date").reset_index(drop=True)

    df = compute_features(df)
    train, val, test = split(df)

    if scaler is None:
        # Fit on SPY training set; reuse for all tickers to keep feature scale consistent
        scaler = fit_scaler(train)

    train = apply_scaler(train, scaler)
    val   = apply_scaler(val,   scaler)
    test  = apply_scaler(test,  scaler)

    ticker_dir = os.path.join(PROC_DIR, ticker)
    os.makedirs(ticker_dir, exist_ok=True)

    for split_name, split_df in [("train", train), ("val", val), ("test", test)]:
        out = os.path.join(ticker_dir, f"{split_name}.csv")
        split_df.to_csv(out, index=False)
        print(f"[{ticker}] {split_name:5s}: {len(split_df):5d} rows → {out}")

    return scaler  # only meaningful when we fitted it (ticker == SPY)


def run() -> None:
    os.makedirs(PROC_DIR, exist_ok=True)

    # 1. Fit scaler on SPY training data
    scaler = process_ticker("SPY", scaler=None)

    # 2. Save scaler
    with open(SCALER_PATH, "w") as f:
        json.dump(scaler, f, indent=2)
    print(f"\nScaler saved → {SCALER_PATH}")

    # 3. Apply same scaler to QQQ and IWM (no retraining)
    for ticker in ["QQQ", "IWM"]:
        process_ticker(ticker, scaler=scaler)


if __name__ == "__main__":
    run()
