"""
data/features.py
Compute derived features for each ticker, fit a scaler on the training split,
apply it to all splits, and save the normalized CSVs + scaler JSON.

Features (6 market features — features 7 & 8 are injected by the Gym env):
  1. ret_1d       — 1-day return:  (close_t / close_t-1) - 1
  2. ret_5d       — 5-day return:  (close_t / close_t-5) - 1
  3. rsi_14       — RSI (14-period)
  4. sma_ratio    — close / SMA_20
  5. vol_20d      — rolling 20-day std of daily returns
  6. vol_ratio    — volume / avg_volume_20d
"""

import json
import os

import numpy as np
import pandas as pd

# ── Configuration ──────────────────────────────────────────────────────────────
RAW_DIR      = os.path.join(os.path.dirname(__file__), "raw")
PROC_DIR     = os.path.join(os.path.dirname(__file__), "processed")
SCALER_PATH  = os.path.join(PROC_DIR, "scaler.json")

TICKERS = ["SPY", "QQQ", "IWM"]

TRAIN_END = "2021-12-31"
VAL_END   = "2023-12-31"
# test is everything after VAL_END

FEATURE_COLS = ["ret_1d", "ret_5d", "rsi_14", "sma_ratio", "vol_20d", "vol_ratio"]


# ── Feature computation ────────────────────────────────────────────────────────

def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain  = delta.clip(lower=0)
    loss  = (-delta).clip(lower=0)
    # Wilder's smoothing (equivalent to EWM with alpha=1/period, adjust=False)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add feature columns to a single-ticker OHLCV DataFrame."""
    df = df.copy()
    df["ret_1d"]    = df["close"].pct_change(1)
    df["ret_5d"]    = df["close"].pct_change(5)
    df["rsi_14"]    = _rsi(df["close"], 14)
    df["sma_20"]    = df["close"].rolling(20).mean()
    df["sma_ratio"] = df["close"] / df["sma_20"]
    df["vol_20d"]   = df["ret_1d"].rolling(20).std()
    df["avg_vol_20"]= df["volume"].rolling(20).mean()
    df["vol_ratio"] = df["volume"] / df["avg_vol_20"]
    # Drop helper columns
    df = df.drop(columns=["sma_20", "avg_vol_20"])
    # Drop rows with NaN features (warmup period)
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
