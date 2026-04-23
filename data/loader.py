"""
data/loader.py
Download daily OHLCV bars for SPY, QQQ, and IWM using FinRL's YahooDownloader.
Saves one CSV per ticker to data/raw/.
"""

import os
from finrl.meta.preprocessor.yahoodownloader import YahooDownloader

# ── Configuration ──────────────────────────────────────────────────────────────
START_DATE = "2005-01-01"
END_DATE   = "2025-12-31"
TICKERS    = ["SPY", "QQQ", "IWM"]
RAW_DIR    = os.path.join(os.path.dirname(__file__), "raw")


def download(tickers: list[str] = TICKERS,
             start: str = START_DATE,
             end: str = END_DATE) -> None:
    """Download OHLCV data and save each ticker to data/raw/<TICKER>.csv."""
    os.makedirs(RAW_DIR, exist_ok=True)

    df = YahooDownloader(
        start_date=start,
        end_date=end,
        ticker_list=tickers,
    ).fetch_data()

    # YahooDownloader returns a combined DataFrame with a 'tic' column.
    for ticker in tickers:
        ticker_df = df[df["tic"] == ticker].copy()
        ticker_df = ticker_df.sort_values("date").reset_index(drop=True)
        out_path = os.path.join(RAW_DIR, f"{ticker}.csv")
        ticker_df.to_csv(out_path, index=False)
        print(f"Saved {len(ticker_df)} rows → {out_path}")


if __name__ == "__main__":
    download()
