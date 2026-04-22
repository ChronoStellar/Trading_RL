"""
env/train_env.py
Custom Gymnasium environment for single-asset (SPY) continuous position sizing.

State space  (17-d): 15 market features (pre-normalized) + current_position + equity_return
Action space  (1-d): continuous allocation in [0, 1]
Episode length:      252 trading days, random start within the split
"""

import os

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

from env.rewards import sharpe_step_reward, TRANSACTION_COST, SLIPPAGE

# ── Constants ──────────────────────────────────────────────────────────────────
EPISODE_LEN   = 252          # trading days per episode
INITIAL_CASH  = 100_000.0   # simulated starting capital
FEATURE_COLS  = [
    "ret_1d", "ret_5d", "sma_ratio", "vol_20d", "vol_ratio",
    "rsi_14",
    "macd_hist",
    "stoch_k", "stoch_d",
    "bb_width", "bb_pct",
    "obv_ret",
    "adx", "adx_di_diff",
    "psar_bull",
]
PROC_DIR      = os.path.join(os.path.dirname(__file__), "..", "data", "processed")


class TradingEnv(gym.Env):
    """
    Single-asset continuous allocation environment.

    Parameters
    ----------
    ticker : str
        Asset ticker (SPY / QQQ / IWM). Loads from data/processed/<ticker>/<split>.csv
    split  : str
        One of 'train', 'val', 'test'
    seed   : int | None
        RNG seed for reproducible episode start dates
    """

    metadata = {"render_modes": []}

    def __init__(self, ticker: str = "SPY", split: str = "train", seed: int | None = None):
        super().__init__()

        csv_path = os.path.join(PROC_DIR, ticker, f"{split}.csv")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(
                f"{csv_path} not found. Run data/loader.py and data/features.py first."
            )

        df = pd.read_csv(csv_path)
        self._features: np.ndarray = df[FEATURE_COLS].to_numpy(dtype=np.float32)
        self._close:    np.ndarray = df["close"].to_numpy(dtype=np.float32)
        self._n_rows = len(df)

        # Spaces
        # Obs: 15 market features (z-scored) + position + equity_return = 17-d
        _n_market = len(FEATURE_COLS)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(_n_market + 2,), dtype=np.float32
        )
        # Action: allocation fraction [0, 1]
        self.action_space = spaces.Box(
            low=np.array([0.0], dtype=np.float32),
            high=np.array([1.0], dtype=np.float32),
            dtype=np.float32,
        )

        self._rng = np.random.default_rng(seed)

        # Episode state (set properly in reset)
        self._start_idx:    int   = 0
        self._current_step: int   = 0
        self._position:     float = 0.0   # current allocation [0, 1]
        self._portfolio_val:float = INITIAL_CASH
        self._ret_history:  list  = []    # for rolling vol

    # ── Gymnasium API ──────────────────────────────────────────────────────────

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        # Random start so that a full episode fits within the data
        max_start = self._n_rows - EPISODE_LEN - 1
        if max_start <= 0:
            raise ValueError(
                f"Split has only {self._n_rows} rows; need at least {EPISODE_LEN + 1}."
            )
        self._start_idx    = int(self._rng.integers(0, max_start + 1))
        self._current_step = 0
        self._position     = 0.0
        self._portfolio_val= INITIAL_CASH

        # Seed rolling vol from the 20 price returns before episode start so
        # early-episode rewards aren't garbage.
        lookback = min(20, self._start_idx)
        if lookback > 1:
            prices = self._close[self._start_idx - lookback : self._start_idx + 1]
            seed_rets = list((prices[1:] / prices[:-1]) - 1.0)
        else:
            seed_rets = []
        self._ret_history = seed_rets

        obs = self._get_obs()
        return obs, {}

    def step(self, action: np.ndarray):
        action = np.clip(action, 0.0, 1.0)
        new_position = float(action[0])
        position_delta = new_position - self._position

        idx = self._start_idx + self._current_step
        current_price = float(self._close[idx])
        next_price    = float(self._close[idx + 1])

        # Portfolio return for this step
        price_return     = (next_price / current_price) - 1.0
        portfolio_return = self._position * price_return  # only invested fraction earns

        # Rolling vol uses portfolio returns (measures portfolio Sharpe correctly).
        # Floored at 0.003 (~0.3% daily) so it can't collapse to zero when the
        # agent holds cash, which would otherwise cause reward blow-up.
        self._ret_history.append(price_return)

        if len(self._ret_history) > 20:
            self._ret_history.pop(0)
        rolling_vol = max(np.std(self._ret_history) if len(self._ret_history) > 1 else 0.01, 0.003)

        net_portofolio_return = portfolio_return - (TRANSACTION_COST + SLIPPAGE) * abs(position_delta)

        reward = sharpe_step_reward(net_portofolio_return, rolling_vol, position_delta)

        # Deduct transaction costs from portfolio value so the equity observation
        # reflects actual realised performance.
        cost_frac = (TRANSACTION_COST + SLIPPAGE) * abs(position_delta)
        self._portfolio_val *= (1.0 + portfolio_return) * (1.0 - cost_frac)

        self._position = new_position

        self._current_step += 1
        terminated = self._current_step >= EPISODE_LEN
        truncated  = False

        obs = self._get_obs()
        info = {
            "portfolio_value": self._portfolio_val,
            "position":        self._position,
            "price":           next_price,
        }
        return obs, reward, terminated, truncated, info

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _get_obs(self) -> np.ndarray:
        idx = self._start_idx + self._current_step
        market_features = self._features[idx]  # shape (15,)

        equity_return = (self._portfolio_val / INITIAL_CASH) - 1.0

        obs = np.append(market_features, [self._position, equity_return]).astype(np.float32)
        return obs
