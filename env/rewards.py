"""
env/rewards.py
Reward function: per-step Sharpe contribution with transaction cost penalty.
"""

import numpy as np


RISK_FREE_DAILY  = 0.00001 / 252  # annualized 0% for simplicity; swap in 0.05/252 if desired
TRANSACTION_COST = 0.001           # fraction of trade size charged per reallocation
SLIPPAGE         = 0.0005          # 5 bps execution slippage, realistic for SPY


def sharpe_step_reward(
    portfolio_return: float,
    rolling_vol: float,
    position_delta: float,
) -> float:
    """
    reward = (r_t - rf) / sigma_t  -  (transaction_cost + slippage) * |delta| / sigma_t

    Penalty is divided by rolling_vol so it lives in the same Sharpe units as
    the signal term — a large rebalance during a calm market is penalised more.

    Args:
        portfolio_return:  single-step portfolio return (float)
        rolling_vol:       rolling 20-day std of portfolio returns (float, > 0)
        position_delta:    change in allocation from previous step (float)

    Returns:
        scalar reward
    """
    vol = max(rolling_vol, 1e-8)
    sharpe_contrib = (portfolio_return - RISK_FREE_DAILY) / vol
    cost = (TRANSACTION_COST + SLIPPAGE) * abs(position_delta) / vol
    return float(sharpe_contrib - cost)
