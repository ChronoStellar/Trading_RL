"""
env/rewards.py
Reward function: per-step Sharpe contribution with transaction cost penalty.
"""

import numpy as np


RISK_FREE_DAILY = 0.0  # annualized 0% for simplicity; swap in 0.05/252 if desired
TRANSACTION_COST = 0.001  # fraction of trade size charged per reallocation


def sharpe_step_reward(
    portfolio_return: float,
    rolling_vol: float,
    position_delta: float,
) -> float:
    """
    reward = (r_t - rf) / sigma_t  -  transaction_cost * |delta_position|

    Args:
        portfolio_return:  single-step portfolio return (float)
        rolling_vol:       rolling 20-day std of portfolio returns (float, > 0)
        position_delta:    change in allocation from previous step (float)

    Returns:
        scalar reward
    """
    excess = portfolio_return - RISK_FREE_DAILY
    sharpe_contrib = excess / max(rolling_vol, 1e-8)
    cost = TRANSACTION_COST * abs(position_delta)
    return float(sharpe_contrib - cost)
