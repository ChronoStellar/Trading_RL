# Trading RL — Claude Guide

## Project Overview
PPO RL agent for SPY paper trading. Full spec: see docs/PRD.md.

## Directory Structure
- Data/         — data pipeline (download, features, normalization)
- Env/          — custom Gym environment
- Agents/       — PPO training scripts
- Export/       — CoreML conversion
- Paper_trade/  — iOS Swift app

## Environment Setup
```bash
pip install stable-baselines3 yfinance coremltools gymnasium

# Data pipeline
python Data/download.py
python Data/features.py

# Train
python Agents/train.py

# Evaluate
python Agents/eval.py --split test

# Export to CoreML
python Export/export_coreml.py
