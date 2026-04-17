# Trading RL — Claude Guide

## Project Overview
Recurrent PPO (R-PPO with LSTM) agent for SPY paper trading. Full spec: see docs/PRD.md.

## Directory Structure
- Data/         — data pipeline (download, features, normalization)
- Env/          — custom Gym environment
- Agents/       — R-PPO training & evaluation scripts
- Export/       — CoreML conversion
- Paper_trade/  — iOS Swift app

## Environment Setup
```bash
pip install stable-baselines3 sb3-contrib yfinance coremltools gymnasium tabulate

# Data pipeline
python Data/download.py
python Data/features.py

# Train (Recurrent PPO with LSTM)
python agents/train.py

# Evaluate
python agents/eval.py --split test

# Export to CoreML
python Export/export_coreml.py
