# Train
python agents/train.py

# Evaluate
python agents/eval.py --split test

# Backtest (single model, train/val/test splits)
python agents/backtest.py

# Tune hyperparameters + feature selection (Optuna)
python agents/tune.py --trials 50 --timesteps 200000

# Walk-forward (trains fresh PPO per fold, expanding window)
python agents/walk_forward.py

# Export to CoreML  (produces export/TradingActor.mlpackage + export/scaler.json)
conda run -n trading-ppo python export/coreml_export.py