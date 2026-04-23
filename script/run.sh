# Train
python agents/train.py

# Evaluate
python agents/eval.py --split test

# Backtest
python agents/backtest.py

# Export to CoreML  (produces export/TradingActor.mlpackage + export/scaler.json)
conda run -n trading-ppo python export/coreml_export.py