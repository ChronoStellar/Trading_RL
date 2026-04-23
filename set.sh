conda create -n trading-ppo python=3.10 -y
conda activate trading-ppo
pip install stable-baselines3 yfinance coremltools gymnasium

# Data pipeline
python data/loader.py
python data/features.py

# Train
python agents/train.py

# Evaluate
python agents/eval.py --split test

# Backtest
python agents/backtest.py

# Export to CoreML  (produces export/TradingActor.mlpackage + export/scaler.json)
conda run -n trading-ppo python export/coreml_export.py