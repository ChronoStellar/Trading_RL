conda create -n trading-ppo python=3.10 -y
conda activate trading-ppo
pip install -r -q requirements.txt

# Data pipeline
python data/loader.py
python data/features.py