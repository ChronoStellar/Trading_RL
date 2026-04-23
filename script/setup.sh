conda create -n trading-ppo python=3.10 -y
conda activate trading-ppo
pip install -r -q requirements.txt

bash script/features.sh

bash script/run.sh