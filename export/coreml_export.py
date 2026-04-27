"""
export/coreml_export.py
Extract the PPO actor network and convert it to a CoreML model.

The actor takes a normalized observation vector and outputs a portfolio
allocation in [0, 1].  The scaler needed to produce that observation is
also saved as JSON so the iOS app can normalise live data identically.

Usage:
    conda run -n trading-ppo python export/coreml_export.py
    conda run -n trading-ppo python export/coreml_export.py --obs-dim 12  # after retraining

Outputs:
    export/TradingActor.mlpackage   — CoreML model
    export/scaler.json              — copy of the feature scaler
"""

import argparse
import os
import shutil
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

MODELS_DIR  = os.path.join(os.path.dirname(__file__), "..", "agents", "models")
EXPORT_DIR  = os.path.dirname(__file__)
SCALER_SRC  = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "scaler.json")
SCALER_DST  = os.path.join(EXPORT_DIR, "scaler.json")


# ── Actor wrapper ──────────────────────────────────────────────────────────────

class TradingActor(nn.Module):
    """
    Thin wrapper that exposes only the actor path of the PPO policy:
        obs → mlp_extractor.policy_net → action_net → clamp(0, 1)

    This is what the iOS app calls at inference time.
    """
    def __init__(self, policy):
        super().__init__()
        self.policy_net = policy.mlp_extractor.policy_net
        self.action_net = policy.action_net

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        latent = self.policy_net(obs)
        action = self.action_net(latent)
        return torch.clamp(action, 0.0, 1.0)


# ── Export ─────────────────────────────────────────────────────────────────────

def export(obs_dim: int, model_name: str = "ppo_spy") -> None:
    try:
        import coremltools as ct
    except ImportError:
        sys.exit("coremltools not found. Run: pip install coremltools")

    try:
        from stable_baselines3 import PPO
    except ImportError:
        sys.exit("stable_baselines3 not found. Activate the trading-ppo conda env first.")

    # 1. Load the trained PPO model
    model_path = os.path.join(MODELS_DIR, f"{model_name}.zip")
    if not os.path.exists(model_path):
        sys.exit(f"Model not found: {model_path}\nRun agents/train.py first.")

    print(f"Loading model: {model_path}")
    ppo = PPO.load(model_path)

    trained_obs_dim = ppo.observation_space.shape[0]
    if trained_obs_dim != obs_dim:
        print(
            f"WARNING: model was trained with obs_dim={trained_obs_dim}, "
            f"but --obs-dim={obs_dim} was requested. Using {trained_obs_dim}."
        )
        obs_dim = trained_obs_dim

    # 2. Build actor module and switch to eval mode
    actor = TradingActor(ppo.policy).eval()

    # 3. Trace with a dummy input
    dummy = torch.zeros(1, obs_dim, dtype=torch.float32)
    with torch.no_grad():
        traced = torch.jit.trace(actor, dummy)
    print("TorchScript trace succeeded.")

    # 4. Convert to CoreML
    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(name="observation", shape=(1, obs_dim), dtype=np.float32)],
        outputs=[ct.TensorType(name="allocation", dtype=np.float32)],
        minimum_deployment_target=ct.target.iOS16,
        convert_to="mlprogram",
    )

    # 5. Add metadata so the Swift app knows what it's dealing with
    mlmodel.short_description = "PPO trading agent — SPY allocation [0, 1]"
    mlmodel.input_description["observation"] = (
        f"Normalised observation vector of length {obs_dim}: "
        "market features (z-scored) + current_position + unrealized_pnl"
    )
    mlmodel.output_description["allocation"] = (
        "Recommended portfolio allocation to SPY (0 = cash, 1 = fully invested)"
    )

    # 6. Save .mlpackage
    out_path = os.path.join(EXPORT_DIR, "TradingActor.mlpackage")
    if os.path.exists(out_path):
        shutil.rmtree(out_path)
    mlmodel.save(out_path)
    print(f"CoreML model saved → {out_path}")

    # 7. Copy the scaler so the iOS app can normalise live features
    if os.path.exists(SCALER_SRC):
        shutil.copy(SCALER_SRC, SCALER_DST)
        print(f"Scaler copied  → {SCALER_DST}")
    else:
        print("WARNING: scaler.json not found — run data/features.py first.")

    # 8. Quick sanity-check: compare PyTorch vs CoreML output
    print("\nSanity check …")
    test_input = np.random.randn(1, obs_dim).astype(np.float32)
    with torch.no_grad():
        pt_out = actor(torch.from_numpy(test_input)).numpy()
    ml_out = mlmodel.predict({"observation": test_input})["allocation"]
    delta = np.abs(pt_out - ml_out).max()
    print(f"  PyTorch  output: {pt_out.flatten()}")
    print(f"  CoreML   output: {ml_out.flatten()}")
    print(f"  Max delta: {delta:.2e}", "✓" if delta < 1e-4 else "✗ CHECK THIS")


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export PPO actor to CoreML")
    p.add_argument("--obs-dim",    type=int, default=10,      help="Observation vector length")
    p.add_argument("--model-name", type=str, default="ppo_spy", help="Basename in agents/models/")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    export(obs_dim=args.obs_dim, model_name=args.model_name)
