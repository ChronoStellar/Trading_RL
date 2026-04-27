"""
agents/tune.py
Optuna-based tuner for PPO hyperparameters AND feature selection.

What is tuned
─────────────
  PPO hyperparameters : learning_rate, n_steps, batch_size, n_epochs,
                        gamma, gae_lambda, ent_coef
  Feature selection   : any subset of FEATURE_COLS with ≥ N_MIN_FEATURES

Objective
─────────
  Train on the training split for --timesteps steps (default 200k — fast
  enough to see signal, cheap enough to run many trials).
  Evaluate with a single sequential pass through the validation split.
  Returns the annualised Sharpe ratio on the validation set.

  Optuna prunes weak trials early using median stopping on in-training
  mean episode reward.

Usage
─────
  python agents/tune.py                             # 50 trials, 200k steps each
  python agents/tune.py --trials 100 --timesteps 300000
  python agents/tune.py --study-name spy_v2         # name / resume
  python agents/tune.py --storage sqlite:///agents/tune.db   # persist to disk
  python agents/tune.py --show-best                 # print best params and exit
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv

try:
    import optuna
    from optuna.pruners import MedianPruner
    from optuna.samplers import TPESampler
except ImportError:
    print("Optuna not installed. Run: pip install optuna")
    sys.exit(1)

from data.features import FEATURE_COLS
from env.rewards import sharpe_step_reward, TRANSACTION_COST, SLIPPAGE

PROC_DIR   = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
TUNE_DIR   = os.path.join(os.path.dirname(__file__), "tune_results")

INITIAL_CASH    = 100_000.0
EPISODE_LEN     = 252
N_MIN_FEATURES  = 5     # at least 5 market features must be selected
N_ENVS_TUNE     = 4     # parallel envs during tuning (lower than full training)


# ── Lightweight env (accepts pre-normalised numpy arrays, no file I/O) ───────

class _TuneEnv(gym.Env):
    """
    Mirrors TradingEnv but takes arrays directly so feature subsets can be
    swapped per trial without touching the filesystem.
    Supports random starts (training) and sequential mode (validation eval).
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        features: np.ndarray,
        close: np.ndarray,
        seed: int | None = None,
        sequential: bool = False,
    ):
        super().__init__()
        self._features   = features.astype(np.float32)
        self._close      = close.astype(np.float32)
        self._n_rows     = len(features)
        self._sequential = sequential

        n_market = features.shape[1]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(n_market + 2,), dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=np.array([0.0], dtype=np.float32),
            high=np.array([1.0], dtype=np.float32),
            dtype=np.float32,
        )

        self._rng           = np.random.default_rng(seed)
        self._start_idx     = 0
        self._current_step  = 0
        self._episode_len   = min(EPISODE_LEN, self._n_rows - 1)
        self._position      = 0.0
        self._portfolio_val = INITIAL_CASH
        self._ret_history: list[float] = []

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        if self._sequential:
            self._start_idx   = 0
            self._episode_len = self._n_rows - 1
        else:
            max_start = self._n_rows - EPISODE_LEN - 1
            if max_start <= 0:
                self._start_idx   = 0
                self._episode_len = self._n_rows - 1
            else:
                self._start_idx   = int(self._rng.integers(0, max_start + 1))
                self._episode_len = EPISODE_LEN

        self._current_step  = 0
        self._position      = 0.0
        self._portfolio_val = INITIAL_CASH

        lookback = min(20, self._start_idx)
        if lookback > 1:
            p = self._close[self._start_idx - lookback : self._start_idx + 1]
            self._ret_history = list((p[1:] / p[:-1]) - 1.0)
        else:
            self._ret_history = []

        return self._get_obs(), {}

    def step(self, action: np.ndarray):
        action         = np.clip(action, 0.0, 1.0)
        new_pos        = float(action[0])
        delta          = new_pos - self._position

        idx            = self._start_idx + self._current_step
        cur_p          = float(self._close[idx])
        nxt_p          = float(self._close[idx + 1])
        price_ret      = (nxt_p / cur_p) - 1.0
        port_ret       = self._position * price_ret

        self._ret_history.append(price_ret)
        if len(self._ret_history) > 20:
            self._ret_history.pop(0)
        vol = max(
            np.std(self._ret_history) if len(self._ret_history) > 1 else 0.01,
            0.003,
        )

        net_ret = port_ret - (TRANSACTION_COST + SLIPPAGE) * abs(delta)
        reward  = sharpe_step_reward(net_ret, vol, delta)

        cost_frac           = (TRANSACTION_COST + SLIPPAGE) * abs(delta)
        self._portfolio_val *= (1.0 + port_ret) * (1.0 - cost_frac)
        self._position       = new_pos
        self._current_step  += 1
        terminated           = self._current_step >= self._episode_len

        return self._get_obs(), reward, terminated, False, {
            "portfolio_value": self._portfolio_val,
            "position":        self._position,
        }

    def _get_obs(self) -> np.ndarray:
        idx    = self._start_idx + self._current_step
        eq_ret = (self._portfolio_val / INITIAL_CASH) - 1.0
        return np.append(
            self._features[idx], [self._position, eq_ret]
        ).astype(np.float32)


# ── Metrics ──────────────────────────────────────────────────────────────────

def _val_sharpe(model: PPO, val_features: np.ndarray, val_close: np.ndarray,
                seed: int) -> float:
    """Single sequential pass through the validation set → annualised Sharpe."""
    env = _TuneEnv(val_features, val_close, seed=seed, sequential=True)
    obs, _ = env.reset()
    portfolio = [INITIAL_CASH]
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        portfolio.append(info["portfolio_value"])

    portfolio = np.array(portfolio)
    rets      = np.diff(portfolio) / portfolio[:-1]
    return float(np.mean(rets) / (np.std(rets) + 1e-8) * np.sqrt(252))


# ── Pruning callback ──────────────────────────────────────────────────────────

class _PruneCallback(BaseCallback):
    """
    Reports mean episode reward to Optuna every *check_freq* timesteps so
    that the MedianPruner can kill weak trials early.
    """

    def __init__(self, trial: "optuna.Trial", check_freq: int = 20_000):
        super().__init__()
        self.trial      = trial
        self.check_freq = check_freq
        self._rewards: list[float] = []
        self._last_check = 0

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self._rewards.append(float(info["episode"]["r"]))

        if (self.num_timesteps - self._last_check) >= self.check_freq:
            self._last_check = self.num_timesteps
            if self._rewards:
                mean_r = float(np.mean(self._rewards[-200:]))
                self.trial.report(mean_r, self.num_timesteps)
                if self.trial.should_prune():
                    raise optuna.TrialPruned()
        return True


# ── Search space ─────────────────────────────────────────────────────────────

def suggest_features(trial: "optuna.Trial") -> list[str]:
    """Sample a feature subset. Always includes at least N_MIN_FEATURES."""
    selected = [
        col for col in FEATURE_COLS
        if trial.suggest_categorical(f"feat_{col}", [True, False])
    ]
    # Fall back to all features if fewer than minimum were selected
    if len(selected) < N_MIN_FEATURES:
        selected = FEATURE_COLS[:]
    return selected


def suggest_ppo_hparams(trial: "optuna.Trial", n_envs: int) -> dict:
    """Sample PPO hyperparameters with SB3 batch_size constraint."""
    n_steps = trial.suggest_categorical("n_steps", [512, 1024, 2048, 4096])

    # batch_size must evenly divide n_steps * n_envs
    total   = n_steps * n_envs
    candidates = [b for b in [64, 128, 256, 512] if total % b == 0]
    if not candidates:
        candidates = [64]
    batch_size = trial.suggest_categorical("batch_size", candidates)

    return dict(
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
        n_steps       = n_steps,
        batch_size    = batch_size,
        n_epochs      = trial.suggest_int("n_epochs", 5, 30),
        gamma         = trial.suggest_float("gamma", 0.97, 0.999),
        gae_lambda    = trial.suggest_float("gae_lambda", 0.8, 0.99),
        ent_coef      = trial.suggest_float("ent_coef", 1e-4, 0.05, log=True),
    )


# ── Objective ────────────────────────────────────────────────────────────────

def make_objective(
    train_df:  pd.DataFrame,
    val_df:    pd.DataFrame,
    timesteps: int,
    n_envs:    int,
    seed:      int,
    device:    str,
):
    """Return the Optuna objective function (closure over data + config)."""

    train_close = train_df["close"].to_numpy(dtype=np.float32)
    val_close   = val_df["close"].to_numpy(dtype=np.float32)

    def objective(trial: "optuna.Trial") -> float:
        selected      = suggest_features(trial)
        ppo_hparams   = suggest_ppo_hparams(trial, n_envs)

        train_feat = train_df[selected].to_numpy(dtype=np.float32)
        val_feat   = val_df[selected].to_numpy(dtype=np.float32)

        def _make(s):
            def _init():
                return _TuneEnv(train_feat, train_close, seed=s)
            return _init

        vec_env = DummyVecEnv([_make(seed + i) for i in range(n_envs)])

        model = PPO(
            "MlpPolicy",
            vec_env,
            seed=seed,
            device=device,
            verbose=0,
            **ppo_hparams,
        )

        try:
            model.learn(
                total_timesteps=timesteps,
                callback=_PruneCallback(trial),
            )
        except optuna.TrialPruned:
            vec_env.close()
            raise
        finally:
            vec_env.close()

        sharpe = _val_sharpe(model, val_feat, val_close, seed)

        # Store extras for inspection
        trial.set_user_attr("val_sharpe",  sharpe)
        trial.set_user_attr("n_features",  len(selected))
        trial.set_user_attr("features",    selected)

        return sharpe

    return objective


# ── Results helpers ───────────────────────────────────────────────────────────

def print_best(study: "optuna.Study") -> None:
    trial = study.best_trial
    print(f"\n{'='*65}")
    print(f"  BEST TRIAL  #{trial.number}  (val Sharpe = {trial.value:.4f})")
    print(f"{'='*65}")

    features = trial.user_attrs.get("features", [])
    print(f"\n  Features ({len(features)}):")
    for f in features:
        print(f"    {f}")

    print(f"\n  PPO hyperparameters:")
    ppo_keys = {"learning_rate", "n_steps", "batch_size", "n_epochs",
                "gamma", "gae_lambda", "ent_coef"}
    for k, v in trial.params.items():
        if k in ppo_keys:
            print(f"    {k:20s} = {v}")
    print()


def save_best(study: "optuna.Study", out_dir: str) -> str:
    """Save best params as agents/models/best_hparams.json."""
    os.makedirs(out_dir, exist_ok=True)
    trial    = study.best_trial
    features = trial.user_attrs.get("features", FEATURE_COLS)

    ppo_keys = {"learning_rate", "n_steps", "batch_size", "n_epochs",
                "gamma", "gae_lambda", "ent_coef"}
    ppo_params = {k: v for k, v in trial.params.items() if k in ppo_keys}

    best = {
        "trial_number": trial.number,
        "val_sharpe":   trial.value,
        "features":     features,
        "n_features":   len(features),
        "ppo":          ppo_params,
    }

    out_path = os.path.join(out_dir, "best_hparams.json")
    with open(out_path, "w") as f:
        json.dump(best, f, indent=2)
    print(f"Best params saved → {out_path}")
    return out_path


def plot_study(study: "optuna.Study", out_dir: str) -> None:
    """Save Optuna visualisation plots (requires optuna[visualization])."""
    try:
        import optuna.visualization as vis
        import plotly
    except ImportError:
        # Fall back to matplotlib importance plot
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            importances = optuna.importance.get_param_importances(study)
            fig, ax = plt.subplots(figsize=(9, 5))
            keys   = list(importances.keys())
            vals   = list(importances.values())
            ax.barh(keys[::-1], vals[::-1], color="steelblue")
            ax.set_xlabel("Importance")
            ax.set_title("Hyperparameter Importances (FAnova)")
            plt.tight_layout()
            out = os.path.join(out_dir, "param_importances.png")
            plt.savefig(out, dpi=120, bbox_inches="tight")
            plt.close(fig)
            print(f"Importance plot → {out}")
        except Exception:
            pass
        return

    os.makedirs(out_dir, exist_ok=True)
    for name, fig_fn in [
        ("optimization_history",  vis.plot_optimization_history),
        ("param_importances",     vis.plot_param_importances),
        ("parallel_coordinate",   vis.plot_parallel_coordinate),
    ]:
        try:
            fig = fig_fn(study)
            out = os.path.join(out_dir, f"{name}.html")
            fig.write_html(out)
            print(f"Plot → {out}")
        except Exception:
            pass


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Optuna tuner: PPO hyperparameters + feature selection",
    )
    p.add_argument("--ticker",      default="SPY")
    p.add_argument("--trials",      type=int, default=50)
    p.add_argument("--timesteps",   type=int, default=2500_000,
                   help="Training timesteps per trial (lower = faster, noisier)")
    p.add_argument("--n-envs",      type=int, default=N_ENVS_TUNE)
    p.add_argument("--seed",        type=int, default=42)
    p.add_argument("--device",      default="cpu", choices=["cpu", "cuda", "mps"])
    p.add_argument("--study-name",  default="ppo_trading",
                   help="Optuna study name (use same name to resume)")
    p.add_argument("--storage",     default=None,
                   help="Optuna storage URL, e.g. sqlite:///agents/tune.db")
    p.add_argument("--n-jobs",      type=int, default=1,
                   help="Parallel Optuna workers (>1 requires --storage)")
    p.add_argument("--show-best",   action="store_true",
                   help="Print best params from existing study and exit")
    p.add_argument("--no-save",     action="store_true")
    p.add_argument("--no-plot",     action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    os.makedirs(TUNE_DIR, exist_ok=True)

    sampler = TPESampler(seed=args.seed)
    pruner  = MedianPruner(n_startup_trials=5, n_warmup_steps=5)

    study = optuna.create_study(
        study_name    = args.study_name,
        direction     = "maximize",
        sampler       = sampler,
        pruner        = pruner,
        storage       = args.storage,
        load_if_exists= True,
    )

    if args.show_best:
        if not study.trials:
            print("No trials in this study yet.")
        else:
            print_best(study)
        return

    # Load pre-normalised train and validation CSVs
    train_df = pd.read_csv(
        os.path.join(PROC_DIR, args.ticker, "train.csv")
    )
    val_df   = pd.read_csv(
        os.path.join(PROC_DIR, args.ticker, "val.csv")
    )

    print(f"\n{'='*65}")
    print(f"  TUNING  {args.study_name}  ({args.trials} trials × "
          f"{args.timesteps:,} steps)")
    print(f"  Train rows : {len(train_df):,}   "
          f"Val rows : {len(val_df):,}")
    print(f"  Feature candidates: {len(FEATURE_COLS)}  "
          f"min selected: {N_MIN_FEATURES}")
    print(f"  Device: {args.device}   n_envs: {args.n_envs}")
    print(f"{'='*65}\n")

    objective = make_objective(
        train_df  = train_df,
        val_df    = val_df,
        timesteps = args.timesteps,
        n_envs    = args.n_envs,
        seed      = args.seed,
        device    = args.device,
    )

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    study.optimize(
        objective,
        n_trials  = args.trials,
        n_jobs    = args.n_jobs,
        show_progress_bar = True,
    )

    # ── Report ──
    print_best(study)

    completed = [t for t in study.trials
                 if t.state == optuna.trial.TrialState.COMPLETE]
    pruned    = [t for t in study.trials
                 if t.state == optuna.trial.TrialState.PRUNED]
    print(f"  Trials: {len(completed)} completed, {len(pruned)} pruned\n")

    if not args.no_save:
        save_best(study, MODELS_DIR)

    if not args.no_plot:
        plot_study(study, TUNE_DIR)


if __name__ == "__main__":
    main()
