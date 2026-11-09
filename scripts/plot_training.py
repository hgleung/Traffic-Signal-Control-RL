"""Plot training curves (mean episode reward vs environment steps) for trained agents.

Usage: python scripts/plot_training.py [NAME=MODEL_DIR ...]
Defaults to the committed models and writes results/training_curves.png.
Rewards are only comparable between runs that use the same reward function.
"""
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from traffic_env import REPO_ROOT  # noqa: E402

DEFAULT_RUNS = {
    "PPO (queue reward)": REPO_ROOT / "models" / "ppo_timed",
    "DQN (queue reward)": REPO_ROOT / "models" / "dqn_timed",
    "PPO (pressure reward)": REPO_ROOT / "models" / "ppo_pressure",
}


def load(model_dir):
    steps, rewards = [], []
    with open(Path(model_dir) / "training_log.csv") as f:
        for row in csv.DictReader(f):
            if row["episode_reward_mean"] not in ("", "nan", "None"):
                steps.append(int(float(row["env_steps"])))
                rewards.append(float(row["episode_reward_mean"]))
    return steps, rewards


def main():
    runs = dict(DEFAULT_RUNS)
    if len(sys.argv) > 1:
        runs = dict(arg.split("=", 1) for arg in sys.argv[1:])
    queue_runs = {k: v for k, v in runs.items() if "pressure" not in k}
    pressure_runs = {k: v for k, v in runs.items() if "pressure" in k}
    panels = [("Queue reward", queue_runs)] + ([("Pressure reward", pressure_runs)] if pressure_runs else [])
    fig, axes = plt.subplots(1, len(panels), figsize=(6 * len(panels), 4), squeeze=False)
    for ax, (title, group) in zip(axes[0], panels):
        for name, model_dir in group.items():
            steps, rewards = load(model_dir)
            ax.plot(steps, rewards, label=name)
        ax.set_title(f"{title} (mean episode reward, higher is better)")
        ax.set_xlabel("Environment steps (5 s decisions)")
        ax.set_ylabel("Mean episode reward")
        ax.legend()
        ax.grid(alpha=0.3)
    fig.tight_layout()
    out = REPO_ROOT / "results" / "training_curves.png"
    fig.savefig(out, dpi=150)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
