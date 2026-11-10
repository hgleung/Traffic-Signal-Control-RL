"""Train DQN traffic signal agents on TimedSignalEnv (realistic signal timing).

Same setup as train_ppo.py (one policy shared by every intersection, random
training scenarios, the same observations, actions and rewards), with RLlib's
DQN instead of PPO: double Q-learning, a dueling network, 3-step returns and a
prioritized replay buffer, exploring with an epsilon-greedy schedule.

Usage:
  python scripts/train_dqn.py --timesteps 600000
Evaluate with: python scripts/evaluate.py --policy dqn=models/dqn_timed
"""
import argparse
import csv
import os
import time
from pathlib import Path

os.environ.setdefault("RLLIB_DISABLE_NEW_API_STACK", "True")

from ray.rllib.algorithms.dqn import DQNConfig  # noqa: E402
from ray.tune.registry import register_env  # noqa: E402

from traffic_env import REPO_ROOT  # noqa: E402
from train_ppo import ENV_NAME, POLICY_ID, env_creator, init_ray, policy_mapping_fn  # noqa: E402


def build_config(env_config, total_timesteps, num_env_runners=3, lr=5e-4, gamma=0.95, hidden=(128, 128), seed=0):
    register_env(ENV_NAME, env_creator)
    return (
        DQNConfig()
        .api_stack(enable_rl_module_and_learner=False, enable_env_runner_and_connector_v2=False)
        .environment(env=ENV_NAME, env_config=env_config)
        .framework("torch")
        .env_runners(
            num_env_runners=num_env_runners,
            rollout_fragment_length=8,
            exploration_config={
                "type": "EpsilonGreedy",
                "initial_epsilon": 1.0,
                "final_epsilon": 0.02,
                # Explore heavily for the first third of training, then act mostly greedily.
                "epsilon_timesteps": total_timesteps // 3,
            },
        )
        .training(
            lr=lr, gamma=gamma, train_batch_size=128, n_step=3, double_q=True, dueling=True,
            target_network_update_freq=2000, num_steps_sampled_before_learning_starts=5000,
            replay_buffer_config={"type": "MultiAgentPrioritizedReplayBuffer", "capacity": 200_000},
            grad_clip=10.0, model={"fcnet_hiddens": list(hidden), "fcnet_activation": "relu"},
        )
        .reporting(min_sample_timesteps_per_iteration=5000)
        .multi_agent(policies={POLICY_ID}, policy_mapping_fn=policy_mapping_fn)
        .debugging(seed=seed)
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--timesteps", type=int, default=600_000, help="environment steps to train for")
    parser.add_argument("--reward", choices=["queue", "pressure", "legacy"], default="queue")
    parser.add_argument("--num-env-runners", type=int, default=3)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--network", choices=["1x2", "4x4", "calaveras"], default="1x2")
    parser.add_argument("--max-red", type=int, default=None,
                        help="longest a lane may hold a stopped vehicle at red (s); enforced by overriding the agent")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "models" / "dqn_timed")
    args = parser.parse_args()

    init_ray()
    env_config = {"reward": args.reward, "network": args.network, "max_red": args.max_red}
    algo = build_config(env_config, args.timesteps, num_env_runners=args.num_env_runners,
                        lr=args.lr, gamma=args.gamma, seed=args.seed).build()

    args.out.mkdir(parents=True, exist_ok=True)
    start = time.time()
    steps, i = 0, 0
    with open(args.out / "training_log.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["iteration", "env_steps", "episodes", "episode_reward_mean", "minutes"])
        while steps < args.timesteps:
            i += 1
            result = algo.train()
            er = result["env_runners"]
            steps = result.get("num_env_steps_sampled_lifetime", result.get("num_env_steps_sampled", 0))
            row = [i, steps, er.get("num_episodes_lifetime", er.get("episodes_total")),
                   er.get("episode_reward_mean"), round((time.time() - start) / 60, 2)]
            writer.writerow(row)
            f.flush()
            print(f"iter {i:4d}  steps {steps}  reward {row[3]}  ({row[4]} min)", flush=True)
            if i % 20 == 0:
                algo.save(str(args.out.resolve()))
    algo.save(str(args.out.resolve()))
    print(f"Saved checkpoint to {args.out}")
    algo.stop()


if __name__ == "__main__":
    main()
