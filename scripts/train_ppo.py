"""Train PPO traffic signal agents on TimedSignalEnv (realistic signal timing).

All intersections share one policy (parameter sharing): each agent acts on its
own observation, and experience from every intersection trains the same network.
With two near-identical intersections this learns faster than one policy per
agent, and the trained policy also works on any intersection with the same layout.

Usage:
  python scripts/train_ppo.py --iterations 100
  python scripts/train_ppo.py --reward pressure --out models/ppo_pressure

Writes a checkpoint to --out and a per-iteration log to <out>/training_log.csv.
Evaluate with: python scripts/evaluate.py --ppo-checkpoint models/ppo_timed
"""
import argparse
import csv
import os
import time
from pathlib import Path

os.environ.setdefault("RLLIB_DISABLE_NEW_API_STACK", "True")

import ray  # noqa: E402
from ray.rllib.algorithms.callbacks import DefaultCallbacks  # noqa: E402
from ray.rllib.algorithms.ppo import PPO, PPOConfig  # noqa: E402
from ray.tune.registry import register_env  # noqa: E402

from traffic_env import REPO_ROOT  # noqa: E402

SCRIPTS_DIR = Path(__file__).resolve().parent
ENV_NAME = "TimedSignalEnv"
POLICY_ID = "shared_policy"


def env_creator(config):
    from signal_env import TimedSignalEnv
    # Give every env runner its own scenario stream.
    worker = getattr(config, "worker_index", 0)
    vector = getattr(config, "vector_index", 0)
    config = dict(config)
    config.setdefault("rng_seed", 1000 * worker + vector)
    return TimedSignalEnv(**config)


class CostMetrics(DefaultCallbacks):
    """Report each episode's constraint cost per agent decision, and traffic outcomes."""

    def on_episode_end(self, *, worker, base_env, policies, episode, env_index=None, **kwargs):
        env = base_env.get_sub_environments()[env_index or 0]
        steps = max(1, episode.length * env.num_intersections)
        episode.custom_metrics["cost_rate"] = env.episode_cost / steps
        episode.custom_metrics["intervention_rate"] = env.interventions / max(1, env.decisions)
        episode.custom_metrics["max_red_s"] = env._max_red_age


def policy_mapping_fn(agent_id, *args, **kwargs):
    return POLICY_ID


def build_config(env_config, num_env_runners=3, lr=3e-4, entropy_coeff=0.01, gamma=0.95, vf_clip_param=1e4,
                 train_batch_size=4000, minibatch_size=256, num_epochs=10, hidden=(128, 128), seed=0,
                 count_steps_by="env_steps"):
    register_env(ENV_NAME, env_creator)
    return (
        PPOConfig()
        .api_stack(enable_rl_module_and_learner=False, enable_env_runner_and_connector_v2=False)
        .environment(env=ENV_NAME, env_config=env_config)
        .framework("torch")
        .env_runners(num_env_runners=num_env_runners, rollout_fragment_length="auto")
        .training(
            gamma=gamma, lr=lr, entropy_coeff=entropy_coeff, train_batch_size=train_batch_size,
            minibatch_size=minibatch_size, num_epochs=num_epochs, clip_param=0.2, grad_clip=0.5,
            # RLlib's default vf_clip_param (10) is far below the size of these
            # returns, which stops the value function from learning.
            vf_clip_param=vf_clip_param,
            model={"fcnet_hiddens": list(hidden), "fcnet_activation": "tanh"},
        )
        .multi_agent(policies={POLICY_ID}, policy_mapping_fn=policy_mapping_fn, count_steps_by=count_steps_by)
        .callbacks(CostMetrics)
        .debugging(seed=seed)
    )


def init_ray():
    if not ray.is_initialized():
        ray.init(include_dashboard=False, runtime_env={"env_vars": {"PYTHONPATH": str(SCRIPTS_DIR)}},
                 log_to_driver=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--reward", choices=["queue", "pressure", "legacy", "fair"], default="queue")
    parser.add_argument("--fair-alpha", type=float, default=1.0, help="weight of the squared-wait term (--reward fair)")
    parser.add_argument("--num-env-runners", type=int, default=3)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--entropy-coeff", type=float, default=0.01)
    parser.add_argument("--gamma", type=float, default=0.95,
                        help="discount per 5 s decision; 0.95 looks roughly 100 s ahead")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--network", choices=["1x2", "4x4", "hangzhou", "jinan"], default="1x2")
    parser.add_argument("--batch-size", type=int, default=4000)
    parser.add_argument("--count-steps-by", choices=["env_steps", "agent_steps"], default="env_steps",
                        help="unit of --batch-size; on the 4x4 grid one env step is 16 agent steps")
    parser.add_argument("--neighbors", action="store_true",
                        help="add the adjacent intersections' queues to each agent's observation")
    parser.add_argument("--max-red", type=int, default=None,
                        help="longest a lane may hold a stopped vehicle at red (s); enforced by overriding the agent")
    parser.add_argument("--wait-obs", action="store_true",
                        help="add each lane's red time so far (as a fraction of --max-red) to the observation")
    parser.add_argument("--shield", choices=["none", "cap", "deadline"], default=None,
                        help="how --max-red is enforced during training (default: cap)")
    parser.add_argument("--cost", choices=["intervention", "overdue"], default=None,
                        help="constraint cost, handled with a Lagrange multiplier on the reward")
    parser.add_argument("--cost-target", type=float, default=0.0,
                        help="allowed average cost per agent decision")
    parser.add_argument("--lambda-init", type=float, default=0.0)
    parser.add_argument("--lambda-max", type=float, default=5.0)
    parser.add_argument("--lambda-lr", type=float, default=5.0,
                        help="dual ascent step: lambda += lr * (cost rate - target) after each iteration")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "models" / "ppo_timed")
    args = parser.parse_args()

    init_ray()
    env_config = {"reward": args.reward, "network": args.network, "neighbors": args.neighbors,
                  "max_red": args.max_red, "wait_obs": args.wait_obs, "shield": args.shield, "cost": args.cost,
                  "cost_weight": args.lambda_init, "fair_alpha": args.fair_alpha}
    algo = build_config(env_config, num_env_runners=args.num_env_runners, lr=args.lr,
                        entropy_coeff=args.entropy_coeff, gamma=args.gamma, seed=args.seed,
                        train_batch_size=args.batch_size, count_steps_by=args.count_steps_by).build()

    args.out.mkdir(parents=True, exist_ok=True)
    start = time.time()
    with open(args.out / "training_log.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["iteration", "env_steps", "episodes", "episode_reward_mean", "minutes", "cost_rate",
                         "intervention_rate", "max_red_s", "lambda"])
        lam = args.lambda_init
        for i in range(1, args.iterations + 1):
            result = algo.train()
            er = result["env_runners"]
            cm = er.get("custom_metrics", {})
            cost_rate = cm.get("cost_rate_mean")
            # Lagrangian dual ascent on the cost multiplier (only after iterations that finished episodes).
            if args.cost and cost_rate is not None:
                lam = min(args.lambda_max, max(0.0, lam + args.lambda_lr * (cost_rate - args.cost_target)))
                algo.env_runner_group.foreach_worker(
                    lambda w, lam=lam: w.foreach_env(lambda e: e.set_cost_weight(lam)))
            row = [i, result.get("num_env_steps_sampled_lifetime", result.get("num_env_steps_sampled")),
                   er.get("num_episodes_lifetime", er.get("episodes_total")), er.get("episode_reward_mean"),
                   round((time.time() - start) / 60, 2), cost_rate, cm.get("intervention_rate_mean"),
                   cm.get("max_red_s_mean"), lam]
            writer.writerow(row)
            f.flush()
            print(f"iter {i:4d}  reward {row[3]}  cost {cost_rate}  lambda {lam:.3f}  ({row[4]} min)", flush=True)
            if i % 25 == 0 or i == args.iterations:
                algo.save(str(args.out.resolve()))
    print(f"Saved checkpoint to {args.out}")
    algo.stop()


if __name__ == "__main__":
    main()
