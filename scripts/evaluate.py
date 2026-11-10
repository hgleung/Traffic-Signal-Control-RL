"""Compare traffic signal controllers on traffic outcomes, not RL reward.

For every (scenario, seed, controller) this runs one full episode and records:
  avg_travel_time_s  CityFlow's average travel time over all vehicles that entered
                     (vehicles still in the network count up to the episode end)
  avg_queue          mean number of waiting vehicles per signalised intersection
  throughput         vehicles that completed their trip
  completion_rate    throughput / vehicles that entered the network
  total_reward       the environment's reward summed over agents, for reference only
                     (trained policies run in TimedSignalEnv, whose reward differs)

Usage (from any directory):
  python scripts/evaluate.py                       # all baselines, all scenarios, 5 seeds
  python scripts/evaluate.py --scenarios medium --seeds 0 1 --horizon 600
  python scripts/evaluate.py --policy ppo=models/ppo_timed   # add a trained policy

Writes runs.csv (one row per episode), summary.csv and summary.md (mean and std
over seeds) and travel_time.png to results/evaluation/ by default.
"""
import argparse
import csv
import time
from pathlib import Path

import numpy as np

from controllers import BASELINES
from scenarios import NETWORKS, ORIGINAL_SCENARIO, generate_flow
from signal_env import TimedSignalEnv
from traffic_env import CONFIGS_DIR, REPO_ROOT, TrafficSignalEnv

METRICS = ["avg_travel_time_s", "avg_queue", "throughput", "completion_rate", "p95_wait_s", "max_wait_s",
           "max_stop_s", "max_red_s", "p95_trip_s", "max_trip_s", "intervention_rate", "total_reward"]
DEFAULT_OUT = REPO_ROOT / "results" / "evaluation"


def run_episode(env, controller):
    env.reset()
    controller.reset(env)
    total_reward = 0.0
    t = 0
    done = False
    while not done:
        actions = controller.act(env, t)
        _, rewards, terminateds, _, _ = env.step(actions)
        total_reward += sum(rewards.values())
        done = terminateds["__all__"]
        t += 1
    return {**env.episode_metrics(), "total_reward": total_reward}


def load_policy(checkpoint_dir, explore=False):
    """Load a trained RLlib policy (see train_ppo.py) and return obs_batch -> actions.

    explore=False takes the most likely action (deterministic); explore=True samples
    from the policy's action distribution, as during training.
    """
    if str(checkpoint_dir) in RULE_POLICIES:
        return RULE_POLICIES[str(checkpoint_dir)]()
    import os
    os.environ.setdefault("RLLIB_DISABLE_NEW_API_STACK", "True")
    from ray.rllib.policy.policy import Policy
    from train_ppo import POLICY_ID

    policy = Policy.from_checkpoint(str(Path(checkpoint_dir).resolve() / "policies" / POLICY_ID))
    if explore:
        import torch
        torch.manual_seed(0)  # reproducible sampled actions

    def act(obs_batch):
        actions, _, _ = policy.compute_actions(np.stack(obs_batch), explore=explore)
        return actions

    # The environment settings the policy was trained with (network, neighbour observations).
    act.env_config = dict(policy.config.get("env_config", {}))
    return act


def max_pressure_policy():
    """Max-pressure as a TimedSignalEnv policy, so it can run under the same shields
    as the RL agents. Every decision it requests the green phase with the largest
    pressure (incoming minus mean outgoing vehicles over its non-right-turn movements);
    the environment applies the minimum green and clearance rules."""
    state = {}

    def act(obs_batch):
        env = state["env"]
        if state.get("built") is not env:
            state["movements"] = []
            for intersection in env.intersections:
                links = intersection["roadLinks"]
                phases = {}
                for p, phase in enumerate(intersection["trafficLight"]["lightphases"]):
                    if p == 0:
                        continue
                    phases[p] = [([f"{links[k]['startRoad']}_{ll['startLaneIndex']}" for ll in links[k]["laneLinks"]],
                                  [f"{links[k]['endRoad']}_{ll['endLaneIndex']}" for ll in links[k]["laneLinks"]])
                                 for k in phase["availableRoadLinks"] if links[k]["type"] != "turn_right"]
                state["movements"].append(phases)
            state["built"] = env
        counts = env.engine.get_lane_vehicle_count()
        actions = []
        for idx, phases in enumerate(state["movements"]):
            def pressure(p):
                return sum(sum(counts.get(l, 0) for l in set(a)) - np.mean([counts.get(l, 0) for l in b])
                           for a, b in phases[p])
            best = max(phases, key=pressure)
            if pressure(env.phase[idx]) >= pressure(best):
                best = env.phase[idx]  # keep the current phase on ties
            actions.append(best - 1)
        return actions

    act.bind = lambda env: state.__setitem__("env", env)
    act.env_config = {}
    return act


RULE_POLICIES = {"@max_pressure": max_pressure_policy}


def run_policy_episode(env, act):
    """Run a trained policy on TimedSignalEnv (decisions every few seconds, timing rules enforced)."""
    if hasattr(act, "bind"):
        act.bind(env)
    obs, _ = env.reset()
    total_reward = 0.0
    done = False
    while not done:
        actions = act([obs[a] for a in env.agents])
        obs, rewards, _, truncateds, _ = env.step(dict(zip(env.agents, actions)))
        total_reward += sum(rewards.values())
        done = truncateds["__all__"]
    return {**env.episode_metrics(), "total_reward": total_reward}


def summarize(rows):
    groups = {}
    for row in rows:
        groups.setdefault((row["scenario"], row["controller"]), []).append(row)
    summary = []
    for (scenario, controller), group in groups.items():
        entry = {"scenario": scenario, "controller": controller, "episodes": len(group)}
        for metric in METRICS:
            values = np.array([r[metric] for r in group], dtype=float)
            entry[f"{metric}_mean"] = float(values.mean())
            entry[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        summary.append(entry)
    return summary


def write_outputs(rows, summary, out_dir, scenarios, controllers):
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "runs.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    with open(out_dir / "summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)

    lookup = {(s["scenario"], s["controller"]): s for s in summary}
    lines = [
        "| Scenario | Controller | Avg travel time (s) | Avg queue (veh) | Throughput (veh) | Completed | Episodes |",
        "|---|---|---|---|---|---|---|",
    ]
    for scenario in scenarios:
        for controller in controllers:
            s = lookup[(scenario, controller)]
            lines.append(
                f"| {scenario} | {controller} "
                f"| {s['avg_travel_time_s_mean']:.1f} ± {s['avg_travel_time_s_std']:.1f} "
                f"| {s['avg_queue_mean']:.2f} ± {s['avg_queue_std']:.2f} "
                f"| {s['throughput_mean']:.0f} ± {s['throughput_std']:.0f} "
                f"| {100 * s['completion_rate_mean']:.1f}% "
                f"| {s['episodes']} |"
            )
    lines += [
        "",
        "Worst-case waits (per vehicle, mean ± std over seeds of each episode's value):",
        "",
        "| Scenario | Controller | 95th pct. time stopped (s) | Max time stopped (s) | Longest single stop (s) "
        "| Longest red with a car waiting (s) | 95th pct. trip (s) | Longest trip (s) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for scenario in scenarios:
        for controller in controllers:
            s = lookup[(scenario, controller)]
            lines.append(
                f"| {scenario} | {controller} "
                + "".join(f"| {s[m + '_mean']:.0f} ± {s[m + '_std']:.0f} "
                          for m in ("p95_wait_s", "max_wait_s", "max_stop_s", "max_red_s", "p95_trip_s", "max_trip_s"))
                + "|"
            )
    table = "\n".join(lines) + "\n"
    (out_dir / "summary.md").write_text(table)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return table
    # The original flow file is so oversaturated (travel times > 1000 s) that it
    # would flatten every other bar, so it is reported in the table only.
    scenarios = [s for s in scenarios if s != ORIGINAL_SCENARIO]
    if not scenarios:
        return table
    fig, ax = plt.subplots(figsize=(8, 4.5))
    width = 0.8 / len(controllers)
    x = np.arange(len(scenarios))
    for k, controller in enumerate(controllers):
        means = [lookup[(s, controller)]["avg_travel_time_s_mean"] for s in scenarios]
        stds = [lookup[(s, controller)]["avg_travel_time_s_std"] for s in scenarios]
        ax.bar(x + (k - (len(controllers) - 1) / 2) * width, means, width, yerr=stds, capsize=3, label=controller)
    ax.set_xticks(x, scenarios)
    ax.set_ylabel("Average travel time (s)")
    ax.set_title("Average travel time (mean ± std over seeds; lower is better)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "travel_time.png", dpi=150)
    plt.close(fig)
    return table


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scenarios", nargs="+", default=None,
                        help="demand levels (low/medium/high/original), or recorded flow files on the real "
                             "networks (default: all of the network's)")
    parser.add_argument("--controllers", nargs="+", default=list(BASELINES), choices=list(BASELINES))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--horizon", type=int, default=3600, help="episode length in simulated seconds")
    parser.add_argument("--policy", nargs="+", default=[], metavar="NAME=CHECKPOINT",
                        help="also evaluate trained policies, e.g. ppo=models/ppo_timed "
                             "(append :sample to sample actions instead of taking the most likely one, "
                             "or :max_red=60 to enforce a max-red limit the policy wasn't trained with)")
    parser.add_argument("--network", choices=list(NETWORKS), default="1x2",
                        help="road network (1x2 corridor or 4x4 grid); see scripts/scenarios.py")
    parser.add_argument("--out", type=Path, default=None,
                        help="output directory (default results/evaluation, or results/evaluation_<network>)")
    args = parser.parse_args()

    policies = {}
    policy_env = {}
    for spec in args.policy:
        name, _, path = spec.partition("=")
        path, *options = path.split(":")
        policies[name] = act = load_policy(path, explore="sample" in options)
        trained_on = act.env_config.get("network", "1x2")
        if trained_on != args.network:
            print(f"warning: policy {name!r} was trained on the {trained_on} network, evaluating on {args.network}")
        # Run each policy under the rules it was trained with, unless overridden.
        policy_env[name] = {"neighbors": bool(act.env_config.get("neighbors", False)),
                            "max_red": act.env_config.get("max_red"),
                            "wait_obs": bool(act.env_config.get("wait_obs", False)),
                            "shield": act.env_config.get("shield")}
        for option in options:
            if option.startswith("max_red="):
                policy_env[name]["max_red"] = int(option.split("=", 1)[1])
            elif option.startswith("shield="):
                policy_env[name]["shield"] = option.split("=", 1)[1]
        if policy_env[name]["shield"] == "none" and not policy_env[name]["wait_obs"]:
            policy_env[name]["shield"] = None  # tracking only; no limit to enforce
    controllers = list(args.controllers) + list(policies)
    if args.out is None:
        args.out = DEFAULT_OUT if args.network == "1x2" else REPO_ROOT / "results" / f"evaluation_{args.network}"
    levels = list(NETWORKS[args.network]["levels"])
    if args.scenarios is None:
        args.scenarios = levels + ([ORIGINAL_SCENARIO] if args.network == "1x2" else [])
    for scenario in args.scenarios:
        if scenario not in levels and not (scenario == ORIGINAL_SCENARIO and args.network == "1x2"):
            parser.error(f"unknown scenario {scenario!r} for network {args.network} (choose from {levels})")

    rows = []
    for scenario in args.scenarios:
        # The original flow file is deterministic, so extra seeds would just repeat it.
        seeds = args.seeds[:1] if scenario == ORIGINAL_SCENARIO else args.seeds
        for seed in seeds:
            flow_file = generate_flow(scenario, seed, args.horizon, network=args.network)
            for name in controllers:
                start = time.time()
                if name in policies:
                    env = TimedSignalEnv(horizon=args.horizon, flow_file=flow_file, seed=seed, network=args.network,
                                         **policy_env[name])
                    metrics = run_policy_episode(env, policies[name])
                else:
                    env = TrafficSignalEnv(args.horizon, config_file=CONFIGS_DIR / NETWORKS[args.network]["config"],
                                           flow_file=flow_file, seed=seed, save_replay=False)
                    metrics = run_episode(env, BASELINES[name]())
                rows.append({"scenario": scenario, "seed": seed, "controller": name,
                             **{m: metrics.get(m, 0.0) for m in METRICS}})
                print(f"{scenario:>8} seed={seed} {name:>16}: travel time {metrics['avg_travel_time_s']:7.1f}s, "
                      f"queue {metrics['avg_queue']:6.2f}, throughput {metrics['throughput']:5d} "
                      f"({time.time() - start:.1f}s)")

    summary = summarize(rows)
    table = write_outputs(rows, summary, args.out, args.scenarios, controllers)
    print()
    print(table)
    print(f"Results written to {args.out}")


if __name__ == "__main__":
    main()
