"""Compare traffic signal controllers on traffic outcomes, not RL reward.

For every (scenario, seed, controller) this runs one full episode and records:
  avg_travel_time_s  CityFlow's average travel time over all vehicles that entered
                     (vehicles still in the network count up to the episode end)
  avg_queue          mean number of waiting vehicles per signalised intersection
  throughput         vehicles that completed their trip
  completion_rate    throughput / vehicles that entered the network
  total_reward       the environment's reward summed over agents, for reference

Usage (from any directory):
  python scripts/evaluate.py                       # all baselines, all scenarios, 5 seeds
  python scripts/evaluate.py --scenarios medium --seeds 0 1 --horizon 600

Writes runs.csv (one row per episode), summary.csv and summary.md (mean and std
over seeds) and travel_time.png to results/evaluation/ by default.
"""
import argparse
import csv
import time
from pathlib import Path

import numpy as np

from controllers import BASELINES
from scenarios import DEMAND_LEVELS, ORIGINAL_SCENARIO, generate_flow
from traffic_env import REPO_ROOT, TrafficSignalEnv

METRICS = ["avg_travel_time_s", "avg_queue", "throughput", "completion_rate", "total_reward"]
DEFAULT_OUT = REPO_ROOT / "results" / "evaluation"


def run_episode(env, controller):
    env.reset()
    controller.reset(env)
    incoming_lanes = [
        [lane for road in env.incoming_roads[i] for lane in env.roads[road]]
        for i in range(env.num_intersections)
    ]
    queue_samples = []
    total_reward = 0.0
    seen = set()
    t = 0
    done = False
    while not done:
        actions = controller.act(env, t)
        _, rewards, terminateds, _, _ = env.step(actions)
        total_reward += sum(rewards.values())
        waiting = env.engine.get_lane_waiting_vehicle_count()
        queue_samples.append(np.mean([sum(waiting.get(l, 0) for l in lanes) for lanes in incoming_lanes]))
        seen.update(env.engine.get_vehicles(include_waiting=False))
        done = terminateds["__all__"]
        t += 1

    in_network = set(env.engine.get_vehicles(include_waiting=False))
    throughput = len(seen - in_network)
    return {
        "avg_travel_time_s": env.engine.get_average_travel_time(),
        "avg_queue": float(np.mean(queue_samples)),
        "throughput": throughput,
        "completion_rate": throughput / len(seen) if seen else float("nan"),
        "total_reward": total_reward,
    }


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
    parser.add_argument("--scenarios", nargs="+", default=list(DEMAND_LEVELS) + [ORIGINAL_SCENARIO],
                        choices=list(DEMAND_LEVELS) + [ORIGINAL_SCENARIO])
    parser.add_argument("--controllers", nargs="+", default=list(BASELINES), choices=list(BASELINES))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--horizon", type=int, default=3600, help="episode length in simulated seconds")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    rows = []
    for scenario in args.scenarios:
        # The original flow file is deterministic, so extra seeds would just repeat it.
        seeds = args.seeds[:1] if scenario == ORIGINAL_SCENARIO else args.seeds
        for seed in seeds:
            flow_file = generate_flow(scenario, seed, args.horizon)
            for name in args.controllers:
                env = TrafficSignalEnv(args.horizon, flow_file=flow_file, seed=seed, save_replay=False)
                start = time.time()
                metrics = run_episode(env, BASELINES[name]())
                rows.append({"scenario": scenario, "seed": seed, "controller": name, **metrics})
                print(f"{scenario:>8} seed={seed} {name:>16}: travel time {metrics['avg_travel_time_s']:7.1f}s, "
                      f"queue {metrics['avg_queue']:6.2f}, throughput {metrics['throughput']:5d} "
                      f"({time.time() - start:.1f}s)")

    summary = summarize(rows)
    table = write_outputs(rows, summary, args.out, args.scenarios, args.controllers)
    print()
    print(table)
    print(f"Results written to {args.out}")


if __name__ == "__main__":
    main()
