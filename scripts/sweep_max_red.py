"""Trade-off between average travel time and the worst red time, for several ways
of bounding red time.

For each method and limit, run every evaluation seed and record the average
travel time and the longest red any lane with a waiting car actually got
(max_red_s, the worst over seeds as well as the mean). A method that sets a
limit R but lets lanes wait longer than R is plotted at the red time it really
delivered, so methods are compared at equal worst-case outcomes.

Methods:
  cap        the naive max-red rule (switch once a lane reaches R) on a policy
  deadline   the deadline shield (scripts/deadline_shield.py) on a policy
  actuated   actuated control with maximum green G (its own way of bounding red)

Usage:
  python scripts/sweep_max_red.py --policy models/ppo_timed --scenarios high
"""
import argparse
import csv
from pathlib import Path

import numpy as np

from controllers import Actuated
from evaluate import load_policy, run_episode, run_policy_episode
from scenarios import NETWORKS, generate_flow
from signal_env import TimedSignalEnv
from traffic_env import CONFIGS_DIR, REPO_ROOT, TrafficSignalEnv


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--policy", action="append", default=[], metavar="[NAME=]CHECKPOINT",
                        help="policy to bound (repeatable); several checkpoints with the same NAME are "
                             "treated as training seeds of one method")
    parser.add_argument("--network", default="1x2", choices=list(NETWORKS))
    parser.add_argument("--scenarios", nargs="+", default=["high"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--cap", nargs="*", type=int, default=[70, 75, 80, 90, 100, 120])
    parser.add_argument("--deadline", nargs="*", type=int, default=[65, 70, 75, 80, 90, 100, 120])
    parser.add_argument("--actuated-max-green", nargs="*", type=int, default=[20, 25, 30, 40, 60])
    parser.add_argument("--native", action="store_true",
                        help="also run each policy under the rules it was trained with")
    parser.add_argument("--horizon", type=int, default=3600)
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "results" / "sweep_max_red")
    args = parser.parse_args()

    runs = []  # (method, limit, policy_name, callable(flow_file, seed) -> metrics)
    for spec in args.policy:
        name, _, path = spec.rpartition("=")
        name = name or Path(path).name
        act = load_policy(path)
        cfg = act.env_config
        base = {"network": args.network, "neighbors": bool(cfg.get("neighbors", False)),
                "wait_obs": bool(cfg.get("wait_obs", False))}

        def make(env_kwargs, act=act):
            def run(flow_file, seed):
                env = TimedSignalEnv(horizon=args.horizon, flow_file=flow_file, seed=seed, **env_kwargs)
                return run_policy_episode(env, act)
            return run

        trained_shield = cfg.get("shield")
        if args.native and cfg.get("max_red"):
            runs.append((f"{name} (as trained)", cfg["max_red"], name,
                         make({**base, "max_red": cfg["max_red"], "shield": trained_shield})))
        if args.native and not cfg.get("max_red"):
            runs.append((f"{name} (no limit)", None, name, make(base)))
        # A policy trained with the deadline flag in its observation needs the deadline shield.
        if trained_shield != "deadline":
            for limit in args.cap:
                runs.append((f"{name} + cap", limit, name, make({**base, "max_red": limit, "shield": "cap"})))
        if not base["wait_obs"] or trained_shield == "deadline":
            for limit in args.deadline:
                runs.append((f"{name} + deadline shield", limit, name,
                             make({**base, "max_red": limit, "shield": "deadline"})))
    for g in args.actuated_max_green:
        def run(flow_file, seed, g=g):
            env = TrafficSignalEnv(args.horizon, config_file=CONFIGS_DIR / NETWORKS[args.network]["config"],
                                   flow_file=flow_file, seed=seed, save_replay=False)
            return run_episode(env, Actuated(max_green=g))
        runs.append(("actuated", g, "actuated", run))

    rows = []
    for scenario in args.scenarios:
        for seed in args.seeds:
            flow_file = generate_flow(scenario, seed, args.horizon, network=args.network)
            for method, limit, name, run in runs:
                m = run(flow_file, seed)
                rows.append({"scenario": scenario, "seed": seed, "method": method, "limit": limit, "policy": name,
                             **{k: m.get(k, 0.0) for k in ("avg_travel_time_s", "max_red_s", "max_stop_s",
                                                         "p95_wait_s", "intervention_rate")}})
                print(f"{scenario} seed={seed} {method:>32} {str(limit):>4}: "
                      f"{m['avg_travel_time_s']:6.1f}s  max red {m['max_red_s']:4.0f}s", flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    with open(args.out / "runs.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    # Summary: per (scenario, method, limit), mean over seeds (and checkpoints).
    groups = {}
    for r in rows:
        groups.setdefault((r["scenario"], r["method"], r["limit"]), []).append(r)
    lines = ["| Scenario | Method | Limit | Avg travel time (s) | Max red, mean (s) | Max red, worst (s) "
             "| Longest stop, mean (s) | 95th pct. stopped (s) |", "|---|---|---|---|---|---|---|---|"]
    summary = []
    for (scenario, method, limit), g in groups.items():
        tt = np.array([r["avg_travel_time_s"] for r in g])
        red = np.array([r["max_red_s"] for r in g])
        stop = np.array([r["max_stop_s"] for r in g])
        p95 = np.array([r["p95_wait_s"] for r in g])
        summary.append({"scenario": scenario, "method": method, "limit": limit, "tt": tt.mean(), "tt_std": tt.std(ddof=1),
                        "red_mean": red.mean(), "red_worst": red.max(), "stop_mean": stop.mean()})
        lines.append(f"| {scenario} | {method} | {limit} | {tt.mean():.1f} ± {tt.std(ddof=1):.1f} | {red.mean():.0f} "
                     f"| {red.max():.0f} | {stop.mean():.0f} | {p95.mean():.0f} |")
    (args.out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for scenario in args.scenarios:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for method in dict.fromkeys(s["method"] for s in summary):
            pts = sorted((s["red_worst"], s["tt"]) for s in summary if s["scenario"] == scenario and s["method"] == method)
            ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="o", label=method)
        ax.set_xlabel("Worst red time with a car waiting, over all seeds (s)")
        ax.set_ylabel("Average travel time (s)")
        ax.set_title(f"Efficiency vs worst-case red time ({args.network}, {scenario} demand)")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(args.out / f"pareto_{scenario}.png", dpi=150)
        plt.close(fig)


if __name__ == "__main__":
    main()
