"""Trade-off between average travel time and the worst red time, for several ways
of bounding red time.

For each method and limit this runs scripts/evaluate.py on every evaluation seed
and records the average travel time and the longest red any lane with a waiting
car actually got (max_red_s). A method that is set to a limit R but lets lanes
wait longer than R is compared at the red time it really delivered.

Methods:
  <name> + cap             the naive max-red rule (switch once a lane reaches R)
  <name> + deadline shield the deadline shield (scripts/deadline_shield.py)
  actuated                 actuated control with maximum green G, its own way of
                           bounding red time

Each (method, limit, checkpoint) runs in its own evaluate.py process with a
timeout: CityFlow very occasionally deadlocks in its thread barrier, and a hung
run is simply retried.

Usage:
  python scripts/sweep_max_red.py --policy ppo=models/ppo_timed --policy mp=@max_pressure
  python scripts/sweep_max_red.py --policy ppo=run_s0 --policy ppo=run_s1   # seeds of one method
"""
import argparse
import csv
import subprocess
import sys
from pathlib import Path

import numpy as np

from traffic_env import REPO_ROOT

SCRIPTS = Path(__file__).resolve().parent


def run_eval(args, out, controllers=(), policy=None, retries=3):
    if (out / "runs.csv").exists():
        return list(csv.DictReader(open(out / "runs.csv")))
    cmd = [sys.executable, str(SCRIPTS / "evaluate.py"), "--network", args.network, "--horizon", str(args.horizon),
           "--scenarios", *args.scenarios, "--seeds", *map(str, args.seeds), "--out", str(out),
           "--controllers", *controllers]
    if policy:
        cmd += ["--policy", policy]
    for attempt in range(retries):
        try:
            subprocess.run(cmd, check=True, timeout=args.timeout, stdout=subprocess.DEVNULL)
            return list(csv.DictReader(open(out / "runs.csv")))
        except subprocess.TimeoutExpired:
            print(f"  timed out (attempt {attempt + 1}), retrying: {out.name}", flush=True)
    raise RuntimeError(f"{out.name} failed {retries} times")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--policy", action="append", default=[], metavar="NAME=CHECKPOINT")
    parser.add_argument("--network", default="1x2")
    parser.add_argument("--scenarios", nargs="+", default=["high"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--cap", nargs="*", type=int, default=[75, 90, 100, 120])
    parser.add_argument("--deadline", nargs="*", type=int, default=[70, 75, 80, 90, 100, 120])
    parser.add_argument("--actuated-max-green", nargs="*", type=int, default=[25, 30, 40, 60])
    parser.add_argument("--native", action="store_true", help="also run each policy as trained")
    parser.add_argument("--horizon", type=int, default=3600)
    parser.add_argument("--timeout", type=int, default=1800, help="seconds before a run counts as hung")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "results" / "sweep_max_red")
    args = parser.parse_args()
    runs_dir = args.out / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    jobs = []  # (method, limit, label, kwargs for run_eval)
    for k, spec in enumerate(args.policy):
        name, _, path = spec.partition("=")
        tag = f"{name}{k}"
        if args.native:
            jobs.append((f"{name} (as trained)", None, f"{tag}_native", {"policy": f"p={path}"}))
        for limit in args.cap:
            jobs.append((f"{name} + cap", limit, f"{tag}_cap{limit}", {"policy": f"p={path}:max_red={limit}"}))
        for limit in args.deadline:
            jobs.append((f"{name} + deadline shield", limit, f"{tag}_ds{limit}",
                         {"policy": f"p={path}:max_red={limit}:shield=deadline"}))
    for g in args.actuated_max_green:
        jobs.append(("actuated", g, f"actuated_g{g}", {"controllers": [f"actuated:max_green={g}"]}))

    rows = []
    for method, limit, label, kwargs in jobs:
        for r in run_eval(args, runs_dir / label, **kwargs):
            rows.append({"method": method, "limit": limit, "label": label, **r})
        print(f"done {label}", flush=True)

    with open(args.out / "runs.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    groups = {}
    for r in rows:
        groups.setdefault((r["scenario"], r["method"], r["limit"]), []).append(r)
    summary = []
    lines = ["| Scenario | Method | Limit | Avg travel time (s) | Longest red, mean (s) | Longest red, worst (s) "
             "| Longest stop, mean (s) | 95th pct. stopped (s) | Episodes |", "|---|---|---|---|---|---|---|---|---|"]
    for (scenario, method, limit), g in groups.items():
        col = {m: np.array([float(r[m]) for r in g]) for m in ("avg_travel_time_s", "max_red_s", "max_stop_s",
                                                               "p95_wait_s")}
        tt, red = col["avg_travel_time_s"], col["max_red_s"]
        summary.append({"scenario": scenario, "method": method, "limit": limit, "tt": tt.mean(),
                        "red_worst": red.max()})
        lines.append(f"| {scenario} | {method} | {limit if limit is not None else '-'} | {tt.mean():.1f} ± "
                     f"{tt.std(ddof=1):.1f} | {red.mean():.0f} | {red.max():.0f} | {col['max_stop_s'].mean():.0f} "
                     f"| {col['p95_wait_s'].mean():.0f} | {len(g)} |")
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
        ax.set_xlabel("Longest red with a car waiting, worst over seeds (s)")
        ax.set_ylabel("Average travel time (s)")
        ax.set_title(f"Average travel time vs worst red ({args.network}, {scenario} demand)")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(args.out / f"tradeoff_{scenario}.png", dpi=150)
        plt.close(fig)


if __name__ == "__main__":
    main()
