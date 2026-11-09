"""Tests for the 4x4 grid network, its traffic and neighbour observations."""
import json
import sys
from pathlib import Path

import pytest

pytest.importorskip("cityflow", reason="CityFlow is not installed (see README)")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from controllers import BASELINES  # noqa: E402
from evaluate import run_episode  # noqa: E402
from scenarios import NETWORKS, generate_flow  # noqa: E402
from signal_env import TimedSignalEnv  # noqa: E402
from traffic_env import CONFIGS_DIR, TrafficSignalEnv  # noqa: E402

HORIZON = 600


def test_grid_traffic_is_uneven_and_peaks_mid_episode():
    flows = json.loads((CONFIGS_DIR / generate_flow("high", 0, 3600, network="4x4")).read_text())
    by_entry = {}
    for f in flows:
        by_entry[f["route"][0]] = by_entry.get(f["route"][0], 0) + 1
    arterial = [n for road, n in by_entry.items() if road.endswith("_0") and road.split("_")[2] in ("2", "3")]
    other = [n for road, n in by_entry.items() if road.endswith("_1")]
    assert min(arterial) > max(other)  # arterials carry about twice the traffic
    middle = sum(1200 <= f["startTime"] < 2400 for f in flows)
    edges = sum(f["startTime"] < 600 or f["startTime"] >= 3000 for f in flows)
    assert middle > 1.5 * edges  # rush hour peak
    assert any(len(f["route"]) > 6 for f in flows)  # vehicles turn more than once


@pytest.mark.parametrize("name", ["fixed_time", "actuated", "max_pressure"])
def test_baselines_run_on_grid(name):
    env = TrafficSignalEnv(HORIZON, config_file=CONFIGS_DIR / NETWORKS["4x4"]["config"],
                           flow_file=generate_flow("medium", 0, HORIZON, network="4x4"), save_replay=False)
    assert env.num_intersections == 16
    metrics = run_episode(env, BASELINES[name]())
    assert metrics["throughput"] > 0


def test_neighbor_observations():
    env = TimedSignalEnv(horizon=HORIZON, network="4x4", neighbors=True,
                         flow_file=generate_flow("high", 0, HORIZON, network="4x4"), seed=0)
    env.reset()
    # Corner intersections have two signalised neighbours, inner ones four.
    counts = sorted(sum(n is not None for n in row) for row in env.neighbor_of)
    assert counts.count(2) == 4 and counts.count(4) == 4 and counts.count(3) == 8
    for _ in range(60):
        obs, *_ = env.step({a: 0 for a in env.agents})
    assert all(env.observation_space[a].contains(o) for a, o in obs.items())
    assert any(o[-16:].sum() > 0 for o in obs.values())
