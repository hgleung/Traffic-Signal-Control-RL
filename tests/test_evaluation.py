"""Tests for the baseline controllers, scenario generator and evaluation harness."""
import math
import sys
from pathlib import Path

import pytest

pytest.importorskip("cityflow", reason="CityFlow is not installed (see README)")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from controllers import BASELINES, CLEARANCE_PHASE, FixedTime, MaxPressure  # noqa: E402
from evaluate import run_episode  # noqa: E402
from scenarios import candidate_routes, generate_flow  # noqa: E402
from traffic_env import CONFIGS_DIR, TrafficSignalEnv  # noqa: E402

HORIZON = 120


def test_routes_cover_every_entry_and_turn():
    routes = candidate_routes()
    assert len(routes) == 6
    for options in routes.values():
        assert len(options) == 3
        assert math.isclose(sum(p for _, p in options), 1.0)


def test_generated_flow_is_seeded():
    a = (CONFIGS_DIR / generate_flow("medium", 3, HORIZON)).read_text()
    b = (CONFIGS_DIR / generate_flow("medium", 3, HORIZON)).read_text()
    c = (CONFIGS_DIR / generate_flow("medium", 4, HORIZON)).read_text()
    assert a == b and a != c


def test_fixed_time_respects_green_and_clearance():
    env = TrafficSignalEnv(HORIZON, flow_file=generate_flow("low", 0, HORIZON), save_replay=False)
    controller = FixedTime(green=30, clearance=5)
    env.reset()
    controller.reset(env)
    phases = [controller.act(env, t)["agent_0"] for t in range(80)]
    assert phases[:30] == [1] * 30
    assert phases[30:35] == [CLEARANCE_PHASE] * 5
    assert phases[35:65] == [2] * 30


def test_max_pressure_holds_minimum_green():
    env = TrafficSignalEnv(HORIZON, flow_file=generate_flow("high", 0, HORIZON), save_replay=False)
    controller = MaxPressure(min_green=10, clearance=5)
    env.reset()
    controller.reset(env)
    phases = []
    for t in range(HORIZON):
        actions = controller.act(env, t)
        phases.append(actions["agent_0"])
        env.step(actions)
    # Every green run (outside the first one, which may be cut by the episode
    # start) lasts at least min_green seconds.
    runs, current, length = [], phases[0], 0
    for p in phases:
        if p == current:
            length += 1
        else:
            runs.append((current, length))
            current, length = p, 1
    assert all(length >= 10 for phase, length in runs[1:] if phase != CLEARANCE_PHASE)


@pytest.mark.parametrize("name", [b for b in BASELINES if b != "fixed_plan"])  # fixed_plan: see test_calaveras.py
def test_every_baseline_completes_an_episode(name):
    env = TrafficSignalEnv(HORIZON, flow_file=generate_flow("medium", 0, HORIZON), save_replay=False)
    metrics = run_episode(env, BASELINES[name]())
    assert metrics["throughput"] > 0
    assert metrics["avg_travel_time_s"] > 0
    assert 0 < metrics["completion_rate"] <= 1
