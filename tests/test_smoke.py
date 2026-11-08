"""Smoke test: build the environment and run one short episode end to end.

Run with `pytest -q tests` from the repository root (or any directory).
"""
import math
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("cityflow", reason="CityFlow is not installed (see README)")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from traffic_env import TrafficSignalEnv, fixed_signal_control  # noqa: E402

MAX_STEPS = 60


@pytest.fixture
def env():
    return TrafficSignalEnv(MAX_STEPS, save_replay=False)


def test_only_signalised_intersections_get_agents(env):
    assert env.num_intersections == 2
    assert all(not i.get("virtual", False) for i in env.intersections)
    assert set(env.action_space.keys()) == set(env.agents)


def test_random_episode_runs(env):
    obs, _ = env.reset()
    assert set(obs) == set(env.agents)
    for agent, value in obs.items():
        assert env.observation_space[agent].contains(value)

    steps = 0
    done = False
    while not done:
        actions = {agent: env.action_space[agent].sample() for agent in env.agents}
        obs, rewards, terminateds, truncateds, _ = env.step(actions)
        for agent in env.agents:
            assert obs[agent].shape == (env.max_lanes,)
            assert math.isfinite(rewards[agent])
        done = terminateds["__all__"]
        steps += 1

    assert steps == MAX_STEPS
    assert env.engine.get_vehicle_count() > 0, "no vehicles entered the network"
    # Vehicles should be observed queueing on incoming lanes at some point.
    assert any(np.any(o > 0) for o in obs.values())


def test_works_from_another_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    env = TrafficSignalEnv(5, save_replay=False)
    env.reset()
    env.step({agent: 1 for agent in env.agents})


def test_fixed_baseline_runs(env, tmp_path):
    total = fixed_signal_control(env, output_file=tmp_path / "fixed.csv")
    assert math.isfinite(total)
    assert (tmp_path / "fixed.csv").read_text().count("\n") == MAX_STEPS + 1
