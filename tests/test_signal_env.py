"""Tests for the timed RL environment and the actuated baseline."""
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("cityflow", reason="CityFlow is not installed (see README)")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from controllers import CLEARANCE_PHASE, Actuated  # noqa: E402
from scenarios import generate_flow  # noqa: E402
from signal_env import TimedSignalEnv  # noqa: E402
from traffic_env import TrafficSignalEnv  # noqa: E402


def make_env(**kwargs):
    return TimedSignalEnv(horizon=300, flow_file=generate_flow("medium", 0, 300), seed=0, **kwargs)


def test_spaces_and_episode_length():
    env = make_env()
    obs, _ = env.reset()
    assert env.action_space["agent_0"].n == 8
    assert all(env.observation_space[a].contains(o) for a, o in obs.items())
    steps, done = 0, False
    while not done:
        obs, rewards, _, truncated, _ = env.step({a: env.action_space[a].sample() for a in env.agents})
        assert all(np.isfinite(r) and r <= 0 for r in rewards.values())
        done = truncated["__all__"]
        steps += 1
    assert steps == 300 // 5
    assert env.episode_metrics()["throughput"] > 0


def test_min_green_and_clearance_are_enforced():
    env = make_env()
    env.reset()
    shown = []
    target = env.intersections[0]["id"]

    class EngineSpy:
        """Records the phases shown at the first intersection (CityFlow methods are read-only)."""

        def __init__(self, engine):
            self._engine = engine

        def set_tl_phase(self, intersection_id, phase):
            if intersection_id == target:
                shown.append(phase)
            self._engine.set_tl_phase(intersection_id, phase)

        def __getattr__(self, name):
            return getattr(self._engine, name)

    env.engine = EngineSpy(env.engine)
    # Ask for a different phase at every decision.
    for k in range(20):
        env.step({a: k % 2 * 3 for a in env.agents})  # alternate phase 1 and phase 4
    runs, current, length = [], shown[0], 0
    for p in shown:
        if p == current:
            length += 1
        else:
            runs.append((current, length))
            current, length = p, 1
    greens = [n for p, n in runs if p != CLEARANCE_PHASE]
    clearances = [n for p, n in runs if p == CLEARANCE_PHASE]
    assert greens and all(n >= 10 for n in greens)
    assert clearances and all(n == 5 for n in clearances)
    # A green phase is never followed directly by a different green phase.
    for (p1, _), (p2, _) in zip(runs, runs[1:]):
        assert CLEARANCE_PHASE in (p1, p2)


def test_training_mode_draws_new_scenarios():
    env = TimedSignalEnv(horizon=60, rng_seed=0)
    flows = set()
    for _ in range(4):
        env.reset()
        flows.add(Path(env._resolved_config.name).read_text())
    assert len(flows) > 1


def test_actuated_respects_min_and_max_green():
    env = TrafficSignalEnv(600, flow_file=generate_flow("high", 0, 600), seed=0, save_replay=False)
    controller = Actuated(min_green=10, max_green=40, clearance=5)
    env.reset()
    controller.reset(env)
    phases = []
    for t in range(600):
        actions = controller.act(env, t)
        phases.append(actions["agent_0"])
        env.step(actions)
    runs, current, length = [], phases[0], 0
    for p in phases:
        if p == current:
            length += 1
        else:
            runs.append((current, length))
            current, length = p, 1
    greens = [n for p, n in runs[1:] if p != CLEARANCE_PHASE]
    assert greens, "actuated controller never changed phase under high demand"
    assert all(10 <= n <= 41 for n in greens)
