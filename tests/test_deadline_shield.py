"""Tests for the deadline shield (scripts/deadline_shield.py) and its use in TimedSignalEnv."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from deadline_shield import DeadlineShield  # noqa: E402

# A toy intersection: four phases, each serving one lane.
PHASES = {1: {"ew"}, 2: {"ns"}, 3: {"ew_left"}, 4: {"ns_left"}}


def test_agent_keeps_control_when_nothing_is_urgent():
    shield = DeadlineShield(PHASES, max_red=90)
    ages = {"ns": 10, "ew_left": 5}
    phase, intervened = shield.filter(1, ages, ["ns", "ew_left"], current=1, green_elapsed=20)
    assert (phase, intervened) == (1, False)
    phase, intervened = shield.filter(3, ages, ["ns", "ew_left"], current=1, green_elapsed=20)
    assert (phase, intervened) == (3, False)


def test_shield_switches_before_the_deadline():
    shield = DeadlineShield(PHASES, max_red=60)
    # "ns" has waited 50 s. Keeping phase 1 for 5 more seconds, plus a 5 s
    # clearance, would serve it at 60 s: still on time. At 51 s it would be late.
    assert shield.filter(1, {"ns": 50}, ["ns"], current=1, green_elapsed=30) == (1, False)
    assert shield.filter(1, {"ns": 51}, ["ns"], current=1, green_elapsed=30) == (2, True)


def test_shield_plans_for_several_waiting_lanes():
    shield = DeadlineShield(PHASES, max_red=90)
    # Three lanes waiting; serving them one after another takes 15 s each, so
    # the shield must act long before any single lane reaches 90 s.
    ages = {"ns": 55, "ew_left": 55, "ns_left": 55}
    phase, intervened = shield.filter(1, ages, list(ages), current=1, green_elapsed=30)
    assert intervened and phase in (2, 3, 4)


def test_agent_switch_that_strands_a_lane_is_overridden():
    shield = DeadlineShield(PHASES, max_red=60)
    # Switching to phase 3 delays "ns" past its deadline; phase 2 serves it now.
    phase, intervened = shield.filter(3, {"ns": 50, "ew_left": 0}, ["ns", "ew_left"], current=1, green_elapsed=30)
    assert (phase, intervened) == (2, True)


@pytest.mark.parametrize("max_red", [65, 90])
def test_env_never_exceeds_max_red(max_red):
    pytest.importorskip("cityflow", reason="CityFlow is not installed (see README)")
    from scenarios import generate_flow
    from signal_env import TimedSignalEnv

    env = TimedSignalEnv(horizon=1200, flow_file=generate_flow("high", 0, 1200), seed=0,
                         max_red=max_red, shield="deadline", wait_obs=True)
    env.reset()
    done = False
    while not done:  # an adversarial agent that never leaves phase 1
        _, _, _, truncated, _ = env.step({a: 0 for a in env.agents})
        done = truncated["__all__"]
    metrics = env.episode_metrics()
    assert metrics["max_red_s"] <= max_red
    assert metrics["shield_interventions"] > 0
