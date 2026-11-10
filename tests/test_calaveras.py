"""Tests for the Calaveras Blvd network, its traffic and its pedestrians."""
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

HORIZON = 900
SPEC = NETWORKS["calaveras"]


def make_env(level="high", seed=0, peds=True, ped_scale=1.0):
    return TrafficSignalEnv(HORIZON, config_file=CONFIGS_DIR / SPEC["config"],
                            flow_file=generate_flow(level, seed, HORIZON, network="calaveras"), seed=seed,
                            save_replay=False, pedestrians=SPEC["pedestrians"] if peds else None,
                            ped_scale=ped_scale, signal_plan=SPEC["signal_plan"])


def test_network_matches_the_crosswalk_inventory():
    peds = json.loads((CONFIGS_DIR / SPEC["pedestrians"]).read_text())["intersections"]
    legs = {i: sorted(cw["leg"] for cw in v["crosswalks"]) for i, v in peds.items()}
    # Caltrans SR 237 TCR: Abel St 4/4 and Milpitas Blvd 4/4 marked crossings, Main/Carlo 1 of 2.
    assert legs == {"intersection_abel": ["E", "N", "S", "W"], "intersection_main_carlo": ["E"],
                    "intersection_milpitas": ["E", "N", "S", "W"]}
    for v in peds.values():
        for cw in v["crosswalks"]:
            assert cw["clearance_s"] >= cw["length_m"] / 1.07  # 3.5 ft/s


def test_traffic_is_heavier_eastbound():
    flows = json.loads((CONFIGS_DIR / generate_flow("high", 0, 3600, network="calaveras")).read_text())
    eb = sum(f["route"][0].startswith("road_west_") for f in flows)
    wb = sum(f["route"][0].startswith("road_east_") for f in flows)
    assert eb > 1.3 * wb
    assert any(len(f["route"]) >= 4 for f in flows)  # trips that cross all three signals


@pytest.mark.parametrize("name", ["fixed_plan", "fixed_time", "actuated", "max_pressure"])
def test_baselines_serve_pedestrians(name):
    env = make_env(ped_scale=3.0)
    metrics = run_episode(env, BASELINES[name]())
    assert metrics["throughput"] > 0
    assert metrics["peds"] > 0
    assert metrics["ped_max_wait_s"] < HORIZON / 2  # everyone gets a WALK eventually


def test_walk_holds_the_green_and_right_turns():
    env = make_env(ped_scale=20.0)
    controller = BASELINES["actuated"]()
    env.reset()
    controller.reset(env)
    seen_variant = False
    for t in range(HORIZON):
        actions = controller.act(env, t)
        before = [env.peds.holding(i, t) for i in range(env.num_intersections)]
        env.step(actions)
        for i in range(env.num_intersections):
            if before[i]:
                # A green with a pedestrian interval running is never cut short.
                assert actions[env.agents[i]] == env.peds.walk_phase[i]
        seen_variant |= any(env.peds.walk_phase[i] is not None and t < env.peds.walk_end[i]
                            for i in range(env.num_intersections))
    assert seen_variant


def test_timed_env_with_pedestrians_and_max_red():
    env = TimedSignalEnv(horizon=HORIZON, network="calaveras", max_red=90, ped_scale=3.0,
                         flow_file=generate_flow("high", 0, HORIZON, network="calaveras"), seed=0)
    obs, _ = env.reset()
    assert env.num_green == 8 and env.peds is not None
    done = False
    while not done:
        obs, rewards, _, truncateds, _ = env.step({a: 0 for a in env.agents})  # always ask for phase 1
        done = truncateds["__all__"]
    assert all(env.observation_space[a].contains(o) for a, o in obs.items())
    metrics = env.episode_metrics()
    # The max-red rule forces the other phases and pedestrian calls through.
    assert env.forced_switches > 0
    assert metrics["ped_max_wait_s"] < 90 + 60
