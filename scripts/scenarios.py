"""Traffic demand scenarios for evaluation.

The original flow file sends one car every 2 s down each of the 6 straight
routes for the whole episode: a single, deterministic, heavily oversaturated
demand with no turning traffic. Comparing controllers on that one case says
little, so this module generates randomized scenarios instead:

* Arrivals on every entry road follow a Poisson process, so different seeds
  give different (but reproducible) traffic.
* Vehicles turn left / go straight / turn right at their first signalised
  intersection with the probabilities in TURN_PROBS, then continue straight
  until they leave the network.
* Demand levels set the arrival rate per entry road.

Generated flow files are written to configs/generated/ (git-ignored) and are
passed to TrafficSignalEnv via its flow_file argument.
"""
import json
from pathlib import Path

import numpy as np

from traffic_env import CONFIGS_DIR

GENERATED_DIR = CONFIGS_DIR / "generated"
DEFAULT_ROADNET = CONFIGS_DIR / "training" / "training_road_network.json"

# Arrival rate per entry road, in vehicles per second (x3600 for veh/h).
DEMAND_LEVELS = {
    # Chosen so the 4-phase fixed-time plan (30 s greens) is comfortably under
    # capacity at "low", noticeably congested at "medium", and close to
    # saturation at "high" (completion rate starts to fall).
    "low": 0.05,     # 180 veh/h per approach
    "medium": 0.10,  # 360 veh/h per approach
    "high": 0.15,    # 540 veh/h per approach
}
# The original hand-written flow file, kept for comparison with the class results.
ORIGINAL_SCENARIO = "original"
ORIGINAL_FLOW_FILE = "training/training_flow.json"

TURN_PROBS = {"turn_left": 0.15, "go_straight": 0.70, "turn_right": 0.15}

VEHICLE = {
    "length": 5.0, "width": 2.0, "maxPosAcc": 2.0, "maxNegAcc": 4.5,
    "usualPosAcc": 2.0, "usualNegAcc": 4.5, "minGap": 2.5,
    "maxSpeed": 16.67, "headwayTime": 1.5,
}


def _load_network(roadnet_file):
    with open(roadnet_file) as f:
        data = json.load(f)
    virtual = {i["id"] for i in data["intersections"] if i.get("virtual", False)}
    roads = {r["id"]: r for r in data["roads"]}
    # (intersection, start road) -> {movement type: end road}
    movements = {}
    for intersection in data["intersections"]:
        for link in intersection.get("roadLinks", []):
            movements.setdefault(link["startRoad"], {})[link["type"]] = link["endRoad"]
    return virtual, roads, movements


def candidate_routes(roadnet_file=DEFAULT_ROADNET):
    """All routes from each entry road, keyed by entry road, with their probabilities."""
    virtual, roads, movements = _load_network(roadnet_file)
    entry_roads = [r for r, road in roads.items() if road["startIntersection"] in virtual]
    routes = {}
    for entry in entry_roads:
        options = []
        for movement, prob in TURN_PROBS.items():
            if movement not in movements.get(entry, {}):
                continue
            route = [entry, movements[entry][movement]]
            # After the first decision, keep going straight until leaving the network.
            while roads[route[-1]]["endIntersection"] not in virtual:
                nxt = movements.get(route[-1], {}).get("go_straight")
                if nxt is None:
                    break
                route.append(nxt)
            options.append((route, prob))
        total = sum(p for _, p in options)
        routes[entry] = [(route, p / total) for route, p in options]
    return routes


def generate_flow(level, seed, horizon, roadnet_file=DEFAULT_ROADNET):
    """Write a randomized flow file and return its path relative to configs/."""
    if level == ORIGINAL_SCENARIO:
        return ORIGINAL_FLOW_FILE
    rate = DEMAND_LEVELS[level]
    rng = np.random.default_rng(seed)
    flows = []
    for entry, options in sorted(candidate_routes(roadnet_file).items()):
        t = rng.exponential(1.0 / rate)
        while t < horizon:
            route = options[rng.choice(len(options), p=[p for _, p in options])][0]
            second = int(t)
            flows.append({
                "vehicle": VEHICLE, "route": route, "interval": 1.0,
                "startTime": second, "endTime": second,
            })
            t += rng.exponential(1.0 / rate)
    flows.sort(key=lambda f: f["startTime"])

    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    path = GENERATED_DIR / f"flow_{level}_seed{seed}_h{horizon}.json"
    with open(path, "w") as f:
        json.dump(flows, f)
    return path.relative_to(CONFIGS_DIR).as_posix()
