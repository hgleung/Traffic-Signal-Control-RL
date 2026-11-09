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

Networks (see NETWORKS):
* "1x2": the two-intersection corridor (the default).
* "4x4": a 16-intersection grid built with scripts/generate_grid_scenario.py.
  Its traffic is deliberately harder: two east-west arterials carry twice the
  demand of the other entry roads, demand follows a one-hour "rush hour" profile
  (60% of the average at the start and end, 140% at the peak), and vehicles may
  turn at every intersection they pass (a random walk using TURN_PROBS), so
  routes cross several signals.

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
NETWORKS = {
    "1x2": {
        "config": "training/training_config.json",
        "roadnet": "training/training_road_network.json",
        "levels": DEMAND_LEVELS,
    },
    "4x4": {
        "config": "grid4x4/config.json",
        "roadnet": "grid4x4/roadnet.json",
        # Average arrival rate per ordinary entry road (arterial entries get
        # ARTERIAL_FACTOR times this). Chosen like the 1x2 levels: the fixed-time
        # plan copes at "low" and is badly congested at "high".
        "levels": {"low": 0.03, "medium": 0.06, "high": 0.09},
        "arterial_rows": (2, 3),
    },
}
ARTERIAL_FACTOR = 2.0
RUSH_HOUR = (0.6, 1.4)  # demand multiplier at the start/end and at the peak
MAX_ROUTE_ROADS = 12

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


def _rush_hour(t, horizon):
    low, peak = RUSH_HOUR
    return low + (peak - low) * (1 - abs(2 * t / horizon - 1))


def _grid_flows(spec, level, seed, horizon):
    """Random-walk routes, arterial weighting and a rush-hour profile (see module docstring)."""
    virtual, roads, movements = _load_network(CONFIGS_DIR / spec["roadnet"])
    base_rate = spec["levels"][level]
    rng = np.random.default_rng(seed)
    names, probs = list(TURN_PROBS), np.array(list(TURN_PROBS.values()))
    flows = []
    entry_roads = sorted(r for r, road in roads.items() if road["startIntersection"] in virtual)
    for entry in entry_roads:
        # Road ids are road_<x>_<y>_<direction>; direction 0/2 is east/west-bound.
        _, x, y, direction = entry.split("_")
        arterial = int(direction) in (0, 2) and int(y) in spec.get("arterial_rows", ())
        rate = base_rate * (ARTERIAL_FACTOR if arterial else 1.0)
        max_rate = rate * RUSH_HOUR[1]
        t = rng.exponential(1.0 / max_rate)
        while t < horizon:
            # Thinning: keep each candidate arrival with probability rate(t) / max_rate.
            if rng.random() < _rush_hour(t, horizon) / RUSH_HOUR[1]:
                route = [entry]
                while roads[route[-1]]["endIntersection"] not in virtual:
                    options = movements.get(route[-1], {})
                    if len(route) >= MAX_ROUTE_ROADS and "go_straight" in options:
                        move = "go_straight"
                    else:
                        move = names[rng.choice(len(names), p=probs)]
                        if move not in options:
                            move = "go_straight"
                    route.append(options[move])
                second = int(t)
                flows.append({"vehicle": VEHICLE, "route": route, "interval": 1.0,
                              "startTime": second, "endTime": second})
            t += rng.exponential(1.0 / max_rate)
    return flows


def generate_flow(level, seed, horizon, roadnet_file=DEFAULT_ROADNET, cache=False, network="1x2"):
    """Write a randomized flow file and return its path relative to configs/.

    With cache=True an existing file for the same (network, level, seed, horizon) is reused.
    """
    if level == ORIGINAL_SCENARIO:
        if network != "1x2":
            raise ValueError("the original flow file only exists for the 1x2 network")
        return ORIGINAL_FLOW_FILE
    prefix = "flow" if network == "1x2" else f"flow_{network}"
    path = GENERATED_DIR / f"{prefix}_{level}_seed{seed}_h{horizon}.json"
    if cache and path.exists():
        return path.relative_to(CONFIGS_DIR).as_posix()
    if network != "1x2":
        flows = _grid_flows(NETWORKS[network], level, seed, horizon)
        flows.sort(key=lambda f: f["startTime"])
        return _write_flows(flows, path)
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
    return _write_flows(flows, path)


def _write_flows(flows, path):
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    # Write to a temporary name first so parallel workers never read a half-written file.
    tmp = path.with_suffix(f".{np.random.randint(1 << 30)}.tmp")
    with open(tmp, "w") as f:
        json.dump(flows, f)
    tmp.replace(path)
    return path.relative_to(CONFIGS_DIR).as_posix()
