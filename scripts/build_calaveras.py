"""Build the Calaveras Blvd (SR 237, Milpitas, CA) network for CityFlow.

Writes, under configs/calaveras/:
  roadnet.json      the road network (3 signalised intersections)
  config.json       the CityFlow config
  pedestrians.json  the marked crosswalks, their lengths, timing and demand
  demand.json       PM-peak vehicle demand: entry volumes and turning shares
  signal_plan.json  an estimated pre-timed plan (Webster's method, coordinated)

Run it again after changing any of the inputs below:
  python scripts/build_calaveras.py

What is real and what is estimated
----------------------------------
Real (public sources):
* The study segment, Abel St to Milpitas Blvd, is the one VTA counts for its
  CMP monitoring and the limits of VTA's Calaveras Boulevard Improvements
  project.
* The controlled intersections in it, west to east: Abel St, Main St/Carlo St
  (the ramps to Main St) and Milpitas Blvd (Caltrans SR 237 Transportation
  Concept Report, Table 4).
* Marked crosswalks: Abel St 4 of 4 legs, Milpitas Blvd 4 of 4, Main/Carlo
  1 of 2 crossings (same table). Between Abel St and Milpitas Blvd the only
  sidewalk is on the north side.
* Calaveras is 6 lanes, except the overpass over the UP railroad and Main St
  "including approaches, between Abel Street and Milpitas Boulevard", which is
  4 lanes (2 each way). Posted speeds 35-40 mph. Heavy eastbound flow in the PM
  peak (2012 Caltrans peak-hour volumes: 5,907 EB vs 3,456 WB on the
  I-880 to I-680 segment).
* 2024 PM-peak level of service: Abel St LOS F, Milpitas Blvd LOS D (VTA 2024
  CMP Monitoring Report).

Estimated (not public, or not reachable from here):
* Turning-movement volumes. VTA's fall 2024 counts are in an appendix that is
  not posted, so the volumes below are engineering estimates consistent with
  the directional split and the LOS above (see demand.json and the README).
* Distances between intersections, lane assignments and turn pockets (OSM could
  not be reached; check against aerial photos).
* Which leg the Main/Carlo ramp and its marked crosswalk are on: modelled as a
  ramp on the south side (to Carlo St/Main St, carrying traffic to and from east
  Calaveras), with the marked crosswalk across Calaveras on the east side.
* Pedestrian volumes (no public counts).
* Signal timing (Caltrans maintains these signals; the timing sheets need a
  records request). signal_plan.json is a standard Webster estimate.
"""
import json
import math
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = REPO_ROOT / "configs" / "calaveras"

LANE_WIDTH = 3.6
SPEED_OUTER = 17.88   # 40 mph
SPEED_BRIDGE = 15.65  # 35 mph
SIDE_SPEED = 15.65
INTERSECTION_WIDTH = 20

# Pedestrian timing (MUTCD 4E.06): walk interval, and a pedestrian clearance
# ("flashing don't walk") long enough to cross at 3.5 ft/s.
WALK_S = 7
PED_SPEED = 1.07  # m/s
MEDIAN_M = 2.0

# Intersections, west to east, at x metres along Calaveras (estimated spacing).
# legs: approach lanes for traffic entering the intersection from that side
# (W = eastbound traffic, E = westbound, S = northbound, N = southbound). Lane 0
# is the leftmost. L = left only, T = through, R = right only.
INTERSECTIONS = [
    {"id": "abel", "name": "Abel St", "x": 0,
     "approach": {"W": "LTTR", "E": "LTTR", "S": "LTR", "N": "LTR"},
     "side_length": 300, "crosswalks": ["N", "S", "E", "W"]},
    {"id": "main_carlo", "name": "Main St / Carlo St ramps", "x": 450,
     # Ramp to and from Main St/Carlo St on the south side, serving east Calaveras:
     # westbound traffic turns left onto it, ramp traffic turns right onto eastbound.
     # Calaveras approaches keep the same 4-lane layout as the rest of the
     # corridor so lane indices line up (CityFlow keeps a vehicle in the lane
     # with the nearest index when it crosses an intersection); the left and
     # right lanes that have no movement here are simply never used.
     "approach": {"W": "LTTR", "E": "LTTR", "S": "R"},
     "side_length": 200, "crosswalks": ["E"]},
    {"id": "milpitas", "name": "Milpitas Blvd", "x": 1000,
     "approach": {"W": "LTTR", "E": "LTTR", "S": "LTTR", "N": "LTTR"},
     "side_length": 300, "crosswalks": ["N", "S", "E", "W"]},
]
WEST_ENTRY_LENGTH = 400   # to the I-880 ramps
EAST_ENTRY_LENGTH = 400   # towards Town Center Dr
EXIT_LANES = {"main": 3, "side": 2, "ramp": 1}

# PM-peak demand (veh/h) at the "high" level, which stands for the 2024 PM peak
# hour. Entry volumes and turning shares are estimates (see module docstring).
ENTRY_VOLUMES = {
    ("abel", "W"): 1650,        # eastbound from I-880
    ("milpitas", "E"): 1000,    # westbound from Town Center Dr
    ("abel", "S"): 1000, ("abel", "N"): 1100,
    ("main_carlo", "S"): 200,   # ramp from Main St / Carlo St
    ("milpitas", "S"): 600, ("milpitas", "N"): 600,
}
# Turning shares per approach: (left, through, right).
TURN_SHARES = {
    ("abel", "W"): (0.08, 0.70, 0.22),
    ("abel", "E"): (0.10, 0.75, 0.15),
    ("abel", "S"): (0.45, 0.35, 0.20),
    ("abel", "N"): (0.25, 0.30, 0.45),
    ("main_carlo", "W"): (0.0, 1.0, 0.0),
    ("main_carlo", "E"): (0.15, 0.85, 0.0),
    ("main_carlo", "S"): (0.0, 0.0, 1.0),
    ("milpitas", "W"): (0.12, 0.72, 0.16),
    ("milpitas", "E"): (0.18, 0.70, 0.12),
    ("milpitas", "S"): (0.25, 0.55, 0.20),
    ("milpitas", "N"): (0.25, 0.55, 0.20),
}
# Pedestrians per hour per crosswalk (both directions), estimated; there are no
# public counts. Milpitas Blvd is next to City Hall and Town Center.
PED_RATES = {"abel": 20, "main_carlo": 6, "milpitas": 30}

DIRS = {"E": 0, "N": 1, "W": 2, "S": 3}       # direction of travel -> id suffix
UNIT = {"E": (1, 0), "N": (0, 1), "W": (-1, 0), "S": (0, -1)}
# Approach side -> direction of travel when entering.
ENTER_DIR = {"W": "E", "E": "W", "S": "N", "N": "S"}
# Movement from an approach side -> the side it leaves through.
LEAVE_SIDE = {
    "W": {"L": "N", "T": "E", "R": "S"},
    "E": {"L": "S", "T": "W", "R": "N"},
    "S": {"L": "W", "T": "N", "R": "E"},
    "N": {"L": "E", "T": "S", "R": "W"},
}
OPPOSITE = {"W": "E", "E": "W", "N": "S", "S": "N"}
TYPES = {"L": "turn_left", "T": "go_straight", "R": "turn_right"}

# Green phases, same numbering as CityFlow's generated intersections (see
# scripts/controllers.py). Right turns are allowed in every phase (right turn on
# red), except during a pedestrian WALK on the crosswalk they cross.
PHASES = {
    1: {("W", "T"), ("E", "T")},
    2: {("S", "T"), ("N", "T")},
    3: {("W", "L"), ("E", "L")},
    4: {("S", "L"), ("N", "L")},
    5: {("W", "T"), ("W", "L")},
    6: {("E", "T"), ("E", "L")},
    7: {("S", "T"), ("S", "L")},
    8: {("N", "T"), ("N", "L")},
}
# Crosswalks run with the vehicle phase parallel to them: crosswalks across the
# side street (N and S legs) with phase 1, across Calaveras (E and W legs) with
# phase 2. Each right turn crosses the crosswalk on the leg it leaves through.
PED_PHASE = {"N": 1, "S": 1, "E": 2, "W": 2}
WALK_VARIANT = {1: 9, 2: 10}  # lightphase shown during WALK (conflicting right turns held)


def _road(rid, start, end, p0, p1, lanes, speed):
    return {"id": rid, "points": [{"x": p0[0], "y": p0[1]}, {"x": p1[0], "y": p1[1]}],
            "lanes": [{"width": LANE_WIDTH, "maxSpeed": speed} for _ in range(lanes)],
            "startIntersection": start, "endIntersection": end}


def build_roadnet():
    nodes = {i["id"]: i for i in INTERSECTIONS}
    pos = {f"intersection_{i['id']}": (i["x"], 0) for i in INTERSECTIONS}
    roads = []
    incoming = {}  # (node id, side) -> road dict
    outgoing = {}
    virtual = []

    def add(rid, start, end, lanes, speed):
        road = _road(rid, start, end, pos[start], pos[end], lanes, speed)
        roads.append(road)
        return road

    first, last = INTERSECTIONS[0], INTERSECTIONS[-1]
    pos["end_west"] = (first["x"] - WEST_ENTRY_LENGTH, 0)
    pos["end_east"] = (last["x"] + EAST_ENTRY_LENGTH, 0)
    virtual += ["end_west", "end_east"]
    fi, li = f"intersection_{first['id']}", f"intersection_{last['id']}"
    incoming[(first["id"], "W")] = add(f"road_west_{first['id']}_0", "end_west", fi,
                                       len(first["approach"]["W"]), SPEED_OUTER)
    outgoing[(first["id"], "W")] = add(f"road_{first['id']}_west_2", fi, "end_west", EXIT_LANES["main"], SPEED_OUTER)
    incoming[(last["id"], "E")] = add(f"road_east_{last['id']}_2", "end_east", li,
                                      len(last["approach"]["E"]), SPEED_OUTER)
    outgoing[(last["id"], "E")] = add(f"road_{last['id']}_east_0", li, "end_east", EXIT_LANES["main"], SPEED_OUTER)
    for a, b in zip(INTERSECTIONS, INTERSECTIONS[1:]):
        ia, ib = f"intersection_{a['id']}", f"intersection_{b['id']}"
        road = add(f"road_{a['id']}_{b['id']}_0", ia, ib, len(b["approach"]["W"]), SPEED_BRIDGE)
        outgoing[(a["id"], "E")] = incoming[(b["id"], "W")] = road
        road = add(f"road_{b['id']}_{a['id']}_2", ib, ia, len(a["approach"]["E"]), SPEED_BRIDGE)
        outgoing[(b["id"], "W")] = incoming[(a["id"], "E")] = road
    for node in INTERSECTIONS:
        iid = f"intersection_{node['id']}"
        for side in ("N", "S"):
            if side not in node["approach"]:
                continue
            ext = f"end_{node['id']}_{side.lower()}"
            sign = 1 if side == "N" else -1
            pos[ext] = (node["x"], sign * node["side_length"])
            virtual.append(ext)
            kind = "ramp" if node["id"] == "main_carlo" else "side"
            incoming[(node["id"], side)] = add(f"road_{ext[4:]}_{node['id']}_{DIRS[ENTER_DIR[side]]}", ext, iid,
                                               len(node["approach"][side]), SIDE_SPEED)
            outgoing[(node["id"], side)] = add(f"road_{node['id']}_{ext[4:]}_{DIRS[side]}", iid, ext,
                                               EXIT_LANES[kind], SIDE_SPEED)

    intersections = []
    for node in INTERSECTIONS:
        iid = f"intersection_{node['id']}"
        links = []
        movement_link = {}
        for side, lanes in node["approach"].items():
            start = incoming[(node["id"], side)]
            for move in "LTR":
                if move not in lanes:
                    continue
                share = TURN_SHARES.get((node["id"], side), (1, 1, 1))["LTR".index(move)]
                out_side = LEAVE_SIDE[side][move]
                if share == 0 or (node["id"], out_side) not in outgoing:
                    continue
                end = outgoing[(node["id"], out_side)]
                lane_links = []
                for k, kind in enumerate(lanes):
                    if kind != move:
                        continue
                    for j in range(len(end["lanes"])):
                        lane_links.append({"startLaneIndex": k, "endLaneIndex": j,
                                           "points": [_lane_point(start, k, True), _lane_point(end, j, False)]})
                movement_link[(side, move)] = len(links)
                links.append({"type": TYPES[move], "startRoad": start["id"], "endRoad": end["id"],
                              "direction": DIRS[ENTER_DIR[side]], "laneLinks": lane_links})
        rights = {k for (side, move), k in movement_link.items() if move == "R"}
        lightphases = [{"time": 5, "availableRoadLinks": sorted(rights)}]
        for p in range(1, 9):
            lightphases.append({"time": 30, "availableRoadLinks": sorted(
                rights | {movement_link[m] for m in PHASES[p] if m in movement_link})})
        # WALK variants of phases 1 and 2: right turns across the walking crosswalks are held.
        for p in (1, 2):
            held = {movement_link[(side, "R")] for side in node["approach"]
                    if (side, "R") in movement_link and LEAVE_SIDE[side]["R"] in node["crosswalks"]
                    and PED_PHASE[LEAVE_SIDE[side]["R"]] == p}
            lightphases.append({"time": 5, "availableRoadLinks": sorted(set(lightphases[p]["availableRoadLinks"]) - held)})
        intersections.append({
            "id": iid, "point": {"x": node["x"], "y": 0}, "width": INTERSECTION_WIDTH,
            "roads": [r["id"] for r in roads if iid in (r["startIntersection"], r["endIntersection"])],
            "roadLinks": links,
            "trafficLight": {"roadLinkIndices": list(range(len(links))), "lightphases": lightphases},
            "virtual": False,
        })
    for vid in virtual:
        intersections.append({
            "id": vid, "point": {"x": pos[vid][0], "y": pos[vid][1]}, "width": 0,
            "roads": [r["id"] for r in roads if vid in (r["startIntersection"], r["endIntersection"])],
            "roadLinks": [], "trafficLight": {"roadLinkIndices": [], "lightphases": []}, "virtual": True,
        })
    return {"intersections": intersections, "roads": roads}, incoming, outgoing


def _lane_point(road, lane, at_end):
    p0, p1 = road["points"][0], road["points"][-1]
    dx, dy = p1["x"] - p0["x"], p1["y"] - p0["y"]
    length = math.hypot(dx, dy)
    ux, uy = dx / length, dy / length
    shift = LANE_WIDTH * (lane + 0.5)
    if at_end:
        x, y = p1["x"] - ux * INTERSECTION_WIDTH, p1["y"] - uy * INTERSECTION_WIDTH
    else:
        x, y = p0["x"] + ux * INTERSECTION_WIDTH, p0["y"] + uy * INTERSECTION_WIDTH
    return {"x": x + uy * shift, "y": y - ux * shift}


def crosswalk_length(node_id, leg, incoming, outgoing):
    lanes = len(incoming[(node_id, leg)]["lanes"]) if (node_id, leg) in incoming else 0
    lanes += len(outgoing[(node_id, leg)]["lanes"]) if (node_id, leg) in outgoing else 0
    median = MEDIAN_M if leg in ("E", "W") else 0.0
    return round(lanes * LANE_WIDTH + median, 1)


def build_pedestrians(incoming, outgoing):
    spec = {"walk_s": WALK_S, "ped_speed_mps": PED_SPEED, "intersections": {}}
    for node in INTERSECTIONS:
        crosswalks = []
        for leg in node["crosswalks"]:
            length = crosswalk_length(node["id"], leg, incoming, outgoing)
            crosswalks.append({"leg": leg, "length_m": length, "phase": PED_PHASE[leg],
                               "clearance_s": math.ceil(length / PED_SPEED),
                               "peds_per_hour": PED_RATES[node["id"]]})
        spec["intersections"][f"intersection_{node['id']}"] = {
            "name": node["name"], "crosswalks": crosswalks, "walk_variant": {str(k): v for k, v in WALK_VARIANT.items()}}
    return spec


def movement_volumes():
    """Expected PM-peak volume of every movement, propagating entry volumes through the turning shares."""
    order = [n["id"] for n in INTERSECTIONS]
    arriving = {k: float(v) for k, v in ENTRY_VOLUMES.items()}
    volumes = {}
    # Eastbound pass (west to east) also computes the side streets, whose demand
    # doesn't depend on Calaveras; the westbound pass then picks up their turns.
    for direction, seq in (("W", order), ("E", order[::-1])):
        for k, nid in enumerate(seq):
            node = next(n for n in INTERSECTIONS if n["id"] == nid)
            for side in node["approach"]:
                if side != direction and side in ("W", "E"):
                    continue
                if side in ("N", "S") and direction == "E":
                    continue  # side streets handled on the first pass
                total = arriving.get((nid, side), 0.0)
                for move, share in zip("LTR", TURN_SHARES[(nid, side)]):
                    if share and move in node["approach"][side]:
                        volumes[(nid, side, move)] = total * share
            if k + 1 < len(seq):
                nxt = seq[k + 1]
                flow = sum(v for (n, s, m), v in volumes.items() if n == nid and LEAVE_SIDE[s][m] == OPPOSITE[direction])
                arriving[(nxt, direction)] = arriving.get((nxt, direction), 0.0) + flow
    return volumes, arriving


def webster_plan(volumes, ped_spec):
    """Pre-timed plan from Webster's method, one common cycle, offsets for eastbound progression.

    Saturation flow 1,800 veh/h per through lane and 1,700 per left lane; lost
    time 5 s clearance + 2 s start-up per phase; cycle (1.5 L + 5) / (1 - Y),
    limited to 60-150 s. Greens are split in proportion to each phase's critical
    flow ratio, with at least 10 s, and at least WALK + pedestrian clearance on
    phases that serve a crosswalk (phase 1 for side-street crosswalks, phase 2
    across Calaveras), as real pre-timed plans with pedestrian timing do.
    """
    plans = {}
    for node in INTERSECTIONS:
        nid = node["id"]
        ratios = {}
        for p in (1, 2, 3, 4):
            y = 0.0
            for side, move in PHASES[p]:
                lanes = node["approach"].get(side, "").count(move)
                if not lanes:
                    continue
                sat = 1700 if move == "L" else 1800
                y = max(y, volumes.get((nid, side, move), 0.0) / (lanes * sat))
            ratios[p] = y
        peds = {cw["phase"]: WALK_S + cw["clearance_s"]
                for cw in ped_spec["intersections"][f"intersection_{nid}"]["crosswalks"]}
        phases = [p for p in (1, 2, 3, 4) if ratios[p] > 0 or p in peds]
        plans[nid] = {"ratios": ratios, "phases": phases, "ped_min": peds}
    cycles = []
    for nid, plan in plans.items():
        n = len(plan["phases"])
        lost = 7 * n
        Y = sum(plan["ratios"][p] for p in plan["phases"])
        cycle = 150 if Y >= 0.95 else (1.5 * lost + 5) / (1 - Y)
        plan["Y"] = round(Y, 3)
        cycles.append(min(150, max(60, cycle)))
    cycle = int(math.ceil(max(cycles) / 5) * 5)
    out = {"cycle_s": cycle, "clearance_s": 5, "intersections": {}}
    distance = 0.0
    prev_x = INTERSECTIONS[0]["x"]
    for node in INTERSECTIONS:
        plan = plans[node["id"]]
        phases = plan["phases"]
        green_total = cycle - 5 * len(phases)
        mins = {p: max(10, plan["ped_min"].get(p, 0)) for p in phases}
        Y = sum(plan["ratios"][p] for p in phases) or 1.0
        greens = {p: max(mins[p], green_total * plan["ratios"][p] / Y) for p in phases}
        # Take any excess from phases that are above their minimum.
        excess = sum(greens.values()) - green_total
        flexible = [p for p in phases if greens[p] > mins[p]]
        while excess > 0.5 and flexible:
            room = sum(greens[p] - mins[p] for p in flexible)
            for p in flexible:
                greens[p] -= excess * (greens[p] - mins[p]) / room
            excess = sum(greens.values()) - green_total
            flexible = [p for p in phases if greens[p] > mins[p] + 0.5]
        greens = {p: int(round(g)) for p, g in greens.items()}
        greens[phases[0]] += green_total - sum(greens.values())
        distance += node["x"] - prev_x
        prev_x = node["x"]
        out["intersections"][f"intersection_{node['id']}"] = {
            "name": node["name"],
            "sequence": [[p, greens[p]] for p in phases],
            "offset_s": int(round(distance / SPEED_BRIDGE)) % cycle,
            "critical_flow_ratio": {str(p): round(plan["ratios"][p], 3) for p in phases},
        }
    return out


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    roadnet, incoming, outgoing = build_roadnet()
    peds = build_pedestrians(incoming, outgoing)
    volumes, arriving = movement_volumes()
    plan = webster_plan(volumes, peds)
    demand = {
        "level_high_means": "estimated 2024 PM peak hour",
        "entry_veh_per_hour": {f"intersection_{n}:{s}": v for (n, s), v in ENTRY_VOLUMES.items()},
        "turn_shares_left_through_right": {f"intersection_{n}:{s}": list(v) for (n, s), v in TURN_SHARES.items()},
        "entry_roads": {f"intersection_{n}:{s}": incoming[(n, s)]["id"] for (n, s) in ENTRY_VOLUMES},
        "expected_movement_veh_per_hour": {f"intersection_{n}:{s}{m}": round(v) for (n, s, m), v in sorted(volumes.items())},
    }
    config = {"interval": 1.0, "seed": 0, "dir": "../configs/", "roadnetFile": "calaveras/roadnet.json",
              "flowFile": "calaveras/flow_sample.json", "rlTrafficLight": True, "laneChange": True,
              "saveReplay": False, "roadnetLogFile": "../logs/calaveras/roadnet_log.json",
              "replayLogFile": "../logs/calaveras/replay.txt"}
    files = {"roadnet.json": roadnet, "pedestrians.json": peds, "demand.json": demand,
             "signal_plan.json": plan, "config.json": config}
    for name, data in files.items():
        (OUT_DIR / name).write_text(json.dumps(data, indent=1 if name != "roadnet.json" else None) + "\n")
    # Placeholder flow file for the config; real traffic comes from scripts/scenarios.py.
    (OUT_DIR / "flow_sample.json").write_text("[]\n")
    print(json.dumps(plan, indent=1))
    print(json.dumps(demand["expected_movement_veh_per_hour"], indent=1))


if __name__ == "__main__":
    main()
