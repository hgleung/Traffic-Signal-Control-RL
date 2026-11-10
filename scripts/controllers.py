"""Rule-based traffic signal controllers used as evaluation baselines.

Every controller is called once per simulated second with the environment and
returns {agent_id: phase_index}, the same action format TrafficSignalEnv.step
expects, so trained RL policies can be evaluated by the same harness.

Phase layout of CityFlow's generated intersections (see the road network file):
  phase 0     : clearance (only right turns run); used here as the yellow/all-red
                interval between two green phases
  phases 1-4  : the standard 4-phase plan (E-W straight, N-S straight,
                E-W left, N-S left)
  phases 5-8  : single-approach phases (straight + left from one direction)

Networks with pedestrians (env.peds, see scripts/pedestrians.py) add two rules
that every controller follows: a green can't end while a pedestrian WALK or
clearance interval runs, and phases with a pedestrian call need serving.
Actuated control treats a pedestrian call like a detector call, max-pressure
counts each waiting pedestrian like a queued vehicle, and the pre-timed plan
gives the pedestrian phases enough green every cycle.
"""
import json

import numpy as np

CLEARANCE_PHASE = 0
MAX_GREEN_PHASE = 8  # lightphases above this are pedestrian WALK variants


def green_phases(intersection):
    return list(range(1, min(MAX_GREEN_PHASE + 1, len(intersection["trafficLight"]["lightphases"]))))


def ped_holding(env, idx, t):
    peds = getattr(env, "peds", None)
    return peds is not None and peds.holding(idx, env.engine.get_current_time())


def ped_calls(env, idx):
    peds = getattr(env, "peds", None)
    return peds.calls(idx) if peds is not None else set()


class Controller:
    name = "controller"

    def reset(self, env):
        pass

    def act(self, env, t):
        raise NotImplementedError


class LegacyCycle(Controller):
    """The class project's original baseline: advance one phase every second.

    Kept only so new results can be compared with the old ones. It is not a
    realistic signal plan (lights change every second and the clearance phase is
    used as a normal phase).
    """
    name = "legacy_1s_cycle"

    def act(self, env, t):
        return {a: t % env.action_space[a].n for a in env.agents}


class _PhaseSwitcher(Controller):
    """Shared timing logic: minimum green time and a clearance interval on every change."""

    def __init__(self, min_green=10, clearance=5):
        self.min_green = min_green
        self.clearance = clearance

    def reset(self, env):
        n = len(env.agents)
        self.phase = [self.initial_phase(env, i) for i in range(n)]
        self.green_start = [0] * n
        self.clearance_until = [None] * n
        self.next_phase = [None] * n

    def initial_phase(self, env, idx):
        return 1

    def choose(self, env, idx, t):
        """Return the green phase this intersection should show next."""
        raise NotImplementedError

    def act(self, env, t):
        actions = {}
        for idx, agent in enumerate(env.agents):
            if self.clearance_until[idx] is not None:
                if t < self.clearance_until[idx]:
                    actions[agent] = CLEARANCE_PHASE
                    continue
                self.phase[idx] = self.next_phase[idx]
                self.green_start[idx] = t
                self.clearance_until[idx] = None
            elif t - self.green_start[idx] >= self.min_green and not ped_holding(env, idx, t):
                desired = self.choose(env, idx, t)
                if desired != self.phase[idx]:
                    if self.clearance > 0:
                        self.next_phase[idx] = desired
                        self.clearance_until[idx] = t + self.clearance
                        actions[agent] = CLEARANCE_PHASE
                        continue
                    self.phase[idx] = desired
                    self.green_start[idx] = t
            actions[agent] = self.phase[idx]
        return actions


class FixedTime(_PhaseSwitcher):
    """Pre-timed plan: cycle through the 4 standard phases with a fixed green time each."""
    name = "fixed_time"

    def __init__(self, green=30, clearance=5, phases=(1, 2, 3, 4)):
        super().__init__(min_green=green, clearance=clearance)
        self.phases = list(phases)

    def initial_phase(self, env, idx):
        return self.phases[0]

    def choose(self, env, idx, t):
        current = self.phases.index(self.phase[idx])
        return self.phases[(current + 1) % len(self.phases)]


class MaxPressure(_PhaseSwitcher):
    """Max-pressure control (Varaiya, 2013).

    Every `min_green` seconds, pick the phase whose permitted movements have the
    largest total pressure, where a movement's pressure is the number of vehicles
    on its incoming lane minus the average on its outgoing lanes. Right turns,
    which run in every phase, are ignored.
    """
    name = "max_pressure"

    def __init__(self, min_green=10, clearance=5, phases=None):
        super().__init__(min_green=min_green, clearance=clearance)
        self.phases = phases

    def reset(self, env):
        self._movements = []
        self._phase_links = []
        for intersection in env.intersections:
            links = intersection["roadLinks"]
            movements = []
            for link in links:
                lane_pairs = [
                    (f"{link['startRoad']}_{ll['startLaneIndex']}", f"{link['endRoad']}_{ll['endLaneIndex']}")
                    for ll in link["laneLinks"]
                ]
                movements.append((link["type"], lane_pairs))
            self._movements.append(movements)
            lightphases = intersection["trafficLight"]["lightphases"]
            candidates = self.phases or green_phases(intersection)
            self._phase_links.append({p: lightphases[p]["availableRoadLinks"] for p in candidates})
        super().reset(env)

    def choose(self, env, idx, t):
        counts = env.engine.get_lane_vehicle_count()
        pressures = {}
        for phase, link_ids in self._phase_links[idx].items():
            pressure = 0.0
            for link_id in link_ids:
                movement_type, lane_pairs = self._movements[idx][link_id]
                if movement_type == "turn_right":
                    continue
                incoming = {a for a, _ in lane_pairs}
                outgoing = {b for _, b in lane_pairs}
                pressure += sum(counts.get(l, 0) for l in incoming) - np.mean([counts.get(l, 0) for l in outgoing])
            pressures[phase] = pressure
        peds = getattr(env, "peds", None)
        if peds is not None:
            # Waiting pedestrians count like queued vehicles on the phase that serves them.
            for k, cw in enumerate(peds.crosswalks[idx]):
                if cw["phase"] in pressures:
                    pressures[cw["phase"]] += len(peds.waiting[idx][k])
        best = max(pressures.values())
        # Keep the current phase on ties to avoid needless switching.
        if pressures.get(self.phase[idx]) == best:
            return self.phase[idx]
        return max(pressures, key=pressures.get)


class Actuated(_PhaseSwitcher):
    """Fully actuated control, the way most real detector-equipped signals run.

    Real signals don't weigh cars: they sense them with inductive loops cut into
    the pavement near the stop line (or with video/radar zones that act like
    loops). An actuated controller (FHWA Traffic Signal Timing Manual, ch. 5):

    * serves phases in a fixed order, but skips any phase with no vehicle
      waiting on its detectors ("no call"),
    * gives every green a minimum time,
    * extends the green each time a vehicle crosses the detector, ending it once
      no vehicle has been detected for `passage_time` seconds ("gap out"),
    * ends the green at `max_green` if traffic keeps coming but another phase
      has a call ("max out"),
    * rests in the current green when no other phase has a call.

    Detectors here are the last `detector_length` metres of each incoming lane,
    i.e. a stop-line loop. Defaults: 10 s minimum green and 5 s clearance (same
    as the other controllers), 3 s passage time (~30 m detector at ~11 m/s,
    FHWA's distance/speed rule), 40 s maximum green.
    """
    name = "actuated"

    def __init__(self, min_green=10, max_green=40, passage_time=3, clearance=5,
                 detector_length=30.0, phases=(1, 2, 3, 4)):
        super().__init__(min_green=min_green, clearance=clearance)
        self.max_green = max_green
        self.passage_time = passage_time
        self.detector_length = detector_length
        self.phases = list(phases)

    def reset(self, env):
        with open(env.road_network_file) as f:
            data = json.load(f)
        widths = {i["id"]: i.get("width", 0) for i in data["intersections"]}
        lane_length = {}
        for road in data["roads"]:
            points = road["points"]
            length = sum(np.hypot(b["x"] - a["x"], b["y"] - a["y"]) for a, b in zip(points, points[1:]))
            # CityFlow trims each road by the width of the intersections at its ends.
            length -= widths[road["startIntersection"]] + widths[road["endIntersection"]]
            for k in range(len(road["lanes"])):
                lane_length[f"{road['id']}_{k}"] = length
        self._lane_length = lane_length

        self._phase_lanes = []
        for intersection in env.intersections:
            links = intersection["roadLinks"]
            lightphases = intersection["trafficLight"]["lightphases"]
            lanes = {}
            for p in self.phases:
                lanes[p] = sorted({
                    f"{links[k]['startRoad']}_{ll['startLaneIndex']}"
                    for k in lightphases[p]["availableRoadLinks"]
                    if links[k]["type"] != "turn_right"
                    for ll in links[k]["laneLinks"]
                })
            self._phase_lanes.append(lanes)
        # Maximum greens: the default everywhere, or, for networks with an
        # estimated signal plan, 1.25 x that plan's green for the phase (at least
        # the default), as practitioners usually set them.
        self._max_green = [dict.fromkeys(self.phases, self.max_green) for _ in env.intersections]
        plan = getattr(env, "signal_plan", None)
        if plan:
            for idx, intersection in enumerate(env.intersections):
                for p, green in plan["intersections"][intersection["id"]]["sequence"]:
                    if p in self._max_green[idx]:
                        self._max_green[idx][p] = max(self.max_green, int(round(1.25 * green)))
        n = len(env.agents)
        self._last_detection = [-np.inf] * n
        self._detected_t = None
        self._detected = None
        super().reset(env)

    def initial_phase(self, env, idx):
        return self.phases[0]

    def _detections(self, env, t):
        """Lanes whose stop-line detector is occupied at time t (computed once per second)."""
        if self._detected_t != t:
            lane_vehicles = env.engine.get_lane_vehicles()
            distance = env.engine.get_vehicle_distance()
            occupied = set()
            for lane, vehicles in lane_vehicles.items():
                limit = self._lane_length.get(lane)
                if limit is None:
                    continue
                if any(distance.get(v, 0.0) >= limit - self.detector_length for v in vehicles):
                    occupied.add(lane)
            self._detected, self._detected_t = occupied, t
        return self._detected

    def act(self, env, t):
        occupied = self._detections(env, t)
        for idx in range(len(env.agents)):
            if self.clearance_until[idx] is None and any(l in occupied for l in self._phase_lanes[idx][self.phase[idx]]):
                self._last_detection[idx] = t
        return super().act(env, t)

    def choose(self, env, idx, t):
        occupied = self._detections(env, t)
        current = self.phase[idx]
        order = self.phases[self.phases.index(current) + 1:] + self.phases[:self.phases.index(current)]
        peds = ped_calls(env, idx)
        calls = [p for p in order if p in peds or any(l in occupied for l in self._phase_lanes[idx][p])]
        if not calls:
            return current  # rest in green
        green_time = t - self.green_start[idx]
        gapped_out = t - self._last_detection[idx] >= self.passage_time
        if gapped_out or green_time >= self._max_green[idx][current]:
            return calls[0]
        return current


class FixedPlan(Controller):
    """A coordinated pre-timed plan read from the network's signal_plan.json.

    Every intersection runs the same cycle with its own phase sequence, green
    times and offset (see scripts/build_calaveras.py, which estimates them with
    Webster's method). A phase that serves only pedestrians (no vehicle
    movement) is skipped when nobody has pushed the button, and its time goes to
    the green before it, so the cycle and offsets don't drift.
    """
    name = "fixed_plan"

    def reset(self, env):
        plan = getattr(env, "signal_plan", None)
        if not plan:
            raise ValueError("fixed_plan needs a network with a signal_plan.json")
        clearance = plan["clearance_s"]
        self.cycle = plan["cycle_s"]
        self.schedule = []
        for intersection in env.intersections:
            entry = plan["intersections"][intersection["id"]]
            ped_only = self._ped_only_phases(intersection)
            segments, start = [], 0
            for p, green in entry["sequence"]:
                segments.append((start, start + green, p, p in ped_only))
                start += green + clearance
            self.schedule.append((entry["offset_s"], segments))
        self.current = [None] * len(env.intersections)

    @staticmethod
    def _ped_only_phases(intersection):
        links = intersection["roadLinks"]
        lightphases = intersection["trafficLight"]["lightphases"]
        return {p for p in green_phases(intersection)
                if all(links[k]["type"] == "turn_right" for k in lightphases[p]["availableRoadLinks"])}

    def act(self, env, t):
        actions = {}
        for idx, agent in enumerate(env.agents):
            offset, segments = self.schedule[idx]
            tau = (t - offset) % self.cycle
            phase = CLEARANCE_PHASE
            for k, (start, end, p, ped_only) in enumerate(segments):
                if start <= tau < end:
                    phase = p
                    # Serve a pedestrian-only phase only if the button was pushed
                    # before it was due; otherwise the green before it continues.
                    if ped_only and self.current[idx] != p and (
                            self.current[idx] != CLEARANCE_PHASE or p not in ped_calls(env, idx)):
                        phase = self._previous_green(segments, k)
                    break
                if tau < start:
                    break
            # The clearance before a skipped pedestrian-only phase is skipped too.
            if phase == CLEARANCE_PHASE and self.current[idx] not in (None, CLEARANCE_PHASE):
                nxt = next((s for s in segments if s[0] > tau), None)
                if nxt is not None and nxt[3] and nxt[2] not in ped_calls(env, idx):
                    phase = self.current[idx]
            self.current[idx] = phase
            actions[agent] = phase
        return actions

    @staticmethod
    def _previous_green(segments, k):
        return segments[k - 1][2] if k > 0 else segments[-1][2]


BASELINES = {
    LegacyCycle.name: LegacyCycle,
    FixedTime.name: FixedTime,
    MaxPressure.name: MaxPressure,
    Actuated.name: Actuated,
    FixedPlan.name: FixedPlan,
}
