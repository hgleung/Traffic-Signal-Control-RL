"""Pedestrians at signalised crosswalks, on top of CityFlow (which has none).

CityFlow simulates vehicles only, so pedestrians are modelled the way a signal
controller sees them: as push-button calls with timing requirements.

* Arrivals: each crosswalk gets pedestrians as a Poisson process (rate from
  pedestrians.json), seeded by the scenario seed. A pedestrian who arrives
  presses the button and waits.
* Service: a crosswalk runs with the vehicle phase parallel to it (phase 1 for
  crosswalks across the side street, phase 2 for crosswalks across the main
  road). When that phase's green starts and someone is waiting on one of its
  crosswalks, the WALK interval starts (7 s), followed by the pedestrian
  clearance ("flashing don't walk"), long enough to cross at 3.5 ft/s (MUTCD).
  Pedestrians who arrive during WALK just go.
* Timing constraint: while WALK or the pedestrian clearance runs, the green
  can't end. A call for a 27 m crossing therefore makes that green at least
  33 s long, whatever the controller wanted. This is the main effect of
  pedestrians on signal control.
* Vehicle conflicts: during WALK, right turns across the walking crosswalks are
  held (the intersection shows a variant of the phase without those turns,
  built by scripts/build_calaveras.py), modelling drivers yielding to people
  stepping off the curb. After WALK, turns run again (pedestrians still on the
  crosswalk are assumed to be past the turning path).
* Calls: controllers can see which phases have waiting pedestrians and how long
  they have waited, so they can serve them (see controllers.py and signal_env.py).

Measured per pedestrian: delay from arrival to the start of WALK (pedestrians
still waiting at the end of the episode count with their wait so far).
"""
import json

import numpy as np


class Pedestrians:
    def __init__(self, spec_file, intersections, rate_scale=1.0):
        with open(spec_file) as f:
            spec = json.load(f)
        self.walk = int(spec["walk_s"])
        self.rate_scale = rate_scale
        self.crosswalks = []   # per intersection: list of dicts
        self.variant = []      # per intersection: {phase: lightphase index during WALK}
        for intersection in intersections:
            entry = spec["intersections"].get(intersection["id"], {"crosswalks": [], "walk_variant": {}})
            self.crosswalks.append([dict(cw) for cw in entry["crosswalks"]])
            self.variant.append({int(k): int(v) for k, v in entry.get("walk_variant", {}).items()})
        self.n = len(intersections)
        self.reset(0)

    # ------------------------------------------------------------------ episode
    def reset(self, seed):
        self.rng = np.random.default_rng(10_000 + int(seed or 0))
        self.waiting = [[[] for _ in cws] for cws in self.crosswalks]  # arrival times
        self.walk_phase = [None] * self.n
        self.walk_end = [-1] * self.n
        self.hold_end = [-1] * self.n
        self.shown = [None] * self.n
        self.delays = []

    def display(self, idx, t, phase):
        """Called every second with the vehicle phase intersection idx shows at time t.

        Starts WALK at the start of a green that has a pedestrian call and returns
        the lightphase to show (a WALK variant while WALK runs).
        """
        if phase != self.shown[idx]:
            self.shown[idx] = phase
            if self.walk_phase[idx] is not None and phase != self.walk_phase[idx]:
                self.walk_phase[idx] = None
                self.hold_end[idx] = -1
            served = [k for k, cw in enumerate(self.crosswalks[idx]) if cw["phase"] == phase]
            if served and any(self.waiting[idx][k] for k in served):
                clearance = max(self.crosswalks[idx][k]["clearance_s"] for k in served)
                self.walk_phase[idx] = phase
                self.walk_end[idx] = t + self.walk
                self.hold_end[idx] = t + self.walk + clearance
                for k in served:
                    self.delays.extend(t - a for a in self.waiting[idx][k])
                    self.waiting[idx][k] = []
        if self.walk_phase[idx] == phase and t < self.walk_end[idx]:
            return self.variant[idx].get(phase, phase)
        return phase

    def arrive(self, t):
        """New pedestrians in second t (call after the simulation step)."""
        for idx, cws in enumerate(self.crosswalks):
            for k, cw in enumerate(cws):
                count = self.rng.poisson(cw["peds_per_hour"] * self.rate_scale / 3600.0)
                if not count:
                    continue
                if self.walk_phase[idx] == cw["phase"] and t < self.walk_end[idx]:
                    self.delays.extend([0] * count)  # WALK is on: cross straight away
                else:
                    self.waiting[idx][k].extend([t] * count)

    # ------------------------------------------------------------------ queries
    def holding(self, idx, t):
        """True while WALK or the pedestrian clearance runs: the green must not end."""
        return t < self.hold_end[idx]

    def calls(self, idx):
        """Phases with at least one waiting pedestrian."""
        return {cw["phase"] for k, cw in enumerate(self.crosswalks[idx]) if self.waiting[idx][k]}

    def longest_wait(self, idx, t, phase=None):
        waits = [t - min(self.waiting[idx][k]) for k, cw in enumerate(self.crosswalks[idx])
                 if self.waiting[idx][k] and (phase is None or cw["phase"] == phase)]
        return max(waits, default=0)

    def waiting_count(self, idx):
        return sum(len(w) for w in self.waiting[idx])

    def metrics(self, t):
        delays = list(self.delays) + [t - a for per in self.waiting for w in per for a in w]
        arr = np.array(delays, dtype=float)
        return {
            "peds": int(arr.size),
            "ped_avg_wait_s": float(arr.mean()) if arr.size else 0.0,
            "ped_p95_wait_s": float(np.percentile(arr, 95)) if arr.size else 0.0,
            "ped_max_wait_s": float(arr.max()) if arr.size else 0.0,
        }
