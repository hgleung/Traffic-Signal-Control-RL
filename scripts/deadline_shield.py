"""Deadline shield: a provable bound on how long any movement waits at red.

The naive max-red rule in signal_env.py waits until a lane has been red for
`max_red` seconds and only then switches to it. By that time several lanes may
be overdue at once, so they are served one after another, each needing a
clearance and a minimum green, and the lane served last waits far longer than
`max_red`. In heavy traffic the forced short greens also waste capacity.

This shield instead plans ahead, like earliest-deadline-first scheduling in a
real-time system. At every decision it checks whether the agent's action still
leaves a schedule that serves every waiting lane before its deadline:

    a lane with a stopped vehicle that is not being served "ages" one second per
    second, and must get a green before its age reaches max_red;
    switching to a phase costs `clearance` seconds, then the phase must stay
    green for `min_green` seconds (rounded up to the decision interval).

A schedule is built greedily: repeatedly take the waiting lane with the least
slack and serve it with the phase that also covers the most other waiting lanes.
If the agent's action leaves no feasible schedule, the shield overrides it with
the first phase of the schedule built from switching now. Otherwise the agent's
action stands, so the shield only acts when it has to (minimal intervention).

Guarantee: if the plan was feasible at the previous decision, following its
first step keeps it feasible, so every lane that was waiting gets a green within
max_red seconds. Vehicles that arrive later start with zero age; they are covered
as long as max_red is at least the longest schedule, one decision interval plus
`n_cover` x (clearance + min green), where n_cover is the number of phases
needed to cover every lane (4 for the standard 4-phase plan). With 10 s minimum
green, 5 s clearance and 5 s decisions that is 5 + 4 x 15 = 65 s, so any
max_red >= 65 s is guaranteed. Below that the shield is best-effort.
"""
import math


class DeadlineShield:
    def __init__(self, phase_lanes, max_red, min_green=10, clearance=5, decision_interval=5):
        """phase_lanes: {phase: set of incoming lanes that phase serves} for one intersection."""
        self.phase_lanes = {p: frozenset(lanes) for p, lanes in phase_lanes.items()}
        self.max_red = max_red
        self.clearance = clearance
        self.decision_interval = decision_interval
        # A green lasts at least min_green, and can only end at a decision.
        self.green_block = decision_interval * math.ceil(min_green / decision_interval)
        self.min_green = min_green

    # ------------------------------------------------------------------ planning
    def plan(self, ages, waiting_lanes, current, green_elapsed, first=None):
        """Greedy schedule for serving every waiting lane before its deadline.

        ages:          {lane: seconds this lane has had a stopped vehicle at red}
        waiting_lanes: lanes with at least one stopped vehicle now
        current:       the green phase shown now
        green_elapsed: seconds the current green has run
        first:         None to keep `current` for (at least) one more decision;
                       a phase to switch to it now.

        Returns (feasible, phases, worst_slack): the planned phase order, and the
        smallest slack (deadline minus time served) over all waiting lanes, which
        is negative if some lane would be served late.
        """
        d, c, g = self.decision_interval, self.clearance, self.green_block
        if first is None or first == current:
            # Hold the current green until the next decision at which it may end,
            # then plan from there. Lanes it serves start ageing when it ends.
            t = d * math.ceil(max(d, self.min_green - green_elapsed) / d)
            served = self.phase_lanes[current]
            pending = {l: (0 if l in served else ages.get(l, 0) + t) + c for l in waiting_lanes}
            order = []
        else:
            # Switch now: `first` turns green after the clearance and holds for a
            # minimum green; the next planned phase turns green c + g + c from now.
            served = self.phase_lanes[first]
            pending = {l: ages.get(l, 0) + c + g + c for l in waiting_lanes if l not in served}
            order = [first]
        # pending: each lane's age when the next planned phase turns green.
        return self._greedy(pending, order)

    def _greedy(self, pending, order):
        """pending: {lane: its age when the next planned phase turns green}."""
        worst = math.inf
        offset = 0
        pending = dict(pending)
        while pending:
            urgent = max(pending, key=pending.get)
            candidates = [p for p, lanes in self.phase_lanes.items() if urgent in lanes]
            if not candidates:  # a lane no phase serves (should not happen)
                pending.pop(urgent)
                continue
            best = max(candidates, key=lambda p: (len(self.phase_lanes[p] & pending.keys()),
                                                  sum(a for l, a in pending.items() if l in self.phase_lanes[p])))
            for lane in self.phase_lanes[best] & pending.keys():
                worst = min(worst, self.max_red - (pending.pop(lane) + offset))
            order.append(best)
            offset += self.green_block + self.clearance
        return worst >= 0, order, worst

    # ------------------------------------------------------------------ decisions
    def filter(self, desired, ages, waiting_lanes, current, green_elapsed):
        """Return (phase to request, intervened) for an agent that may switch now.

        `desired` is the agent's requested green; it is kept unless it leaves no
        feasible schedule.
        """
        ok, _, _ = self.plan(ages, waiting_lanes, current, green_elapsed,
                             first=None if desired == current else desired)
        if ok:
            return desired, False
        # Earliest-deadline-first: switch to the first phase of the plan made
        # from switching now. Among the phases that make a feasible plan, take
        # the one with the most slack; otherwise the one with the least lateness.
        best, best_key = None, None
        for p in self.phase_lanes:
            feasible, _, slack = self.plan(ages, waiting_lanes, current, green_elapsed,
                                           first=None if p == current else p)
            key = (feasible, slack)
            if best_key is None or key > best_key:
                best, best_key = p, key
        return best, best != desired

    def keep_is_feasible(self, ages, waiting_lanes, current, green_elapsed):
        return self.plan(ages, waiting_lanes, current, green_elapsed)[0]
