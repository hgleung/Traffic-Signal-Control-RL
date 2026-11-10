"""RL environment with realistic signal timing.

TrafficSignalEnv lets an agent pick any phase every simulated second. CityFlow
has no yellow time or start-up loss, so flicking phases every second is an
unrealistic shortcut (see the README's evaluation results). TimedSignalEnv holds
RL agents to the same rules as the baseline controllers:

* an agent decides every `decision_interval` seconds (default 5),
* it chooses which green phase it wants (phases 1-8; phase 0 is reserved for
  clearance),
* a green must run for at least `min_green` seconds (default 10) before it can
  change; requests made earlier are ignored, and the observation tells the agent
  whether it can switch,
* every change goes through `clearance` seconds (default 5) of phase 0,
* optionally (`max_red`, off by default), no lane may hold a stopped vehicle at
  red for more than `max_red` seconds. Once a lane reaches that limit and the
  current green has run its minimum, the environment overrides the agent and
  switches to the green that serves the most overdue lanes. Real controllers
  bound red times the same way (a maximum green per phase, or a maximum wait),
  and without it the agents can leave a quiet approach at red for minutes.
  With `wait_obs`, the observation also gives each incoming lane's red time
  so far as a fraction of `max_red`, so the agent can see the limit coming.
* `shield` chooses how `max_red` is enforced: "cap" is the rule above (the
  default when max_red is set), "deadline" is the planning shield in
  deadline_shield.py (it switches early enough that no lane exceeds max_red),
  and "none" only tracks red times (for the cost below and the observation).
* `cost` adds a penalty to the reward, weighted by `cost_weight` (which the
  Lagrangian training loop in train_ppo.py adapts): "intervention" is 1 for each
  decision the shield overrides, "overdue" is the number of lanes held at red
  longer than max_red, averaged over the seconds of the decision.

Rewards (per agent, averaged over the seconds in a decision step):
  "queue"    -(waiting vehicles on the intersection's incoming lanes) / 10
  "pressure" -|vehicles on incoming lanes - vehicles on outgoing lanes| / 10
  "legacy"   the class project's reward (normalized counts plus throughput bonus)
  "fair"     delay-based fairness reward (Raeis & Leon-Garcia, 2021): every
             stopped vehicle costs 1 + alpha (2d - 1) / fair_scale per second, d
             being how long it has been stopped, so a vehicle's total cost is
             w + alpha w^2 / fair_scale for a stop of w seconds (then / 10)

Training traffic: when no flow_file is given, every reset draws a new random
scenario (demand level from `train_levels`, seed from `train_seeds`), so the
agent never sees the evaluation seeds (0-4) during training.
"""
import numpy as np
from gymnasium import spaces

from controllers import CLEARANCE_PHASE
from deadline_shield import DeadlineShield
from scenarios import generate_flow
from scenarios import NETWORKS
from traffic_env import CONFIGS_DIR, TrafficSignalEnv

LANE_SCALE = 10.0
TRAIN_SEEDS = range(1000, 1050)


class TimedSignalEnv(TrafficSignalEnv):
    def __init__(self, horizon=3600, decision_interval=5, min_green=10, clearance=5, reward="queue",
                 flow_file=None, seed=None, train_levels=None, train_seeds=TRAIN_SEEDS,
                 rng_seed=None, save_replay=False, network="1x2", neighbors=False, max_red=None, wait_obs=False,
                 shield=None, cost=None, cost_weight=0.0, fair_alpha=1.0, fair_scale=30.0, **kwargs):
        self.horizon = horizon
        self.network = network
        if "config_file" not in kwargs:
            kwargs["config_file"] = CONFIGS_DIR / NETWORKS[network]["config"]
        self.fixed_scenario = flow_file is not None
        # By default train on every demand level (or recorded flow file) of the network.
        self.train_levels = list(train_levels or NETWORKS[network]["levels"])
        self.train_seeds = list(train_seeds)
        self.rng = np.random.default_rng(rng_seed)
        if not self.fixed_scenario:
            flow_file, seed = self._sample_scenario()
        super().__init__(horizon, flow_file=flow_file, seed=seed, save_replay=save_replay, **kwargs)
        if reward not in ("queue", "pressure", "legacy", "fair"):
            raise ValueError(f"unknown reward {reward!r}")
        self.decision_interval = decision_interval
        self.min_green = min_green
        self.clearance = clearance
        self.reward_type = reward
        if wait_obs and not max_red:
            raise ValueError("wait_obs needs max_red")
        self.max_red = max_red
        self.wait_obs = wait_obs
        if shield is None:
            shield = "cap" if max_red else "none"
        if shield not in ("none", "cap", "deadline"):
            raise ValueError(f"unknown shield {shield!r}")
        if shield != "none" and not max_red:
            raise ValueError("a shield needs max_red")
        if cost not in (None, "intervention", "overdue"):
            raise ValueError(f"unknown cost {cost!r}")
        if cost and not max_red:
            raise ValueError("a cost needs max_red")
        self.shield = shield
        self.cost = cost
        self.cost_weight = float(cost_weight)
        self.fair_alpha = fair_alpha
        self.fair_scale = fair_scale

        self.num_phases = [len(i["trafficLight"]["lightphases"]) for i in self.intersections]
        self.num_green = max(self.num_phases) - 1  # phases 1..n-1
        self.outgoing_lanes = []
        for idx, intersection in enumerate(self.intersections):
            out_roads = [r for r in intersection["roads"] if r not in self.incoming_roads[idx]]
            self.outgoing_lanes.append([lane for road in out_roads for lane in self.roads[road]])
        self.n_lanes = max(len(lanes) for lanes in self.incoming_lanes)
        # Incoming lanes that each green phase lets through.
        self.phase_lanes = []
        for intersection in self.intersections:
            links = intersection["roadLinks"]
            lightphases = intersection["trafficLight"]["lightphases"]
            self.phase_lanes.append({
                p: {f"{links[k]['startRoad']}_{ll['startLaneIndex']}"
                    for k in lightphases[p]["availableRoadLinks"] for ll in links[k]["laneLinks"]}
                for p in range(1, len(lightphases))
            })
        # Per incoming lane: vehicle count and waiting count; then current-phase
        # one-hot, "can switch now" and "in clearance" flags.
        self.obs_size = 2 * self.n_lanes + self.num_green + 2
        # Optional neighbour information: for each direction (east, north, west,
        # south), the waiting vehicles on each incoming road of the adjacent
        # signalised intersection (zeros where there is none). This lets an agent
        # see queues building up next door before they reach it.
        self.neighbors = neighbors
        self.neighbor_of = self._find_neighbors()
        if neighbors:
            self.obs_size += 4 * 4
        if wait_obs:
            self.obs_size += self.n_lanes
            if shield == "deadline":
                self.obs_size += 1  # "the shield would override keeping the current green"
        self.shields = [
            DeadlineShield(self.phase_lanes[i], max_red, min_green, clearance, decision_interval)
            for i in range(self.num_intersections)
        ] if shield == "deadline" else None
        self.observation_space = spaces.Dict({
            agent: spaces.Box(low=0.0, high=np.inf, shape=(self.obs_size,), dtype=np.float32) for agent in self.agents
        })
        self.action_space = spaces.Dict({agent: spaces.Discrete(self.num_green) for agent in self.agents})

    def set_cost_weight(self, weight):
        """Called by the Lagrangian training loop to update the cost multiplier."""
        self.cost_weight = float(weight)

    # ------------------------------------------------------------------ scenarios
    def _sample_scenario(self):
        level = self.train_levels[self.rng.integers(len(self.train_levels))]
        seed = int(self.train_seeds[self.rng.integers(len(self.train_seeds))])
        return generate_flow(level, seed, self.horizon, cache=True, network=self.network), seed

    # ------------------------------------------------------------------ gym API
    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        if not self.fixed_scenario:
            self._make_engine(*self._sample_scenario())
        else:
            self.engine.reset()
        self._start_episode_stats()
        n = self.num_intersections
        self.phase = [1] * n
        self.green_elapsed = [0] * n
        self.clearance_left = [0] * n
        self.next_phase = [None] * n
        self.forced_switches = 0
        self.interventions = 0
        self.decisions = 0
        self.episode_cost = 0.0
        self._waiting = self.engine.get_lane_waiting_vehicle_count()
        self.previous_vehicle_count = self.engine.get_vehicle_count()
        return self._observations(), {}

    def step(self, actions):
        n = self.num_intersections
        step_cost = np.zeros(n)
        for agent, action in actions.items():
            idx = int(agent.split("_")[1])
            desired = int(action) + 1
            if desired >= self.num_phases[idx]:
                desired = self.phase[idx]
            if self.clearance_left[idx] > 0 or self.green_elapsed[idx] < self.min_green:
                continue  # the current green can't end yet
            self.decisions += 1
            intervened = False
            if self.shield == "cap":
                forced = self._forced_phase(idx)
                if forced is not None:
                    intervened = forced != desired
                    desired = forced
            elif self.shield == "deadline":
                waiting_lanes = [l for l in self.incoming_lanes[idx] if self._waiting.get(l, 0) > 0]
                desired, intervened = self.shields[idx].filter(
                    desired, self.red_age[idx], waiting_lanes, self.phase[idx], self.green_elapsed[idx])
            self.forced_switches += int(intervened)
            self.interventions += int(intervened)
            if self.cost == "intervention":
                step_cost[idx] += float(intervened)
            if desired != self.phase[idx]:
                if self.clearance > 0:
                    self.clearance_left[idx] = self.clearance
                    self.next_phase[idx] = desired
                else:
                    self.phase[idx] = desired
                    self.green_elapsed[idx] = 0

        reward_sum = np.zeros(n)
        overdue_sum = np.zeros(n)
        seconds = 0
        for _ in range(self.decision_interval):
            shown = []
            for idx, intersection in enumerate(self.intersections):
                phase = CLEARANCE_PHASE if self.clearance_left[idx] > 0 else self.phase[idx]
                shown.append(phase)
                self.engine.set_tl_phase(intersection["id"], phase)
            self.engine.next_step()
            for idx in range(n):
                if self.clearance_left[idx] > 0:
                    self.clearance_left[idx] -= 1
                    if self.clearance_left[idx] == 0:
                        self.phase[idx] = self.next_phase[idx]
                        self.green_elapsed[idx] = 0
                else:
                    self.green_elapsed[idx] += 1
            waiting = self.engine.get_lane_waiting_vehicle_count()
            self._record_second(waiting, shown)
            reward_sum += self._step_rewards(waiting)
            if self.cost == "overdue":
                overdue_sum += [sum(a > self.max_red for a in ages.values()) for ages in self.red_age]
            seconds += 1
            if self.engine.get_current_time() >= self.horizon:
                break
        self._waiting = waiting
        if self.cost == "overdue":
            step_cost += overdue_sum / seconds
        self.episode_cost += float(step_cost.sum())

        rewards = {agent: float(reward_sum[i] / self.decision_interval - self.cost_weight * step_cost[i])
                   for i, agent in enumerate(self.agents)}
        done = self.engine.get_current_time() >= self.horizon
        terminateds = {agent: False for agent in self.agents}
        terminateds["__all__"] = False
        # The episode ends because of the time limit, which is a truncation.
        truncateds = {agent: done for agent in self.agents}
        truncateds["__all__"] = done
        return self._observations(), rewards, terminateds, truncateds, {}

    def episode_metrics(self):
        metrics = super().episode_metrics()
        metrics["shield_interventions"] = self.interventions
        metrics["intervention_rate"] = self.interventions / self.decisions if self.decisions else 0.0
        return metrics

    # ------------------------------------------------------------------ helpers
    def _forced_phase(self, idx):
        """The green the max-red rule imposes on intersection idx now, or None."""
        if not self.max_red or self.clearance_left[idx] > 0 or self.green_elapsed[idx] < self.min_green:
            return None
        ages = self.red_age[idx]
        if max(ages.values(), default=0) < self.max_red:
            return None
        overdue = {lane: age for lane, age in ages.items() if age >= self.max_red}
        # Among phases serving an overdue lane, pick the one with the most overdue red time.
        def score(p):
            lanes = self.phase_lanes[idx][p]
            return sum(a for l, a in overdue.items() if l in lanes), sum(a for l, a in ages.items() if l in lanes)
        best = max(self.phase_lanes[idx], key=score)
        return best if score(best)[0] > 0 else None

    def _find_neighbors(self):
        """neighbor_of[idx][d] = index of the signalised intersection reached by
        leaving idx in direction d (0 east, 1 north, 2 west, 3 south), or None."""
        index = {intersection["id"]: k for k, intersection in enumerate(self.intersections)}
        with open(self.road_network_file) as f:
            import json
            roads = {r["id"]: r for r in json.load(f)["roads"]}
        result = []
        for idx, intersection in enumerate(self.intersections):
            row = [None] * 4
            for road_id in intersection["roads"]:
                road = roads[road_id]
                if road["startIntersection"] == intersection["id"] and road["endIntersection"] in index:
                    direction = int(road_id.rsplit("_", 1)[1])
                    if 0 <= direction < 4:
                        row[direction] = index[road["endIntersection"]]
            result.append(row)
        return result

    def _observations(self):
        counts = self.engine.get_lane_vehicle_count()
        waiting = self.engine.get_lane_waiting_vehicle_count()
        obs = {}
        n_lanes = self.n_lanes
        for idx, agent in enumerate(self.agents):
            lanes = self.incoming_lanes[idx]
            vec = np.zeros(self.obs_size, dtype=np.float32)
            n = len(lanes)
            vec[:n] = [counts.get(l, 0) / LANE_SCALE for l in lanes]
            vec[n_lanes:n_lanes + n] = [waiting.get(l, 0) / LANE_SCALE for l in lanes]
            phase = self.next_phase[idx] if self.clearance_left[idx] > 0 else self.phase[idx]
            base = 2 * n_lanes
            vec[base + phase - 1] = 1.0
            vec[base + self.num_green] = float(self.clearance_left[idx] == 0 and self.green_elapsed[idx] >= self.min_green)
            vec[base + self.num_green + 1] = float(self.clearance_left[idx] > 0)
            if self.neighbors:
                offset = base + self.num_green + 2
                for d, other in enumerate(self.neighbor_of[idx]):
                    if other is None:
                        continue
                    for k, road in enumerate(self.incoming_roads[other][:4]):
                        vec[offset + 4 * d + k] = sum(waiting.get(l, 0) for l in self.roads[road]) / LANE_SCALE
            if self.wait_obs:
                offset = self.obs_size - n_lanes
                vec[offset:offset + n] = [min(self.red_age[idx][l] / self.max_red, 1.0) for l in lanes]
                if self.shield == "deadline" and self.clearance_left[idx] == 0:
                    waiting_lanes = [l for l in lanes if waiting.get(l, 0) > 0]
                    vec[-1] = float(not self.shields[idx].keep_is_feasible(
                        self.red_age[idx], waiting_lanes, self.phase[idx], self.green_elapsed[idx]))
            obs[agent] = vec
        return obs

    def _step_rewards(self, waiting):
        if self.reward_type == "queue":
            return np.array([-sum(waiting.get(l, 0) for l in lanes) / LANE_SCALE for lanes in self.incoming_lanes])
        if self.reward_type == "fair":
            lane_vehicles = self.engine.get_lane_vehicles()
            out = []
            for lanes in self.incoming_lanes:
                total = 0.0
                for lane in lanes:
                    for v in lane_vehicles.get(lane, ()):
                        d = self._stop_run.get(v, 0)
                        if d > 0:
                            total += 1 + self.fair_alpha * (2 * d - 1) / self.fair_scale
                out.append(-total / LANE_SCALE)
            return np.array(out)
        if self.reward_type == "pressure":
            counts = self.engine.get_lane_vehicle_count()
            return np.array([
                -abs(sum(counts.get(l, 0) for l in inc) - sum(counts.get(l, 0) for l in out)) / LANE_SCALE
                for inc, out in zip(self.incoming_lanes, self.outgoing_lanes)
            ])
        counts = self.engine.get_lane_vehicle_count()
        return np.array(self._get_rewards(counts, waiting))
