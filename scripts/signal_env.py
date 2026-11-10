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

Pedestrians (networks with a pedestrians.json, e.g. "calaveras"; see
scripts/pedestrians.py): a green can't end while its WALK or pedestrian
clearance runs, the observation adds how long pedestrians have waited for
phases 1 and 2 and whether a pedestrian interval is running, the "queue"
reward counts waiting pedestrians like waiting vehicles, and `max_red` also
caps pedestrian waits. `pedestrians=False` turns them off.

Rewards (per agent, averaged over the seconds in a decision step):
  "queue"    -(waiting vehicles on the intersection's incoming lanes) / 10
  "pressure" -|vehicles on incoming lanes - vehicles on outgoing lanes| / 10
  "legacy"   the class project's reward (normalized counts plus throughput bonus)

Training traffic: when no flow_file is given, every reset draws a new random
scenario (demand level from `train_levels`, seed from `train_seeds`), so the
agent never sees the evaluation seeds (0-4) during training.
"""
import numpy as np
from gymnasium import spaces

from controllers import CLEARANCE_PHASE
from scenarios import generate_flow
from scenarios import NETWORKS
from traffic_env import CONFIGS_DIR, TrafficSignalEnv

LANE_SCALE = 10.0
TRAIN_SEEDS = range(1000, 1050)


class TimedSignalEnv(TrafficSignalEnv):
    def __init__(self, horizon=3600, decision_interval=5, min_green=10, clearance=5, reward="queue",
                 flow_file=None, seed=None, train_levels=("low", "medium", "high"), train_seeds=TRAIN_SEEDS,
                 rng_seed=None, save_replay=False, network="1x2", neighbors=False, max_red=None, wait_obs=False,
                 pedestrians=True, ped_scale=1.0, **kwargs):
        self.horizon = horizon
        self.network = network
        if "config_file" not in kwargs:
            kwargs["config_file"] = CONFIGS_DIR / NETWORKS[network]["config"]
        if pedestrians and NETWORKS[network].get("pedestrians"):
            kwargs["pedestrians"] = NETWORKS[network]["pedestrians"]
            kwargs["ped_scale"] = ped_scale
        if NETWORKS[network].get("signal_plan"):
            kwargs["signal_plan"] = NETWORKS[network]["signal_plan"]
        self.fixed_scenario = flow_file is not None
        self.train_levels = list(train_levels)
        self.train_seeds = list(train_seeds)
        self.rng = np.random.default_rng(rng_seed)
        if not self.fixed_scenario:
            flow_file, seed = self._sample_scenario()
        super().__init__(horizon, flow_file=flow_file, seed=seed, save_replay=save_replay, **kwargs)
        if reward not in ("queue", "pressure", "legacy"):
            raise ValueError(f"unknown reward {reward!r}")
        self.decision_interval = decision_interval
        self.min_green = min_green
        self.clearance = clearance
        self.reward_type = reward
        if wait_obs and not max_red:
            raise ValueError("wait_obs needs max_red")
        self.max_red = max_red
        self.wait_obs = wait_obs

        # Lightphases beyond 8 are pedestrian WALK variants, never chosen directly.
        self.num_phases = [min(9, len(i["trafficLight"]["lightphases"])) for i in self.intersections]
        self.num_green = max(self.num_phases) - 1  # phases 1..n-1
        self.incoming_lanes = [
            [lane for road in self.incoming_roads[i] for lane in self.roads[road]]
            for i in range(self.num_intersections)
        ]
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
                for p in range(1, min(9, len(lightphases)))
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
        # Pedestrians: longest wait for phase 1 and for phase 2 (in 100 s), and "pedestrian interval running".
        if self.peds is not None:
            self.obs_size += 3
        self.observation_space = spaces.Dict({
            agent: spaces.Box(low=0.0, high=np.inf, shape=(self.obs_size,), dtype=np.float32) for agent in self.agents
        })
        self.action_space = spaces.Dict({agent: spaces.Discrete(self.num_green) for agent in self.agents})

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
        # Seconds each incoming lane has had a stopped vehicle while not served by a green.
        self.red_age = [dict.fromkeys(lanes, 0) for lanes in self.incoming_lanes]
        self.forced_switches = 0
        self.previous_vehicle_count = self.engine.get_vehicle_count()
        return self._observations(), {}

    def step(self, actions):
        for agent, action in actions.items():
            idx = int(agent.split("_")[1])
            desired = int(action) + 1
            forced = self._forced_phase(idx)
            if forced is not None:
                self.forced_switches += int(forced != desired)
                desired = forced
            if desired >= self.num_phases[idx]:
                continue
            if self._can_switch(idx) and desired != self.phase[idx]:
                if self.clearance > 0:
                    self.clearance_left[idx] = self.clearance
                    self.next_phase[idx] = desired
                else:
                    self.phase[idx] = desired
                    self.green_elapsed[idx] = 0

        reward_sum = np.zeros(self.num_intersections)
        for _ in range(self.decision_interval):
            now = self.engine.get_current_time()
            for idx, intersection in enumerate(self.intersections):
                shown = CLEARANCE_PHASE if self.clearance_left[idx] > 0 else self.phase[idx]
                if self.peds is not None:
                    shown = self.peds.display(idx, now, shown)
                self.engine.set_tl_phase(intersection["id"], shown)
            self.engine.next_step()
            if self.peds is not None:
                self.peds.arrive(now)
            for idx in range(self.num_intersections):
                if self.clearance_left[idx] > 0:
                    self.clearance_left[idx] -= 1
                    if self.clearance_left[idx] == 0:
                        self.phase[idx] = self.next_phase[idx]
                        self.green_elapsed[idx] = 0
                else:
                    self.green_elapsed[idx] += 1
            waiting = self.engine.get_lane_waiting_vehicle_count()
            if self.max_red:
                self._update_red_age(waiting)
            self._record_second(waiting)
            reward_sum += self._step_rewards(waiting)
            if self.engine.get_current_time() >= self.horizon:
                break

        rewards = {agent: float(reward_sum[i] / self.decision_interval) for i, agent in enumerate(self.agents)}
        done = self.engine.get_current_time() >= self.horizon
        terminateds = {agent: False for agent in self.agents}
        terminateds["__all__"] = False
        # The episode ends because of the time limit, which is a truncation.
        truncateds = {agent: done for agent in self.agents}
        truncateds["__all__"] = done
        return self._observations(), rewards, terminateds, truncateds, {}

    # ------------------------------------------------------------------ helpers
    def _can_switch(self, idx):
        if self.clearance_left[idx] > 0 or self.green_elapsed[idx] < self.min_green:
            return False
        return self.peds is None or not self.peds.holding(idx, self.engine.get_current_time())

    def _update_red_age(self, waiting):
        for idx, ages in enumerate(self.red_age):
            served = () if self.clearance_left[idx] > 0 else self.phase_lanes[idx][self.phase[idx]]
            for lane in ages:
                ages[lane] = 0 if lane in served or waiting.get(lane, 0) == 0 else ages[lane] + 1

    def _forced_phase(self, idx):
        """The green the max-red rule imposes on intersection idx now, or None."""
        if not self.max_red or not self._can_switch(idx):
            return None
        ages = self.red_age[idx]
        now = self.engine.get_current_time()
        # Pedestrian waits count like lane red times, for the phase that serves them.
        ped_ages = {}
        if self.peds is not None:
            ped_ages = {p: self.peds.longest_wait(idx, now, p) for p in self.peds.calls(idx)}
        if max(ages.values(), default=0) < self.max_red and max(ped_ages.values(), default=0) < self.max_red:
            return None
        overdue = {lane: age for lane, age in ages.items() if age >= self.max_red}
        # Among phases serving an overdue lane or pedestrian, pick the one with the most overdue red time.
        def score(p):
            lanes = self.phase_lanes[idx][p]
            ped = ped_ages.get(p, 0)
            return (sum(a for l, a in overdue.items() if l in lanes) + (ped if ped >= self.max_red else 0),
                    sum(a for l, a in ages.items() if l in lanes) + ped)
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
            vec[base + self.num_green] = float(self._can_switch(idx))
            vec[base + self.num_green + 1] = float(self.clearance_left[idx] > 0)
            if self.neighbors:
                offset = base + self.num_green + 2
                for d, other in enumerate(self.neighbor_of[idx]):
                    if other is None:
                        continue
                    for k, road in enumerate(self.incoming_roads[other][:4]):
                        vec[offset + 4 * d + k] = sum(waiting.get(l, 0) for l in self.roads[road]) / LANE_SCALE
            if self.wait_obs:
                offset = self.obs_size - n_lanes - (3 if self.peds is not None else 0)
                vec[offset:offset + n] = [min(self.red_age[idx][l] / self.max_red, 1.0) for l in lanes]
            if self.peds is not None:
                now = self.engine.get_current_time()
                vec[-3] = self.peds.longest_wait(idx, now, 1) / 100.0
                vec[-2] = self.peds.longest_wait(idx, now, 2) / 100.0
                vec[-1] = float(self.peds.holding(idx, now))
            obs[agent] = vec
        return obs

    def _step_rewards(self, waiting):
        if self.reward_type == "queue":
            peds = [self.peds.waiting_count(i) if self.peds is not None else 0 for i in range(self.num_intersections)]
            return np.array([-(sum(waiting.get(l, 0) for l in lanes) + p) / LANE_SCALE
                             for lanes, p in zip(self.incoming_lanes, peds)])
        if self.reward_type == "pressure":
            counts = self.engine.get_lane_vehicle_count()
            return np.array([
                -abs(sum(counts.get(l, 0) for l in inc) - sum(counts.get(l, 0) for l in out)) / LANE_SCALE
                for inc, out in zip(self.incoming_lanes, self.outgoing_lanes)
            ])
        counts = self.engine.get_lane_vehicle_count()
        return np.array(self._get_rewards(counts, waiting))
