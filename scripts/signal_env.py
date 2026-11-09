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
* every change goes through `clearance` seconds (default 5) of phase 0.

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
from traffic_env import TrafficSignalEnv

LANE_SCALE = 10.0
TRAIN_SEEDS = range(1000, 1050)


class TimedSignalEnv(TrafficSignalEnv):
    def __init__(self, horizon=3600, decision_interval=5, min_green=10, clearance=5, reward="queue",
                 flow_file=None, seed=None, train_levels=("low", "medium", "high"), train_seeds=TRAIN_SEEDS,
                 rng_seed=None, save_replay=False, **kwargs):
        self.horizon = horizon
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

        self.num_phases = [len(i["trafficLight"]["lightphases"]) for i in self.intersections]
        self.num_green = max(self.num_phases) - 1  # phases 1..n-1
        self.incoming_lanes = [
            [lane for road in self.incoming_roads[i] for lane in self.roads[road]]
            for i in range(self.num_intersections)
        ]
        self.outgoing_lanes = []
        for idx, intersection in enumerate(self.intersections):
            out_roads = [r for r in intersection["roads"] if r not in self.incoming_roads[idx]]
            self.outgoing_lanes.append([lane for road in out_roads for lane in self.roads[road]])
        n_lanes = max(len(lanes) for lanes in self.incoming_lanes)
        # Per incoming lane: vehicle count and waiting count; then current-phase
        # one-hot, "can switch now" and "in clearance" flags.
        self.obs_size = 2 * n_lanes + self.num_green + 2
        self.observation_space = spaces.Dict({
            agent: spaces.Box(low=0.0, high=np.inf, shape=(self.obs_size,), dtype=np.float32) for agent in self.agents
        })
        self.action_space = spaces.Dict({agent: spaces.Discrete(self.num_green) for agent in self.agents})

    # ------------------------------------------------------------------ scenarios
    def _sample_scenario(self):
        level = self.train_levels[self.rng.integers(len(self.train_levels))]
        seed = int(self.train_seeds[self.rng.integers(len(self.train_seeds))])
        return generate_flow(level, seed, self.horizon, cache=True), seed

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
        self.previous_vehicle_count = self.engine.get_vehicle_count()
        return self._observations(), {}

    def step(self, actions):
        for agent, action in actions.items():
            idx = int(agent.split("_")[1])
            desired = int(action) + 1
            if desired >= self.num_phases[idx]:
                continue
            if self.clearance_left[idx] == 0 and self.green_elapsed[idx] >= self.min_green and desired != self.phase[idx]:
                if self.clearance > 0:
                    self.clearance_left[idx] = self.clearance
                    self.next_phase[idx] = desired
                else:
                    self.phase[idx] = desired
                    self.green_elapsed[idx] = 0

        reward_sum = np.zeros(self.num_intersections)
        for _ in range(self.decision_interval):
            for idx, intersection in enumerate(self.intersections):
                shown = CLEARANCE_PHASE if self.clearance_left[idx] > 0 else self.phase[idx]
                self.engine.set_tl_phase(intersection["id"], shown)
            self.engine.next_step()
            for idx in range(self.num_intersections):
                if self.clearance_left[idx] > 0:
                    self.clearance_left[idx] -= 1
                    if self.clearance_left[idx] == 0:
                        self.phase[idx] = self.next_phase[idx]
                        self.green_elapsed[idx] = 0
                else:
                    self.green_elapsed[idx] += 1
            waiting = self.engine.get_lane_waiting_vehicle_count()
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
    def _observations(self):
        counts = self.engine.get_lane_vehicle_count()
        waiting = self.engine.get_lane_waiting_vehicle_count()
        obs = {}
        for idx, agent in enumerate(self.agents):
            lanes = self.incoming_lanes[idx]
            vec = np.zeros(self.obs_size, dtype=np.float32)
            n = len(lanes)
            vec[:n] = [counts.get(l, 0) / LANE_SCALE for l in lanes]
            n_lanes = (self.obs_size - self.num_green - 2) // 2
            vec[n_lanes:n_lanes + n] = [waiting.get(l, 0) / LANE_SCALE for l in lanes]
            phase = self.next_phase[idx] if self.clearance_left[idx] > 0 else self.phase[idx]
            vec[2 * n_lanes + phase - 1] = 1.0
            vec[-2] = float(self.clearance_left[idx] == 0 and self.green_elapsed[idx] >= self.min_green)
            vec[-1] = float(self.clearance_left[idx] > 0)
            obs[agent] = vec
        return obs

    def _step_rewards(self, waiting):
        if self.reward_type == "queue":
            return np.array([-sum(waiting.get(l, 0) for l in lanes) / LANE_SCALE for lanes in self.incoming_lanes])
        if self.reward_type == "pressure":
            counts = self.engine.get_lane_vehicle_count()
            return np.array([
                -abs(sum(counts.get(l, 0) for l in inc) - sum(counts.get(l, 0) for l in out)) / LANE_SCALE
                for inc, out in zip(self.incoming_lanes, self.outgoing_lanes)
            ])
        counts = self.engine.get_lane_vehicle_count()
        return np.array(self._get_rewards(counts, waiting))
