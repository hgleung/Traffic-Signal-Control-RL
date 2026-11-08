import cityflow
from gymnasium import spaces
from ray.rllib.env.multi_agent_env import MultiAgentEnv
import numpy as np
import json
import os
import csv
import tempfile
from pathlib import Path

# Resolve every path from the repository root so the environment works no matter
# which directory it is launched from (scripts/, notebooks/, the repo root, or a
# Ray worker's working directory).
REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIGS_DIR = REPO_ROOT / "configs"
DATA_DIR = REPO_ROOT / "data"
DEFAULT_CONFIG_FILE = CONFIGS_DIR / "training" / "training_config.json"


def _resolve_path(path):
    """Resolve a path given relative to scripts/ (the original convention) or absolute."""
    path = Path(path)
    if path.is_absolute():
        return path
    for base in (Path.cwd(), REPO_ROOT / "scripts", REPO_ROOT):
        candidate = (base / path).resolve()
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not find {path} (looked relative to cwd, scripts/ and repo root)")


class TrafficSignalEnv(MultiAgentEnv):
    def __init__(self, max_steps, config_file=DEFAULT_CONFIG_FILE, road_network_file=None, save_replay=None):
        super().__init__()
        self.max_steps = max_steps
        self.config_file = _resolve_path(config_file)

        # CityFlow resolves "dir" against the process's working directory, which made
        # the original setup break unless it was started from scripts/. Rewrite the
        # config with an absolute "dir" and hand CityFlow that copy instead.
        with open(self.config_file, "r") as f:
            config = json.load(f)
        config["dir"] = str(CONFIGS_DIR) + os.sep
        if save_replay is not None:
            config["saveReplay"] = bool(save_replay)
        if config.get("saveReplay"):
            for key in ("roadnetLogFile", "replayLogFile"):
                (CONFIGS_DIR / config[key]).resolve().parent.mkdir(parents=True, exist_ok=True)
        self._resolved_config = tempfile.NamedTemporaryFile("w", suffix=".json", prefix="cityflow_config_", delete=False)
        json.dump(config, self._resolved_config)
        self._resolved_config.close()
        self.engine = cityflow.Engine(self._resolved_config.name, thread_num=1)

        if road_network_file is None:
            self.road_network_file = (CONFIGS_DIR / config["roadnetFile"]).resolve()
        else:
            self.road_network_file = _resolve_path(road_network_file)

        self.intersections, self.roads, self.incoming_roads = self._parse_network(self.road_network_file)
        self.num_intersections = len(self.intersections)
        self.num_roads = len(self.roads)
        self.num_lanes = sum(len(lanes) for lanes in self.roads.values())
        # Observation per lane on incoming roads: vehicle count and waiting count.
        self.max_lanes = max(
            2 * sum(len(self.roads[road]) for road in self.incoming_roads[i])
            for i in range(self.num_intersections)
        )

        self.previous_vehicle_count = self.engine.get_vehicle_count()

        self.agents = self.possible_agents = [f"agent_{i}" for i in range(self.num_intersections)]
        self.observation_space = spaces.Dict({
            agent: spaces.Box(low=0, high=np.inf, shape=(self.max_lanes,), dtype=np.float32) for agent in self.agents
        })
        self.action_space = spaces.Dict({
            agent: spaces.Discrete(len(self.intersections[i]["trafficLight"]["lightphases"])) for i, agent in enumerate(self.agents)
        })

    def reset(self, *, seed=None, options=None):
        self.engine.reset()
        self.previous_vehicle_count = self.engine.get_vehicle_count()
        lane_vehicle_counts = self.engine.get_lane_vehicle_count()
        lane_waiting_counts = self.engine.get_lane_waiting_vehicle_count()
        observations = {
            f"agent_{i}": self._get_observation_for_agent(i, lane_vehicle_counts, lane_waiting_counts)
            for i in range(self.num_intersections)
        }
        return observations, {}

    def step(self, actions):
        for agent_id, action in actions.items():
            intersection_idx = int(agent_id.split("_")[1])
            self.engine.set_tl_phase(self.intersections[intersection_idx]["id"], int(action))
        self.engine.next_step()

        # Query lane statistics once per step instead of once per lane.
        lane_vehicle_counts = self.engine.get_lane_vehicle_count()
        lane_waiting_counts = self.engine.get_lane_waiting_vehicle_count()
        observations = {
            f"agent_{i}": self._get_observation_for_agent(i, lane_vehicle_counts, lane_waiting_counts)
            for i in range(self.num_intersections)
        }
        rewards = {
            f"agent_{i}": reward for i, reward in enumerate(self._get_rewards(lane_vehicle_counts, lane_waiting_counts))
        }
        done = self._is_done()
        terminateds = {f"agent_{i}": done for i in range(self.num_intersections)}
        terminateds["__all__"] = done
        truncateds = {f"agent_{i}": False for i in range(self.num_intersections)}
        truncateds["__all__"] = done
        infos = {}

        return observations, rewards, terminateds, truncateds, infos

    def _parse_network(self, road_network_file):
        with open(road_network_file, "r") as f:
            data = json.load(f)
        # Virtual intersections are the network's entry/exit points: they have no
        # traffic light, so they must not get an agent.
        intersections = [i for i in data["intersections"] if not i.get("virtual", False)]
        # Use the real lane count of each road instead of assuming three lanes.
        lanes_by_road = {
            road["id"]: [f"{road['id']}_{lane_idx}" for lane_idx in range(len(road["lanes"]))]
            for road in data["roads"]
        }
        # Observe only roads that flow into each intersection; outgoing roads do not
        # queue at its light.
        road_end = {road["id"]: road["endIntersection"] for road in data["roads"]}
        incoming_roads = [
            [road for road in intersection["roads"] if road_end[road] == intersection["id"]]
            for intersection in intersections
        ]
        return intersections, lanes_by_road, incoming_roads

    def _get_observation_for_agent(self, agent_idx, lane_vehicle_counts, lane_waiting_counts):
        # Currently retrieves vehicle count and waiting vehicle count per incoming lane.
        lanes = [lane for road in self.incoming_roads[agent_idx] for lane in self.roads[road]]
        vehicle_counts = [lane_vehicle_counts.get(lane, 0) for lane in lanes]
        waiting_counts = [lane_waiting_counts.get(lane, 0) for lane in lanes]
        state = vehicle_counts + waiting_counts
        padded_state = state + [0] * (self.max_lanes - len(state))
        return np.array(padded_state, dtype=np.float32)

    def _get_rewards(self, lane_vehicle_counts, lane_waiting_counts):
        rewards = []
        throughput = max(0, (self.previous_vehicle_count - self.engine.get_vehicle_count()) / self.num_intersections)

        # Pre-defined constants (useful for now, eventually need to implement dynamic normalization during runtime when switching to larger grid sizes)
        MAX_VEHICLE_COUNT = 87 * self.num_intersections
        MAX_WAITING_COUNT = 44 * self.num_lanes
        MAX_THROUGHPUT = 0.5 * self.num_intersections

        for agent_idx in range(self.num_intersections):
            lanes = [lane for road in self.incoming_roads[agent_idx] for lane in self.roads[road]]
            total_vehicle_count = sum(lane_vehicle_counts.get(lane, 0) for lane in lanes)
            total_waiting_count = sum(lane_waiting_counts.get(lane, 0) for lane in lanes)
            normalized_waiting_count = total_waiting_count / MAX_WAITING_COUNT
            normalized_vehicle_count = total_vehicle_count / MAX_VEHICLE_COUNT
            normalized_throughput = throughput / MAX_THROUGHPUT
            reward = 0.1 - 0.5 * normalized_waiting_count - 0.5 * normalized_vehicle_count + 5.0 * normalized_throughput
            rewards.append(reward)

        self.previous_vehicle_count = self.engine.get_vehicle_count()
        return rewards

    def _is_done(self):
        return self.engine.get_current_time() >= self.max_steps


# Baseline: fixed signal durations
def fixed_signal_control(env, output_file=DATA_DIR / "fixed_rewards.csv"):
    env.reset()
    total_reward = 0
    done = False
    step = 0  # Initialize a step counter
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Step", "Total Reward"])
        while not done:
            # Cycle through fixed phases based on the step counter
            actions = {f"agent_{i}": step % env.action_space[f"agent_{i}"].n for i in range(env.num_intersections)}
            obs, reward, terminated, truncated, info = env.step(actions)
            total_reward += sum(reward.values())
            writer.writerow([step, total_reward])
            done = terminated["__all__"]
            step += 1  # Increment the step counter
    return total_reward


if __name__ == "__main__":
    max_steps = 100
    env = TrafficSignalEnv(max_steps)
    fixed_rewards = fixed_signal_control(env)
    print(fixed_rewards)
