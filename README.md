# Traffic Signal Control with Reinforcement Learning

## Overview
This project explores the application of Reinforcement Learning (RL) to optimize traffic signal control in urban road networks. By leveraging the CityFlow traffic simulator, the system aims to reduce congestion and improve traffic flow by dynamically adjusting traffic lights based on real-time traffic conditions.

The project provides a comparison between a rule-based fixed signal control baseline and a Proximal Policy Optimization (PPO) RL agent.

## Features
- **Multi-agent RL environment** for traffic signal control using CityFlow
- **PPO (Proximal Policy Optimization)** implementation for learning optimal traffic light policies
- **Baseline fixed-signal control** for benchmarking
- **Configurable simulation scenarios** via JSON files
- **Training, evaluation, and visualization tools**

## Project Structure
```
├── configs/           # Simulation and network configuration files
├── data/              # Collected rewards and results (e.g., fixed_rewards.csv)
├── frontend/          # Visualization and replay tools
├── logs/              # Training and evaluation logs
├── models/            # Saved RL models
├── notebooks/         # Jupyter notebook for experiments and analysis
├── results/           # Evaluation outputs (scripts/evaluate.py)
├── scripts/           # Core RL and simulation scripts
│   ├── evaluate.py    # Compare controllers on travel time, queues, throughput
│   ├── controllers.py # Baselines: fixed-time, max-pressure, legacy
│   ├── scenarios.py   # Randomized low/medium/high demand scenarios
│   ├── ppo.py         # PPO RL agent implementation
│   ├── traffic_env.py # Environment wrapper for CityFlow
│   ├── traffic_env_ppo.py # Policy evaluation and utility functions
│   └── results/       # Graphs and training logs
└── requirements.txt   # Python dependencies
```

## Getting Started
### Prerequisites
- Python 3.10 (Ray 2.39 and the CityFlow build are tested against it)
- A C++ toolchain and CMake, to build [CityFlow](https://cityflow-project.github.io/) (it is not on PyPI)

### Installation
**Option A: Docker (recommended).** Builds CityFlow and all dependencies in one step:
```bash
docker build -t traffic-rl .
docker run --rm traffic-rl            # runs the smoke test
docker run --rm -it traffic-rl bash   # interactive shell
```

**Option B: local virtual environment.**
```bash
git clone <repo_url>
cd Traffic-Signal-Control-RL
python3.10 -m venv venv && source venv/bin/activate
pip install "git+https://github.com/cityflow-project/CityFlow.git@81ee0f47659ca66177a71f81676691c58ee89184"
pip install -r requirements.txt
pytest -q tests                        # smoke test: one short episode end to end
```
`requirements-freeze-2024.txt` is the original full `pip freeze` from the class project, kept for reference.

Scripts resolve config, data, log and model paths from the repository root, so they can be run from any directory.

### Running Experiments
#### 1. Baseline: Fixed Signal Control
```bash
python scripts/traffic_env.py
```
or from Python:
```python
from traffic_env import TrafficSignalEnv, fixed_signal_control
env = TrafficSignalEnv(max_steps=1000)
fixed_rewards = fixed_signal_control(env)
```

#### 2. RL Agent: PPO (RLlib, multi-agent)
Training, tuning and evaluation live in `scripts/traffic_env_ppo.py` (`train_PPO()`, `tune_hyperparameters()`, `evaluate_policy()`).
```python
from traffic_env_ppo import train_PPO, evaluate_policy
train_PPO()
ppo_rewards = evaluate_policy()
```
> **Note:** the checkpoint in `models/ppo_trained_model` was trained with one agent per intersection *including* the six virtual (light-less) entry points. The environment now only creates agents for the two signalised intersections, so that checkpoint no longer loads and the model needs to be retrained.

`scripts/ppo.py` is a single-agent CleanRL PPO reference implementation and is not wired to this environment.

#### 3. Evaluate controllers on traffic outcomes
```bash
python scripts/evaluate.py                                   # all baselines, low/medium/high/original, 5 seeds, 1 h episodes
python scripts/evaluate.py --scenarios medium --seeds 0 1 --horizon 600
```
This runs each controller on randomized traffic scenarios (`scripts/scenarios.py`: Poisson arrivals with turning traffic at low, medium and high demand, plus the original flow file) and reports average travel time, average queue length, throughput and completion rate as mean ± std over seeds. Results go to `results/evaluation/` (`summary.md`, `summary.csv`, `runs.csv`, `travel_time.png`).

Baselines (`scripts/controllers.py`):
- `fixed_time`: pre-timed 4-phase plan, 30 s green per phase with a 5 s clearance between phases
- `max_pressure`: max-pressure control, re-decided every 10 s (minimum green) with a 5 s clearance on every change
- `legacy_1s_cycle`: the class project's original baseline, which changes phase every second (kept for reference only; see below)

#### 4. Visualize Results
Use the notebook or `scripts/results/create_graph.py` to visualize reward curves:
```bash
python scripts/results/create_graph.py
```
This generates `rewards.png` comparing mean rewards over training iterations.

## Configuration
- Edit files in `configs/training/` and `configs/evaluation/` to modify the road network, flow, and simulation parameters.
- Example config: `configs/training/training_config.json`

## Results
### Baseline evaluation (`python scripts/evaluate.py`)
Averages over 5 seeds of 1-hour episodes on the 2-intersection network (the original flow file is deterministic, so it runs once):

| Scenario | Controller | Avg travel time (s) | Avg queue (veh) | Throughput (veh) | Completed | Episodes |
|---|---|---|---|---|---|---|
| low | legacy_1s_cycle | 49.3 ± 0.3 | 0.03 ± 0.00 | 1067 ± 19 | 98.7% | 5 |
| low | fixed_time | 93.7 ± 2.4 | 6.44 ± 0.29 | 1062 ± 20 | 98.2% | 5 |
| low | max_pressure | 61.5 ± 0.3 | 1.64 ± 0.03 | 1065 ± 19 | 98.4% | 5 |
| medium | legacy_1s_cycle | 52.0 ± 0.4 | 0.07 ± 0.01 | 2160 ± 27 | 98.6% | 5 |
| medium | fixed_time | 119.6 ± 9.0 | 20.31 ± 2.49 | 2120 ± 18 | 96.8% | 5 |
| medium | max_pressure | 70.2 ± 0.9 | 5.51 ± 0.24 | 2146 ± 22 | 97.9% | 5 |
| high | legacy_1s_cycle | 59.1 ± 0.8 | 0.58 ± 0.08 | 3231 ± 45 | 98.3% | 5 |
| high | fixed_time | 314.5 ± 15.3 | 70.83 ± 3.06 | 2784 ± 39 | 92.6% | 5 |
| high | max_pressure | 112.5 ± 6.1 | 25.64 ± 2.72 | 3160 ± 44 | 96.1% | 5 |
| original | legacy_1s_cycle | 1344.6 ± 0.0 | 59.97 ± 0.00 | 2754 ± 0 | 91.9% | 1 |
| original | fixed_time | 1426.3 ± 0.0 | 83.32 ± 0.00 | 2300 ± 0 | 92.4% | 1 |
| original | max_pressure | 1072.3 ± 0.0 | 48.48 ± 0.00 | 4444 ± 0 | 96.1% | 1 |

![Average travel time](results/evaluation/travel_time.png)

Takeaways:
- **Max-pressure beats the fixed-time plan everywhere**, cutting average travel time by about a third at low demand and by almost two thirds at high demand.
- **The original "change phase every second" baseline looks best, but only because CityFlow doesn't penalize it.** CityFlow has no yellow time or start-up loss, so flicking through phases every second lets vehicles through almost without stopping. At medium demand, cycling the four main phases once per second gives about 50 s average travel time, but cycling them once every 3 s gives over 400 s. Real signals can't switch like that, so the RL agents need to be held to the same minimum green and clearance times as the baselines before they can be compared fairly. Step 3 makes that change.
- The original flow file (one car every 2 s on every straight route, with no turns) is far beyond the network's capacity for every controller, which made it a poor test case on its own.

### Class project results (reward-based, superseded)
These compare the environment's own reward rather than traffic outcomes. They were produced with agents on the 6 light-less intersections and a phase-every-second baseline, so they aren't comparable with the table above.

#### Baseline (Fixed Signal Control)
- **Total reward** (sample run): ~992.33
- Reward increases steadily with simulation steps, reaching ~885 at step 893 (see `data/fixed_rewards.csv`).

#### PPO RL Agent
- **Mean reward** improves over training:
    - Iteration 1: 1014.80
    - Iteration 10: 1205.67
    - Iteration 20: 1241.89
    - Iteration 30: 1270.44
    - Iteration 40: 1293.09
    - Iteration 46: 1311.09
- PPO outperforms the fixed-signal baseline, as shown in `scripts/results/rewards.png`.

![Reward Curve](scripts/results/rewards.png)

## References
- [CityFlow: An Open-source Traffic Simulator](https://cityflow-project.github.io/)
- RL Algorithm: PPO ([Schulman et al., 2017](https://arxiv.org/abs/1707.06347))

## Acknowledgements
- This project uses CityFlow for traffic simulation and builds on open-source RL libraries.

## License
MIT License

---

For questions or contributions, please open an issue or pull request.
