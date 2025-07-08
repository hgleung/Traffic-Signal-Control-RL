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
├── scripts/           # Core RL and simulation scripts
│   ├── ppo.py         # PPO RL agent implementation
│   ├── traffic_env.py # Environment wrapper for CityFlow
│   ├── traffic_env_ppo.py # Policy evaluation and utility functions
│   └── results/       # Graphs and training logs
└── requirements.txt   # Python dependencies
```

## Getting Started
### Prerequisites
- Python 3.10+
- [CityFlow traffic simulator](https://cityflow-project.github.io/)

### Installation
1. Clone the repository:
   ```bash
   git clone <repo_url>
   cd Traffic-Signal-Control-RL
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running Experiments
#### 1. Baseline: Fixed Signal Control
Run the fixed signal control simulation from the notebook or script:
```python
from traffic_env import TrafficSignalEnv, fixed_signal_control
# ... (see notebooks/project.ipynb)
fixed_rewards = fixed_signal_control(env)
```

#### 2. RL Agent: PPO
Train or evaluate the PPO agent:
```bash
python scripts/ppo.py
```
Or run evaluation:
```python
from traffic_env_ppo import evaluate_policy
ppo_rewards = evaluate_policy()
```

#### 3. Visualize Results
Use the notebook or `scripts/results/create_graph.py` to visualize reward curves:
```bash
python scripts/results/create_graph.py
```
This generates `rewards.png` comparing mean rewards over training iterations.

## Configuration
- Edit files in `configs/training/` and `configs/evaluation/` to modify the road network, flow, and simulation parameters.
- Example config: `configs/training/training_config.json`

## Results
### Baseline (Fixed Signal Control)
- **Total reward** (sample run): ~992.33
- Reward increases steadily with simulation steps, reaching ~885 at step 893 (see `data/fixed_rewards.csv`).

### PPO RL Agent
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
