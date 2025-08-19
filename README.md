# Mean-Field Subsample Q-Learning for Multi-Agent Systems

This repository implements a scalable **Mean-Field Subsample Q-Learning** algorithm for multi-agent reinforcement learning. The system uses mean-field approximations to handle large populations while allowing agents to sample only k neighbors for computational efficiency.

## Key Insights

**Mean-Field Subsample Q-Learning** combines:
- **Mean-field approximation**: Represents population states as distributions rather than joint states
- **Subsample learning**: Each agent samples k << n other agents for decision making
- **Statistical equivalence**: Empirical distribution of k agents approximates full population
- **Massive scalability**: Reduces state space from O(|S|^n) to O(|S| × C(n,|S|))

## Algorithm Overview

The system consists of:
- **Global Agent**: Coordinates population behavior, state `s_g ∈ S_g`, action `a_g ∈ A_g`
- **Local Agents**: Population of n agents, each with state `s_i ∈ S_l`, action `a_i ∈ A_l`
- **Mean-Field States**: `(s_g, distribution_counts)` instead of `(s_g, s_1, ..., s_n)`
- **Subsample Deployment**: Each agent samples k others for Q-value maximization

### Key Features

- **🎯 Scalable**: Handles large populations (tested with n=20 agents)
- **⚡ Fast Training**: Vectorized Q-updates with pre-computed reward matrices
- **🔧 Configurable**: All parameters, functions, and spaces defined in `config.json`
- **📊 Performance Tracking**: Training curves comparing different subsample sizes k
- **🧠 Smart Sampling**: Statistical approximation using k-agent subsamples
- **💾 Persistent**: Save/load trained models and configurations
- **📈 Visualization**: Automatic plotting of cumulative rewards vs subsample size

## Installation

1. Clone the repository:
```bash
git clone https://github.com/emiletimothy/Mean-Field-Subsample-Q-Learning.git
cd Mean-Field-Subsample-Q-Learning
```

2. Create virtual environment:
```bash
python3 -m venv mean-field-venv
source mean-field-venv/bin/activate  # On Windows: mean-field-venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

Run the mean-field subsample Q-learning experiment:

```bash
python environment.py
```

This will:
1. **Train** Q-functions for different subsample sizes k ∈ {2, 3, ..., n}
2. **Deploy** each trained Q-function on the full n-agent population
3. **Compare** performance across different k values
4. **Generate** cumulative reward curves and save plot as `cumulative_rewards.png`

### Expected Output
```
Mean-field Q-table: 231 states x 231 actions
Pre-computing mean-field reward matrix...
Starting vectorized Q-learning update...
Completed optimized Q-update in 0.85s (53361 updates)
...
Deployment completed! Total cumulative reward over 200 steps: 29.85
```

## Architecture

### Core Components

#### 1. Training Phase (`centralized_training.py`)
- **Mean-Field Q-Function**: Learns optimal policies using population distributions
- **Vectorized Updates**: Efficient Bellman updates with pre-computed rewards
- **Monte Carlo Sampling**: Approximates expectations for complex transition dynamics

#### 2. Deployment Phase (`environment.py`)
- **Subsample Decision Making**: Each agent samples k others for action selection
- **Statistical Approximation**: k-sample distribution approximates full population
- **Performance Evaluation**: Measures cumulative rewards over H time steps

#### 3. Configuration (`config.json`)
- **Environment Setup**: State/action spaces, reward/transition functions
- **Simulation Parameters**: Population size, horizon, training steps
- **Visualization Settings**: Plot generation and file naming

### Algorithm Flow

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Training      │    │   Deployment     │    │   Analysis      │
│                 │    │                  │    │                 │
│ • Mean-field    │───▶│ • k-subsample    │───▶│ • Compare k     │
│   Q-learning    │    │   decision       │    │   performance   │
│ • Population    │    │ • Full n-agent   │    │ • Generate      │
│   distributions │    │   simulation     │    │   plots         │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

#### Key Classes
- **`Q_function`**: Mean-field Q-learning with vectorized updates
- **`deployment_environment`**: Subsample-based agent coordination
- **Configuration System**: JSON-based parameter and function management

## Configuration

All environment behavior is controlled via `config.json`:

```json
{
  "environment": {
    "global_agent": {
      "state_space": [-2, -1, 0, 1, 2],
      "action_space": [-2, -1, 0, 1, 2]
    },
    "local_agent": {
      "state_space": [-1, 0, 1],
      "action_space": [-1, 0, 1]
    },
    "functions": {
      "local_agent_reward": [
        "def local_agent_reward(local_state, global_state, local_action):",
        "    # Multi-objective reward for local agents",
        "    desired_spacing = 0  # Target formation position",
        "    formation_reward = 1.0 - abs(local_state - desired_spacing)",
        "    movement_reward = 0.5 - 0.1 * abs(local_action)",
        "    global_alignment_reward = 0.3 - 0.1 * abs(local_state - global_state)",
        "    coordination_bonus = 0.2 if abs(local_state - global_state) <= 1 else 0",
        "    total_reward = formation_reward + movement_reward + global_alignment_reward + coordination_bonus",
        "    return max(0.1, total_reward)"
      ]
    }
  },
  "simulation": {
    "horizon": 200,
    "num_local_agents": 14,
    "monte_carlo_samples": 10,
    "discount_factor": 0.9,
    "training_steps": 100,
    "deployment_runs": 20
  }
}
```

### Customizing the Environment

1. **Modify reward functions** directly in the JSON
2. **Adjust population size** with `num_local_agents`
3. **Change state/action spaces** for different complexity
4. **Tune training parameters** for convergence

## Modifying Reward & Transition Functions

### Reward Functions

Reward functions are defined as arrays of Python code strings in `config.json`. Each function can be completely customized:

```json
"functions": {
  "local_agent_reward": [
    "def local_agent_reward(local_state, global_state, local_action):",
    "    # Example: Distance-based coordination reward",
    "    distance_penalty = -0.1 * abs(local_state - global_state)",
    "    action_cost = -0.05 * abs(local_action)",
    "    ",
    "    # Bonus for specific state combinations",
    "    if local_state == 0 and global_state == 0:",
    "        coordination_bonus = 1.0",
    "    else:",
    "        coordination_bonus = 0.0",
    "    ",
    "    return distance_penalty + action_cost + coordination_bonus"
  ],
  "global_agent_reward": [
    "def global_agent_reward(global_state, global_action):",
    "    # Example: State-dependent reward",
    "    if abs(global_state) <= 1:",
    "        state_reward = 2.0  # High reward for central states",
    "    else:",
    "        state_reward = 0.5  # Lower reward for extreme states",
    "    ",
    "    # Action preferences",
    "    action_penalty = -0.1 * abs(global_action)",
    "    ",
    "    return state_reward + action_penalty"
  ]
}
```

### Transition Functions

Transition functions define how agents move between states:

```json
"functions": {
  "local_agent_transition": [
    "def local_agent_transition(local_state, global_state, local_action):",
    "    # Example: Stochastic transitions",
    "    import random",
    "    ",
    "    # Base transition",
    "    new_state = local_state + local_action",
    "    ",
    "    # Add noise with 20% probability",
    "    if random.random() < 0.2:",
    "        noise = random.choice([-1, 1])",
    "        new_state += noise",
    "    ",
    "    # Clamp to bounds",
    "    return max(-1, min(1, new_state))"
  ],
  "global_agent_transition": [
    "def global_agent_transition(global_state, global_action):",
    "    # Example: Non-linear transition",
    "    if global_state == 0:",
    "        # Special behavior at origin",
    "        new_state = global_action * 2",
    "    else:",
    "        # Normal additive transition",
    "        new_state = global_state + global_action",
    "    ",
    "    return max(-2, min(2, new_state))"
  ]
}
```

### Function Modification Examples

#### 1. Sparse Rewards
```json
"local_agent_reward": [
  "def local_agent_reward(local_state, global_state, local_action):",
  "    # Only reward when perfectly aligned",
  "    if local_state == global_state and local_action == 0:",
  "        return 10.0",
  "    else:",
  "        return 0.0"
]
```

#### 2. Momentum-Based Transitions
```json
"local_agent_transition": [
  "def local_agent_transition(local_state, global_state, local_action):",
  "    # Transition depends on global agent's influence",
  "    influence = 0.3 * (global_state - local_state)",
  "    new_state = local_state + local_action + influence",
  "    return max(-1, min(1, int(round(new_state))))"
]
```

#### 3. Multi-Objective Rewards
```json
"local_agent_reward": [
  "def local_agent_reward(local_state, global_state, local_action):",
  "    # Objective 1: Formation maintenance",
  "    formation_reward = 1.0 - 0.5 * abs(local_state)",
  "    ",
  "    # Objective 2: Global coordination  ",
  "    coordination_reward = 0.8 - 0.2 * abs(local_state - global_state)",
  "    ",
  "    # Objective 3: Energy efficiency",
  "    energy_reward = 0.6 - 0.1 * abs(local_action)",
  "    ",
  "    # Weighted combination",
  "    return 0.4*formation_reward + 0.4*coordination_reward + 0.2*energy_reward"
]
```

## Algorithm Details

### Mean-Field Approximation

Instead of tracking joint states `(s_g, s_1, s_2, ..., s_n)`, we use:
- **Mean-field states**: `(s_g, [count_state_-1, count_state_0, count_state_1])`
- **State space reduction**: From `|S_g| × |S_l|^n` to `|S_g| × C(n + |S_l| - 1, |S_l| - 1)`
- **Example**: For n=14, |S_l|=3: `5 × 3^14 = 23,914,845` → `5 × 120 = 600` states

### Subsample Q-Learning

During deployment:
1. **Agent i samples k-1 others** from the population
2. **Creates empirical distribution** from sample + self
3. **Looks up optimal action** in mean-field Q-table
4. **Statistical guarantee**: Large k → accurate population approximation

### Training Process

```python
# Vectorized Q-update
Q_new = R + γ × E[max Q(s', a')]

# Where:
# R = pre-computed reward matrix
# E[·] = Monte Carlo expectation over transitions
# Optimized with caching and batch operations
```

### Performance Optimizations

- **🚀 Vectorized operations**: NumPy batch processing
- **💾 Reward caching**: Pre-compute all reward combinations
- **🎯 Transition caching**: Store frequently accessed transitions
- **⚡ Reduced sampling**: 3 Monte Carlo samples (vs 10) for speed
- **📊 Progress tracking**: Real-time training metrics

### Scalability Results

| Population Size | State Space | Reduction Factor | Training Time |
|----------------|-------------|------------------|---------------|
| n=7            | 45 → 30     | 1.5x             | ~5s           |
| n=14           | 4.8M → 600  | 8000x            | ~30s          |
| n=20           | 1.9B → 1330 | 1.4M×            | ~2min         |

## Results & Analysis

The system generates:

- **📈 Performance Curves**: Cumulative rewards for each k value
- **⏱️ Training Metrics**: Q-update timing and convergence
- **💾 Saved Models**: Trained Q-functions for each subsample size
- **🖼️ Visualization**: `cumulative_rewards.png` comparing all k values

### Key Insights

1. **Subsample Efficiency**: Larger k generally improves performance
2. **Diminishing Returns**: Performance plateaus as k approaches n
3. **Computational Trade-off**: Larger k requires more training time
4. **Statistical Validity**: Even small k can achieve a good performance

## Use Cases

- **Swarm Robotics**: Coordinate large robot teams with limited communication
- **Traffic Control**: Manage vehicle coordination with local sensing
- **Resource Allocation**: Distribute resources among competing agents
- **Formation Control**: Maintain group formations with partial observations

## License

MIT License - see LICENSE file for details.

### If you found this repository useful, please consider giving it a star 🌟 and consider citing it in your research.

@article{anand2025meanfieldsamplingforcooperative, title={Mean-Field Sampling for Cooperative Multi-Agent Reinforcement Learning}, author={Emile Anand, Ishani Karmarkar, Guannan Qu}, year={2025}, eprint={2412.00661}, archivePrefix={arXiv}, primaryClass={cs.ML}, url={https://arxiv.org/abs/2412.00661}}