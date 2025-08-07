# Centralized Q-Learning for Multi-Agent Systems

This repository implements a modular and extensible centralized Q-learning algorithm for multi-agent reinforcement learning systems. The implementation supports both global and local agents with customizable state spaces, transition functions, and reward structures.

## Overview

The algorithm implements a centralized Q-function that considers:
- **Global Agent**: State `s_g ∈ S_g`, Action `a_g ∈ A_g`
- **Local Agents**: States `s_1, s_2, ..., s_n ∈ S_l^n`, Actions `a_1, a_2, ..., a_n ∈ A_l^n`

### Key Features

- **Modular Design**: Easy to extend with custom states, actions, transitions, and rewards
- **Centralized Q-Learning**: Expected Bellman backup `E_{s'}[max_{a'} Q(s',a')]` using the transition model (exact enumeration when feasible; Monte Carlo sampling otherwise)
- **Training Metrics**: Tracks rewards and computation time during training
- **Visualization**: Built-in plotting for training curves and performance analysis
- **Model Persistence**: Save and load trained models
- **Customizable Transitions**: 
  - Global: `s_g' ~ P_g(.|s_g, a_g)`
  - Local: `s_i' ~ P_l(.|s_g, a_g, a_i)`
- **Composite Rewards**: `r(s,a) = r_g(s_g, a_g) + (1/n) * Σ r_l(s_i, s_g, a_i)`

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd Mean-Field-Subsample-Q-Learning
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

Run the example implementation:

```bash
python centralized_training.py
```

This will:
1. Create a simple multi-agent environment
2. Train the centralized Q-learning algorithm
3. Generate training curves
4. Save the trained model

## Architecture

### Core Components

#### Abstract Base Classes
- `State`: Base class for state representations
- `Action`: Base class for action representations
- `TransitionFunction`: Base class for transition dynamics
- `RewardFunction`: Base class for reward functions

#### Agent Classes
- `GlobalAgent`: Manages global agent state and actions
- `LocalAgent`: Manages individual local agent states and actions

#### Environment
- `CentralizedEnvironment`: Orchestrates multi-agent interactions
- `GlobalAgentTransition`: Implements global state transitions
- `LocalAgentTransition`: Implements local state transitions
- `GlobalAgentReward`: Computes global rewards
- `LocalAgentReward`: Computes local rewards

#### Learning Algorithm
- `CentralizedQLearning`: Main Q-learning implementation with Bellman updates

### Example Usage

```python
import numpy as np
from centralized_training import *

# Create state and action spaces
global_states = [DiscreteState(i) for i in range(5)]
global_actions = [DiscreteAction(i) for i in range(3)]
local_states = [DiscreteState(i) for i in range(3)]
local_actions = [DiscreteAction(i) for i in range(2)]

# Create agents
global_agent = GlobalAgent(global_states, global_actions)
local_agents = [LocalAgent(i, local_states, local_actions) for i in range(2)]

# Define custom reward functions
def custom_global_reward(global_state, global_action):
    return 1.0 if global_action.value == 0 else -0.1

def custom_local_reward(local_state, global_state, local_action):
    return 0.5 if local_action.value == 1 else -0.1

# Create environment with custom transitions and rewards
env = CentralizedEnvironment(
    global_agent, local_agents, 
    global_agent_transition, local_agent_transition,
    GlobalAgentReward(custom_global_reward),
    LocalAgentReward(custom_local_reward)
)

# Initialize Q-learning
q_learner = CentralizedQLearning(
    env, 
    learning_rate=0.1,
    discount_factor=0.9
)

# Train the model
results = q_learner.train(
    initial_global_state, initial_local_states,
    num_episodes=1000, max_steps_per_episode=100
)

# Visualize training progress
q_learner.plot_training_curves("my_training_curves.png")

# Save the trained model
q_learner.save_model("my_model.pkl")
```

## Customization

### Custom States

Extend the `State` base class:

```python
class CustomState(State):
    def __init__(self, features):
        self.features = features
    
    def __hash__(self):
        return hash(tuple(self.features))
    
    def __eq__(self, other):
        return isinstance(other, CustomState) and self.features == other.features
    
    def __repr__(self):
        return f"CustomState({self.features})"
```

### Custom Transitions

Define transition probabilities:

```python
# For global transitions: P_g(s'|s,a)
global_transitions = {
    (state, action): {next_state: probability, ...}
    for state in states for action in actions
}

# For local transitions: P_l(s'|s_g,a_g,a_i)
local_transitions = {
    (global_state, global_action, local_action): {next_state: probability, ...}
    for global_state, global_action, local_action in combinations
}
```

### Custom Rewards

Implement custom reward logic:

```python
def my_global_reward(global_state, global_action):
    # Your reward logic here
    return reward_value

def my_local_reward(local_state, global_state, local_action):
    # Your reward logic here
    return reward_value
```

## Algorithm Details

### Bellman Update

The Q-function is updated using:

```
Q(s,a) = r(s,a) + γ * E_{s'} [max_{a'} Q(s',a')]
```

Where:
- `s = (s_g, s_1, ..., s_n)` is the joint state
- `a = (a_g, a_1, ..., a_n)` is the joint action
- `r(s,a) = r_g(s_g, a_g) + (1/n) * Σ r_l(s_i, s_g, a_i)` is the composite reward
- `γ` is the discount factor

Expectation computation uses the known transition model. If the joint next-state space is small enough, we enumerate all `s'` exactly. Otherwise, we approximate with Monte Carlo samples. You can control this with:
- `expectation_enumeration_limit` (default: 5000)
- `expectation_num_samples` (default: 64)

### Training Process

1. **Initialization**: Reset environment to initial states
2. **Action Selection**: Greedy policy (no exploration)
3. **Environment Step**: Execute actions and observe rewards/next states
4. **Q-Update**: Apply Bellman update equation
5. **Metrics**: Track episode rewards and computation time
6. **Repeat**: Continue for specified number of episodes

## Outputs

The implementation generates:

- **Training Curves**: Episode rewards over time with moving averages
- **Computation Time**: Time per episode analysis
- **Trained Model**: Saved Q-table and hyperparameters
- **Console Logs**: Progress updates during training

## Performance Monitoring

The algorithm tracks:
- Episode rewards
- Computation time per episode
- Moving averages for performance assessment

## Contributing

This implementation is designed to be modular and extensible. To contribute:

1. Follow the abstract base class patterns for new components
2. Ensure all custom classes implement required abstract methods
3. Add comprehensive docstrings
4. Include example usage in documentation

## License

MIT License - see LICENSE file for details.

## References

Based on concepts from:
- Multi-Agent Reinforcement Learning
- Centralized Training with Decentralized Execution
- Q-Learning and Temporal Difference Methods
