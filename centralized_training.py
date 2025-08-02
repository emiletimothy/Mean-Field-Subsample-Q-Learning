"""Centralized Q-Learning for Multi-Agent Systems.

Main implementation of the centralized Q-learning algorithm with environment
management and training capabilities.
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
import pickle
import os
from base_classes import (
    State, Action, DiscreteState, DiscreteAction,
    GlobalAgent, LocalAgent,
    GlobalAgentTransition, LocalAgentTransition,
    GlobalAgentReward, LocalAgentReward
)
import itertools


class CentralizedEnvironment:
    """Environment managing global and local agents with custom transitions and rewards."""
    def __init__(self, 
                 global_agent: GlobalAgent,
                 local_agents: List[LocalAgent],
                 global_transition: GlobalAgentTransition,
                 local_transition: LocalAgentTransition,
                 global_reward: GlobalAgentReward,
                 local_reward: LocalAgentReward):
        self.global_agent = global_agent
        self.local_agents = local_agents
        self.n_agents = len(local_agents)
        self.global_transition = global_transition
        self.local_transition = local_transition
        self.global_reward = global_reward
        self.local_reward = local_reward
    
    def reset(self, initial_global_state: State, initial_local_states: List[State]):
        """Reset environment to initial states."""
        self.global_agent.reset(initial_global_state)
        for i, local_state in enumerate(initial_local_states):
            self.local_agents[i].reset(local_state)
    
    def step(self, global_action: Action, local_actions: List[Action]) -> Tuple[List[State], float]:
        """Execute one step in the environment."""
        # Get current states
        current_global_state = self.global_agent.get_state()
        current_local_states = [agent.get_state() for agent in self.local_agents]
        
        ###### something is weird here... getting reward before choosing actions?? should first choose actions then get rewards and finally transition states
        # Compute reward
        reward = self.compute_reward(current_global_state, global_action, 
                                   current_local_states, local_actions)
        
        # Transition to next states
        next_global_state = self.global_transition.sample_next_state(current_global_state, global_action)
        next_local_states = []
        
        # Local agent's state evolves as s_i' ~ P_l(.|s_i, s_g, a_i)
        for i, local_action in enumerate(local_actions):
            current_local_state = current_local_states[i]
            next_local_state = self.local_transition.sample_next_state(
                current_local_state, current_global_state, local_action)
            next_local_states.append(next_local_state)
        
        # Update agent states
        self.global_agent.set_state(next_global_state)
        for i, agent in enumerate(self.local_agents):
            agent.set_state(next_local_states[i])
        
        return [next_global_state] + next_local_states, reward
    
    def compute_reward(self, global_state: State, global_action: Action, 
                      local_states: List[State], local_actions: List[Action]) -> float:
        """Compute stage reward: r(s,a) = r_g(s_g, a_g) + 1/n * sum r_l(s_i, s_g, a_i)"""
        global_reward = self.global_reward.compute_reward(global_state, global_action)
        
        local_reward_sum = 0.0
        for i, (local_state, local_action) in enumerate(zip(local_states, local_actions)):
            local_reward_sum += self.local_reward.compute_reward(local_state, global_state, local_action)
        
        return global_reward + (1.0 / self.n_agents) * local_reward_sum
    
    def get_current_states(self) -> Tuple[State, List[State]]:
        """Get current states of all agents."""
        global_state = self.global_agent.get_state()
        local_states = [agent.get_state() for agent in self.local_agents]
        return global_state, local_states


class CentralizedQLearning:
    """Centralized Q-learning algorithm for multi-agent system."""
    def __init__(self, 
                environment: CentralizedEnvironment,
                learning_rate: float = 0.3,
                discount_factor: float = 0.9,
                epsilon: float = 0.1,
                epsilon_decay: float = 0.995,
                min_epsilon: float = 0.01):
        self.env = environment
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        
        # Q-table: Q(s, a) where s = (s_g, s_1, ..., s_n) and a = (a_g, a_1, ..., a_n)
        self.q_table = defaultdict(float)
        
        # Training metrics
        self.training_rewards = []
        self.training_times = []
        self.episodes_completed = 0
    
    def get_state_action_key(self, states: List[State], actions: List[Action]) -> Tuple:
        """Create hashable key for Q-table from states and actions."""
        return (tuple(states), tuple(actions))
    
    def get_all_actions(self) -> List[Tuple[Action, List[Action]]]:
        """Get all possible action combinations."""
        all_actions = []
        for global_action in self.env.global_agent.action_space:
            for local_action_combo in self._get_local_action_combinations():
                all_actions.append((global_action, local_action_combo))
        return all_actions
    
    def _get_local_action_combinations(self) -> List[List[Action]]:
        """Get all combinations of local actions."""
        action_spaces = [agent.action_space for agent in self.env.local_agents]
        return list(itertools.product(*action_spaces))
    
    def select_action(self, states: List[State]) -> Tuple[Action, List[Action]]:
        """Epsilon-greedy action selection."""
        if np.random.random() < self.epsilon:
            # Random action
            global_action = np.random.choice(self.env.global_agent.action_space)
            local_actions = [np.random.choice(agent.action_space) for agent in self.env.local_agents]
            return global_action, local_actions
        else:
            # Greedy action
            return self.get_best_action(states)
    
    def get_best_action(self, states: List[State]) -> Tuple[Action, List[Action]]:
        """Get action with highest Q-value."""
        best_value = float('-inf')
        best_action = None
        
        for global_action, local_actions in self.get_all_actions():
            actions = [global_action] + list(local_actions)
            state_action_key = self.get_state_action_key(states, actions)
            q_value = self.q_table[state_action_key]
            
            if q_value > best_value:
                best_value = q_value
                best_action = (global_action, list(local_actions))
        
        if best_action is None:
            # Fallback to random action
            global_action = np.random.choice(self.env.global_agent.action_space)
            local_actions = [np.random.choice(agent.action_space) for agent in self.env.local_agents]
            return global_action, local_actions
        
        return best_action
    
    def update_q_value(self, states: List[State], actions: List[Action], 
                      reward: float, next_states: List[State]):
        """Bellman update: Q(s,a) = r(s,a) + gamma * E[max Q(s',a')]"""
        current_key = self.get_state_action_key(states, actions)
        
        # Find max Q-value for next state
        max_next_q = float('-inf')
        for global_action, local_actions in self.get_all_actions():
            next_actions = [global_action] + list(local_actions)
            next_key = self.get_state_action_key(next_states, next_actions)
            next_q = self.q_table[next_key]
            max_next_q = max(max_next_q, next_q)
        
        if max_next_q == float('-inf'):
            max_next_q = 0.0
        
        # Bellman update
        current_q = self.q_table[current_key]
        updated_q = current_q + self.lr * (reward + self.gamma * max_next_q - current_q)
        self.q_table[current_key] = updated_q
    
    def train_episode(self, initial_global_state: State, initial_local_states: List[State], 
                     max_steps: int = 100) -> float:
        """Train for one episode."""
        episode_start_time = time.time()
        
        # Reset environment
        self.env.reset(initial_global_state, initial_local_states)
        
        total_reward = 0.0
        for step in range(max_steps):
            # Get current states
            global_state, local_states = self.env.get_current_states()
            current_states = [global_state] + local_states
            
            # Select actions
            global_action, local_actions = self.select_action(current_states)
            current_actions = [global_action] + local_actions
            
            # Take step
            next_states, reward = self.env.step(global_action, local_actions)
            total_reward += reward
            
            # Update Q-value
            self.update_q_value(current_states, current_actions, reward, next_states)
        
        # Decay epsilon
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
        
        # Record metrics
        episode_time = time.time() - episode_start_time
        self.training_rewards.append(total_reward)
        self.training_times.append(episode_time)
        self.episodes_completed += 1
        
        return total_reward
    
    def train(self, initial_global_state: State, initial_local_states: List[State],
             num_episodes: int = 1000, max_steps_per_episode: int = 100) -> Dict[str, List[float]]:
        """Train the Q-learning algorithm."""
        print(f"Starting training for {num_episodes} episodes...")
        
        for episode in range(num_episodes):
            reward = self.train_episode(initial_global_state, initial_local_states, max_steps_per_episode)
            
            if (episode + 1) % 100 == 0:
                avg_reward = np.mean(self.training_rewards[-100:])
                print(f"Episode {episode + 1}/{num_episodes}, "
                      f"Average Reward (last 100): {avg_reward:.2f}, "
                      f"Epsilon: {self.epsilon:.3f}")
        
        print("Training completed!")
        return {
            'rewards': self.training_rewards.copy(),
            'times': self.training_times.copy()
        }
    
    def plot_training_curves(self, save_path: Optional[str] = None):
        """Plot training curves and computation time."""
        if not self.training_rewards:
            print("No training data to plot.")
            return
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Plot rewards
        episodes = range(1, len(self.training_rewards) + 1)
        ax1.plot(episodes, self.training_rewards, alpha=0.6, label='Episode Reward')
        
        # Plot moving average
        window = min(100, len(self.training_rewards) // 10)
        if window > 1:
            moving_avg = [np.mean(self.training_rewards[max(0, i-window):i+1]) 
                         for i in range(len(self.training_rewards))]
            ax1.plot(episodes, moving_avg, 'r-', linewidth=2, label=f'Moving Average ({window})')
        
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Total Reward')
        ax1.set_title('Training Rewards Over Time')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot computation time
        ax2.plot(episodes, self.training_times, 'g-', alpha=0.6, label='Episode Time')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Time (seconds)')
        ax2.set_title('Computation Time per Episode')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        
        plt.show()
    
    def save_model(self, filepath: str):
        """Save the trained Q-table and training metrics."""
        model_data = {
            'q_table': dict(self.q_table),
            'training_rewards': self.training_rewards,
            'training_times': self.training_times,
            'episodes_completed': self.episodes_completed,
            'parameters': {
                'learning_rate': self.lr,
                'discount_factor': self.gamma,
                'epsilon': self.epsilon,
                'epsilon_decay': self.epsilon_decay,
                'min_epsilon': self.min_epsilon
            }
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a trained Q-table and training metrics."""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.q_table = defaultdict(float, model_data['q_table'])
        self.training_rewards = model_data['training_rewards']
        self.training_times = model_data['training_times']
        self.episodes_completed = model_data['episodes_completed']
        
        # Load parameters
        params = model_data['parameters']
        self.lr = params['learning_rate']
        self.gamma = params['discount_factor']
        self.epsilon = params['epsilon']
        self.epsilon_decay = params['epsilon_decay']
        self.min_epsilon = params['min_epsilon']
        
        print(f"Model loaded from {filepath}")


def create_simple_example():
    """Create a simple example to demonstrate the algorithm."""
    # Create simple discrete state and action spaces
    global_agent_states = [DiscreteState(i) for i in range(5)]
    global_agent_actions = [DiscreteAction(i) for i in range(3)]
    local_agent_states = [DiscreteState(i) for i in range(3)]
    local_agent_actions = [DiscreteAction(i) for i in range(2)]
    
    # Create agents
    n = 8
    global_agent = GlobalAgent(global_agent_states, global_agent_actions)
    local_agents = [LocalAgent(i, local_agent_states, local_agent_actions) for i in range(n)]
    
    # Create simple transition functions
    global_agent_transitions = {}
    for gs in global_agent_states:
        for ga in global_agent_actions:
            # Simple random transitions
            next_state_probs = {s: 1.0/len(global_agent_states) for s in global_agent_states}
            global_agent_transitions[(gs, ga)] = next_state_probs
    
    local_agent_transitions = {}
    for ls in local_agent_states:
        for gs in global_agent_states:
            for la in local_agent_actions:
                # Simple random transitions
                next_state_probs = {s: 1.0/len(local_agent_states) for s in local_agent_states}
                local_agent_transitions[(ls, gs, la)] = next_state_probs
    
    global_agent_transition = GlobalAgentTransition(global_agent_transitions)
    local_agent_transition = LocalAgentTransition(local_agent_transitions)
    
    # Create simple reward functions
    def simple_global_agent_reward(global_state: State, global_action: Action) -> float:
        return 1.0 if global_action.value == 0 else -0.1
    
    def simple_local_agent_reward(local_state: State, global_state: State, local_action: Action) -> float:
        return 0.5 if local_action.value == 1 else -0.1
    
    global_agent_reward = GlobalAgentReward(simple_global_agent_reward)
    local_agent_reward = LocalAgentReward(simple_local_agent_reward)
    
    # Create environment
    env = CentralizedEnvironment(
        global_agent, local_agents, global_agent_transition, local_agent_transition,
        global_agent_reward, local_agent_reward
    )
    
    # Create Q-learning algorithm
    q_learner = CentralizedQLearning(env, learning_rate=0.3, discount_factor=0.9)
    
    return q_learner, global_agent_states[0], [local_agent_states[0], local_agent_states[0]]


if __name__ == "__main__":
    # Example usage
    print("Creating simple example...")
    q_learner, initial_global_state, initial_local_states = create_simple_example()
    
    print("Starting training...")
    training_results = q_learner.train(
        initial_global_state, initial_local_states,
        num_episodes=750, max_steps_per_episode=100
    )
    
    print("Plotting results...")
    q_learner.plot_training_curves("training_curves.png")
    
    print("Saving model...")
    q_learner.save_model("centralized_q_model.pkl")
    
    print("Done!")
