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

try:
    from numba import jit, njit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    # Fallback decorator
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator


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
        """Execute one step in the environment.
        Standard RL environment sequence:
        1. Agent observes current state (before calling this method)
        2. Agent chooses actions (passed as parameters to this method)
        3. Environment computes reward for current state-action pair
        4. Environment transitions to next state
        5. Environment returns next state and reward to agent
        """
        # Get current states
        current_global_state = self.global_agent.get_state()
        current_local_states = [agent.get_state() for agent in self.local_agents]
        
        # Step 3: Compute reward for current state-action pair
        # r(s,a) = r_g(s_g, a_g) + (1/n) * sum r_l(s_i, s_g, a_i)
        reward = self.compute_reward(current_global_state, global_action, 
                                   current_local_states, local_actions)
        
        # Step 4: Transition to next states
        # Global agent: s_g' ~ P_g(.|s_g, a_g)
        next_global_state = self.global_transition.sample_next_state(current_global_state, global_action)
        
        # Local agents: s_i' ~ P_l(.|s_i, s_g, a_i)
        next_local_states = []
        for i, local_action in enumerate(local_actions):
            current_local_state = current_local_states[i]
            next_local_state = self.local_transition.sample_next_state(
                current_local_state, current_global_state, local_action)
            next_local_states.append(next_local_state)
        
        # Update internal agent states for next iteration
        self.global_agent.set_state(next_global_state)
        for i, agent in enumerate(self.local_agents):
            agent.set_state(next_local_states[i])
        
        # Step 5: Return next state and reward to agent
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
                discount_factor: float = 0.9):
        self.env = environment
        self.gamma = discount_factor
        
        # Always use Monte Carlo sampling for expectation approximation
        self.expectation_num_samples = 20
        
        # Create integer mappings for states and actions for fast NumPy indexing
        self._create_state_action_mappings()
        
        # Q-table as NumPy array: Q[joint_state_idx, joint_action_idx]
        self.q_table = np.zeros((self.n_joint_states, self.n_joint_actions))
        
        # Training metrics
        self.training_rewards = []
        self.training_times = []
        self.episodes_completed = 0
    
    def _create_state_action_mappings(self):
        """Create integer mappings for states and actions for fast NumPy indexing."""
        # Global state/action mappings
        self.global_state_to_idx = {state: i for i, state in enumerate(self.env.global_agent.state_space)}
        self.global_action_to_idx = {action: i for i, action in enumerate(self.env.global_agent.action_space)}
        
        # Local state/action mappings (same for all agents)
        self.local_state_to_idx = {state: i for i, state in enumerate(self.env.local_agents[0].state_space)}
        self.local_action_to_idx = {action: i for i, action in enumerate(self.env.local_agents[0].action_space)}
        
        # Dimensions
        self.n_global_states = len(self.env.global_agent.state_space)
        self.n_global_actions = len(self.env.global_agent.action_space)
        self.n_local_states = len(self.env.local_agents[0].state_space)
        self.n_local_actions = len(self.env.local_agents[0].action_space)
        self.n_agents = len(self.env.local_agents)
        
        # Joint space dimensions
        self.n_joint_states = self.n_global_states * (self.n_local_states ** self.n_agents)
        self.n_joint_actions = self.n_global_actions * (self.n_local_actions ** self.n_agents)
        
        # Precompute all valid joint actions for vectorization
        self._precompute_joint_actions()
        
        # Precompute transition matrices for faster lookups
        self._precompute_transition_matrices()
    
    def _precompute_joint_actions(self):
        """Precompute all joint actions as integer arrays."""
        joint_actions = []
        for global_action in self.env.global_agent.action_space:
            for local_action_combo in itertools.product(*[agent.action_space for agent in self.env.local_agents]):
                g_idx = self.global_action_to_idx[global_action]
                l_idxs = [self.local_action_to_idx[la] for la in local_action_combo]
                joint_actions.append([g_idx] + l_idxs)
        
        self.joint_actions_array = np.array(joint_actions)  # Shape: (n_joint_actions, n_agents+1)
    
    def _precompute_transition_matrices(self):
        """Precompute transition matrices for faster lookups."""
        # Global transition matrix: [global_state, global_action] -> probability distribution
        self.global_transition_matrix = np.zeros((self.n_global_states, self.n_global_actions, self.n_global_states))
        
        for (g_state, g_action), dist in self.env.global_transition.transition_probs.items():
            g_state_idx = self.global_state_to_idx[g_state]
            g_action_idx = self.global_action_to_idx[g_action]
            for next_state, prob in dist.items():
                next_state_idx = self.global_state_to_idx[next_state]
                self.global_transition_matrix[g_state_idx, g_action_idx, next_state_idx] = prob
        
        # Local transition matrix: [local_state, global_state, local_action] -> probability distribution
        self.local_transition_matrix = np.zeros((self.n_local_states, self.n_global_states, 
                                               self.n_local_actions, self.n_local_states))
        
        for (l_state, g_state, l_action), dist in self.env.local_transition.transition_probs.items():
            l_state_idx = self.local_state_to_idx[l_state]
            g_state_idx = self.global_state_to_idx[g_state]
            l_action_idx = self.local_action_to_idx[l_action]
            for next_state, prob in dist.items():
                next_l_state_idx = self.local_state_to_idx[next_state]
                self.local_transition_matrix[l_state_idx, g_state_idx, l_action_idx, next_l_state_idx] = prob
    
    def get_state_idx(self, states: List[State]) -> int:
        """Convert joint state to integer index."""
        global_idx = self.global_state_to_idx[states[0]]
        local_idxs = [self.local_state_to_idx[state] for state in states[1:]]
        
        # Joint state index calculation
        joint_idx = global_idx
        multiplier = self.n_global_states
        for local_idx in local_idxs:
            joint_idx += local_idx * multiplier
            multiplier *= self.n_local_states
        return joint_idx
    
    def get_action_idx(self, actions: List[Action]) -> int:
        """Convert joint action to integer index."""
        global_idx = self.global_action_to_idx[actions[0]]
        local_idxs = [self.local_action_to_idx[action] for action in actions[1:]]
        
        # Joint action index calculation
        joint_idx = global_idx
        multiplier = self.n_global_actions
        for local_idx in local_idxs:
            joint_idx += local_idx * multiplier
            multiplier *= self.n_local_actions
        return joint_idx
    
    def get_all_actions(self) -> List[Tuple[Action, List[Action]]]:
        """Get all possible action combinations (for backward compatibility)."""
        all_actions = []
        for global_action in self.env.global_agent.action_space:
            for local_action_combo in itertools.product(*[agent.action_space for agent in self.env.local_agents]):
                all_actions.append((global_action, list(local_action_combo)))
        return all_actions
    
    def select_action(self, states: List[State]) -> Tuple[Action, List[Action]]:
        """Greedy action selection (no exploration) - vectorized."""
        state_idx = self.get_state_idx(states)
        
        # Vectorized Q-value lookup for all actions
        q_values = self.q_table[state_idx, :]
        best_action_idx = np.argmax(q_values)
        
        # Convert back to action objects
        joint_action_idxs = self.joint_actions_array[best_action_idx]
        global_action = self.env.global_agent.action_space[joint_action_idxs[0]]
        local_actions = [self.env.local_agents[i].action_space[joint_action_idxs[i+1]] 
                        for i in range(self.n_agents)]
        
        return global_action, local_actions
    
    def get_best_action(self, states: List[State]) -> Tuple[Action, List[Action]]:
        """Get action with highest Q-value (alias for select_action)."""
        return self.select_action(states)
    
    def update_q_value(self, states: List[State], actions: List[Action], 
                      reward: float, next_states: List[State]):
        """Bellman update: Q(s,a) = r(s,a) + gamma * E[max Q(s',a')]"""
        # Unpack current state and action
        global_state = states[0]
        local_states = states[1:]
        global_action = actions[0]
        local_actions = actions[1:]

        # Get transition probabilities using precomputed matrices
        global_state_idx = self.global_state_to_idx[global_state]
        global_action_idx = self.global_action_to_idx[global_action]
        local_state_idxs = [self.local_state_to_idx[ls] for ls in local_states]
        local_action_idxs = [self.local_action_to_idx[la] for la in local_actions]
        
        # Check if transitions exist (non-zero probability mass)
        g_transition_probs = self.global_transition_matrix[global_state_idx, global_action_idx, :]
        missing = np.sum(g_transition_probs) == 0
        
        if not missing:
            for i, (ls_idx, la_idx) in enumerate(zip(local_state_idxs, local_action_idxs)):
                l_transition_probs = self.local_transition_matrix[ls_idx, global_state_idx, la_idx, :]
                if np.sum(l_transition_probs) == 0:
                    missing = True
                    break
        
        if missing:
            # Fallback: use the observed next_states to compute a sample-based backup
            next_state_idx = self.get_state_idx(next_states)
            max_q = np.max(self.q_table[next_state_idx, :])
            expected_max_next_q = max_q
            
            # Bellman update with sample-based backup
            current_state_idx = self.get_state_idx(states)
            current_action_idx = self.get_action_idx(actions)
            self.q_table[current_state_idx, current_action_idx] = reward + self.gamma * expected_max_next_q
            return

        # Fully vectorized Monte Carlo approximation
        # Sample global next states (batch)
        valid_g_states = np.where(g_transition_probs > 0)[0]
        valid_g_probs = g_transition_probs[valid_g_states]
        valid_g_probs /= np.sum(valid_g_probs)  # Normalize
        
        sampled_g_idxs = np.random.choice(valid_g_states, size=self.expectation_num_samples, p=valid_g_probs)
        
        # Sample local next states (batch)
        sampled_local_idxs_batch = np.zeros((self.expectation_num_samples, self.n_agents), dtype=int)
        for agent_idx, (ls_idx, la_idx) in enumerate(zip(local_state_idxs, local_action_idxs)):
            l_transition_probs = self.local_transition_matrix[ls_idx, global_state_idx, la_idx, :]
            valid_l_states = np.where(l_transition_probs > 0)[0]
            valid_l_probs = l_transition_probs[valid_l_states]
            valid_l_probs /= np.sum(valid_l_probs)  # Normalize
            
            sampled_local_idxs_batch[:, agent_idx] = np.random.choice(
                valid_l_states, size=self.expectation_num_samples, p=valid_l_probs
            )
        
        # Vectorized joint state index calculation
        joint_state_idxs = sampled_g_idxs.copy()
        multiplier = self.n_global_states
        for agent_idx in range(self.n_agents):
            joint_state_idxs += sampled_local_idxs_batch[:, agent_idx] * multiplier
            multiplier *= self.n_local_states
        
        # Vectorized max Q-value computation
        max_q_samples = np.max(self.q_table[joint_state_idxs, :], axis=1)
        expected_max_next_q = np.mean(max_q_samples)
        
        # Bellman update with expected backup
        current_state_idx = self.get_state_idx(states)
        current_action_idx = self.get_action_idx(actions)
        self.q_table[current_state_idx, current_action_idx] = reward + self.gamma * expected_max_next_q
    


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
                      f"Average Reward (last 100): {avg_reward:.2f}")
        
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
            'q_table': self.q_table,  # NumPy array
            'training_rewards': self.training_rewards,
            'training_times': self.training_times,
            'episodes_completed': self.episodes_completed,
            'parameters': {
                'discount_factor': self.gamma,
                'expectation_num_samples': self.expectation_num_samples
            },
            # Store mappings for reconstruction
            'state_action_mappings': {
                'n_global_states': self.n_global_states,
                'n_global_actions': self.n_global_actions,
                'n_local_states': self.n_local_states,
                'n_local_actions': self.n_local_actions,
                'n_agents': self.n_agents
            }
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a trained Q-table and training metrics."""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        # Handle both old (dict) and new (NumPy) Q-table formats
        if isinstance(model_data['q_table'], np.ndarray):
            self.q_table = model_data['q_table']
        else:
            # Convert old dict format to NumPy array
            old_q_table = model_data['q_table']
            self.q_table = np.zeros((self.n_joint_states, self.n_joint_actions))
            for (states, actions), value in old_q_table.items():
                state_idx = self.get_state_idx(list(states))
                action_idx = self.get_action_idx(list(actions))
                self.q_table[state_idx, action_idx] = value
        
        self.training_rewards = model_data['training_rewards']
        self.training_times = model_data['training_times']
        self.episodes_completed = model_data['episodes_completed']
        
        # Load parameters (backward compatible)
        params = model_data.get('parameters', {})
        self.gamma = params.get('discount_factor', self.gamma)
        self.expectation_num_samples = params.get(
            'expectation_num_samples', getattr(self, 'expectation_num_samples', 20)
        )
        
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
    q_learner = CentralizedQLearning(env, discount_factor=0.9)
    
    # Initialize ALL local agents with a valid starting state
    initial_local_states = [local_agent_states[0] for _ in range(n)]
    return q_learner, global_agent_states[0], initial_local_states


if __name__ == "__main__":
    # Example usage
    print("Creating simple example...")
    q_learner, initial_global_state, initial_local_states = create_simple_example()
    
    print("Starting training...")
    training_results = q_learner.train(
        initial_global_state, initial_local_states,
        num_episodes=300, max_steps_per_episode=100
    )
    
    print("Plotting results...")
    q_learner.plot_training_curves("training_curves.png")
    
    print("Saving model...")
    q_learner.save_model("centralized_q_model.pkl")
    
    print("Done!")
