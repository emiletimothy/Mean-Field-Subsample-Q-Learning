import numpy as np
import matplotlib.pyplot as plt
import time
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
import pickle
import itertools
from itertools import combinations_with_replacement

class Q_function:
    def __init__(self, n, state_spaces, action_spaces, samples, transition_system, reward_system, gamma):
        self.n = n
        self.local_agent_state_space = state_spaces[0]
        self.local_agent_action_space = action_spaces[0]
        self.global_agent_state_space = state_spaces[1]
        self.global_agent_action_space = action_spaces[1]
        self.samples = samples
        self.global_agent_transition_function = transition_system[0]
        self.local_agent_transition_function = transition_system[1]
        self.global_agent_reward_function = reward_system[0]
        self.local_agent_reward_function = reward_system[1]
        self.gamma = gamma
        self._create_index_mappings()
        self.Q = np.zeros((self.n_joint_states, self.n_joint_actions), dtype=np.float32)
        self.joint_actions_array = np.array(self.mean_field_actions, dtype=object)
        self._precompute_rewards()
        self._transition_cache = {}

    def _create_index_mappings(self):
        """Create mean-field state/action mappings using distribution counts."""
        self.mean_field_states = []
        self.state_to_idx = {}
        idx = 0
        
        for global_state in self.global_agent_state_space:
            for distribution in self._generate_distributions(self.n, len(self.local_agent_state_space)):
                mean_field_state = (global_state, tuple(distribution))
                self.mean_field_states.append(mean_field_state)
                self.state_to_idx[mean_field_state] = idx
                idx += 1
        
        self.n_joint_states = len(self.mean_field_states)
        self.mean_field_actions = []
        self.action_to_idx = {}
        idx = 0
        
        for global_action in self.global_agent_action_space:
            for action_distribution in self._generate_distributions(self.n, len(self.local_agent_action_space)):
                mean_field_action = (global_action, tuple(action_distribution))
                self.mean_field_actions.append(mean_field_action)
                self.action_to_idx[mean_field_action] = idx
                idx += 1
        
        self.n_joint_actions = len(self.mean_field_actions)
        print(f"Mean-field Q-table: {self.n_joint_states} states x {self.n_joint_actions} actions")
    
    def _generate_distributions(self, n_agents, n_states):
        """Generate all possible ways to distribute n_agents across n_states."""        
        distributions = []
        for combo in combinations_with_replacement(range(n_states), n_agents):
            distribution = [0] * n_states
            for state_idx in combo:
                distribution[state_idx] += 1
            distributions.append(distribution)
        
        unique_distributions = []
        for dist in distributions:
            if dist not in unique_distributions:
                unique_distributions.append(dist)
        
        return unique_distributions
    
    def get_state_idx(self, joint_state):
        """Convert joint state tuple to flat index for numpy array access."""
        return self.state_to_idx[joint_state]
    
    def get_action_idx(self, joint_action):
        """Convert joint action tuple to flat index for numpy array access."""
        return self.action_to_idx[joint_action]
    
    def get_all_joint_actions(self):
        """Generate all possible mean-field actions."""
        return self.mean_field_actions
    
    def _joint_state_to_mean_field(self, joint_state):
        """Convert joint state to mean-field representation."""
        global_state = joint_state[0]
        local_states = joint_state[1:]
        
        distribution = [0] * len(self.local_agent_state_space)
        for local_state in local_states:
            state_idx = self.local_agent_state_space.index(local_state)
            distribution[state_idx] += 1
        
        return (global_state, tuple(distribution))
    
    def _joint_action_to_mean_field(self, joint_action):
        """Convert joint action to mean-field representation."""
        global_action = joint_action[0]
        local_actions = joint_action[1:]
        
        distribution = [0] * len(self.local_agent_action_space)
        for local_action in local_actions:
            action_idx = self.local_agent_action_space.index(local_action)
            distribution[action_idx] += 1
        
        return (global_action, tuple(distribution))
    
    def _find_nearest_mean_field_state(self, target_state):
        """Find the nearest valid mean-field state to the target state."""
        target_global_state = target_state[0]
        target_local_dist = target_state[1]
        
        best_distance = float('inf')
        nearest_state = None
        
        for mean_field_state in self.mean_field_states:
            if mean_field_state[0] == target_global_state:
                local_dist = mean_field_state[1]
                
                distance = sum(abs(a - b) for a, b in zip(target_local_dist, local_dist))
                distance = sum(abs(a - b) for a, b in zip(target_local_dist, local_dist))
                
                if distance < best_distance:
                    best_distance = distance
                    nearest_state = mean_field_state
        
        return nearest_state

    def reward_function(self, global_agent_state, global_agent_action, local_agent_states, local_agent_actions):
        local_reward_sum = 0
        for i in range(self.n):
            local_reward_sum += self.local_agent_reward_function(local_agent_states[i], global_agent_state, local_agent_actions[i])
        return self.global_agent_reward_function(global_agent_state, global_agent_action) + (1/self.n) * local_reward_sum

    def update_Q(self):
        """Optimized Q-function update with aggressive vectorization."""
        start_time = time.time()
        Q_old = self.Q.copy()
        
        rewards = self.reward_matrix
        
        expected_max_q_values = self._fast_monte_carlo_sampling(Q_old)
        
        self.Q = rewards + self.gamma * expected_max_q_values
        
        elapsed_time = time.time() - start_time
        total_updates = self.n_joint_states * self.n_joint_actions
        print(f"Completed optimized Q-update in {elapsed_time:.3f}s ({total_updates} updates)")
    
    def _precompute_rewards(self):
        """Pre-compute all reward values for mean-field states/actions."""
        print("Pre-computing mean-field reward matrix...")
        self.reward_matrix = np.zeros((self.n_joint_states, self.n_joint_actions), dtype=np.float32)
        
        for state_idx, mean_field_state in enumerate(self.mean_field_states):
            global_state = mean_field_state[0]
            local_distribution = mean_field_state[1]
            
            for action_idx, mean_field_action in enumerate(self.mean_field_actions):
                global_action = mean_field_action[0]
                action_distribution = mean_field_action[1]
                
                self.reward_matrix[state_idx, action_idx] = self._mean_field_reward(
                    global_state, global_action, local_distribution, action_distribution
                )
        print("Mean-field reward matrix pre-computed.")
    
    def _mean_field_reward(self, global_state, global_action, local_state_dist, local_action_dist):
        """Compute expected reward for mean-field state/action distributions."""
        global_reward = self.global_agent_reward_function(global_state, global_action)  
        local_reward_sum = 0
        for state_idx, state_count in enumerate(local_state_dist):
            if state_count > 0:
                local_state = self.local_agent_state_space[state_idx]
                for action_idx, action_count in enumerate(local_action_dist):
                    if action_count > 0:
                        local_action = self.local_agent_action_space[action_idx]
                        prob = (state_count / self.n) * (action_count / self.n)
                        local_reward_sum += prob * self.local_agent_reward_function(
                            local_state, global_state, local_action
                        )
        
        return global_reward + local_reward_sum
    
    def _fast_monte_carlo_sampling(self, Q_old):
        """Mean-field Monte Carlo sampling for expected Q-values."""
        expected_values = np.zeros((self.n_joint_states, self.n_joint_actions), dtype=np.float32)
        
        for state_idx, mean_field_state in enumerate(self.mean_field_states):
            global_state = mean_field_state[0]
            local_state_dist = mean_field_state[1]
            
            for action_idx, mean_field_action in enumerate(self.mean_field_actions):
                global_action = mean_field_action[0]
                local_action_dist = mean_field_action[1]
                
                cache_key = (state_idx, action_idx)
                if cache_key in self._transition_cache:
                    next_state_indices = self._transition_cache[cache_key]
                else:
                    next_state_indices = []
                    for _ in range(self.samples):
                        next_global_state = self.global_agent_transition_function(global_state, global_action)
                        next_local_dist = self._sample_next_local_distribution(
                            local_state_dist, local_action_dist, global_state
                        )
                        
                        next_mean_field_state = (next_global_state, tuple(next_local_dist))
                        if next_mean_field_state in self.state_to_idx:
                            next_state_indices.append(self.get_state_idx(next_mean_field_state))
                    
                    self._transition_cache[cache_key] = next_state_indices
                
                if next_state_indices:
                    max_q_values = [np.max(Q_old[idx, :]) for idx in next_state_indices]
                    expected_values[state_idx, action_idx] = np.mean(max_q_values)
        
        return expected_values
    
    def _sample_next_local_distribution(self, current_dist, action_dist, global_state):
        """Sample next local agent distribution based on transitions."""
        next_dist = [0] * len(self.local_agent_state_space)
        
        for curr_state_idx, curr_count in enumerate(current_dist):
            if curr_count > 0:
                curr_state = self.local_agent_state_space[curr_state_idx]
                for action_idx, action_count in enumerate(action_dist):
                    if action_count > 0:
                        action = self.local_agent_action_space[action_idx]
                        agents_in_transition = min(curr_count, action_count)
                        next_state = self.local_agent_transition_function(curr_state, global_state, action)
                        next_state_idx = self.local_agent_state_space.index(next_state)
                        next_dist[next_state_idx] += agents_in_transition
        
        total = sum(next_dist)
        if total != self.n and total > 0:
            scale = self.n / total
            next_dist = [int(count * scale) for count in next_dist]
            diff = self.n - sum(next_dist)
            if diff > 0:
                next_dist[0] += diff
        
        return next_dist

    def learn(self, steps):
        print("Starting vectorized Q-learning update...")
        for step in range(steps):
            self.update_Q()
        print("Completed Q-learning over {} steps".format(steps))

    def save_model(self, filepath):
        model_data = {
            'Q': self.Q,
            'state_to_idx': self.state_to_idx,
            'action_to_idx': self.action_to_idx,
            'joint_states': self.joint_states,
            'joint_actions': self.joint_actions,
            'n_joint_states': self.n_joint_states,
            'n_joint_actions': self.n_joint_actions,
            'gamma': self.gamma
        }
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"Model saved to {filepath}")

    def load_model(self, filepath):
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.Q = model_data['Q']
        self.state_to_idx = model_data['state_to_idx']
        self.action_to_idx = model_data['action_to_idx']
        self.joint_states = model_data['joint_states']
        self.joint_actions = model_data['joint_actions']
        self.n_joint_states = model_data['n_joint_states']
        self.n_joint_actions = model_data['n_joint_actions']
        self.gamma = model_data['gamma']
        
        print(f"Model loaded from {filepath}")
