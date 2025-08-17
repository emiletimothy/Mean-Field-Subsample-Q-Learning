import numpy as np
import matplotlib.pyplot as plt
import time
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
import pickle
import itertools

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
        self.joint_actions_array = np.array(self.joint_actions, dtype=np.int32)

    def _create_index_mappings(self):
        """Create mappings between joint states/actions and flat indices for numpy array access."""
        self.joint_states = []
        self.state_to_idx = {}
        idx = 0
        for global_state in self.global_agent_state_space:
            for local_states in itertools.product(self.local_agent_state_space, repeat=self.n):
                joint_state = (global_state,) + local_states
                self.joint_states.append(joint_state)
                self.state_to_idx[joint_state] = idx
                idx += 1
        self.n_joint_states = len(self.joint_states)
        self.joint_actions = []
        self.action_to_idx = {}
        idx = 0
        for global_action in self.global_agent_action_space:
            for local_actions in itertools.product(self.local_agent_action_space, repeat=self.n):
                joint_action = (global_action,) + local_actions
                self.joint_actions.append(joint_action)
                self.action_to_idx[joint_action] = idx
                idx += 1
        self.n_joint_actions = len(self.joint_actions)
        print(f"Initialized numpy Q-table: {self.n_joint_states} states x {self.n_joint_actions} actions")
    
    def get_state_idx(self, joint_state):
        """Convert joint state tuple to flat index for numpy array access."""
        return self.state_to_idx[joint_state]
    
    def get_action_idx(self, joint_action):
        """Convert joint action tuple to flat index for numpy array access."""
        return self.action_to_idx[joint_action]
    
    def get_all_joint_actions(self):
        """Generate all possible joint actions for the multi-agent system."""
        return self.joint_actions

    def reward_function(self, global_agent_state, global_agent_action, local_agent_states, local_agent_actions):
        local_reward_sum = 0
        for i in range(self.n):
            local_reward_sum += self.local_agent_reward_function(local_agent_states[i], global_agent_state, local_agent_actions[i])
        return self.global_agent_reward_function(global_agent_state, global_agent_action) + (1/self.n) * local_reward_sum

    def update_Q(self):
        """Update Q-function using Bellman equation with vectorized Monte Carlo expectation."""
        start_time = time.time()
        for global_agent_state in self.global_agent_state_space:
            for local_agent_states in itertools.product(self.local_agent_state_space, repeat=self.n):
                for global_agent_action in self.global_agent_action_space:
                    for local_agent_actions in itertools.product(self.local_agent_action_space, repeat=self.n):
                        joint_state = (global_agent_state,) + local_agent_states
                        joint_action = (global_agent_action,) + local_agent_actions
                        
                        state_idx = self.get_state_idx(joint_state)
                        action_idx = self.get_action_idx(joint_action)
                        
                        r = self.reward_function(global_agent_state, global_agent_action, local_agent_states, local_agent_actions)
                        
                        mc_samples = np.zeros(self.samples, dtype=np.float32)
                        
                        for iter in range(self.samples):
                            next_global_state_sample = self.global_agent_transition_function(global_agent_state, global_agent_action)
                            next_local_states_sample = []
                            for i in range(self.n):
                                next_local_state_sample = self.local_agent_transition_function(
                                    local_agent_states[i], global_agent_state, local_agent_actions[i]
                                )
                                next_local_states_sample.append(next_local_state_sample)
                            
                            next_joint_state_sample = (next_global_state_sample,) + tuple(next_local_states_sample)
                            next_state_idx = self.get_state_idx(next_joint_state_sample)
                            
                            max_q_next = np.max(self.Q[next_state_idx, :])
                            mc_samples[iter] = max_q_next
                        
                        expected_max_q = np.mean(mc_samples)
                        
                        self.Q[state_idx, action_idx] = r + self.gamma * expected_max_q
        
        elapsed_time = time.time() - start_time
        total_updates = self.n_joint_states * self.n_joint_actions
        print(f"Completed Q-value update in {elapsed_time:.3f} seconds")                        

    def learn(self, steps):
        print("Starting vectorized Q-learning update...")
        for step in range(steps):
            self.update_Q()

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
