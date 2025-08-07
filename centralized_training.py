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
        self.Q = {}

        # Initialize Q-function for joint state-action space
        # Joint state: (global_state, local_state_1, local_state_2, ..., local_state_n)
        # Joint action: (global_action, local_action_1, local_action_2, ..., local_action_n)
        for global_state in self.global_agent_state_space:
            for local_states in itertools.product(self.local_agent_state_space, repeat=self.n):
                for global_action in self.global_agent_action_space:
                    for local_actions in itertools.product(self.local_agent_action_space, repeat=self.n):
                        # Key format: (global_state, (local_states...), global_action, (local_actions...))
                        joint_state = (global_state,) + local_states
                        joint_action = (global_action,) + local_actions
                        self.Q[(joint_state, joint_action)] = 0.0

    def get_all_joint_actions(self):
        """Generate all possible joint actions for the multi-agent system."""
        joint_actions = []
        for global_action in self.global_agent_action_space:
            for local_actions in itertools.product(self.local_agent_action_space, repeat=self.n):
                joint_action = (global_action,) + local_actions
                joint_actions.append(joint_action)
        return joint_actions

    def reward_function(self, global_agent_state, global_agent_action, local_agent_states, local_agent_actions):
        local_reward_sum = 0
        for i in range(self.n):
            local_reward_sum += self.local_agent_reward_function(local_agent_states[i], global_agent_state, local_agent_actions[i])
        return self.global_agent_reward_function(global_agent_state, global_agent_action) + (1/self.n) * local_reward_sum

    def update_Q(self):
        """Update Q-function using Bellman equation with Monte Carlo expectation."""
        for global_agent_state in self.global_agent_state_space:
            for local_agent_states in itertools.product(self.local_agent_state_space, repeat=self.n):
                for global_agent_action in self.global_agent_action_space:
                    for local_agent_actions in itertools.product(self.local_agent_action_space, repeat=self.n):
                        joint_state = (global_agent_state,) + local_agent_states
                        joint_action = (global_agent_action,) + local_agent_actions
                        r = self.reward_function(global_agent_state, global_agent_action, local_agent_states, local_agent_actions)
                        
                        # Monte Carlo estimate of E[max Q(s',a')]
                        mc_sum = 0.0
                        for iter in range(self.samples):
                            next_global_state_sample = self.global_agent_transition_function(global_agent_state, global_agent_action)
                            next_local_states_sample = []
                            for i in range(self.n):
                                next_local_state_sample = self.local_agent_transition_function(local_agent_states[i], global_agent_state, local_agent_actions[i])
                                next_local_states_sample.append(next_local_state_sample)
                            next_joint_state_sample = (next_global_state_sample,) + tuple(next_local_states_sample)
                            max_q_next = max(self.Q[(next_joint_state_sample, next_joint_action)] 
                                           for next_joint_action in self.get_all_joint_actions())
                            mc_sum += max_q_next
                        expected_max_q = mc_sum / self.samples
                        self.Q[(joint_state, joint_action)] = r + self.gamma * expected_max_q
                        

    def Q_learning(self, steps):
        for step in range(steps):
            self.update_Q()



