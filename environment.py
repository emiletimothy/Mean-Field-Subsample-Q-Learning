import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from centralized_training import Q_function
import matplotlib.pyplot as plt
import json

# helper functions for deployment
def random_state_sample(state_tuple, k):
    return np.random.choice(state_tuple, k, replace=False)

# deployment environment
def deployment_environment(H, k, n, q_func):
    cumulative_rewards = []
    total_reward = 0
    # randomly sample initial states for each agent
    current_global_state = np.random.choice(Sg)
    current_joint_local_states = tuple(np.random.choice(Sl) for _ in range(n))

    for t in range(H):
        # Each agent samples k-1 other agents and finds the action that maximizes Q-value
        joint_actions = []
        
        for i in range(n):
            # Sample k-1 other local agent states for agent i
            other_agents_indices = [j for j in range(n) if j != i]
            if len(other_agents_indices) >= k-1:
                sampled_indices = np.random.choice(other_agents_indices, k-1, replace=False)
            else:
                sampled_indices = other_agents_indices
            
            sampled_other_states = [current_joint_local_states[j] for j in sampled_indices]
            
            # Create mean-field state representation for this agent's perspective
            # Count distribution including agent i and sampled others
            local_state_dist = [0] * len(Sl)
            # Add agent i's state
            agent_i_state_idx = Sl.index(current_joint_local_states[i])
            local_state_dist[agent_i_state_idx] += 1
            # Add sampled other states
            for other_state in sampled_other_states:
                other_state_idx = Sl.index(other_state)
                local_state_dist[other_state_idx] += 1
            
            # Keep the sampled distribution as-is (it already sums to k)
            # No scaling needed since Q-function was trained on k agents
            total_sampled = len(sampled_other_states) + 1  # This equals k
            # local_state_dist already represents the k-agent distribution
            
            mean_field_state = (current_global_state, tuple(local_state_dist))
            
            # Find action that maximizes Q-value for this mean-field state
            if mean_field_state in q_func.state_to_idx:
                state_idx = q_func.get_state_idx(mean_field_state)
                    
                # Find the best action by checking all possible mean-field actions
                best_q_value = float('-inf')
                best_local_action = None
                    
                for action_idx, mean_field_action in enumerate(q_func.mean_field_actions):
                    q_value = q_func.Q[state_idx, action_idx]
                    if q_value > best_q_value:
                        best_q_value = q_value
                        # Extract the most common local action from this mean-field action
                        local_action_dist = mean_field_action[1]
                        best_local_action = Al[np.argmax(local_action_dist)]
                    
                joint_actions.append(best_local_action if best_local_action is not None else np.random.choice(Al))
            else:
                # Find nearest valid mean-field state
                nearest_state = q_func._find_nearest_mean_field_state(mean_field_state)
                if nearest_state:
                    state_idx = q_func.get_state_idx(nearest_state)
                    best_q_value = float('-inf')
                    best_local_action = None
                    for action_idx, mean_field_action in enumerate(q_func.mean_field_actions):
                        q_value = q_func.Q[state_idx, action_idx]
                        if q_value > best_q_value:
                            best_q_value = q_value
                            local_action_dist = mean_field_action[1]
                            best_local_action = Al[np.argmax(local_action_dist)]
                    joint_actions.append(best_local_action)
        
        # Global agent uses a k-sample of the population for decision making
        sampled_global_indices = np.random.choice(range(n), k, replace=False)
        sampled_global_states = [current_joint_local_states[j] for j in sampled_global_indices]
        
        global_state_dist = [0] * len(Sl)
        for local_state in sampled_global_states:
            state_idx = Sl.index(local_state)
            global_state_dist[state_idx] += 1
        
        mean_field_state = (current_global_state, tuple(global_state_dist))
        
        state_idx = q_func.get_state_idx(mean_field_state)
        best_action_idx = np.argmax(q_func.Q[state_idx, :])
        best_mean_field_action = q_func.mean_field_actions[best_action_idx]
        global_action = best_mean_field_action[0]
    
        # the agents collect the reward r_t
        step_reward = 0
        for i in range(n):
            step_reward += local_agent_reward(current_joint_local_states[i], current_global_state, joint_actions[i]) / n
        step_reward += global_agent_reward(current_global_state, global_action)
        
        # the cumulative_reward is updated by adding gamma**t * r_t
        discounted_reward = (gamma ** t) * step_reward
        total_reward += discounted_reward
        cumulative_rewards.append(total_reward)
    
        # then the agents transition
        next_global_state = global_agent_transition(current_global_state, global_action)
        next_joint_local_states = []
        for i in range(n):
            next_local_state = local_agent_transition(current_joint_local_states[i], current_global_state, joint_actions[i])
            next_joint_local_states.append(next_local_state)
    
        # update current states
        current_global_state = next_global_state
        current_joint_local_states = tuple(next_joint_local_states)

    print(f"Deployment completed! Total cumulative reward over {H} steps: {total_reward:.4f}")
    return cumulative_rewards


# Load configuration from JSON file
with open('config.json', 'r') as f:
    config = json.load(f)

# Extract environment parameters
Sg = config['environment']['global_agent']['state_space']
Sl = config['environment']['local_agent']['state_space']
Ag = config['environment']['global_agent']['action_space']
Al = config['environment']['local_agent']['action_space']

# Execute function definitions from config
for func_name, func_lines in config['environment']['functions'].items():
    func_code = '\n'.join(func_lines)
    exec(func_code)

state_spaces = [Sl, Sg]
action_spaces = [Al, Ag]
transition_system = [global_agent_transition, local_agent_transition]
reward_system = [global_agent_reward, local_agent_reward]

# Extract simulation parameters
H = config['simulation']['horizon']
n = config['simulation']['num_local_agents']
samples = config['simulation']['monte_carlo_samples']
gamma = config['simulation']['discount_factor']
T = config['simulation']['training_steps']
deployment_runs = config['simulation']['deployment_runs']

cum_rewards = []
plt.figure()
for k in range(2,n+1):
    q_func = Q_function(n=k, state_spaces=state_spaces, action_spaces=action_spaces, samples=samples, transition_system=transition_system, reward_system=reward_system, gamma=gamma)
    q_func.learn(steps=T)
    k_cum_rewards = []
    for _ in range(deployment_runs):
        k_cum_rewards.append(deployment_environment(H, k, n, q_func))
    k_cum_rewards = np.array(k_cum_rewards)
    k_cum_rewards_mean = np.mean(k_cum_rewards, axis=0)
    cum_rewards.append(k_cum_rewards_mean)
    plt.plot([i for i in range(H)], k_cum_rewards_mean, label=f"k={k}")

plt.title("Cumulative Rewards for k=2 to k={}".format(n-1))
plt.xlabel("Time-step")
plt.ylabel("Cumulative Reward")
plt.legend()

if config['visualization']['save_plot']:
    plt.savefig(config['visualization']['plot_filename'])
plt.show()
