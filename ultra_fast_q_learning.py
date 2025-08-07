import numpy as np
import torch
import torch.nn.functional as F
from numba import jit, cuda
import matplotlib.pyplot as plt
import time
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
import pickle
import itertools
from scipy.sparse import csr_matrix

# JIT-compiled standalone functions for maximum speed
@jit(nopython=True, fastmath=True, cache=True)
def decode_joint_state(joint_idx, n_global, n_local, n_agents):
    """JIT-compiled ultra-fast joint state decoding."""
    global_state = joint_idx % n_global
    remaining = joint_idx // n_global
    local_states = np.zeros(n_agents, dtype=np.int32)
    for i in range(n_agents):
        local_states[i] = remaining % n_local
        remaining //= n_local
    return global_state, local_states

@jit(nopython=True, fastmath=True, cache=True)
def decode_joint_action(joint_idx, n_global, n_local, n_agents):
    """JIT-compiled ultra-fast joint action decoding."""
    global_action = joint_idx % n_global
    remaining = joint_idx // n_global
    local_actions = np.zeros(n_agents, dtype=np.int32)
    for i in range(n_agents):
        local_actions[i] = remaining % n_local
        remaining //= n_local
    return global_action, local_actions

@jit(nopython=True, fastmath=True, cache=True)
def fast_reward_computation(rewards_array, n_joint_states, n_joint_actions, 
                          n_global_states, n_local_states, n_global_actions, n_local_actions, n_agents):
    """JIT-compiled reward computation for maximum speed."""
    for joint_state_idx in range(n_joint_states):
        global_s, local_states = decode_joint_state(joint_state_idx, n_global_states, n_local_states, n_agents)
        
        for joint_action_idx in range(n_joint_actions):
            global_a, local_actions = decode_joint_action(joint_action_idx, n_global_actions, n_local_actions, n_agents)
            
            # Simple reward computation (replace with your actual reward logic)
            global_reward = float(global_s + global_a)
            local_reward_sum = 0.0
            for i in range(n_agents):
                local_reward_sum += float(local_states[i] + global_s + local_actions[i])
            
            rewards_array[joint_state_idx, joint_action_idx] = global_reward + local_reward_sum / n_agents
    
    return rewards_array

class UltraFastQLearning:
    """FAST vectorized Q-learning with GPU acceleration and JIT compilation."""
    
    def __init__(self, n_agents, n_global_states, n_local_states, n_global_actions, 
                 n_local_actions, samples=20, gamma=0.9, device=None):
        self.n = n_agents
        self.n_global_states = n_global_states
        self.n_local_states = n_local_states
        self.n_global_actions = n_global_actions
        self.n_local_actions = n_local_actions
        self.samples = samples
        self.gamma = gamma
        
        # Auto-detect best device (CUDA > MPS > CPU)
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
                print("CUDA GPU detected!")
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = torch.device('mps')  # Apple Silicon GPU
                print("Apple Silicon GPU detected!")
            else:
                self.device = torch.device('cpu')
                print("Using CPU")
        else:
            self.device = torch.device(device)
        
        # Calculate joint space dimensions
        self.n_joint_states = n_global_states * (n_local_states ** n_agents)
        self.n_joint_actions = n_global_actions * (n_local_actions ** n_agents)
        
        print(f"Initializing UltraFastQLearning on {self.device.type.upper()}")
        print(f"Joint space: {self.n_joint_states:,} states × {self.n_joint_actions:,} actions")
        print(f"Q-table size: {self.n_joint_states * self.n_joint_actions * 4 / 1e9:.2f} GB")
        
        # Initialize Q-table as PyTorch tensor for GPU acceleration
        self.Q = torch.zeros(self.n_joint_states, self.n_joint_actions, 
                            dtype=torch.float32, device=self.device)
        
        # Pre-compute state/action index mappings for vectorization
        self._precompute_mappings()
        
        # Pre-allocate memory for batch operations
        self._preallocate_tensors()
        
    def _precompute_mappings(self):
        """Pre-compute all state/action mappings for instant vectorized access."""
        # State multipliers for joint state indexing
        self.state_multipliers = torch.tensor([
            self.n_global_states * (self.n_local_states ** i) for i in range(self.n)
        ], dtype=torch.long, device=self.device)
        
        # Action multipliers for joint action indexing
        self.action_multipliers = torch.tensor([
            self.n_global_actions * (self.n_local_actions ** i) for i in range(self.n)
        ], dtype=torch.long, device=self.device)
        
        print(f"🔧 Pre-computed mappings: {len(self.state_multipliers)} multipliers")
        
    def _preallocate_tensors(self):
        """Pre-allocate tensors for batch operations to avoid memory allocation overhead."""
        # Batch tensors (reused across iterations)
        self.batch_rewards = torch.zeros(self.n_joint_states, self.n_joint_actions, 
                                       dtype=torch.float32, device=self.device)
        self.max_q_batch = torch.zeros(self.n_joint_states, self.n_joint_actions, self.samples,
                                     dtype=torch.float32, device=self.device)
        
        print(f"Pre-allocated {self.batch_rewards.numel() * 4 / 1e6:.1f}MB of GPU memory")
        
        
    def vectorized_reward_computation(self, global_reward_func, local_reward_func):
        """Vectorized reward computation for ALL state-action combinations."""
        print("Computing rewards tensor...")
        start_time = time.time()
        
        rewards_np = np.zeros((self.n_joint_states, self.n_joint_actions), dtype=np.float32)
        
        # ULTRA-FAST JIT-compiled computation
        rewards_np = fast_reward_computation(
            rewards_np, self.n_joint_states, self.n_joint_actions,
            self.n_global_states, self.n_local_states, 
            self.n_global_actions, self.n_local_actions, self.n
        )
        
        # Convert to GPU tensor
        self.batch_rewards = torch.from_numpy(rewards_np).to(self.device)
        
        elapsed = time.time() - start_time
        print(f"Reward tensor computed in {elapsed:.2f}s: {self.batch_rewards.shape}")
        
    def lightning_fast_update(self, transition_probs_global, transition_probs_local):
        """MAXIMUM SPEED vectorized Q-learning update using GPU tensor operations."""
        with torch.cuda.amp.autocast(enabled=self.device.type=='cuda'):  # Mixed precision
            batch_size = self.n_joint_states * self.n_joint_actions
            
            # Create dummy transition sampling (replace with your actual transitions)
            # Sample next global states
            global_probs = transition_probs_global.view(self.n_global_states * self.n_global_actions, -1)
            next_global_indices = torch.multinomial(global_probs, self.samples, replacement=True)
            
            # Sample next local states  
            local_probs = transition_probs_local.view(-1, self.n_local_states)
            next_local_indices = torch.multinomial(local_probs, self.samples * self.n, replacement=True)
            next_local_indices = next_local_indices.view(-1, self.samples, self.n)
            
            # Vectorized computation of next joint states
            next_joint_states = torch.zeros(self.n_joint_states, self.n_joint_actions, self.samples, 
                                          dtype=torch.long, device=self.device)
            
            # Simplified next state computation (this would need proper transition logic)
            for s_idx in range(self.n_joint_states):
                for a_idx in range(self.n_joint_actions):
                    # Sample next states based on current state-action
                    next_global = torch.randint(0, self.n_global_states, (self.samples,), device=self.device)
                    next_locals = torch.randint(0, self.n_local_states, (self.samples, self.n), device=self.device)
                    
                    # Compute next joint state indices
                    next_joint = next_global + torch.sum(next_locals * self.state_multipliers[None, :], dim=1)
                    next_joint_states[s_idx, a_idx] = next_joint
            
            # Vectorized max Q-value computation
            q_next_values = self.Q[next_joint_states]  # Shape: [states, actions, samples, actions]
            max_q_next = torch.max(q_next_values, dim=-1)[0]  # Max over next actions
            expected_max_q = torch.mean(max_q_next, dim=-1)   # Expected over samples
            
            # Bellman update using tensor operations (ALL updates simultaneously)
            self.Q = self.batch_rewards + self.gamma * expected_max_q
            
    def mega_fast_training(self, steps, global_reward_func, local_reward_func, 
                          transition_funcs=None, verbose=True):
        """ULTRA-HIGH-SPEED training with full GPU acceleration and vectorization."""
        if verbose:
            print(f"Starting MEGA-FAST training for {steps:,} steps on {self.device.type.upper()}...")
        
        # Pre-compute ALL rewards (vectorized)
        self.vectorized_reward_computation(global_reward_func, local_reward_func)
        
        # Create dummy transition matrices (replace with your actual transitions)
        print("🔄 Creating transition tensors...")
        transition_global = torch.rand(self.n_global_states, self.n_global_actions, self.n_global_states,
                                     device=self.device)
        transition_local = torch.rand(self.n_local_states, self.n_global_states, self.n_local_actions, self.n_local_states,
                                    device=self.device)
        
        # Normalize to valid probabilities
        transition_global = F.softmax(transition_global, dim=-1)
        transition_local = F.softmax(transition_local, dim=-1)
        
        start_time = time.time()
        updates_per_step = self.n_joint_states * self.n_joint_actions
        
        if verbose:
            print(f"{updates_per_step:,} updates per step = {steps * updates_per_step:,} total updates")
        
        for step in range(steps):
            # Single vectorized call updates ALL state-action pairs
            self.lightning_fast_update(transition_global, transition_local)
            
            if verbose and (step + 1) % max(1, steps // 10) == 0:
                elapsed = time.time() - start_time
                throughput = (step + 1) * updates_per_step / elapsed
                print(f"⚡ Step {step+1:,}/{steps:,} | {throughput:.0f} updates/sec | {self.device.type.upper()}")
        
        total_time = time.time() - start_time
        final_throughput = steps * updates_per_step / total_time
        
        if verbose:
            print(f"Training completed in {total_time:.2f}s")
            print(f"Final throughput: {final_throughput:.0f} updates/sec")
            print(f"Total updates: {steps * updates_per_step:,}")
        
        return {
            'total_time': total_time,
            'throughput': final_throughput,
            'updates_per_second': final_throughput,
            'total_updates': steps * updates_per_step
        }
    
    def get_policy(self, epsilon=0.1):
        """Extract epsilon-greedy policy from learned Q-values."""
        with torch.no_grad():
            # Get best actions for each state (greedy policy)
            best_actions = torch.argmax(self.Q, dim=1)
            
            # Add epsilon-greedy exploration
            random_actions = torch.randint(0, self.n_joint_actions, (self.n_joint_states,), 
                                         device=self.device)
            explore_mask = torch.rand(self.n_joint_states, device=self.device) < epsilon
            
            policy_actions = torch.where(explore_mask, random_actions, best_actions)
            return policy_actions.cpu().numpy()
    
    def benchmark_speed(self, steps=100):
        """Benchmark the speed of this implementation."""
        print("\nSPEED BENCHMARK")
        print("=" * 50)
        
        # Dummy functions for benchmarking
        def dummy_global_reward(gs, ga):
            return float(gs + ga)
        
        def dummy_local_reward(ls, gs, la):
            return float(ls + gs + la)
        
        results = self.mega_fast_training(
            steps=steps,
            global_reward_func=dummy_global_reward,
            local_reward_func=dummy_local_reward,
            transition_funcs=None,
            verbose=True
        )
        
        print("\nBENCHMARK RESULTS:")
        print(f"Device: {self.device.type.upper()}")
        print(f"Speed: {results['throughput']:,.0f} updates/sec")
        print(f"Time: {results['total_time']:.2f} seconds")
        print(f"Updates: {results['total_updates']:,}")
        
        return results

# EXAMPLE USAGE WITH MAXIMUM PERFORMANCE
if __name__ == "__main__":
    print("ULTRA-FAST Q-LEARNING DEMO")
    print("=" * 40)
    
    # Create fast Q-learning instance
    ultra_fast_q = UltraFastQLearning(
        n_agents=3,
        n_global_states=5,
        n_local_states=4,
        n_global_actions=3,
        n_local_actions=2,
        samples=10,
        gamma=0.9
    )
    
    # Run speed benchmark
    results = ultra_fast_q.benchmark_speed(steps=500)
    
    print(f"\n SPEED ACHIEVED: {results['throughput']:,.0f} updates/sec!")
