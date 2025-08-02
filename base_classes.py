"""
Base classes and core components for centralized multi-agent Q-learning.
This module contains the abstract base classes and basic implementations
for states, actions, agents, and functions used in the centralized Q-learning framework.
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Callable


class State(ABC):
    """Abstract base class for states."""
    
    @abstractmethod
    def __hash__(self):
        pass
    
    @abstractmethod
    def __eq__(self, other):
        pass
    
    @abstractmethod
    def __repr__(self):
        pass


class Action(ABC):
    """Abstract base class for actions."""
    
    @abstractmethod
    def __hash__(self):
        pass
    
    @abstractmethod
    def __eq__(self, other):
        pass
    
    @abstractmethod
    def __repr__(self):
        pass


class DiscreteState(State):
    """Simple discrete state representation."""
    
    def __init__(self, value: int):
        self.value = value
    
    def __hash__(self):
        return hash(self.value)
    
    def __eq__(self, other):
        return isinstance(other, DiscreteState) and self.value == other.value
    
    def __repr__(self):
        return f"State({self.value})"


class DiscreteAction(Action):
    """Simple discrete action representation."""
    
    def __init__(self, value: int):
        self.value = value
    
    def __hash__(self):
        return hash(self.value)
    
    def __eq__(self, other):
        return isinstance(other, DiscreteAction) and self.value == other.value
    
    def __repr__(self):
        return f"Action({self.value})"


class GlobalAgent:
    """Represents the global agent with its own state and action space."""
    
    def __init__(self, state_space: List[State], action_space: List[Action]):
        self.state_space = state_space
        self.action_space = action_space
        self.current_state = None
    
    def reset(self, initial_state: State):
        """Reset agent to initial state."""
        self.current_state = initial_state
    
    def get_state(self) -> State:
        return self.current_state
    
    def set_state(self, state: State):
        self.current_state = state


class LocalAgent:
    """Represents a local agent with its own state and action space."""
    
    def __init__(self, agent_id: int, state_space: List[State], action_space: List[Action]):
        self.agent_id = agent_id
        self.state_space = state_space
        self.action_space = action_space
        self.current_state = None
    
    def reset(self, initial_state: State):
        """Reset agent to initial state."""
        self.current_state = initial_state
    
    def get_state(self) -> State:
        return self.current_state
    
    def set_state(self, state: State):
        self.current_state = state


class TransitionFunction(ABC):
    """Abstract base class for transition functions."""
    
    @abstractmethod
    def sample_next_state(self, *args) -> State:
        """Sample next state given current conditions."""
        pass


class GlobalAgentTransition(TransitionFunction):
    """Transition function for global agent: s_g' ~ P_g(.|s_g, a_g)"""
    
    def __init__(self, transition_probs: Dict[Tuple[State, Action], Dict[State, float]]):
        self.transition_probs = transition_probs
    
    def sample_next_state(self, current_state: State, action: Action) -> State:
        """Sample next global state."""
        key = (current_state, action)
        next_states = list(self.transition_probs[key].keys())
        probs = list(self.transition_probs[key].values())
        return np.random.choice(next_states, p=probs)

class LocalAgentTransition(TransitionFunction):
    """Transition function for local agent: s_i' ~ P_l(.|s_g, a_g, a_i)"""
    def __init__(self, transition_probs: Dict[Tuple[State, Action, Action], Dict[State, float]]):
        self.transition_probs = transition_probs
    
    def sample_next_state(self, global_state: State, global_action: Action, local_action: Action) -> State:
        """Sample next local state given global state, global action, and local action."""
        key = (global_state, global_action, local_action)        
        next_states = list(self.transition_probs[key].keys())
        probs = list(self.transition_probs[key].values())
        return np.random.choice(next_states, p=probs)


class RewardFunction(ABC):
    """Abstract base class for reward functions."""
    
    @abstractmethod
    def compute_reward(self, *args) -> float:
        pass


class GlobalAgentReward(RewardFunction):
    """Global agent reward function: r_g(s_g, a_g)"""
    
    def __init__(self, reward_func: Callable[[State, Action], float]):
        self.reward_func = reward_func
    
    def compute_reward(self, global_state: State, global_action: Action) -> float:
        return self.reward_func(global_state, global_action)


class LocalAgentReward(RewardFunction):
    """Local agent reward function: r_l(s_i, s_g, a_i)"""
    
    def __init__(self, reward_func: Callable[[State, State, Action], float]):
        self.reward_func = reward_func
    
    def compute_reward(self, local_state: State, global_state: State, local_action: Action) -> float:
        return self.reward_func(local_state, global_state, local_action)
