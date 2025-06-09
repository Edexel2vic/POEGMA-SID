# algorithms.py

import random
import numpy as np
from numba import njit, prange # Import prange for potential parallel loops in Numba
from solution_concepts import SolutionConcept
from game_model import GameModel
import abc
from typing import Dict, Any, Tuple, List # For type hinting the experience batch

# --- Numba JITed functions (IQL select_action corrected, others same) ---
@njit
def _iql_select_action_numba(state_idx, Q_table_agent, n_individual_actions, epsilon, 
                             random_float_for_epsilon_check, 
                             random_int_for_exploration, 
                             random_int_for_tie_breaking): # random_int_for_tie_breaking will be used with modulo
    if random_float_for_epsilon_check <= epsilon:
        return random_int_for_exploration
    else:
        q_values_for_state = Q_table_agent[state_idx]
        max_q = -np.inf 
        has_finite_q = False
        for i in range(q_values_for_state.shape[0]):
            if q_values_for_state[i] > -np.inf: 
                if q_values_for_state[i] > max_q:
                    max_q = q_values_for_state[i]
                has_finite_q = True
        
        if not has_finite_q: 
            return random_int_for_exploration

        count = 0
        for i in range(q_values_for_state.shape[0]):
            if np.abs(q_values_for_state[i] - max_q) < 1e-9: 
                count += 1
        
        if count == 0: 
            return random_int_for_exploration

        best_actions_indices = np.empty(count, dtype=np.int64)
        current_best_idx = 0
        for i in range(q_values_for_state.shape[0]):
            if np.abs(q_values_for_state[i] - max_q) < 1e-9:
                if current_best_idx < count: 
                    best_actions_indices[current_best_idx] = i
                    current_best_idx += 1
        
        if current_best_idx == 0: 
             return random_int_for_exploration
        
        selected_index_in_best_arr = random_int_for_tie_breaking % current_best_idx
        return best_actions_indices[selected_index_in_best_arr]

@njit
def _iql_update_q_numba(Q_table_agent, state_idx, action_idx, reward, next_state_idx, is_done, alpha, gamma):
    q_next_max = 0.0
    if not is_done: # Keep explicit done check for IQL's robustness
        max_val = -np.inf
        for i in range(Q_table_agent[next_state_idx].shape[0]): 
            if Q_table_agent[next_state_idx, i] > max_val:
                max_val = Q_table_agent[next_state_idx, i]
        q_next_max = max_val
    td_error = reward + gamma * q_next_max - Q_table_agent[state_idx, action_idx]
    Q_table_agent[state_idx, action_idx] += alpha * td_error
    return Q_table_agent, td_error

@njit(cache=True)
def _jalgt_calculate_value_numba(q_table_agent_s_flat, joint_policy_s_for_numba,
                                 game_action_space_numba_array, num_agents,
                                 num_total_joint_actions, num_individual_actions):
    value = 0.0
    for ja_idx in prange(num_total_joint_actions):
        joint_action_tuple = game_action_space_numba_array[ja_idx]
        payoff = q_table_agent_s_flat[ja_idx] 
        joint_probability = 1.0
        for i in range(num_agents):
            agent_i_action = joint_action_tuple[i]
            joint_probability *= joint_policy_s_for_numba[i, agent_i_action]
        value += payoff * joint_probability
    return value

# Define a type hint for the experience batch if you like
ExperienceBatch = Dict[str, Any]

class MARLAlgorithm(abc.ABC):
    @abc.abstractmethod
    def learn(self, experience: ExperienceBatch):
        pass

    @abc.abstractmethod
    def select_action(self, state_for_this_agent: int, train: bool):
        # For JALGT, state_for_this_agent is global_state_int
        # For IQL, state_for_this_agent is local_state_int for this agent
        pass

    @abc.abstractmethod
    def set_epsilon(self, epsilon: float):
        pass

    @abc.abstractmethod
    def explain(self) -> str:
        pass


class JALGT(MARLAlgorithm):
    def __init__(self, agent_id: int, game: GameModel, solution_concept: SolutionConcept,
                 gamma: float =0.95, alpha: float =0.5, epsilon: float =0.2, seed: int =42): # Added type hints
        self.agent_id = agent_id
        self.game = game
        self.solution_concept = solution_concept
        self.alpha = alpha
        self.gamma = gamma # Gamma is used
        self.epsilon = epsilon
        self.rng = random.Random(seed)

        self.q_table = np.zeros((self.game.num_agents, self.game.num_states,
                                 len(self.game.action_space)))
        self.joint_policy = np.ones((self.game.num_agents, self.game.num_states,
                                     self.game.num_actions)) / self.game.num_actions
        self.metrics = {"td_error": []}

        if not hasattr(self.game, 'action_space_numba_array'):
            self.game.action_space_numba_array = np.array(self.game.action_space, dtype=np.int64)

    def value(self, q_table_owner_agent_id: int, global_state_idx: int) -> float:
        q_table_agent_s_flat = self.q_table[q_table_owner_agent_id, global_state_idx, :]
        joint_policy_s_for_numba = np.empty((self.game.num_agents, self.game.num_actions), dtype=np.float64)
        for i in range(self.game.num_agents):
            joint_policy_s_for_numba[i, :] = self.joint_policy[i, global_state_idx, :]
        return _jalgt_calculate_value_numba(q_table_agent_s_flat, joint_policy_s_for_numba,
                                            self.game.action_space_numba_array, self.game.num_agents,
                                            len(self.game.action_space), self.game.num_actions)

    def update_policy(self, q_table_owner_agent_id: int, global_state_idx: int):
        self.joint_policy[q_table_owner_agent_id][global_state_idx] = \
            self.solution_concept.solution_policy(q_table_owner_agent_id, global_state_idx,
                                                  self.game, self.q_table)

    def learn(self, experience: ExperienceBatch):
        joint_action_tuple: Tuple[int, ...] = experience['joint_action']
        rewards_list: List[float] = experience['rewards']
        current_global_state: int = experience['current_global_state']
        next_global_state: int = experience['next_global_state']
        # agent_dones_list is available in experience if needed, but JALGT ignores it.

        joint_action_index = self.game.action_space_index[joint_action_tuple]

        for q_table_owner_agent_id in range(self.game.num_agents):
            agent_reward = rewards_list[q_table_owner_agent_id]
            agent_game_value_next_state = self.value(q_table_owner_agent_id, next_global_state)
            current_q_value = self.q_table[q_table_owner_agent_id][current_global_state][joint_action_index]
            
            td_error = agent_reward + self.gamma * agent_game_value_next_state - current_q_value
            self.q_table[q_table_owner_agent_id][current_global_state][joint_action_index] += self.alpha * td_error
            self.update_policy(q_table_owner_agent_id, current_global_state)
            
            if q_table_owner_agent_id == self.agent_id: # Only log TD error for the agent this instance represents
                self.metrics['td_error'].append(td_error)

    def set_epsilon(self, epsilon: float):
        self.epsilon = epsilon

    def solve(self, agent_id_solving: int, global_state_idx: int) -> np.ndarray:
        return self.joint_policy[agent_id_solving][global_state_idx]

    def select_action(self, global_state_for_select: int, train: bool = True) -> int:
        if train and self.rng.random() < self.epsilon:
            return self.rng.choice(range(self.game.num_actions))
        else:
            probs = self.solve(self.agent_id, global_state_for_select)
            # Use Python's random.choices for weighted selection, or numpy's with careful seeding
            # Using self.rng to seed a temporary np.random.RandomState is a good pattern
            current_seed = self.rng.randint(0, 1_000_000)
            temp_rng = np.random.RandomState(current_seed)
            return temp_rng.choice(range(self.game.num_actions), p=probs)

    def explain(self, global_state_idx_to_explain: int = 0) -> str: # Added type hint and default
        return self.solution_concept.debug(self.agent_id, global_state_idx_to_explain, self.game, self.q_table)


class IQLAgent(MARLAlgorithm):
    def __init__(self, agent_id: int, num_local_states: int, num_individual_actions: int,
                 gamma: float = 0.99, alpha: float = 0.1,
                 epsilon_start: float = 1.0, epsilon_end: float = 0.01, 
                 seed: int = 42):
        self.agent_id = agent_id
        self.num_local_states = num_local_states
        self.num_individual_actions = num_individual_actions

        self.py_rng = random.Random(seed + agent_id) # Seed per agent instance
        # For Numba's np.random calls, ensure proper seeding if relying on global np.random state
        # Or pass a Numba-compatible RNG object if Numba version supports it well.
        # Initialize Q-table with small random values or zeros
        # np_rng_for_q_init = np.random.RandomState(seed + 100 + agent_id)
        # self.Q_table = np_rng_for_q_init.uniform(low=-0.001, high=0.001, size=(self.num_local_states, self.num_individual_actions)).astype(np.float64)
        self.Q_table = np.zeros((self.num_local_states, self.num_individual_actions), dtype=np.float64)


        self.gamma = gamma
        self.alpha = alpha
        self.epsilon_start = epsilon_start # Epsilon is set by external controller via set_epsilon
        self.current_epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.metrics = {"td_error": []}

    def set_epsilon(self, epsilon_value: float):
        self.current_epsilon = max(self.epsilon_end, epsilon_value) # Ensure not below min

    def select_action(self, agent_local_state_idx: int, train: bool = True) -> int:
        epsilon_to_use = self.current_epsilon if train else 0.0
        
        random_float_for_epsilon_check = self.py_rng.random()
        random_int_for_exploration = self.py_rng.randint(0, self.num_individual_actions - 1)
        # random_int_for_tie_breaking is used with modulo inside Numba, so its exact range isn't critical
        # as long as it's a non-negative integer.
        random_int_for_tie_breaking = self.py_rng.randint(0, self.num_individual_actions * 100) # Larger range is fine

        return _iql_select_action_numba(agent_local_state_idx, self.Q_table,
                                        self.num_individual_actions, epsilon_to_use,
                                        random_float_for_epsilon_check,
                                        random_int_for_exploration,
                                        random_int_for_tie_breaking)

    def learn(self, experience: ExperienceBatch):
        joint_action_tuple: Tuple[int, ...] = experience['joint_action']
        rewards_list: List[float] = experience['rewards']
        # IQL uses its own local states from the lists
        current_local_states: List[int] = experience['current_local_states']
        next_local_states: List[int] = experience['next_local_states']
        agent_dones_list: List[bool] = experience['dones']

        my_action = joint_action_tuple[self.agent_id]
        my_reward = rewards_list[self.agent_id]
        my_current_local_state_idx = current_local_states[self.agent_id]
        my_next_local_state_idx = next_local_states[self.agent_id]
        my_done_flag = agent_dones_list[self.agent_id]

        self.Q_table, td_err = _iql_update_q_numba(
            self.Q_table, my_current_local_state_idx, my_action, my_reward,
            my_next_local_state_idx, my_done_flag, self.alpha, self.gamma
        )
        self.metrics["td_error"].append(td_err)

    def explain(self) -> str:
        return (f"IQL Agent {self.agent_id}: Epsilon={self.current_epsilon:.3f}. "
                f"Q-table L1 norm: {np.linalg.norm(self.Q_table, ord=1):.2f}")