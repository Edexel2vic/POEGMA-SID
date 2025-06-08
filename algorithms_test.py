# algorithms.py

import random
import numpy as np
from numba import njit, prange
from solution_concepts import SolutionConcept
from game_model import GameModel
import abc

# --- Numba JITed functions for IQL ( 그대로 유지 ) ---
@njit
def _iql_select_action_numba(state_idx, Q_table_agent, n_individual_actions, epsilon, rng_randint_val_1, rng_randint_val_2):
    if np.random.random() <= epsilon:
        return rng_randint_val_1
    else:
        q_values_for_state = Q_table_agent[state_idx]
        max_q = -np.inf
        for i in range(q_values_for_state.shape[0]):
            if q_values_for_state[i] > max_q:
                max_q = q_values_for_state[i]
        count = 0
        for i in range(q_values_for_state.shape[0]):
            if q_values_for_state[i] == max_q:
                count += 1
        if count == 0: 
            return rng_randint_val_1
        best_actions = np.empty(count, dtype=np.int64)
        idx = 0
        for i in range(q_values_for_state.shape[0]):
            if q_values_for_state[i] == max_q:
                best_actions[idx] = i
                idx += 1
        return best_actions[rng_randint_val_2]

@njit
def _iql_update_q_numba(Q_table_agent, state_idx, action_idx, reward, next_state_idx, is_done, alpha, gamma):
    q_next_max = 0.0
    if not is_done:
        max_val = -np.inf
        for i in range(Q_table_agent[next_state_idx].shape[0]): 
            if Q_table_agent[next_state_idx, i] > max_val:
                max_val = Q_table_agent[next_state_idx, i]
        q_next_max = max_val
    td_error = reward + gamma * q_next_max - Q_table_agent[state_idx, action_idx]
    Q_table_agent[state_idx, action_idx] += alpha * td_error
    return Q_table_agent, td_error

# --- Numba JITed function for JALGT.value ( 그대로 유지 ) ---
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


class MARLAlgorithm(abc.ABC):
    @abc.abstractmethod
    def learn(self, joint_action_tuple, rewards_list, current_state_representation, next_state_representation, agent_dones_list):
        # state_representation can be a single global int for JALGT,
        # or a list of local state ints for IQL (IQL will pick its own).
        pass

    @abc.abstractmethod
    def select_action(self, agent_state_for_select: int, train: bool):
        # agent_state_for_select is the specific state (local for IQL, global for JALGT's policy)
        # this agent instance uses to make its decision.
        pass

    @abc.abstractmethod
    def set_epsilon(self, epsilon):
        pass

    @abc.abstractmethod
    def explain(self):
        pass


class JALGT(MARLAlgorithm):
    def __init__(self, agent_id, game: GameModel, solution_concept: SolutionConcept,
                 gamma=0.95, alpha=0.5, epsilon=0.2, seed=42):
        self.agent_id = agent_id # ID of the agent this instance primarily acts for/represents
        self.game = game
        self.solution_concept = solution_concept
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.rng = random.Random(seed)

        # Q-table: N_agents x N_GLOBAL_STATES x N_joint_actions
        self.q_table = np.zeros((self.game.num_agents, self.game.num_states, # game.num_states is the global state space size
                                 len(self.game.action_space)))
        # Joint policy: N_agents x N_GLOBAL_STATES x N_individual_actions
        self.joint_policy = np.ones((self.game.num_agents, self.game.num_states,
                                     self.game.num_actions)) / self.game.num_actions
        self.metrics = {"td_error": []}

        if not hasattr(self.game, 'action_space_numba_array'):
            self.game.action_space_numba_array = np.array(self.game.action_space, dtype=np.int64)

    def value(self, q_table_owner_agent_id, global_state_idx):
        """Calculates V_k(S) for agent 'k' in global state 'S'."""
        q_table_agent_s_flat = self.q_table[q_table_owner_agent_id, global_state_idx, :]
        
        joint_policy_s_for_numba = np.empty((self.game.num_agents, self.game.num_actions), dtype=np.float64)
        for i in range(self.game.num_agents):
            # All agents' policies are conditioned on the same global_state_idx
            joint_policy_s_for_numba[i, :] = self.joint_policy[i, global_state_idx, :]

        return _jalgt_calculate_value_numba(q_table_agent_s_flat,
                                            joint_policy_s_for_numba,
                                            self.game.action_space_numba_array,
                                            self.game.num_agents,
                                            len(self.game.action_space),
                                            self.game.num_actions)

    def update_policy(self, q_table_owner_agent_id, global_state_idx):
        """Updates policy for agent 'k' in global state 'S'."""
        self.joint_policy[q_table_owner_agent_id][global_state_idx] = \
            self.solution_concept.solution_policy(q_table_owner_agent_id,
                                                  global_state_idx, # Global state
                                                  self.game,
                                                  self.q_table)

    def learn(self, joint_action_tuple, rewards_list, current_global_state: int, next_global_state: int, _agent_dones_list=None):
        # _agent_dones_list is ignored by JALGT as V(terminal_state) should be 0 via Q-values.
        joint_action_index = self.game.action_space_index[joint_action_tuple]

        for q_table_owner_agent_id in range(self.game.num_agents):
            agent_reward = rewards_list[q_table_owner_agent_id]
            
            # V(S') calculated using the global next state
            agent_game_value_next_state = self.value(q_table_owner_agent_id, next_global_state)
            current_q_value = self.q_table[q_table_owner_agent_id][current_global_state][joint_action_index]
            
            td_error = agent_reward + self.gamma * agent_game_value_next_state - current_q_value
            self.q_table[q_table_owner_agent_id][current_global_state][joint_action_index] += self.alpha * td_error
            
            self.update_policy(q_table_owner_agent_id, current_global_state)
            
            if q_table_owner_agent_id == self.agent_id:
                self.metrics['td_error'].append(td_error)

    def set_epsilon(self, epsilon):
        self.epsilon = epsilon

    def solve(self, agent_id_solving, global_state_idx):
        """Returns policy for agent_id_solving in global_state_idx."""
        return self.joint_policy[agent_id_solving][global_state_idx]

    def select_action(self, global_state_for_select: int, train: bool = True):
        """Selects action for self.agent_id based on the global_state_for_select."""
        if train:
            if self.rng.random() < self.epsilon:
                return self.rng.choice(range(self.game.num_actions))
            else:
                probs = self.solve(self.agent_id, global_state_for_select)
                current_seed = self.rng.randint(0, 1_000_000)
                temp_rng = np.random.RandomState(current_seed)
                return temp_rng.choice(range(self.game.num_actions), p=probs)
        else:
            return np.argmax(self.solve(self.agent_id, global_state_for_select))

    def explain(self, global_state_idx_to_explain=0):
        return self.solution_concept.debug(self.agent_id, global_state_idx_to_explain, self.game, self.q_table)


class IQLAgent(MARLAlgorithm):
    def __init__(self, agent_id: int, num_local_states: int, num_individual_actions: int,
                 gamma: float = 0.99, alpha: float = 0.1,
                 epsilon_start: float = 1.0, epsilon_end: float = 0.01, 
                 seed: int = 42):
        self.agent_id = agent_id
        self.num_local_states = num_local_states # Size of state space from this agent's obs_to_state
        self.num_individual_actions = num_individual_actions

        self.Q_table = np.zeros((self.num_local_states, self.num_individual_actions), dtype=np.float64)
        
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon_start = epsilon_start
        self.current_epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.py_rng = random.Random(seed)
        self.metrics = {"td_error": []}

    def set_epsilon(self, epsilon_value: float):
        self.current_epsilon = max(self.epsilon_end, epsilon_value)

    def select_action(self, agent_local_state_idx: int, train: bool = True):
        # agent_local_state_idx is the output of obs_to_state for this agent
        epsilon_to_use = self.current_epsilon if train else 0.0
        rng_int_for_random_action = self.py_rng.randint(0, self.num_individual_actions - 1)
        rng_int_for_tie_breaking = self.py_rng.randint(0, self.num_individual_actions) 
        return _iql_select_action_numba(agent_local_state_idx, self.Q_table,
                                        self.num_individual_actions, epsilon_to_use,
                                        rng_int_for_random_action, rng_int_for_tie_breaking)

    def learn(self, joint_action_tuple, rewards_list, current_state_input, next_state_input, agent_dones_list):
        # For IQL, current_state_input and next_state_input are expected to be LISTS of local states
        # (even if JALGT receives them as single ints, the interface should allow lists for IQL)
        if not isinstance(current_state_input, (list, np.ndarray)) or \
           not isinstance(next_state_input, (list, np.ndarray)):
            raise ValueError("IQL expects current_state_input and next_state_input to be lists/arrays of local states.")

        my_action = joint_action_tuple[self.agent_id]
        my_reward = rewards_list[self.agent_id]
        my_current_local_state_idx = current_state_input[self.agent_id]
        my_next_local_state_idx = next_state_input[self.agent_id]
        my_done_flag = agent_dones_list[self.agent_id] # IQL uses its own done flag

        self.Q_table, td_err = _iql_update_q_numba(
            self.Q_table, my_current_local_state_idx, my_action, my_reward,
            my_next_local_state_idx, my_done_flag, self.alpha, self.gamma
        )
        self.metrics["td_error"].append(td_err)

    def explain(self):
        return (f"IQL Agent {self.agent_id}: Epsilon={self.current_epsilon:.3f}. "
                f"Q-table L1 norm: {np.linalg.norm(self.Q_table, ord=1):.2f}")