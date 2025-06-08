# =============      TODO: Epsilon decay se realiza por epoch, no por total_training_episodes
# =============            Ahora mismo la selección de algoritmo es por optuna, no hardcodeada (igual preferimos hardcodear para dividirnoslo)
# =============            Solo hay una sample de entrenamiento por trial, no varias, esto es igual que en la practica 2, pero al parecer es buena práctica ir guardando samples de entrenamiento
# =============            Todo el output se guarda en un solo CSV al final, no en varios ficheros, también hay opción de guardar en SQLite pero igual es matada 








import os
import time
import random
import numpy as np
from tqdm import tqdm
import optuna # Essential import
import csv # For final CSV export if needed, though Optuna has built-in

# Assuming algorithms_fixed.py contains the corrected IQL and JALGT
from algorithms_test import JALGT, IQLAgent # Make sure to import IQLAgent
from solution_concepts import MinimaxSolutionConcept, ParetoSolutionConcept, NashSolutionConcept, WelfareSolutionConcept
from game_model import GameModel
from gymnasium import Wrapper
from pogema import pogema_v0, GridConfig
from pogema.animation import AnimationMonitor, AnimationConfig
# from utils import draw_history # Comment out if not immediately needed for HPO runs

# --- Helper: obs_to_state (keep as is from your file) ---
def obs_to_state(obs):
    matrix_obstacles = obs[0]
    matrix_agents = obs[1]
    matrix_target = obs[2]
    target = np.max(matrix_target[2]) * 1 + \
             matrix_target[1][0] * 2 + matrix_target[1][2] * 3
    obstacles = matrix_obstacles[0][1] * 2 ** 9 + \
                matrix_obstacles[1][0] * 2 ** 8 + \
                matrix_obstacles[1][2] * 2 ** 7 + \
                matrix_obstacles[2][1] * 2 ** 6
    agents_obs_part = matrix_agents[0][1] * 2 ** 5 + \
                      matrix_agents[1][0] * 2 ** 4 + \
                      matrix_agents[1][2] * 2 ** 3 + \
                      matrix_agents[2][1] * 2 ** 2
    return int(obstacles + agents_obs_part + target)


# --- Helper: RewardWrapper (keep as is from your file) ---
class RewardWrapper(Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.raw_rewards_this_step = []

    def step(self, joint_action):
        # previous_observations = self.env.unwrapped._obs() # If needed
        observations, rewards, terminated, truncated, infos = self.env.step(joint_action)
        self.raw_rewards_this_step = list(rewards) # Store raw rewards

        modified_rewards = list(rewards)
        for i in range(len(joint_action)):
            if not terminated[i] and not truncated[i]:
                if modified_rewards[i] == 0:
                    modified_rewards[i] = modified_rewards[i] - 0.01 # Time penalty
        return observations, modified_rewards, terminated, truncated, infos

    def get_unmodified_rewards(self): # To get actual task rewards for collective metrics
        return self.raw_rewards_this_step


# --- Helper: create_env (keep as is from your file, but adapt render_mode) ---
def create_pogema_env(num_agents_cfg, size_cfg, obstacle_density_cfg, episode_length_cfg,
                      env_seed, save_every_nth_episode_render=None, render_dir='renders_optuna/'):
    # For HPO, usually render_mode=None unless specifically debugging one trial
    render_mode_for_pogema = 'rgb_array' if save_every_nth_episode_render is not None else None

    grid_config = GridConfig(num_agents=num_agents_cfg,
                             size=size_cfg,
                             density=obstacle_density_cfg,
                             seed=env_seed,
                             max_episode_steps=episode_length_cfg,
                             obs_radius=1,
                             on_target="finish",
                             render_mode=render_mode_for_pogema) # Use the determined render_mode

    env = pogema_v0(grid_config)

    if save_every_nth_episode_render is not None:
        # Ensure render directory exists (handle potential race conditions if parallelizing Optuna trials later)
        trial_specific_render_dir = os.path.join(render_dir, f"trial_{optuna.trial.Trial().number if optuna.trial.FixedTrial._current_trial else 'default'}")
        os.makedirs(trial_specific_render_dir, exist_ok=True)
        
        animation_config = AnimationConfig(directory=trial_specific_render_dir,
                                           save_every_idx_episode=save_every_nth_episode_render,
                                           filename_prefix=f"map{env_seed}")
        env = AnimationMonitor(env, animation_config=animation_config)
    return RewardWrapper(env)


# --- Optuna Objective Function ---
def objective(trial: optuna.trial.Trial):
    # --- I. Suggest Hyperparameters ---
    # Algorithm Choice
    try:
        algorithm_name = trial.suggest_categorical("algorithm_name", ["JALGT", "IQL"])

        # Shared Environment/Setup Params
        # num_agents = trial.suggest_int("num_agents", 2, 2) # For POGEMA assignment, often fixed at 2
        num_agents = 2 # Fixed for this assignment's baseline
        map_size = trial.suggest_int("map_size", 6, 10)
        obstacle_density = trial.suggest_float("obstacle_density", 0.0, 0.4)
        
        # Training Structure
        # Using "epochs" and "episodes_per_epoch" like original main.py
        # total_training_episodes = trial.suggest_int("total_training_episodes", 200, 2000)
        # episodes_per_evaluation = trial.suggest_int("episodes_per_evaluation", 20, 100)
        # num_epochs = total_training_episodes // episodes_per_evaluation
        # For simplicity, let's use the original main.py's structure:
        num_epochs = trial.suggest_int("num_epochs", 50, 300) # Number of main "outer loops"
        episodes_per_epoch = trial.suggest_int("episodes_per_epoch", 10, 50) # Training episodes per outer loop

        max_steps_per_episode = trial.suggest_int("max_steps_per_episode", 20, 100) # t_max

        # Learning Parameters (Common)
        learning_rate = trial.suggest_float("learning_rate", 1e-4, 0.1, log=True) # alpha
        # gamma = trial.suggest_float("gamma", 0.9, 0.999) # Assuming gamma is in algos
        # Your JALGT and IQL have gamma in __init__, so we should tune it.
        gamma_val = trial.suggest_float("gamma", 0.90, 0.999)


        # Epsilon Parameters (Common)
        epsilon_start = trial.suggest_float("epsilon_start", 0.6, 1.0) # epsilon_max
        epsilon_end = trial.suggest_float("epsilon_end", 0.01, 0.2)   # epsilon_min
        if epsilon_end >= epsilon_start: # Ensure logical range
            epsilon_end = epsilon_start * 0.1
        # Epsilon decay: linear over episodes_per_epoch, resetting each epoch (like original)
        # OR linear over total_training_episodes. Let's stick to original per-epoch decay for now.

        # JALGT-Specific
        solution_concept_name = "Nash" # Default, or make categorical if algorithm_name is JALGT
        if algorithm_name == "JALGT":
            solution_concept_name = trial.suggest_categorical("solution_concept", ["Nash", "Pareto", "Welfare"])
            # Add alpha_decay for JALGT if implemented
            # alpha_decay_rate = trial.suggest_float("jalgt_alpha_decay", 0.99, 1.0) # 1.0 = no decay

        # Fixed for obs_radius=1 and current obs_to_state
        num_states_for_q_tables = 1024 # 16*16*4
        num_individual_actions = 5 # POGEMA: STAY, U, D, L, R
        
        # Create GameModel (needed by JALGT)
        game = GameModel(num_agents=num_agents, num_states=num_states_for_q_tables, num_actions=num_individual_actions)

        # Instantiate Algorithms
        algorithms = []
        if algorithm_name == "JALGT":
            sc_map = {"Nash": NashSolutionConcept, "Pareto": ParetoSolutionConcept, "Welfare": WelfareSolutionConcept}
            solution_concept_class = sc_map[solution_concept_name]
            for i in range(num_agents):
                algorithms.append(JALGT(agent_id=i, game=game, solution_concept=solution_concept_class(),
                                        gamma=gamma_val, alpha=learning_rate, epsilon=epsilon_start,
                                        seed=trial.number * num_agents + i)) # Unique seed
        elif algorithm_name == "IQL":
            for i in range(num_agents):
                algorithms.append(IQLAgent(agent_id=i, num_local_states=num_states_for_q_tables, # IQL uses this as its local state space size
                                        num_individual_actions=num_individual_actions,
                                        gamma=gamma_val, alpha=learning_rate,
                                        epsilon_start=epsilon_start, epsilon_end=epsilon_end,
                                        seed=trial.number * num_agents + i)) # Unique seed
        
        # --- II. Training and Evaluation Loop (adapting original main.py structure) ---
        # Metrics for this Optuna trial
        trial_collective_rewards_per_epoch = []
        trial_individual_rewards_per_epoch_agent0 = [] # Example
        trial_individual_rewards_per_epoch_agent1 = [] # Example
        trial_td_errors_per_epoch_agent0 = [] # Example for agent 0
        total_training_time_s = 0.0
        
        # Epsilon decay (linear per epoch, like original main.py)
        # This means epsilon is effectively reset/recalculated at the start of each epoch's training.
        epsilon_diff_per_episode_in_epoch = 0
        if episodes_per_epoch > 0:
            epsilon_diff_per_episode_in_epoch = (epsilon_start - epsilon_end) / episodes_per_epoch


        # Main loop over "epochs" (outer loops of training + evaluation)
        for epoch_idx in range(num_epochs):
            epoch_start_time = time.time()
            # --- Training Phase for this epoch ---
            epoch_training_td_errors_agent0 = [] # Collect TD errors for this epoch's training
            
            for ep_idx_in_epoch in range(episodes_per_epoch):
                # Set current epsilon for all agents based on decay within this epoch
                current_epsilon_for_epoch_episode = max(epsilon_end, epsilon_start - epsilon_diff_per_episode_in_epoch * ep_idx_in_epoch)
                for alg in algorithms:
                    alg.set_epsilon(current_epsilon_for_epoch_episode)

                env_seed_train = trial.number * 10000 + epoch_idx * episodes_per_epoch + ep_idx_in_epoch # Unique map seed
                env = create_pogema_env(num_agents, map_size, obstacle_density, max_steps_per_episode, env_seed_train)
                
                current_observations, _ = env.reset(seed=env_seed_train)
                
                terminated_list = [False] * num_agents
                truncated_list = [False] * num_agents # Pogema also uses truncated
                
                for step_in_episode in range(max_steps_per_episode):
                    # 1. Get current states for action selection
                    current_local_states_list = [obs_to_state(obs) for obs in current_observations]
                    # Assuming obs_to_state(local_obs) can serve as global_state if needed by JALGT
                    current_global_state_for_jalgt = current_local_states_list[0] 

                    # 2. Select actions
                    actions_list = []
                    for i in range(num_agents):
                        if isinstance(algorithms[i], JALGT):
                            action = algorithms[i].select_action(current_global_state_for_jalgt, train=True)
                        else: # IQL or other
                            action = algorithms[i].select_action(current_local_states_list[i], train=True)
                        actions_list.append(action)
                    actions_tuple = tuple(actions_list)

                    # 3. Step environment
                    next_observations, step_rewards, terminated_list, truncated_list, _ = env.step(actions_tuple)
                    
                    # 4. Prepare states for learning
                    next_local_states_list = [obs_to_state(obs) for obs in next_observations]
                    next_global_state_for_jalgt = next_local_states_list[0] 
                    
                    experience_batch = {
                        'joint_action': actions_tuple,
                        'rewards': step_rewards, # Rewards from wrapper (potentially with time penalty)
                        'current_global_state': current_global_state_for_jalgt,
                        'next_global_state': next_global_state_for_jalgt,
                        'current_local_states': current_local_states_list,
                        'next_local_states': next_local_states_list,
                        'dones': terminated_list # Primary done signal for Q-learning target
                    }

                    # 5. Learn
                    for alg in algorithms:
                        alg.learn(experience_batch)
                    
                    # Collect TD error for agent 0 (example)
                    if algorithms[0].metrics["td_error"]:
                        epoch_training_td_errors_agent0.append(algorithms[0].metrics["td_error"][-1])
                        algorithms[0].metrics["td_error"].clear() # Clear after consuming to keep list small per step

                    current_observations = next_observations # For next step

                    if all(terminated_list) or all(truncated_list):
                        break
                env.close()
            
            if epoch_training_td_errors_agent0:
                trial_td_errors_per_epoch_agent0.append(np.mean(epoch_training_td_errors_agent0))
            else:
                trial_td_errors_per_epoch_agent0.append(0.0) # No training steps in epoch

            total_training_time_s += (time.time() - epoch_start_time)

            # --- Evaluation Phase for this epoch ---
            # (This uses new maps each time, like original main.py's evaluation section)
            num_eval_episodes = 10 # Fixed number of evaluation episodes
            epoch_eval_collective_rewards = 0
            epoch_eval_individual_rewards_agent0 = 0
            epoch_eval_individual_rewards_agent1 = 0

            for alg in algorithms: # Set to greedy for evaluation
                alg.set_epsilon(0.0)

            for ep_eval_idx in range(num_eval_episodes):
                eval_env_seed = trial.number * 1000 + epoch_idx * num_eval_episodes + ep_eval_idx + 50000 # Different seeds
                env = create_pogema_env(num_agents, map_size, obstacle_density, max_steps_per_episode, eval_env_seed)
                current_observations, _ = env.reset(seed=eval_env_seed)
                
                terminated_list = [False] * num_agents
                truncated_list = [False] * num_agents
                
                episode_collective_reward = 0
                episode_individual_rewards = [0.0] * num_agents

                for step_in_episode in range(max_steps_per_episode):
                    current_local_states_list = [obs_to_state(obs) for obs in current_observations]
                    current_global_state_for_jalgt = current_local_states_list[0]

                    actions_list = []
                    for i in range(num_agents):
                        if isinstance(algorithms[i], JALGT):
                            action = algorithms[i].select_action(current_global_state_for_jalgt, train=False)
                        else: # IQL or other
                            action = algorithms[i].select_action(current_local_states_list[i], train=False)
                        actions_list.append(action)
                    actions_tuple = tuple(actions_list)

                    next_observations, _, terminated_list, truncated_list, _ = env.step(actions_tuple)
                    # Use raw rewards from wrapper for evaluation metrics
                    raw_step_rewards = env.get_unmodified_rewards()
                    
                    for i in range(num_agents):
                        episode_individual_rewards[i] += raw_step_rewards[i]
                    episode_collective_reward += sum(raw_step_rewards)
                    
                    current_observations = next_observations
                    if all(terminated_list) or all(truncated_list):
                        break
                env.close()
                epoch_eval_collective_rewards += episode_collective_reward
                epoch_eval_individual_rewards_agent0 += episode_individual_rewards[0]
                if num_agents > 1:
                    epoch_eval_individual_rewards_agent1 += episode_individual_rewards[1]

            # Average rewards for this epoch's evaluation
            avg_epoch_eval_collective = epoch_eval_collective_rewards / num_eval_episodes
            trial_collective_rewards_per_epoch.append(avg_epoch_eval_collective)
            trial_individual_rewards_per_epoch_agent0.append(epoch_eval_individual_rewards_agent0 / num_eval_episodes)
            if num_agents > 1:
                trial_individual_rewards_per_epoch_agent1.append(epoch_eval_individual_rewards_agent1 / num_eval_episodes)
            else:
                trial_individual_rewards_per_epoch_agent1.append(0.0)


            # --- Optuna Pruning & Intermediate Reporting ---
            trial.report(avg_epoch_eval_collective, epoch_idx) # Report main metric for pruning
            if trial.should_prune():
                # Store partial metrics if needed before pruning
                trial.set_user_attr("pruned_at_epoch", epoch_idx)
                trial.set_user_attr("total_training_time_s_at_prune", total_training_time_s)
                raise optuna.exceptions.TrialPruned()

        # --- III. Log final metrics for this Optuna trial ---
        # Metrics to log (as per your list)
        final_collective_reward = np.mean(trial_collective_rewards_per_epoch[-max(1, num_epochs // 5):]) if trial_collective_rewards_per_epoch else 0.0
        
        trial.set_user_attr("total_training_episodes_ran", num_epochs * episodes_per_epoch)
        trial.set_user_attr("total_training_time_seconds", total_training_time_s)
        if num_epochs * episodes_per_epoch > 0:
            trial.set_user_attr("avg_time_per_training_episode_ms", (total_training_time_s / (num_epochs * episodes_per_epoch)) * 1000)
        else:
            trial.set_user_attr("avg_time_per_training_episode_ms", 0)
        
        trial.set_user_attr("final_avg_collective_reward_last_20pct_epochs", final_collective_reward)
        if trial_individual_rewards_per_epoch_agent0:
            trial.set_user_attr("final_avg_individual_reward_agent0", np.mean(trial_individual_rewards_per_epoch_agent0[-max(1, num_epochs // 5):]))
        if trial_individual_rewards_per_epoch_agent1:
            trial.set_user_attr("final_avg_individual_reward_agent1", np.mean(trial_individual_rewards_per_epoch_agent1[-max(1, num_epochs // 5):]))
        
        if trial_td_errors_per_epoch_agent0: # This is avg TD error per epoch for agent 0
            trial.set_user_attr("final_avg_td_error_agent0", np.mean(trial_td_errors_per_epoch_agent0[-max(1, num_epochs // 5):]))

        # Optimalidad de la política resultante (proxied by final collective reward)
        trial.set_user_attr("policy_optimality_proxy_collective_reward", final_collective_reward)

        return final_collective_reward # This is what Optuna will maximize
    
    except OverflowError as e:
        if "Can't create task" in str(e):
            print(f"Trial {trial.number} failed due to POGEMA map generation error: {e}. Pruning trial.")
            # Report a very bad value or raise Pruned.
            # Raising Pruned is better as Optuna knows this trial didn't "complete" with a bad score,
            # but rather was infeasible.
            raise optuna.exceptions.TrialPruned()
        else:
            raise # Re-raise other OverflowErrors

    except Exception as e: # Catch other potential errors during a trial
        print(f"Trial {trial.number} failed with an unexpected error: {e}")
        # Log the full traceback if needed for debugging
        import traceback
        traceback.print_exc()
        # Return a very bad score for other errors to penalize them
        return -float('inf') # Or raise optuna.exceptions.TrialPruned()


# --- Main HPO Execution ---
if __name__ == '__main__':
    study_name = "pogema_marl_hpo_simplified"
    # Optuna by default uses an in-memory storage if storage_name is None.
    # For longer runs, SQLite is good: storage_name = f"sqlite:///{study_name}.db"
    storage_name = None # In-memory for quick tests; use SQLite for actual runs
    
    # If you want to use the parallel execution like your qlearningstudy.py (N_SAMPLES_PER_TRIAL > 1)
    # you would wrap the call to objective() inside another function that uses ProcessPoolExecutor.
    # This current 'objective' function defines ONE Optuna trial = ONE full training run.

    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10, interval_steps=1)

    study = optuna.create_study(
        study_name=study_name,
        storage=storage_name,
        load_if_exists=True, # For SQLite: resumes study if DB exists
        direction="maximize",
        pruner=pruner
    )

    try:
        study.optimize(objective, n_trials=50, # Number of HPO trials
                       timeout=3600 * 1) # Example: 1 hour timeout
    except KeyboardInterrupt:
        print("Study optimization interrupted.")
    finally:
        print(f"\n--- Study {study.study_name} Summary ---")
        print(f"Number of finished trials: {len(study.trials)}")
        
        df_results = study.trials_dataframe(attrs=('number', 'value', 'params', 'user_attrs', 'state', 'duration'))
        
        # Save to CSV
        csv_file_path = f"{study_name}_results.csv"
        try:
            df_results.to_csv(csv_file_path, index=False)
            print(f"Results saved to {csv_file_path}")
        except Exception as e:
            print(f"Error saving results to CSV: {e}")

        if any(trial.state == optuna.trial.TrialState.COMPLETE for trial in study.trials):
            print("\nBest trial:")
            best_trial = study.best_trial
            print(f"  Trial Number: {best_trial.number}")
            print(f"  Value (Max Avg Collective Reward): {best_trial.value:.4f}")
            print("  Params: ")
            for key, value in best_trial.params.items():
                print(f"    {key}: {value}")
            print("  User Attributes for Best Trial: ")
            for key, value in best_trial.user_attrs.items():
                if isinstance(value, float): print(f"    {key}: {value:.4f}")
                else: print(f"    {key}: {value}")
        else:
            print("No trials completed successfully.")