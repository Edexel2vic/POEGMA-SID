import os
import time
import random
import numpy as np
from tqdm import tqdm # For progress bar
import optuna
import csv # For final CSV export

# Assuming algorithms.py contains the corrected IQL and JALGT (as per our previous versions)
from algorithms import JALGT, IQLAgent, MARLAlgorithm, ExperienceBatch
from solution_concepts import MinimaxSolutionConcept, ParetoSolutionConcept, NashSolutionConcept, WelfareSolutionConcept
from game_model import GameModel
from gymnasium import Wrapper
from pogema import pogema_v0, GridConfig
from pogema.animation import AnimationMonitor, AnimationConfig
# from utils import draw_history # Usually not called during HPO runs

# --- Helper: obs_to_state (from your original file) ---
def obs_to_state(obs):
    matrix_obstacles = obs[0]; matrix_agents = obs[1]; matrix_target = obs[2]
    target = np.max(matrix_target[2])*1 + matrix_target[1][0]*2 + matrix_target[1][2]*3
    obstacles = matrix_obstacles[0][1]*2**9 + matrix_obstacles[1][0]*2**8 + \
                matrix_obstacles[1][2]*2**7 + matrix_obstacles[2][1]*2**6
    agents_obs_part = matrix_agents[0][1]*2**5 + matrix_agents[1][0]*2**4 + \
                      matrix_agents[1][2]*2**3 + matrix_agents[2][1]*2**2
    return int(obstacles + agents_obs_part + target)

# --- Helper: RewardWrapper (from your original file) ---
class RewardWrapper(Wrapper):
    def __init__(self, env):
        super().__init__(env)
    def step(self, joint_action):
        observations, rewards, terminated, truncated, infos = self.env.step(joint_action)
        for i in range(len(joint_action)):
            if not terminated[i] and not truncated[i]:
                if rewards[i] == 0: rewards[i] -= 0.01
        return observations, rewards, terminated, truncated, infos

# --- Helper: create_env (adapted from your original file) ---
def create_pogema_env_for_hpo(num_agents_cfg, size_cfg, obstacle_density_cfg, episode_length_cfg,
                              env_seed, save_render_flag=False, render_dir='renders_optuna/', trial_num=0, epoch_num=0, eval_ep_num=0):
    render_mode_pogema = 'rgb_array' if save_render_flag else None
    grid_config = GridConfig(num_agents=num_agents_cfg, size=size_cfg, density=obstacle_density_cfg,
                             seed=env_seed, max_episode_steps=episode_length_cfg, obs_radius=1,
                             on_target="finish", render_mode=render_mode_pogema)
    env = pogema_v0(grid_config)
    if save_render_flag:
        trial_render_dir = os.path.join(render_dir, f"trial_{trial_num}")
        os.makedirs(trial_render_dir, exist_ok=True)
        # Save only specific eval episodes for the best trial (example: last epoch, first eval map)
        animation_config = AnimationConfig(directory=trial_render_dir, save_every_idx_episode=0, # Save if this ep is targeted
                                           filename_prefix=f"eval_epoch{epoch_num}_map{eval_ep_num}_seed{env_seed}")
        env = AnimationMonitor(env, animation_config=animation_config)
    return RewardWrapper(env)

# --- Optuna Objective Function ---
def objective(trial: optuna.trial.Trial):
    # --- I. Suggest Hyperparameters ---
    # Algorithm Choice - Let's assume this is fixed per student run for now
    # For a combined run, you would use:
    # algorithm_name = trial.suggest_categorical("algorithm_name", ["JALGT", "IQL"])
    algorithm_name = "JALGT" # Hardcode for this student's run (e.g., JALGT specialist)
    # algorithm_name = "IQL" # Hardcode for IQL specialist's run
    trial.set_user_attr("algorithm_name_fixed", algorithm_name) # Log which one is fixed

    # Shared Env/Setup
    num_agents = 2 # Fixed per assignment spec
    map_size = trial.suggest_int("map_size", 6, 10) # Min 6x6 for 2 agents
    obstacle_density = trial.suggest_float("obstacle_density", 0.0, 0.3) # Max 30% for smaller maps

    # Training Structure (like original main.py)
    num_epochs = trial.suggest_int("num_epochs", 30, 150) # Outer loops (e.g., 30 to 150)
    episodes_per_epoch = trial.suggest_int("episodes_per_epoch", 10, 30) # Training eps per outer loop
    max_steps_per_episode = trial.suggest_int("max_steps_per_episode", 40, 150) # t_max

    # Learning Parameters
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 0.1, log=True)
    gamma_val = trial.suggest_float("gamma", 0.90, 0.995)

    # Epsilon Parameters
    epsilon_start = trial.suggest_float("epsilon_start", 0.7, 1.0)
    epsilon_end = trial.suggest_float("epsilon_end", 0.01, 0.15)
    if epsilon_end >= epsilon_start: epsilon_end = epsilon_start * 0.1

    # JALGT-Specific (only if algorithm_name is JALGT)
    solution_concept_name_for_jalgt = "Nash" # Default if not tuning this for JALGT
    if algorithm_name == "JALGT":
        solution_concept_name_for_jalgt = trial.suggest_categorical("solution_concept", ["Nash", "Pareto", "Welfare"])

    # Fixed values based on POGEMA and obs_to_state
    num_states_for_q_tables = 1024
    num_individual_actions = 5

    # Create GameModel (for JALGT)
    game_model_instance = GameModel(num_agents=num_agents, num_states=num_states_for_q_tables, num_actions=num_individual_actions)

    # Instantiate Algorithms
    algorithms: list[MARLAlgorithm] = []
    unique_seed_offset = trial.number * num_agents # Base seed for this trial's agents
    if algorithm_name == "JALGT":
        sc_map = {"Nash": NashSolutionConcept, "Pareto": ParetoSolutionConcept, "Welfare": WelfareSolutionConcept}
        solution_concept_class = sc_map[solution_concept_name_for_jalgt]
        for i in range(num_agents):
            algorithms.append(JALGT(agent_id=i, game=game_model_instance, solution_concept=solution_concept_class(),
                                    gamma=gamma_val, alpha=learning_rate, epsilon=epsilon_start,
                                    seed=unique_seed_offset + i))
    elif algorithm_name == "IQL":
        for i in range(num_agents):
            algorithms.append(IQLAgent(agent_id=i, num_local_states=num_states_for_q_tables,
                                       num_individual_actions=num_individual_actions,
                                       gamma=gamma_val, alpha=learning_rate,
                                       epsilon_start=epsilon_start, epsilon_end=epsilon_end,
                                       seed=unique_seed_offset + i))
    
    # --- II. Training and Evaluation Loop ---
    # Metrics for this Optuna trial
    trial_metrics = {
        "collective_rewards_per_epoch": [], "indiv_rewards_agent0_per_epoch": [],
        "indiv_rewards_agent1_per_epoch": [], "td_errors_agent0_per_epoch": [],
        "total_training_time_s": 0.0, "total_steps_trained": 0
    }
    
    # Epsilon decay (linear over episodes_per_epoch, like original main.py)
    epsilon_decay_step = (epsilon_start - epsilon_end) / episodes_per_epoch if episodes_per_epoch > 0 else 0

    # TQDM Progress bar for epochs
    epoch_pbar = tqdm(range(num_epochs), desc=f"Trial {trial.number} ({algorithm_name})", unit="epoch", leave=False)

    try: # Main try-except for the whole trial processing
        for epoch_idx in epoch_pbar:
            epoch_start_time = time.time()
            current_epoch_td_errors_agent0 = []

            # --- Training Phase ---
            for ep_idx_in_epoch in range(episodes_per_epoch):
                current_epsilon = max(epsilon_end, epsilon_start - epsilon_decay_step * ep_idx_in_epoch)
                for alg in algorithms: alg.set_epsilon(current_epsilon)

                train_env_seed = trial.number * 10000 + epoch_idx * 100 + ep_idx_in_epoch # Unique map seed
                env = create_pogema_env_for_hpo(num_agents, map_size, obstacle_density, max_steps_per_episode, train_env_seed)
                
                current_observations, _ = env.reset(seed=train_env_seed) # This can raise POGEMA OverflowError
                
                for step_in_ep in range(max_steps_per_episode):
                    trial_metrics["total_steps_trained"] += 1
                    current_local_s_list = [obs_to_state(obs) for obs in current_observations]
                    current_global_s_for_jalgt = current_local_s_list[0]

                    actions_list = []
                    for i in range(num_agents):
                        agent_state = current_global_s_for_jalgt if isinstance(algorithms[i], JALGT) else current_local_s_list[i]
                        actions_list.append(algorithms[i].select_action(agent_state, train=True))
                    actions_tuple = tuple(actions_list)

                    next_obs, step_rewards_wrapper, terminated, truncated, _ = env.step(actions_tuple)
                    
                    next_local_s_list = [obs_to_state(obs) for obs in next_obs]
                    next_global_s_for_jalgt = next_local_s_list[0]
                    
                    experience = ExperienceBatch({
                        'joint_action': actions_tuple, 'rewards': step_rewards_wrapper,
                        'current_global_state': current_global_s_for_jalgt, 'next_global_state': next_global_s_for_jalgt,
                        'current_local_states': current_local_s_list, 'next_local_states': next_local_s_list,
                        'dones': terminated }) # Use `terminated` as the primary done signal

                    for alg in algorithms: alg.learn(experience)
                    
                    if algorithms[0].metrics["td_error"]: # Agent 0 TD error
                        current_epoch_td_errors_agent0.append(algorithms[0].metrics["td_error"][-1])
                        algorithms[0].metrics["td_error"].clear()

                    current_observations = next_obs
                    if all(terminated) or all(truncated): break
                env.close()
            
            trial_metrics["total_training_time_s"] += (time.time() - epoch_start_time)
            trial_metrics["td_errors_agent0_per_epoch"].append(np.mean(current_epoch_td_errors_agent0) if current_epoch_td_errors_agent0 else 0.0)

            # --- Evaluation Phase ---
            num_eval_maps = 10 # Fixed, as in original for consistency
            epoch_eval_collective_sum, epoch_eval_indiv0_sum, epoch_eval_indiv1_sum = 0.0, 0.0, 0.0
            for alg in algorithms: alg.set_epsilon(0.0) # Greedy

            for eval_ep_idx in range(num_eval_maps):
                eval_env_seed = trial.number * 1000 + epoch_idx * 100 + eval_ep_idx + 70000 # Unique eval map seeds
                # save_render = (trial.number == study.best_trial.number and epoch_idx == num_epochs -1 and eval_ep_idx == 0) if study and study.best_trial else False # For best trial
                env = create_pogema_env_for_hpo(num_agents, map_size, obstacle_density, max_steps_per_episode, eval_env_seed, save_render_flag=False) # No rendering during HPO
                current_observations, _ = env.reset(seed=eval_env_seed)
                ep_collective_r, ep_indiv_r = 0.0, [0.0]*num_agents

                for _ in range(max_steps_per_episode):
                    current_local_s_list = [obs_to_state(obs) for obs in current_observations]
                    current_global_s_for_jalgt = current_local_s_list[0]
                    actions_list = []
                    for i in range(num_agents):
                        agent_state = current_global_s_for_jalgt if isinstance(algorithms[i], JALGT) else current_local_s_list[i]
                        actions_list.append(algorithms[i].select_action(agent_state, train=False))
                    actions_tuple = tuple(actions_list)
                    
                    next_obs, eval_step_rewards_wrapper, terminated, truncated, _ = env.step(actions_tuple)
                    # Use rewards from wrapper directly for evaluation (to match original main.py)
                    for i in range(num_agents): ep_indiv_r[i] += eval_step_rewards_wrapper[i]
                    ep_collective_r += sum(eval_step_rewards_wrapper)
                    current_observations = next_obs
                    if all(terminated) or all(truncated): break
                env.close()
                epoch_eval_collective_sum += ep_collective_r
                epoch_eval_indiv0_sum += ep_indiv_r[0]
                if num_agents > 1: epoch_eval_indiv1_sum += ep_indiv_r[1]
            
            avg_epoch_eval_collective = epoch_eval_collective_sum / num_eval_maps if num_eval_maps > 0 else 0.0
            trial_metrics["collective_rewards_per_epoch"].append(avg_epoch_eval_collective)
            trial_metrics["indiv_rewards_agent0_per_epoch"].append(epoch_eval_indiv0_sum / num_eval_maps if num_eval_maps > 0 else 0.0)
            trial_metrics["indiv_rewards_agent1_per_epoch"].append(epoch_eval_indiv1_sum / num_eval_maps if num_eval_maps > 0 else 0.0)
            
            epoch_pbar.set_description(f"T{trial.number}({algorithm_name}) E{epoch_idx+1} [EvalR: {avg_epoch_eval_collective:.2f}]")
            trial.report(avg_epoch_eval_collective, epoch_idx)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

    except optuna.exceptions.TrialPruned as e_pruned:
        epoch_pbar.close() # Ensure progress bar is closed
        trial.set_user_attr("status_detail", "Pruned by Optuna or POGEMA error")
        raise # Re-raise to Optuna to mark as PRUNED
    except OverflowError as e_overflow: # Catch POGEMA map generation error
        epoch_pbar.close()
        if "Can't create task" in str(e_overflow):
            trial.set_user_attr("status_detail", f"POGEMA map error: {e_overflow}")
            raise optuna.exceptions.TrialPruned(f"POGEMA map generation failed: {e_overflow}")
        else: # Other OverflowError
            trial.set_user_attr("status_detail", f"Other OverflowError: {e_overflow}")
            raise optuna.exceptions.TrialPruned(f"Unexpected OverflowError: {e_overflow}")
    except Exception as e_generic: # Catch any other unhandled errors
        epoch_pbar.close()
        import traceback
        tb_str = traceback.format_exc()
        trial.set_user_attr("status_detail", f"Generic error: {type(e_generic).__name__} - {e_generic}")
        trial.set_user_attr("traceback", tb_str[:1000]) # Log part of traceback
        # print(f"Trial {trial.number} FAILED with generic error:\n{tb_str}") # For console
        raise optuna.exceptions.TrialPruned(f"Generic unhandled error: {type(e_generic).__name__}") # Prune on any error

    epoch_pbar.close() # Ensure progress bar is closed on successful completion of all epochs

    # --- III. Log final metrics for this Optuna trial ---
    num_meaningful_epochs = max(1, num_epochs // 5) # Average over last 20% or at least 1
    final_collective_reward = np.mean(trial_metrics["collective_rewards_per_epoch"][-num_meaningful_epochs:]) if trial_metrics["collective_rewards_per_epoch"] else 0.0
    
    trial.set_user_attr("total_training_episodes_ran", num_epochs * episodes_per_epoch)
    trial.set_user_attr("total_training_time_seconds", trial_metrics["total_training_time_s"])
    if num_epochs * episodes_per_epoch > 0:
        avg_time_per_ep_ms = (trial_metrics["total_training_time_s"] / (num_epochs * episodes_per_epoch)) * 1000
        trial.set_user_attr("avg_time_per_training_episode_ms", avg_time_per_ep_ms)
    
    trial.set_user_attr("final_avg_collective_reward_last_20pct", final_collective_reward)
    if trial_metrics["indiv_rewards_agent0_per_epoch"]:
        trial.set_user_attr("final_avg_indiv_reward_agent0_last_20pct", np.mean(trial_metrics["indiv_rewards_agent0_per_epoch"][-num_meaningful_epochs:]))
    if trial_metrics["indiv_rewards_agent1_per_epoch"]:
        trial.set_user_attr("final_avg_indiv_reward_agent1_last_20pct", np.mean(trial_metrics["indiv_rewards_agent1_per_epoch"][-num_meaningful_epochs:]))
    if trial_metrics["td_errors_agent0_per_epoch"]:
        trial.set_user_attr("final_avg_td_error_agent0_last_20pct", np.mean(trial_metrics["td_errors_agent0_per_epoch"][-num_meaningful_epochs:]))
    trial.set_user_attr("policy_optimality_proxy", final_collective_reward) # Proxy for optimality

    return final_collective_reward

# --- Main HPO Execution ---
if __name__ == '__main__':
    study_name = "pogema_marl_hpo_v_simple" # Changed study name
    # Use SQLite for persistence across runs
    storage_path = f"sqlite:///{study_name}.db"
    # To reset study, delete the .db file: if os.path.exists(f"{study_name}.db"): os.remove(f"{study_name}.db")

    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10, interval_steps=1)

    study = optuna.create_study(
        study_name=study_name, storage=storage_path, load_if_exists=True,
        direction="maximize", pruner=pruner )

    n_optuna_trials = 50 # Number of HPO trials to run
    timeout_seconds = 3600 * 2 # e.g., 2 hours
    
    start_hpo_time = time.time()
    try:
        study.optimize(objective, n_trials=n_optuna_trials, timeout=timeout_seconds)
    except KeyboardInterrupt:
        print("Study optimization interrupted by user.")
    finally:
        hpo_duration = time.time() - start_hpo_time
        print(f"\n--- Study '{study.study_name}' Summary (Duration: {hpo_duration:.2f}s) ---")
        print(f"Number of finished trials: {len(study.trials)}")
        
        # Filter trials for analysis (e.g., only COMPLETE or PRUNED by pruner)
        finished_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE or \
                                                       (t.state == optuna.trial.TrialState.PRUNED and \
                                                        not (t.user_attrs.get("status_detail","").startswith("POGEMA map error") or \
                                                             t.user_attrs.get("status_detail","").startswith("Generic error") ) ) ]
        
        print(f"Number of trials considered 'finished' (COMPLETE or PRUNED by median pruner): {len(finished_trials)}")

        df_results = study.trials_dataframe(attrs=('number', 'value', 'params', 'user_attrs', 'state', 'duration'))
        csv_file_path = f"out/{study_name}_all_trials.csv"
        try:
            df_results.to_csv(csv_file_path, index=False)
            print(f"All trial results saved to {csv_file_path}")
        except Exception as e: print(f"Error saving all trial results to CSV: {e}")

        if finished_trials:
            # Find best trial among those not pruned due to critical errors
            # Optuna's study.best_trial considers only COMPLETE trials by default.
            # If all are pruned, study.best_trial might raise an error.
            try:
                best_trial_overall = study.best_trial
                print("\nBest trial overall (from Optuna, considers only COMPLETE):")
                print(f"  Trial Number: {best_trial_overall.number}, Value: {best_trial_overall.value:.4f}")
                print("  Params:", best_trial_overall.params)
                print("  User Attributes:", {k: (f"{v:.4f}" if isinstance(v,float) else v) for k,v in best_trial_overall.user_attrs.items()})
            except ValueError: # No best trial if all failed or pruned
                print("No successfully completed trials found by Optuna to determine a 'best_trial'.")
        else:
            print("No trials finished (neither COMPLETE nor PRUNED by median pruner).")