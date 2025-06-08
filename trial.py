import os
import time
import random # For top-level seed generation
import numpy as np
from tqdm import tqdm
import optuna # Make sure to import optuna

from algorithms import JALGT # Assuming IQL might be added later
from solution_concepts import MinimaxSolutionConcept, ParetoSolutionConcept, NashSolutionConcept, WelfareSolutionConcept
from game_model import GameModel
from gymnasium import Wrapper # Corrected import
from pogema import pogema_v0, GridConfig
from pogema.animation import AnimationMonitor, AnimationConfig
from utils import draw_history # You might not use this directly in the HPO loop

# --- Helper: obs_to_state (keep as is) ---
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
    agents = matrix_agents[0][1] * 2 ** 5 + \
             matrix_agents[1][0] * 2 ** 4 + \
             matrix_agents[1][2] * 2 ** 3 + \
             matrix_agents[2][1] * 2 ** 2
    return int(obstacles + agents + target)

# --- Helper: RewardWrapper (keep as is) ---
class RewardWrapper(Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.current_rewards_for_step = []

    def step(self, joint_action):
        previous_observations = self.env.unwrapped._obs()
        observations, rewards, terminated, truncated, infos = self.env.step(joint_action)
        self.current_rewards_for_step = list(rewards) # Store raw rewards before modification

        modified_rewards = list(rewards) # Create a mutable copy
        for i in range(len(joint_action)):
            if not terminated[i] and not truncated[i]:
                if modified_rewards[i] == 0:
                    modified_rewards[i] = modified_rewards[i] - 0.01
        return observations, modified_rewards, terminated, truncated, infos

    def get_unmodified_rewards(self):
        return self.current_rewards_for_step


# --- Helper: create_env (modified slightly for clarity) ---
def create_env_pogema(num_agents_cfg, size_cfg, obstacle_density_cfg, episode_length_cfg,
                  env_seed, save_renders_for_episode_num=None, render_dir='renders/'):
    grid_config = GridConfig(num_agents=num_agents_cfg,
                             size=size_cfg,
                             density=obstacle_density_cfg,
                             seed=env_seed,
                             max_episode_steps=episode_length_cfg,
                             obs_radius=1, # Fixed for current obs_to_state
                             on_target="finish",
                             render_mode='rgb_array' if save_renders_for_episode_num is not None else None) # 'rgb_array' for AnimationMonitor

    env = pogema_v0(grid_config)
    if save_renders_for_episode_num is not None:
        if not os.path.exists(render_dir):
            os.makedirs(render_dir, exist_ok=True)
        animation_config = AnimationConfig(directory=render_dir,
                                           save_every_idx_episode=0, # Save for the one episode we specify
                                           filename_prefix=f"map{env_seed}_ep{save_renders_for_episode_num}")
        env = AnimationMonitor(env, animation_config=animation_config)
    return RewardWrapper(env)


# --- Core Training and Evaluation Function (for one full run/seed) ---
def run_single_training_evaluation_cycle(trial_config, run_seed, optuna_trial_object=None, report_step_offset=0):
    """
    Runs a complete training and evaluation cycle for a given configuration and seed.
    optuna_trial_object and report_step_offset are for potential intermediate reporting to Optuna.
    """
    # --- Setup based on trial_config and run_seed ---
    num_agents = trial_config["num_agents"]
    game = GameModel(num_agents=num_agents,
                     num_states=trial_config["num_states"], # Assumes obs_radius=1
                     num_actions=5)
    
    solution_concept_instance = trial_config["solution_concept_class"]()
    
    algorithms = [JALGT(agent_id, game, solution_concept_instance,
                        gamma=trial_config["gamma"],
                        alpha=trial_config["learning_rate"],
                        epsilon=trial_config["epsilon_start"], # Epsilon starts high for each run
                        seed=run_seed + agent_id) # Vary seed for algorithm initialization
                  for agent_id in range(num_agents)]

    # Epsilon decay parameters
    epsilon_decay_per_training_episode = 0
    if trial_config["total_training_episodes"] > 0:
        epsilon_decay_per_training_episode = \
            (trial_config["epsilon_start"] - trial_config["epsilon_end"]) / trial_config["total_training_episodes"]

    # Metrics for this single run
    history_collective_eval_rewards = []
    history_avg_td_error_per_eval_cycle = [] # TD error from training preceding an eval
    total_training_time_this_run = 0.0
    current_total_training_episode_count = 0

    # --- Training and Evaluation Loop (structured by evaluation_interval) ---
    num_evaluation_cycles = trial_config["total_training_episodes"] // trial_config["episodes_per_evaluation"]
    if trial_config["total_training_episodes"] % trial_config["episodes_per_evaluation"] != 0:
        num_evaluation_cycles +=1 # Ensure last episodes are also evaluated if not a perfect multiple

    for eval_cycle_idx in range(num_evaluation_cycles):
        # --- Training Phase ---
        training_phase_start_time = time.time()
        current_cycle_td_errors = []
        for ep_train_idx in range(trial_config["episodes_per_evaluation"]):
            if current_total_training_episode_count >= trial_config["total_training_episodes"]:
                break # Stop if total desired training episodes reached

            current_epsilon = max(trial_config["epsilon_end"],
                                  trial_config["epsilon_start"] - epsilon_decay_per_training_episode * current_total_training_episode_count)
            for alg in algorithms:
                alg.set_epsilon(current_epsilon)

            # Use a different map seed for each training episode to encourage generalization
            # You could cycle through a smaller set of map seeds if preferred.
            train_env_seed = run_seed * 1000 + current_total_training_episode_count # Make unique env seed
            
            env = create_env_pogema(num_agents_cfg=num_agents,
                                size_cfg=trial_config["size"],
                                obstacle_density_cfg=trial_config["obstacle_density"],
                                episode_length_cfg=trial_config["max_steps_per_episode"],
                                env_seed=train_env_seed)
            observations, _ = env.reset(seed=train_env_seed) # Pass seed to reset too
            
            states = [obs_to_state(obs) for obs in observations]
            terminated = [False] * num_agents
            truncated = [False] * num_agents
            
            episode_steps = 0
            while not (all(terminated) or all(truncated)) and episode_steps < trial_config["max_steps_per_episode"]:
                actions = tuple(alg.select_action(states[idx], train=True) for idx, alg in enumerate(algorithms))
                next_observations, rewards, terminated, truncated, _ = env.step(actions)
                next_states = [obs_to_state(obs) for obs in next_observations]

                # Learn call - assumes JALGT.learn takes current states and next_states
                # The current JALGT learn takes state[i] but uses a global next_state (from obs_to_state(observations[i]))
                # This needs to be consistent. Let's assume JALGT handles its state indexing.
                # A common pattern for JALGT might be for the learn function to take obs and next_obs
                # and internally call obs_to_state.
                # For simplicity, we stick to the provided JALGT interface.
                # JALGT.learn expects state (int) and next_state (int) PER AGENT
                # So the call inside the loop below is more appropriate
                # But the state update for "states = next_states" should be outside the agent loop.

                raw_rewards_this_step = env.get_unmodified_rewards() # Get rewards before wrapper modification

                for agent_idx in range(num_agents):
                    # JALGT.learn(self, joint_action, rewards, state, next_state)
                    # The JALGT learn signature expects a single state and next_state, implying agent's own.
                    # However, it processes for all agents internally using its agent_id.
                    # This is a bit unusual. Let's assume the first agent's TD error is representative for now.
                    # Or JALGT.learn should only be called once with the full joint action and rewards.
                    # The current JALGT learn loop over agent_id inside is fine.
                    pass # The learn call will be done once after actions

                # Corrected learn call (one call for the joint experience)
                # The JALGT `learn` method internally loops through agents.
                # It needs the *current state* of each agent and the *next state* of each agent.
                # The provided `JALGT.learn` is: learn(self, joint_action, rewards, state, next_state)
                # This implies it's called for *each agent* with *its own state and next_state*.
                # This means the loop for algorithms.learn is correct in the original main.py.
                
                # Let's stick to the original main.py's way of calling learn for each agent
                temp_td_errors_this_step = []
                for agent_idx in range(num_agents):
                    # The JALGT.learn takes individual state and next_state for the Q-table update
                    # The Q-table itself is q_table[agent_id][state_index_for_q_table][joint_action_index]
                    # The `state` argument to `learn` should be the state used for Q-table lookup for that agent.
                    # If using global state, all agents see the same global state.
                    # If using local states for Q-table, then states[agent_idx] is correct.
                    # Your JALGT has q_table[agent_id][STATE_INDEX_GLOBAL_OR_LOCAL][...]
                    # The current obs_to_state creates a single integer, likely intended as a global state representation
                    # or a local one if each agent gets a different obs.
                    # Pogema gives local obs. obs_to_state converts local obs to an int.
                    # So states[agent_idx] is the integer representation of agent_idx's local observation.

                    algorithms[agent_idx].learn(actions, rewards, states[agent_idx], next_states[agent_idx])
                    if algorithms[agent_idx].metrics["td_error"]: # Check if list is not empty
                        temp_td_errors_this_step.append(algorithms[agent_idx].metrics["td_error"][-1])

                if temp_td_errors_this_step:
                    current_cycle_td_errors.append(np.mean(temp_td_errors_this_step))

                states = next_states
                episode_steps += 1
            env.close()
            current_total_training_episode_count += 1
        
        training_phase_end_time = time.time()
        total_training_time_this_run += (training_phase_end_time - training_phase_start_time)
        
        if current_cycle_td_errors:
            history_avg_td_error_per_eval_cycle.append(np.mean(current_cycle_td_errors))
        else:
            history_avg_td_error_per_eval_cycle.append(0) # Or some other placeholder if no training happened

        # --- Evaluation Phase ---
        collective_eval_reward_this_cycle = 0
        num_eval_maps = trial_config["num_eval_maps"] # e.g., 10 maps for evaluation
        
        # Make sure algorithms are in evaluation mode (epsilon=0)
        for alg in algorithms:
            alg.set_epsilon(0) # Greedypolicy for evaluation

        for ep_eval_idx in range(num_eval_maps):
            eval_env_seed = run_seed * 100 + ep_eval_idx # Different set of seeds for eval maps
            env = create_env_pogema(num_agents_cfg=num_agents,
                                size_cfg=trial_config["size"],
                                obstacle_density_cfg=trial_config["obstacle_density"],
                                episode_length_cfg=trial_config["max_steps_per_episode"],
                                env_seed=eval_env_seed,
                                save_renders_for_episode_num=(eval_env_seed if trial_config.get("save_final_renders", False) and eval_cycle_idx == num_evaluation_cycles -1 else None),
                                render_dir=trial_config.get("render_dir", "renders_final/"))
            observations, _ = env.reset(seed=eval_env_seed)
            states = [obs_to_state(obs) for obs in observations]
            terminated = [False] * num_agents
            truncated = [False] * num_agents
            
            episode_collective_reward = 0
            episode_steps = 0
            while not (all(terminated) or all(truncated)) and episode_steps < trial_config["max_steps_per_episode"]:
                actions = tuple(alg.select_action(states[idx], train=False) for idx, alg in enumerate(algorithms))
                next_observations, rewards, terminated, truncated, _ = env.step(actions)
                next_states = [obs_to_state(obs) for obs in next_observations]
                
                episode_collective_reward += sum(env.get_unmodified_rewards()) # Sum of raw rewards
                states = next_states
                episode_steps += 1
            collective_eval_reward_this_cycle += episode_collective_reward
            env.close()
        
        avg_collective_eval_reward_this_cycle = collective_eval_reward_this_cycle / num_eval_maps
        history_collective_eval_rewards.append(avg_collective_eval_reward_this_cycle)

        # --- Optional: Report to Optuna for pruning ---
        if optuna_trial_object:
            # Report the average collective reward of this evaluation cycle
            optuna_trial_object.report(avg_collective_eval_reward_this_cycle, step=report_step_offset + eval_cycle_idx)
            if optuna_trial_object.should_prune():
                # Clean up if needed
                raise optuna.exceptions.TrialPruned()
    
    # --- Return final metrics for this single run ---
    final_avg_collective_reward = np.mean(history_collective_eval_rewards[-max(1, num_evaluation_cycles // 5):]) # Avg of last 20% eval cycles
    # Or just the very last one: history_collective_eval_rewards[-1] if list is not empty else 0

    metrics = {
        "final_avg_collective_reward": final_avg_collective_reward if history_collective_eval_rewards else 0,
        "avg_td_error_overall": np.mean(history_avg_td_error_per_eval_cycle) if history_avg_td_error_per_eval_cycle else 0,
        "total_training_time_s": total_training_time_this_run,
        "history_collective_eval_rewards": history_collective_eval_rewards # For plotting learning curve of this run
    }
    return metrics


# --- Optuna Objective Function ---
def objective(trial: optuna.trial.Trial):
    # --- I. Suggest Hyperparameters ---
    # Basic env params
    num_agents = trial.suggest_int("num_agents", 2, 2) # Keep it 2 for JALGT complexity for now
    map_size = trial.suggest_int("map_size", 4, 6)
    obstacle_density = trial.suggest_float("obstacle_density", 0.05, 0.3)
    max_steps_per_episode = trial.suggest_int("max_steps_per_episode", 20, 60)

    # Training length / structure
    total_training_episodes = trial.suggest_int("total_training_episodes", 200, 1000) # Total learning experiences
    episodes_per_evaluation = trial.suggest_int("episodes_per_evaluation", 20, 100) # How often to evaluate

    # Algorithm params
    solution_concept_name = trial.suggest_categorical("solution_concept", ["Nash", "Pareto", "Welfare"]) # Minimax can be slow/complex
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-1, log=True)
    gamma = trial.suggest_float("gamma", 0.90, 0.995)
    epsilon_start = trial.suggest_float("epsilon_start", 0.7, 1.0)
    epsilon_end = trial.suggest_float("epsilon_end", 0.01, 0.1)
    if epsilon_end >= epsilon_start: # Ensure end is less than start
        epsilon_end = epsilon_start * 0.1 

    # Map solution concept string to class
    sc_map = {"Nash": NashSolutionConcept, "Pareto": ParetoSolutionConcept, "Welfare": WelfareSolutionConcept}
    solution_concept_class = sc_map[solution_concept_name]

    # Fixed params for now
    num_states_fixed_obs_radius_1 = 1024 # 4 (target) * 16 (obstacles) * 16 (agents)
    num_eval_maps_fixed = 5 # Number of maps to average over for evaluation

    trial_config = {
        "num_agents": num_agents,
        "size": map_size,
        "obstacle_density": obstacle_density,
        "max_steps_per_episode": max_steps_per_episode,
        "total_training_episodes": total_training_episodes,
        "episodes_per_evaluation": episodes_per_evaluation,
        "solution_concept_class": solution_concept_class,
        "learning_rate": learning_rate,
        "gamma": gamma,
        "epsilon_start": epsilon_start,
        "epsilon_end": epsilon_end,
        "num_states": num_states_fixed_obs_radius_1,
        "num_eval_maps": num_eval_maps_fixed,
        # "save_final_renders": False, # Set to True for best trial run
        # "render_dir": f"optuna_renders/trial_{trial.number}/"
    }

    # --- II. Run multiple independent short runs for robustness ---
    n_seeds_per_trial = 3 # Number of different seeds to average over for this HPO trial
    all_runs_final_rewards = []
    all_runs_total_time = 0
    all_runs_avg_td_errors = []
    
    # Store detailed learning curves for each seed if needed for later analysis,
    # but for Optuna's primary objective, we'll average.
    # trial_learning_curves = [] 

    for i in range(n_seeds_per_trial):
        run_seed = trial.number * n_seeds_per_trial + i # Unique seed for each short run
        # Pass optuna_trial and an offset for step if using pruning within run_single_training_evaluation_cycle
        # report_offset = i * (total_training_episodes // episodes_per_evaluation +1)
        try:
            # If not using pruning within run_single_training_evaluation_cycle, optuna_trial_object can be None
            run_metrics = run_single_training_evaluation_cycle(trial_config, run_seed,
                                                               optuna_trial_object=None) # Pass 'trial' for pruning
            all_runs_final_rewards.append(run_metrics["final_avg_collective_reward"])
            all_runs_total_time += run_metrics["total_training_time_s"]
            all_runs_avg_td_errors.append(run_metrics["avg_td_error_overall"])
            # trial_learning_curves.append(run_metrics["history_collective_eval_rewards"])

        except optuna.exceptions.TrialPruned:
            # If run_single_training_evaluation_cycle itself prunes, re-raise
            raise
        except Exception as e:
            print(f"Error in trial {trial.number}, seed run {i}: {e}")
            all_runs_final_rewards.append(-float('inf')) # Penalize errors heavily
            # Or re-raise the exception if Optuna should handle it as a fail.
            # For now, just record a very bad score.

    # --- III. Aggregate results and set user attributes ---
    avg_final_reward_for_trial = np.mean(all_runs_final_rewards) if all_runs_final_rewards else -float('inf')
    trial.set_user_attr("avg_final_collective_reward_across_seeds", avg_final_reward_for_trial)
    trial.set_user_attr("std_final_collective_reward_across_seeds", np.std(all_runs_final_rewards) if all_runs_final_rewards else 0)
    trial.set_user_attr("avg_total_training_time_s_across_seeds", all_runs_total_time / n_seeds_per_trial if n_seeds_per_trial > 0 else 0)
    trial.set_user_attr("avg_td_error_across_seeds", np.mean(all_runs_avg_td_errors) if all_runs_avg_td_errors else 0)
    # trial.set_user_attr("all_seed_learning_curves", trial_learning_curves) # Can make DB large

    # The main objective Optuna will try to maximize
    return avg_final_reward_for_trial


# --- Main HPO Execution ---
if __name__ == '__main__':
    study_name = "pogema_jalgt_hpo_v2"
    storage_name = f"sqlite:///{study_name}.db"

    # Ensure render directory for final (best trial) renders exists
    # best_trial_render_dir = "renders_best_trial/"
    # os.makedirs(best_trial_render_dir, exist_ok=True)

    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, # Prune after this many initial trials
                                         n_warmup_steps=2,   # Prune after this many intermediate reports (eval_cycle_idx)
                                         interval_steps=1)    # Prune every intermediate report

    study = optuna.create_study(
        study_name=study_name,
        storage=storage_name,
        load_if_exists=True,
        direction="maximize",
        # pruner=pruner # Add pruner if run_single_training_evaluation_cycle reports to optuna_trial_object
    )

    try:
        # Adjust n_trials or timeout as needed
        study.optimize(objective, n_trials=100, timeout=3600 * 8) # e.g., 100 trials or 8 hours
    except KeyboardInterrupt:
        print("Study optimization interrupted by user.")
    finally:
        print("\n--- Study Summary ---")
        print(f"Study Name: {study.study_name}")
        print(f"Number of finished trials: {len(study.trials)}")
        
        pruned_trials = study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.PRUNED])
        complete_trials = study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.COMPLETE])
        print(f"Number of pruned trials: {len(pruned_trials)}")
        print(f"Number of complete trials: {len(complete_trials)}")

        if complete_trials: # Check if there's a best trial
            print("\nBest trial:")
            best_trial = study.best_trial
            print(f"  Value (Avg Collective Reward): {best_trial.value:.4f}")
            print("  Params: ")
            for key, value in best_trial.params.items():
                print(f"    {key}: {value}")
            print("  User Attributes for Best Trial: ")
            for key, value in best_trial.user_attrs.items():
                if isinstance(value, float):
                    print(f"    {key}: {value:.4f}")
                else:
                    print(f"    {key}: {value}")
            
            # --- Optional: Run the best trial again with rendering and for longer ---
            print("\n--- Re-running best trial for visualization (example) ---")
            best_params_config = best_trial.params.copy() # Get a copy of the best hyperparameters
            
            # Construct the full trial_config for the best run
            # This mirrors the structure inside the objective function
            final_run_config = {
                "num_agents": best_params_config["num_agents"],
                "size": best_params_config["map_size"],
                "obstacle_density": best_params_config["obstacle_density"],
                "max_steps_per_episode": best_params_config["max_steps_per_episode"],
                "total_training_episodes": best_params_config.get("total_training_episodes_final_run", 2000), # Longer run
                "episodes_per_evaluation": best_params_config.get("episodes_per_evaluation_final_run", 50),
                "solution_concept_class": sc_map[best_params_config["solution_concept"]],
                "learning_rate": best_params_config["learning_rate"],
                "gamma": best_params_config["gamma"],
                "epsilon_start": best_params_config["epsilon_start"],
                "epsilon_end": best_params_config["epsilon_end"],
                "num_states": 1024, # Fixed
                "num_eval_maps": 10, # More eval maps for final
                "save_final_renders": True,
                "render_dir": f"renders_{study_name}_best_trial/"
            }
            os.makedirs(final_run_config["render_dir"], exist_ok=True)

            # Run multiple seeds for the best trial to get robust final metrics
            num_final_seeds = 5
            final_seed_rewards = []
            for final_seed_idx in range(num_final_seeds):
                print(f"Running final evaluation seed {final_seed_idx+1}/{num_final_seeds}...")
                final_metrics = run_single_training_evaluation_cycle(final_run_config, run_seed=10000 + final_seed_idx) # Use fresh high seeds
                final_seed_rewards.append(final_metrics["final_avg_collective_reward"])
                # Plot learning curve for this specific best trial run
                # draw_history(final_metrics["history_collective_eval_rewards"],
                #              f"Best Trial (Seed {final_seed_idx}) - Collective Reward Over Eval Cycles")
            
            print(f"\nPerformance of Best HPs over {num_final_seeds} seeds:")
            print(f"  Avg Collective Reward: {np.mean(final_seed_rewards):.4f}")
            print(f"  Std Collective Reward: {np.std(final_seed_rewards):.4f}")

        # Save all trial results to CSV
        try:
            df_results = study.trials_dataframe(attrs=('value', 'params', 'user_attrs', 'state', 'datetime_start', 'datetime_complete', 'duration'))
            df_results.to_csv(f"{study_name}_all_trials_results.csv", index=False)
            print(f"\nFull study results saved to {study_name}_all_trials_results.csv")
        except Exception as e:
            print(f"Could not save full study results to CSV: {e}")