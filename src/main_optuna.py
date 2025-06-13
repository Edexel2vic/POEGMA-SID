# main_optuna.py

import optuna
import os
import pandas as pd
import numpy as np
import time
import random
from concurrent.futures import ProcessPoolExecutor
from main import run_experiment # Import the function we just created
from solution_concepts import ParetoSolutionConcept, MinimaxSolutionConcept, NashSolutionConcept, WelfareSolutionConcept

ALGORITHM = "IQL"   # JALGT, IQL o NN
NUM_PARALLEL_RUNS = 6  # Number of parallel runs per trial

def run_single_trial(config, trial_number, run_id, base_seed):
    """
    Run a single trial and return detailed metrics.
    """
    start_time = time.time()
    
    # Create unique seed for this run
    run_seed = base_seed + run_id * 1000
    
    # Set random seeds for reproducibility but uniqueness across runs
    random.seed(run_seed)
    np.random.seed(run_seed)
    
    # Track training time per episode
    training_times = []
    
    # Modify config to track episode training times and use unique seed
    config_with_timing = config.copy()
    config_with_timing['track_timing'] = True
    config_with_timing['run_id'] = run_id
    config_with_timing['show_progress'] = False  # Disable progress bars for parallel runs
    config_with_timing['base_seed'] = run_seed  # Pass unique seed to the experiment
    
    # Run the experiment
    result = run_experiment(config_with_timing, trial_number=f"{trial_number}_{run_id}")
    
    total_training_time = time.time() - start_time
    
    # Extract metrics from result
    if isinstance(result, dict):
        collective_reward = result.get('collective_reward', 0)
        individual_rewards = result.get('individual_rewards', [0] * config['num_agents'])
        episode_training_times = result.get('episode_training_times', [])
        num_episodes = result.get('num_episodes', config['epochs'] * config['episodes_per_epoch'])
        
        # Calculate total individual rewards for fair comparison with collective
        total_individual_rewards = [r * config['maps'] for r in individual_rewards]
    else:
        # Fallback if run_experiment returns just a score
        collective_reward = result
        individual_rewards = [result / config['num_agents']] * config['num_agents']
        total_individual_rewards = [result / config['num_agents']] * config['num_agents']
        episode_training_times = []
        num_episodes = config['epochs'] * config['episodes_per_epoch']
    
    # Calculate average training time per episode
    avg_training_time_per_episode = (total_training_time / num_episodes) if num_episodes > 0 else 0
    
    return {
        'collective_reward': collective_reward,
        'individual_rewards': individual_rewards,  # Keep as averages for compatibility
        'total_individual_rewards': total_individual_rewards,  # Add totals for comparison
        'total_training_time': total_training_time,
        'avg_training_time_per_episode': avg_training_time_per_episode,
        'num_episodes': num_episodes,
        'run_id': run_id,
        'seed_used': run_seed
    }

def save_results_to_csv(study, algorithm):
    """
    Save current study results to CSV file.
    """
    try:
        df = study.trials_dataframe()
        
        # Add user attributes to the dataframe
        if len(study.trials) > 0 and hasattr(study.trials[0], 'user_attrs'):
            user_attr_columns = set()
            for trial in study.trials:
                if hasattr(trial, 'user_attrs'):
                    user_attr_columns.update(trial.user_attrs.keys())
            
            for attr in user_attr_columns:
                df[f'user_attrs_{attr}'] = [
                    trial.user_attrs.get(attr, None) if hasattr(trial, 'user_attrs') else None
                    for trial in study.trials
                ]

        csv_file_path = f"out/{algorithm.lower()}_optuna_results_detailed.csv"
        df.to_csv(csv_file_path, index=False)
        return csv_file_path
    except Exception as e:
        print(f"Warning: Could not save CSV file: {e}")
        return None


def objective(trial: optuna.trial.Trial, solution_concept: str = None) -> float:
    """
    The objective function for Optuna.
    - Suggests hyperparameters.
    - Runs the experiment multiple times in parallel.
    - Returns the mean performance score and stores statistics as user attributes.
    """

    # 1. Define the base configuration with fixed parameters
    base_config = {
        "num_agents": 2,
        "size": 4,
        "maps": 10,
        "num_states": 16 * 16 * 4,
        "epochs": 10,  # Use fewer epochs for faster tuning
        "episodes_per_epoch": 20,
        "episode_length": 16,
        "obstacle_density": 0.1,
        "save_every": None,  # Disable rendering to speed up
        "epsilon_max": 1.0,
        "renders": "renders/",
        "algorithm": ALGORITHM,
        "solution_concept": solution_concept,  # Use the solution concept from the trial
    }

    # 2. Suggest hyperparameters to be tuned using the 'trial' object
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-1, log=True)
    gamma = trial.suggest_float("gamma", 0.9, 0.999)
    epsilon_max = trial.suggest_float("epsilon_max", 0.5, 1.0)
    epsilon_min = trial.suggest_float("epsilon_min", 0.01, 0.2)
    num_episodes = trial.suggest_int("num_episodes", 100, 500)
    episode_length = trial.suggest_int("episode_length", 10, 50)

    epochs = 10
    episodes_per_epoch = num_episodes // epochs

    # 3. Create the final configuration for this trial
    trial_config = base_config.copy()
    trial_config.update({
        "learning_rate": learning_rate,
        "gamma": gamma,
        "epsilon_min": epsilon_min,
        "epsilon_max": epsilon_max,
        "episodes_per_epoch": episodes_per_epoch,
        "episode_length": episode_length,
    })

    # 4. Generate a base seed for this trial to ensure different runs have different seeds
    base_seed = trial.number * 10000 + int(time.time()) % 10000

    # 5. Run the experiment multiple times in parallel
    try:
        with ProcessPoolExecutor(max_workers=NUM_PARALLEL_RUNS) as executor:
            futures = [
                executor.submit(run_single_trial, trial_config, trial.number, run_id, base_seed)
                for run_id in range(NUM_PARALLEL_RUNS)
            ]
            
            results = [future.result() for future in futures]
    except Exception as e:
        print(f"Parallel execution failed, falling back to sequential: {e}")
        # Fallback to sequential execution
        results = [
            run_single_trial(trial_config, trial.number, run_id, base_seed)
            for run_id in range(NUM_PARALLEL_RUNS)
        ]

    # 6. Calculate statistics across all runs
    collective_rewards = [r['collective_reward'] for r in results]
    individual_rewards_all = [r['individual_rewards'] for r in results]
    total_individual_rewards_all = [r['total_individual_rewards'] for r in results]
    training_times_total = [r['total_training_time'] for r in results]
    training_times_per_episode = [r['avg_training_time_per_episode'] for r in results]
    num_episodes_all = [r['num_episodes'] for r in results]
    seeds_used = [r['seed_used'] for r in results]

    # Calculate means and standard deviations
    mean_collective_reward = np.mean(collective_rewards)
    std_collective_reward = np.std(collective_rewards)
    
    # Individual rewards statistics (for agent 0 as example) - using averages
    individual_rewards_agent0 = [rewards[0] for rewards in individual_rewards_all]
    mean_individual_reward_agent0 = np.mean(individual_rewards_agent0)
    std_individual_reward_agent0 = np.std(individual_rewards_agent0)
    
    # Total individual rewards statistics for comparison with collective
    total_individual_rewards_agent0 = [rewards[0] for rewards in total_individual_rewards_all]
    mean_total_individual_reward_agent0 = np.mean(total_individual_rewards_agent0)
    std_total_individual_reward_agent0 = np.std(total_individual_rewards_agent0)
    
    mean_training_time_total = np.mean(training_times_total)
    std_training_time_total = np.std(training_times_total)
    
    mean_training_time_per_episode = np.mean(training_times_per_episode)
    std_training_time_per_episode = np.std(training_times_per_episode)
    
    mean_num_episodes = np.mean(num_episodes_all)

    # 7. Store statistics as user attributes
    trial.set_user_attr("collective_reward_mean", mean_collective_reward)
    trial.set_user_attr("collective_reward_std", std_collective_reward)
    trial.set_user_attr("individual_reward_agent0_mean", mean_individual_reward_agent0)
    trial.set_user_attr("individual_reward_agent0_std", std_individual_reward_agent0)
    trial.set_user_attr("total_individual_reward_agent0_mean", mean_total_individual_reward_agent0)
    trial.set_user_attr("total_individual_reward_agent0_std", std_total_individual_reward_agent0)
    trial.set_user_attr("training_time_total_mean", mean_training_time_total)
    trial.set_user_attr("training_time_total_std", std_training_time_total)
    trial.set_user_attr("training_time_per_episode_mean", mean_training_time_per_episode)
    trial.set_user_attr("training_time_per_episode_std", std_training_time_per_episode)
    trial.set_user_attr("num_episodes_mean", mean_num_episodes)
    trial.set_user_attr("num_parallel_runs", NUM_PARALLEL_RUNS)
    trial.set_user_attr("base_seed", base_seed)
    
    # FIXED: Store solution concept as user attribute for CSV output
    if solution_concept is not None:
        # Store the class name as a string for readability
        solution_concept_name = solution_concept.__name__ if hasattr(solution_concept, '__name__') else str(solution_concept)
        trial.set_user_attr("solution_concept", solution_concept_name)
    
    # Store individual run results for detailed analysis
    for i, result in enumerate(results):
        trial.set_user_attr(f"run_{i}_collective_reward", result['collective_reward'])
        trial.set_user_attr(f"run_{i}_individual_rewards", result['individual_rewards'])
        trial.set_user_attr(f"run_{i}_total_individual_rewards", result['total_individual_rewards'])
        trial.set_user_attr(f"run_{i}_training_time_total", result['total_training_time'])
        trial.set_user_attr(f"run_{i}_training_time_per_episode", result['avg_training_time_per_episode'])
        trial.set_user_attr(f"run_{i}_seed", result['seed_used'])

    # Print debug info for first few trials
    if trial.number < 3:
        print(f"\nDEBUG Trial {trial.number}:")
        print(f"  Solution concept: {solution_concept_name if solution_concept else 'None'}")
        print(f"  Seeds used: {seeds_used}")
        print(f"  Collective rewards: {collective_rewards}")
        print(f"  Collective reward std: {std_collective_reward:.6f}")
        print(f"  Individual agent0 rewards (avg): {individual_rewards_agent0}")
        print(f"  Individual agent0 rewards (total): {total_individual_rewards_agent0}")

    # 8. Return the mean collective reward as the objective value
    return mean_collective_reward


if __name__ == "__main__":
    # Create directories if they don't exist
    try:
        os.makedirs("renders/", exist_ok=True)
        os.makedirs("out/", exist_ok=True)
    except Exception as e:
        print(f"Warning: Could not create directories: {e}")
    
    print(f"Running Optuna optimization with {NUM_PARALLEL_RUNS} parallel runs per trial")
    print(f"Algorithm: {ALGORITHM}")
    
    # FIXED: Create separate studies for each solution concept to provide better tracking
    if ALGORITHM == "JALGT":
        solution_concepts = [WelfareSolutionConcept, MinimaxSolutionConcept, ParetoSolutionConcept, NashSolutionConcept]
        
        for i, solution_concept in enumerate(solution_concepts):
            solution_concept_name = solution_concept.__name__
            print(f"\n{'='*60}")
            print(f"OPTIMIZING SOLUTION CONCEPT {i+1}/{len(solution_concepts)}: {solution_concept_name}")
            print(f"{'='*60}")
            
            # Create a study for this specific solution concept
            study = optuna.create_study(
                direction="maximize",
                study_name=f"{ALGORITHM.lower()}_{solution_concept_name.lower()}_pogema_tuning",
                storage=f"sqlite:///out/{ALGORITHM.lower()}_{solution_concept_name.lower()}_pogema_tuning.db",
                load_if_exists=True
            )
            
            # Callback function to save results periodically
            save_interval = 5  # Save every 5 trials
            
            def callback_after_trial(study, trial, concept_name=solution_concept_name):
                """Callback function to save results periodically."""
                if trial.number % save_interval == 0:
                    csv_path = save_results_to_csv(study, f"{ALGORITHM}_{concept_name}")
                    if csv_path:
                        print(f"\n[{concept_name} - Trial {trial.number}] Intermediate results saved to: {csv_path}")
            
            # Optimize for this solution concept
            study.optimize(
                lambda trial: objective(trial, solution_concept), 
                n_trials = 300,
                callbacks=[callback_after_trial]
            )
            
            # Print results for this solution concept
            print(f"\n{'-'*40}")
            print(f"RESULTS FOR {solution_concept_name}")
            print(f"{'-'*40}")
            print(f"Number of finished trials: {len(study.trials)}")
            
            if len(study.trials) > 0:
                best_trial = study.best_trial
                print(f"Best trial number: {best_trial.number}")
                print(f"Best collective reward: {best_trial.value:.4f}")
                
                if hasattr(best_trial, 'user_attrs') and best_trial.user_attrs:
                    print(f"Collective reward std: {best_trial.user_attrs.get('collective_reward_std', 'N/A'):.4f}")
                    print(f"Solution concept: {best_trial.user_attrs.get('solution_concept', 'N/A')}")
                
                print(f"Best hyperparameters:")
                for key, value in best_trial.params.items():
                    print(f"  {key}: {value}")
                
                # Save final results for this solution concept
                final_csv_path = save_results_to_csv(study, f"{ALGORITHM}_{solution_concept_name}")
                if final_csv_path:
                    print(f"Results saved to: {final_csv_path}")
                print(f"Database saved to: out/{ALGORITHM.lower()}_{solution_concept_name.lower()}_pogema_tuning.db")
    
    elif ALGORITHM == "IQL":
        # 1. Create a study object for IQL
        study = optuna.create_study(
            direction="maximize",
            study_name=f"{ALGORITHM.lower()}_pogema_tuning",
            storage=f"sqlite:///out/{ALGORITHM.lower()}_pogema_tuning.db",
            load_if_exists=True
        )
        
        # 2. Start the optimization with periodic CSV saving
        save_interval = 5  # Save every 5 trials
        
        def callback_after_trial(study, trial):
            """Callback function to save results periodically."""
            if trial.number % save_interval == 0:
                csv_path = save_results_to_csv(study, ALGORITHM)
                if csv_path:
                    print(f"\n[Trial {trial.number}] Intermediate results saved to: {csv_path}")
        
        study.optimize(objective, n_trials=2000, timeout=10800, callbacks=[callback_after_trial])
        
        # 3. Print the results
        print("\n" + "="*50)
        print("OPTIMIZATION RESULTS")
        print("="*50)
        print(f"Study statistics:")
        print(f"  Number of finished trials: {len(study.trials)}")
        print(f"  Number of parallel runs per trial: {NUM_PARALLEL_RUNS}")
        
        if len(study.trials) > 0:
            print(f"\nBest trial:")
            best_trial = study.best_trial
            print(f"  Trial number: {best_trial.number}")
            print(f"  Collective Reward (mean): {best_trial.value:.4f}")
            
            # Print user attributes if available
            if hasattr(best_trial, 'user_attrs') and best_trial.user_attrs:
                print(f"  Collective Reward (std): {best_trial.user_attrs.get('collective_reward_std', 'N/A'):.4f}")
                print(f"  Individual Reward Agent 0 (avg): {best_trial.user_attrs.get('individual_reward_agent0_mean', 'N/A'):.4f}")
                print(f"  Individual Reward Agent 0 (avg std): {best_trial.user_attrs.get('individual_reward_agent0_std', 'N/A'):.4f}")
                print(f"  Individual Reward Agent 0 (total): {best_trial.user_attrs.get('total_individual_reward_agent0_mean', 'N/A'):.4f}")
                print(f"  Individual Reward Agent 0 (total std): {best_trial.user_attrs.get('total_individual_reward_agent0_std', 'N/A'):.4f}")
                print(f"  Training Time Total (mean): {best_trial.user_attrs.get('training_time_total_mean', 'N/A'):.2f}s")
                print(f"  Training Time Total (std): {best_trial.user_attrs.get('training_time_total_std', 'N/A'):.2f}s")
                print(f"  Training Time per Episode (mean): {best_trial.user_attrs.get('training_time_per_episode_mean', 'N/A'):.4f}s")
                print(f"  Training Time per Episode (std): {best_trial.user_attrs.get('training_time_per_episode_std', 'N/A'):.4f}s")
                print(f"  Number of Episodes (mean): {best_trial.user_attrs.get('num_episodes_mean', 'N/A'):.0f}")
                print(f"  Base seed used: {best_trial.user_attrs.get('base_seed', 'N/A')}")
            
            print(f"\n  Best hyperparameters:")
            for key, value in best_trial.params.items():
                print(f"    {key}: {value}")

        # 4. Save the final results to a CSV file with enhanced information
        final_csv_path = save_results_to_csv(study, ALGORITHM)
        
        if final_csv_path:
            print(f"\nFinal detailed results saved to: {final_csv_path}")
        print(f"Database saved to: out/{ALGORITHM.lower()}_pogema_tuning.db")
        print("="*50)
