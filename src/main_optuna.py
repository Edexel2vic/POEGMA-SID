# main_optuna.py

import optuna
import os
import pandas as pd  # <-- ADD THIS LINE to import pandas
from main import run_experiment # Import the function we just created

ALGORITHM = "IQL"   # JALGT, IQL o NN

def objective(trial: optuna.trial.Trial, solution_concept: str = None) -> float:
    """
    The objective function for Optuna.
    - Suggests hyperparameters.
    - Runs the experiment.
    - Returns the performance score.
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

    # 4. Run the experiment and return the score
    final_score = run_experiment(trial_config, trial_number=trial.number)

    return final_score


if __name__ == "__main__":
    # Create a directory for renders if it doesn't exist
    try:
        os.mkdir("renders/")
    except FileExistsError:
        pass
        
    # 1. Create a study object.
    study = optuna.create_study(
        direction="maximize",
        study_name="iql_pogema_tuning",         # cambiar el nombre del estudio jalgt o nn
        storage="out/sqlite:///iql_pogema_tuning.db",
        load_if_exists=True
    )
    
    # 2. Start the optimization.
    if ALGORITHM == "IQL":
        study.optimize(objective, timeout=7200)
    elif ALGORITHM == "JALGT":
        for solution_concept in ["Pareto", "Minimax", "Nash","Wellfare"]:
            study.optimize(objective, timeout=7200)

    # 3. Print the results
    print("Study statistics: ")
    print(f"  Number of finished trials: {len(study.trials)}")
    
    print("Best trial:")
    best_trial = study.best_trial
    print(f"  Value (Max Reward): {best_trial.value}")
    
    print("  Params: ")
    for key, value in best_trial.params.items():
        print(f"    {key}: {value}")

    # --- START OF NEW CODE ---
    # 4. Save the results to a CSV file.
    # The trials_dataframe() method converts the study results into a nice table.
    df = study.trials_dataframe()

    # We save the DataFrame to a csv file.
    # `index=False` is used to prevent pandas from writing row indices into the file.
    csv_file_path = "out/optuna_results.csv"
    df.to_csv(csv_file_path, index=False)

    print(f"\nOptimization results have been saved to {csv_file_path}")
    # --- END OF NEW CODE ---