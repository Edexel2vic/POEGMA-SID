# plot_results.py

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import optuna
from itertools import combinations

# --- CONFIGURATION ---
# Change these values to match the experiment you want to plot
ALGORITHM = "JALGT"  # or "JALGT"
# For JALGT, the script will loop through all concepts. For IQL, this is ignored.
SOLUTION_CONCEPTS = ["WelfareSolutionConcept", "MinimaxSolutionConcept", "ParetoSolutionConcept", "NashSolutionConcept"] 

# Directory to save the generated plots
PLOTS_OUTPUT_DIR = "plots"

# --- HELPER FUNCTIONS ---

def load_study_and_df(algorithm, solution_concept=None):
    """Loads the Optuna study and the corresponding CSV DataFrame."""
    
    if algorithm == "IQL":
        study_name = f"iql_pogema_tuning"
        db_file = f"out/iql_pogema_tuning.db"
        csv_file = f"out/iql_optuna_results_detailed.csv"
    elif algorithm == "JALGT":
        concept_name_lower = solution_concept.lower()
        study_name = f"jalgt_{concept_name_lower}_pogema_tuning"
        db_file = f"out/{study_name}.db"
        csv_file = f"out/jalgt_{concept_name_lower}_optuna_results_detailed.csv"
    else:
        raise ValueError(f"Algorithm '{algorithm}' not recognized.")

    # Check if files exist
    if not os.path.exists(db_file) or not os.path.exists(csv_file):
        print(f"Warning: Could not find results for {algorithm} {solution_concept or ''}. Skipping.")
        print(f"  - Looked for DB: {db_file}")
        print(f"  - Looked for CSV: {csv_file}")
        return None, None
        
    storage_url = f"sqlite:///{db_file}"
    
    try:
        study = optuna.load_study(study_name=study_name, storage=storage_url)
        df = pd.read_csv(csv_file)
        return study, df
    except Exception as e:
        print(f"Error loading study '{study_name}' from '{db_file}': {e}")
        return None, None

def create_output_dir(base_dir, sub_dir=None):
    """Creates the output directory for plots if it doesn't exist."""
    path = os.path.join(base_dir, sub_dir) if sub_dir else base_dir
    os.makedirs(path, exist_ok=True)
    return path

# --- PLOTTING FUNCTIONS ---

def plot_reward_distribution(df, output_dir, file_prefix):
    """
    Plots the distribution of the mean collective rewards across all trials.
    This helps visualize the overall performance landscape.
    """
    plt.figure(figsize=(10, 6))
    
    # Use the 'value' column which is the mean collective reward for the trial
    reward_col = 'value' 
    if reward_col not in df.columns:
        print(f"Warning: '{reward_col}' column not found for distribution plot. Skipping.")
        return

    sns.histplot(df[reward_col], kde=True, bins=30)
    
    mean_reward = df[reward_col].mean()
    plt.axvline(mean_reward, color='r', linestyle='--', label=f'Mean: {mean_reward:.2f}')
    
    plt.title('Distribution of Mean Collective Rewards Across Trials')
    plt.xlabel('Mean Collective Reward')
    plt.ylabel('Frequency (Number of Trials)')
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    plot_path = os.path.join(output_dir, f"{file_prefix}_reward_distribution.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"  Saved reward distribution plot to: {plot_path}")

def plot_optuna_visualizations(study, output_dir, file_prefix):
    """
    Generates and saves a suite of standard, highly informative Optuna plots.
    These are interactive HTML files.
    """
    if not study:
        return

    # 1. Optimization History: Shows how the best score improves over trials.
    try:
        fig = optuna.visualization.plot_optimization_history(study)
        path = os.path.join(output_dir, f"{file_prefix}_optimization_history.html")
        fig.write_html(path)
        print(f"  Saved optimization history plot to: {path}")
    except Exception as e:
        print(f"Could not generate optimization history plot: {e}")

    # 2. Parameter Importance: Shows which hyperparameters were most influential.
    try:
        fig = optuna.visualization.plot_param_importances(study)
        path = os.path.join(output_dir, f"{file_prefix}_param_importances.html")
        fig.write_html(path)
        print(f"  Saved parameter importance plot to: {path}")
    except Exception as e:
        print(f"Could not generate parameter importance plot: {e}")

    # 3. Slice Plots (addresses your "line plot" request)
    # This is the best way to see how the objective function changes with each parameter.
    # The x-axis is the parameter, and the y-axis is the reward.
    try:
        fig = optuna.visualization.plot_slice(study)
        # Note: Optuna automatically handles the log scale for 'learning_rate'!
        path = os.path.join(output_dir, f"{file_prefix}_slice_plots.html")
        fig.write_html(path)
        print(f"  Saved slice plots to: {path}")
    except Exception as e:
        print(f"Could not generate slice plots: {e}")

    # 4. Contour Plots (Bonus): Shows interactions between pairs of hyperparameters.
    params_to_plot = [p for p in study.best_params.keys() if len(study.best_params) > 1]
    if len(params_to_plot) >= 2:
        param_pairs = list(combinations(params_to_plot, 2))
        for pair in param_pairs:
            try:
                fig = optuna.visualization.plot_contour(study, params=list(pair))
                path = os.path.join(output_dir, f"{file_prefix}_contour_{pair[0]}_vs_{pair[1]}.html")
                fig.write_html(path)
                print(f"  Saved contour plot for {pair} to: {path}")
            except Exception as e:
                # This can fail if there isn't enough data diversity for a pair
                print(f"Could not generate contour plot for {pair}: {e}")

    # 5. Parallel Coordinate Plot (Bonus): Great for seeing high-performing regions.
    try:
        fig = optuna.visualization.plot_parallel_coordinate(study)
        path = os.path.join(output_dir, f"{file_prefix}_parallel_coordinate.html")
        fig.write_html(path)
        print(f"  Saved parallel coordinate plot to: {path}")
    except Exception as e:
        print(f"Could not generate parallel coordinate plot: {e}")


def main():
    """Main function to generate plots based on the configuration."""
    print("--- Starting Result Plotting Script ---")
    
    if ALGORITHM == "IQL":
        print(f"\nProcessing algorithm: {ALGORITHM}")
        study, df = load_study_and_df(ALGORITHM)
        if study and df is not None:
            output_dir = create_output_dir(PLOTS_OUTPUT_DIR, ALGORITHM.lower())
            
            print(f"Generating plots for {ALGORITHM}...")
            # Plot 1: Reward Distribution
            plot_reward_distribution(df, output_dir, ALGORITHM.lower())
            
            # Plot 2: Suite of Optuna's interactive plots
            plot_optuna_visualizations(study, output_dir, ALGORITHM.lower())
            
    elif ALGORITHM == "JALGT":
        print(f"\nProcessing algorithm: {ALGORITHM}")
        for concept in SOLUTION_CONCEPTS:
            print(f"\n--- Processing Solution Concept: {concept} ---")
            study, df = load_study_and_df(ALGORITHM, concept)
            if study and df is not None:
                # Create a sub-directory for each concept's plots
                output_dir = create_output_dir(PLOTS_OUTPUT_DIR, f"{ALGORITHM.lower()}/{concept.lower()}")
                file_prefix = f"{ALGORITHM.lower()}_{concept.lower()}"
                
                print(f"Generating plots for {ALGORITHM} - {concept}...")
                # Plot 1: Reward Distribution
                plot_reward_distribution(df, output_dir, file_prefix)
                
                # Plot 2: Suite of Optuna's interactive plots
                plot_optuna_visualizations(study, output_dir, file_prefix)
    
    print("\n--- Plotting complete! ---")
    print(f"All plots saved in the '{PLOTS_OUTPUT_DIR}/' directory.")


if __name__ == "__main__":
    main()