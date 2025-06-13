# plot_results.py (version 3 - static-only optimization)

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import optuna
from itertools import combinations

# --- CONFIGURATION ---
ALGORITHM = "IQL"
SOLUTION_CONCEPTS = ["WelfareSolutionConcept", "MinimaxSolutionConcept", "ParetoSolutionConcept", "NashSolutionConcept"]
PLOTS_OUTPUT_DIR = "plots"

# <<< MODIFIED: Set the desired static image format (pdf, svg, or png)
STATIC_IMAGE_FORMAT = "pdf" 

# <<< MODIFIED: Control which types of plots to generate for speed
GENERATE_INTERACTIVE_PLOTS = False  # Set to False to skip slow HTML generation
GENERATE_STATIC_PLOTS = True      # Set to True to get images for reports

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
        csv_file = f"out/JALGT_{solution_concept}_optuna_results_detailed.csv"
    else:
        raise ValueError(f"Algorithm '{algorithm}' not recognized.")

    if not os.path.exists(db_file) or not os.path.exists(csv_file):
        print(f"Warning: Could not find results for {algorithm} {solution_concept or ''}. Skipping.")
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
    """Plots the distribution of the mean collective rewards and saves static images."""
    if not GENERATE_STATIC_PLOTS:
        return

    plt.figure(figsize=(10, 6))
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
    
    # Save in both PNG (for quick viewing) and the high-quality static format (for reports)
    plot_path_png = os.path.join(output_dir, f"{file_prefix}_reward_distribution.png")
    plt.savefig(plot_path_png)
    
    plot_path_static = os.path.join(output_dir, f"{file_prefix}_reward_distribution.{STATIC_IMAGE_FORMAT}")
    plt.savefig(plot_path_static, format=STATIC_IMAGE_FORMAT)
    print(f"  Saved reward distribution plot to: {plot_path_static}")

    plt.close()


def plot_optuna_visualizations(study, output_dir, file_prefix):
    """
    Generates and saves a suite of standard Optuna plots based on the configuration flags.
    """
    if not study or (not GENERATE_INTERACTIVE_PLOTS and not GENERATE_STATIC_PLOTS):
        return

    # A helper to save in the configured formats
    def save_fig(fig, name):
        if GENERATE_INTERACTIVE_PLOTS:
            html_path = os.path.join(output_dir, f"{file_prefix}_{name}.html")
            fig.write_html(html_path)
            print(f"  Saved interactive plot to: {html_path}")

        if GENERATE_STATIC_PLOTS:
            static_path = os.path.join(output_dir, f"{file_prefix}_{name}.{STATIC_IMAGE_FORMAT}")
            try:
                fig.write_image(static_path)
                print(f"  Saved static image to: {static_path}")
            except ValueError as e: # Catch kaleido-specific errors
                 print(f"  [!] Failed to save static image '{static_path}'.")
                 print(f"  [!] Make sure 'kaleido' is installed and working: pip install kaleido")
                 print(f"  [!] Original error: {e}")
            except Exception as e:
                print(f"  [!] An unexpected error occurred while saving static image: {e}")

    # 1. Optimization History
    try:
        fig = optuna.visualization.plot_optimization_history(study)
        save_fig(fig, "optimization_history")
    except Exception as e:
        print(f"Could not generate optimization history plot: {e}")

    # 2. Parameter Importance
    try:
        fig = optuna.visualization.plot_param_importances(study)
        save_fig(fig, "param_importances")
    except Exception as e:
        print(f"Could not generate parameter importance plot: {e}")

    # 3. Slice Plots
    try:
        fig = optuna.visualization.plot_slice(study)
        save_fig(fig, "slice_plots")
    except Exception as e:
        print(f"Could not generate slice plots: {e}")

    # 4. Contour Plots
    params_to_plot = [p for p in study.best_params.keys() if len(study.best_params) > 1]
    if len(params_to_plot) >= 2:
        param_pairs = list(combinations(params_to_plot, 2))
        for pair in param_pairs:
            try:
                fig = optuna.visualization.plot_contour(study, params=list(pair))
                save_fig(fig, f"contour_{pair[0]}_vs_{pair[1]}")
            except Exception as e:
                print(f"Could not generate contour plot for {pair}: {e}")

    # 5. Parallel Coordinate Plot
    try:
        fig = optuna.visualization.plot_parallel_coordinate(study)
        save_fig(fig, "parallel_coordinate")
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
            plot_reward_distribution(df, output_dir, ALGORITHM.lower())
            plot_optuna_visualizations(study, output_dir, ALGORITHM.lower())
            
    elif ALGORITHM == "JALGT":
        print(f"\nProcessing algorithm: {ALGORITHM}")
        for concept in SOLUTION_CONCEPTS:
            print(f"\n--- Processing Solution Concept: {concept} ---")
            study, df = load_study_and_df(ALGORITHM, concept)
            if study and df is not None:
                output_dir = create_output_dir(PLOTS_OUTPUT_DIR, f"{ALGORITHM.lower()}/{concept.lower()}")
                file_prefix = f"{ALGORITHM.lower()}_{concept.lower()}"
                print(f"Generating plots for {ALGORITHM} - {concept}...")
                plot_reward_distribution(df, output_dir, file_prefix)
                plot_optuna_visualizations(study, output_dir, file_prefix)
    
    print("\n--- Plotting complete! ---")
    if GENERATE_STATIC_PLOTS:
        print(f"Static images saved as .{STATIC_IMAGE_FORMAT} files in the '{PLOTS_OUTPUT_DIR}/' directory.")
    if not GENERATE_INTERACTIVE_PLOTS:
        print("Skipped generation of interactive HTML files as configured.")


if __name__ == "__main__":
    main()