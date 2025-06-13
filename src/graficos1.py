# plot_results.py (version 4 - with custom metric plotting)

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import optuna
from itertools import combinations

# --- CONFIGURATION ---
ALGORITHM = "IQL" # JALGT o ILQ
SOLUTION_CONCEPTS = ["WelfareSolutionConcept", "MinimaxSolutionConcept", "ParetoSolutionConcept", "NashSolutionConcept"]
PLOTS_OUTPUT_DIR = "plots"

STATIC_IMAGE_FORMAT = "pdf"

# --- PLOT GENERATION FLAGS ---
GENERATE_OPTUNA_PLOTS = False       # Generate the standard Optuna plots (for collective reward)
GENERATE_CUSTOM_METRIC_PLOTS = True # <<< NEW: Generate plots for other metrics (training time, etc.)
# --- SUB-FLAGS for speed/control ---
GENERATE_INTERACTIVE_PLOTS = False  # Set to False to skip slow HTML generation
GENERATE_STATIC_PLOTS = True      # Set to True to get images for reports


# --- HELPER FUNCTIONS (unchanged) ---
def load_study_and_df(algorithm, solution_concept=None):
    if algorithm == "IQL":
        study_name, db_file, csv_file = "iql_pogema_tuning", "out/iql_pogema_tuning.db", "out/iql_optuna_results_detailed.csv"
    elif algorithm == "JALGT":
        concept_name_lower = solution_concept.lower()
        study_name = f"jalgt_{concept_name_lower}_pogema_tuning"
        db_file = f"out/{study_name}.db"
        csv_file = f"out/JALGT_{solution_concept}_optuna_results_detailed.csv"
    else: raise ValueError(f"Algorithm '{algorithm}' not recognized.")
    if not os.path.exists(db_file) or not os.path.exists(csv_file):
        print(f"Warning: Could not find results for {algorithm} {solution_concept or ''}. Skipping."); return None, None
    try:
        study = optuna.load_study(study_name=study_name, storage=f"sqlite:///{db_file}")
        df = pd.read_csv(csv_file)
        return study, df
    except Exception as e: print(f"Error loading study '{study_name}': {e}"); return None, None

def create_output_dir(base_dir, sub_dir=None):
    path = os.path.join(base_dir, sub_dir) if sub_dir else base_dir
    os.makedirs(path, exist_ok=True)
    return path

# --- OPTUNA VISUALIZATION PLOTTING ---
def plot_optuna_visualizations(study, output_dir, file_prefix):
    # This function remains the same as version 3...
    if not study or (not GENERATE_INTERACTIVE_PLOTS and not GENERATE_STATIC_PLOTS): return
    def save_fig(fig, name):
        if GENERATE_INTERACTIVE_PLOTS: fig.write_html(os.path.join(output_dir, f"{file_prefix}_{name}.html"))
        if GENERATE_STATIC_PLOTS:
            try: fig.write_image(os.path.join(output_dir, f"{file_prefix}_{name}.{STATIC_IMAGE_FORMAT}"))
            except Exception as e: print(f"  [!] Failed to save static image for {name}. Ensure 'kaleido' is installed. Error: {e}")
    plots = {
        "optimization_history": optuna.visualization.plot_optimization_history,
        "param_importances": optuna.visualization.plot_param_importances,
        "slice": optuna.visualization.plot_slice,
        "parallel_coordinate": optuna.visualization.plot_parallel_coordinate,
    }
    for name, plot_func in plots.items():
        try: save_fig(plot_func(study), name); print(f"  Generated Optuna plot: {name}")
        except Exception as e: print(f"Could not generate Optuna plot '{name}': {e}")
    params_to_plot = [p for p in study.best_params.keys() if len(study.best_params) > 1]
    if len(params_to_plot) >= 2:
        for pair in combinations(params_to_plot, 2):
            try: save_fig(optuna.visualization.plot_contour(study, params=list(pair)), f"contour_{pair[0]}_vs_{pair[1]}"); print(f"  Generated Optuna plot: contour_{pair[0]}_vs_{pair[1]}")
            except Exception as e: print(f"Could not generate contour plot for {pair}: {e}")

# <<< NEW FUNCTION TO PLOT CUSTOM METRICS ---
def plot_custom_metrics(df, output_dir, file_prefix):
    """
    Generates custom plots for user-defined metrics from the results DataFrame.
    """
    if not GENERATE_CUSTOM_METRIC_PLOTS or not GENERATE_STATIC_PLOTS:
        return

    print("\n  --- Generating custom metric plots ---")

    # Define the metrics you want to plot and their corresponding column names
    metrics_to_plot = {
        "Total Training Time (s)": "user_attrs_training_time_total_mean",
        "Avg Time per Episode (s)": "user_attrs_training_time_per_episode_mean",
        "Individual Reward (Agent 0)": "user_attrs_individual_reward_agent0_mean",
    }

    # Identify hyperparameter columns
    hyperparameter_cols = [col for col in df.columns if col.startswith('params_')]
    
    for metric_name, metric_col in metrics_to_plot.items():
        if metric_col not in df.columns:
            print(f"  Skipping metric '{metric_name}': Column '{metric_col}' not found.")
            continue
        
        # Create a subdirectory for each metric's plots for better organization
        metric_plot_dir = create_output_dir(output_dir, f"custom_{metric_col.replace('user_attrs_', '').replace('_mean','')}")
        print(f"  Plotting for metric: {metric_name}")

        # 1. Generate 2D "Slice" plots (Hyperparameter vs. Metric)
        for param_col in hyperparameter_cols:
            param_name = param_col.replace('params_', '')
            plt.figure(figsize=(10, 6))
            
            # Use a regplot to show the trend
            sns.regplot(data=df, x=param_col, y=metric_col, scatter_kws={'alpha':0.4}, line_kws={'color':'red'})

            if 'learning_rate' in param_col:
                plt.xscale('log')

            plt.xlabel(f"Hyperparameter: {param_name}")
            plt.ylabel(metric_name)
            plt.title(f"{metric_name} vs. {param_name}")
            plt.grid(True, which='both', linestyle='--', linewidth=0.5)
            
            plot_path = os.path.join(metric_plot_dir, f"2d_scatter_{param_name}.{STATIC_IMAGE_FORMAT}")
            plt.savefig(plot_path)
            plt.close()

        # 2. Generate 3D "Contour" like plots (Hyperparam1 vs. Hyperparam2, color=Metric)
        for param1_col, param2_col in combinations(hyperparameter_cols, 2):
            param1_name = param1_col.replace('params_', '')
            param2_name = param2_col.replace('params_', '')
            
            plt.figure(figsize=(12, 8))
            sc = plt.scatter(data=df, x=param1_col, y=param2_col, c=metric_col, cmap='viridis', alpha=0.7)
            
            if 'learning_rate' in param1_col: plt.xscale('log')
            if 'learning_rate' in param2_col: plt.yscale('log')

            plt.xlabel(f"Hyperparameter: {param1_name}")
            plt.ylabel(f"Hyperparameter: {param2_name}")
            cbar = plt.colorbar(sc)
            cbar.set_label(metric_name)
            plt.title(f"{param1_name} vs. {param2_name}\n(Color shows {metric_name})")
            plt.grid(True, which='both', linestyle='--', linewidth=0.5)

            plot_path = os.path.join(metric_plot_dir, f"3d_scatter_{param1_name}_vs_{param2_name}.{STATIC_IMAGE_FORMAT}")
            plt.savefig(plot_path)
            plt.close()
            
# --- MAIN EXECUTION LOGIC ---
def main():
    print("--- Starting Result Plotting Script ---")
    algorithms_to_run = [ALGORITHM] if ALGORITHM != "JALGT" else [(ALGORITHM, concept) for concept in SOLUTION_CONCEPTS]

    for item in algorithms_to_run:
        if isinstance(item, tuple):
            alg, concept = item
            print(f"\n--- Processing: {alg} - {concept} ---")
            study, df = load_study_and_df(alg, concept)
            if not study: continue
            output_dir = create_output_dir(PLOTS_OUTPUT_DIR, f"{alg.lower()}/{concept.lower()}")
            file_prefix = f"{alg.lower()}_{concept.lower()}"
        else:
            alg = item
            print(f"\n--- Processing: {alg} ---")
            study, df = load_study_and_df(alg)
            if not study: continue
            output_dir = create_output_dir(PLOTS_OUTPUT_DIR, alg.lower())
            file_prefix = alg.lower()

        # Generate standard Optuna plots for the main objective (Collective Reward)
        if GENERATE_OPTUNA_PLOTS:
            plot_optuna_visualizations(study, output_dir, file_prefix)
        
        # <<< NEW: Generate custom plots for other important metrics
        if GENERATE_CUSTOM_METRIC_PLOTS:
            plot_custom_metrics(df, output_dir, file_prefix)

    print("\n--- Plotting complete! ---")

if __name__ == "__main__":
    main()