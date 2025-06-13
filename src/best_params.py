import os
import pandas as pd

def find_best_runs_in_directory(directory_path):
    # List all CSV files in the given directory
    csv_files = [f for f in os.listdir(directory_path) if f.endswith('.csv')]

    if not csv_files:
        print("❌ No CSV files found in the directory.")
        return

    for csv_file in csv_files:
        csv_path = os.path.join(directory_path, csv_file)
        try:
            df = pd.read_csv(csv_path)

            if 'value' not in df.columns:
                print(f"⚠️ Skipping '{csv_file}' (no 'value' column found)")
                continue

            df['value'] = pd.to_numeric(df['value'], errors='coerce')
            df = df.dropna(subset=['value'])

            if df.empty:
                print(f"⚠️ Skipping '{csv_file}' (no valid rows with numeric 'value')")
                continue

            best_row = df.loc[df['value'].idxmax()]
            best_value = best_row['value']
            params_used = {col: best_row[col] for col in df.columns if col.startswith('params_')}

            print(f"\n📄 File: {csv_file}")
            print(f"🏆 Best value: {best_value}")
            print("🔧 Parameters used:")
            for k, v in params_used.items():
                print(f"  {k}: {v}")

        except Exception as e:
            print(f"❌ Error processing '{csv_file}': {e}")

# Example usage
# Replace './results/' with the actual path to your directory
if __name__ == "__main__":
    directory_path = "./src/out/"
    find_best_runs_in_directory(directory_path)
