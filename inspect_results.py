
from pathlib import Path
import pandas as pd

results_dir = Path(__file__).resolve().parent / "results"

files_to_inspect = [
    "statistical_significance_results.csv",
    "master_experiment_comparison.csv",
    "final_optimized_model_summary.csv",
    "final_lstm_model_summary.csv",
    "final_half_space_trees_model_summary.csv",
]

for filename in files_to_inspect:
    path = results_dir / filename

    print("\n" + "=" * 80)
    print(filename)
    print("=" * 80)

    df = pd.read_csv(path)
    print("Shape:", df.shape)
    print("Columns:", df.columns.tolist())
    print(df.to_string(index=False))
