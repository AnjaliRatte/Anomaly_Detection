import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


# ============================================================
# LOAD FINAL SIX-ALGORITHM RESULTS
# ============================================================

RESULTS_DIR = Path("results")

file_path = (
    RESULTS_DIR /
    "final_six_algorithm_comparison.csv"
)

df = pd.read_csv(file_path)

print("=" * 70)
print("FINAL MODEL VISUALIZATIONS")
print("=" * 70)

print("\nLoaded results:")
print(df.to_string(index=False))


# ============================================================
# 1. MEAN F1 SCORE
# ============================================================

f1_df = df.sort_values(
    "mean_f1",
    ascending=True
)

plt.figure(figsize=(10, 6))

plt.barh(
    f1_df["algorithm"],
    f1_df["mean_f1"]
)

plt.xlabel("Mean F1 Score")
plt.ylabel("Algorithm")
plt.title(
    "Mean F1 Score Comparison of Anomaly Detection Algorithms"
)

plt.tight_layout()

f1_file = (
    RESULTS_DIR /
    "final_mean_f1_comparison.png"
)

plt.savefig(
    f1_file,
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print(
    f"\nSaved F1 graph to: {f1_file}"
)


# ============================================================
# 2. PRECISION VS RECALL
# ============================================================

plt.figure(figsize=(10, 6))

plt.scatter(
    df["mean_recall"],
    df["mean_precision"],
    s=120
)

for _, row in df.iterrows():

    plt.annotate(
        row["algorithm"],
        (
            row["mean_recall"],
            row["mean_precision"]
        ),
        xytext=(6, 6),
        textcoords="offset points"
    )

plt.xlabel("Mean Recall")
plt.ylabel("Mean Precision")

plt.title(
    "Precision vs Recall Across Anomaly Detection Algorithms"
)

plt.tight_layout()

precision_recall_file = (
    RESULTS_DIR /
    "final_precision_recall_comparison.png"
)

plt.savefig(
    precision_recall_file,
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print(
    f"Saved Precision-Recall graph to: "
    f"{precision_recall_file}"
)


# ============================================================
# 3. MEAN PR-AUC
# ============================================================

pr_df = df.sort_values(
    "mean_pr_auc",
    ascending=True
)

plt.figure(figsize=(10, 6))

plt.barh(
    pr_df["algorithm"],
    pr_df["mean_pr_auc"]
)

plt.xlabel("Mean PR-AUC")
plt.ylabel("Algorithm")

plt.title(
    "Mean PR-AUC Comparison of Anomaly Detection Algorithms"
)

plt.tight_layout()

pr_file = (
    RESULTS_DIR /
    "final_pr_auc_comparison.png"
)

plt.savefig(
    pr_file,
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print(
    f"Saved PR-AUC graph to: {pr_file}"
)


# ============================================================
# 4. RUNTIME
# ============================================================

runtime_df = df.sort_values(
    "mean_runtime",
    ascending=True
)

plt.figure(figsize=(10, 6))

plt.barh(
    runtime_df["algorithm"],
    runtime_df["mean_runtime"]
)

plt.xlabel("Mean Runtime (seconds)")
plt.ylabel("Algorithm")

plt.title(
    "Mean Computational Runtime Comparison"
)

plt.tight_layout()

runtime_file = (
    RESULTS_DIR /
    "final_runtime_comparison.png"
)

plt.savefig(
    runtime_file,
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print(
    f"Saved Runtime graph to: {runtime_file}"
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("VISUALIZATION COMPLETE")
print("=" * 70)

print("\nGenerated files:")

print(f"1. {f1_file}")
print(f"2. {precision_recall_file}")
print(f"3. {pr_file}")
print(f"4. {runtime_file}")