import subprocess
import glob
import json
import os

import pandas as pd
import matplotlib.pyplot as plt


num_leaves_values = [2, 4, 5, 8, 10, 31, 60, 150]

results = []

# --------------------------------------------------
# 2. Create a folder for CSV files and plots
# --------------------------------------------------

output_folder = "experiments/results"
os.makedirs(output_folder, exist_ok=True)


# --------------------------------------------------
# 3. Test every num_leaves value
# --------------------------------------------------

for num_leaves in num_leaves_values:

    print("\n-----------------------------------")
    print(f"Testing num_leaves: {num_leaves}")
    print("-----------------------------------")


    # ==================================================
    # REGRESSION
    # ==================================================

    subprocess.run([
        "python",
        "finetune.py",
        "model=morgan_rdkit_lgbm",
        "task=regression",
        f"model.params.num_leaves={num_leaves}"
    ], check=True)


    # --------------------------------------------------
    # Find the newest regression run
    # --------------------------------------------------

    regression_runs = glob.glob(
        "runs/regression/morgan_rdkit_lgbm/*"
    )

    regression_run = sorted(regression_runs)[-1]


    # --------------------------------------------------
    # Read the regression validation results
    # These are used for model selection
    # --------------------------------------------------

    regression_metrics_path = os.path.join(
        regression_run,
        "validation_metrics.json"
    )

    with open(regression_metrics_path) as f:
        regression_metrics = json.load(f)


    # Main regression metric:
    # LOWER macro ST-RAE is better

    macro_st_rae = regression_metrics["macro_ST-RAE"]


    # ==================================================
    # CLASSIFICATION
    # ==================================================

    subprocess.run([
        "python",
        "finetune.py",
        "model=morgan_rdkit_lgbm",
        "task=classification",
        f"model.params.num_leaves={num_leaves}"
    ], check=True)


    # --------------------------------------------------
    # Find the newest classification run
    # --------------------------------------------------

    classification_runs = glob.glob(
        "runs/classification/morgan_rdkit_lgbm/*"
    )

    classification_run = sorted(classification_runs)[-1]

    # --------------------------------------------------
    # Read the classification validation results
    # --------------------------------------------------

    classification_metrics_path = os.path.join(
        classification_run,
        "validation_metrics.json"
    )

    with open(classification_metrics_path) as f:
        classification_metrics = json.load(f)


    # Main classification metric:
    # HIGHER macro MCC is better

    macro_mcc = classification_metrics["macro_MCC"]


    # ==================================================
    # SAVE THIS NUM_LEAVES VALUE'S RESULTS
    # ==================================================

    results.append({
        "num_leaves": num_leaves,
        "macro_ST-RAE": macro_st_rae,
        "macro_MCC": macro_mcc
    })


# --------------------------------------------------
# 4. Convert all results into a table
# --------------------------------------------------

results_df = pd.DataFrame(results)


# --------------------------------------------------
# 5. Save the results as a CSV file
# --------------------------------------------------

csv_path = os.path.join(
    output_folder,
    "tree_complexity_results.csv"
)

results_df.to_csv(csv_path, index=False)

print("\nFinal results:")
print(results_df)

print(f"\nResults saved to: {csv_path}")


# --------------------------------------------------
# 6. Find the best regression num_leaves value
# --------------------------------------------------
# LOWER macro ST-RAE is better

best_regression = results_df.loc[
    results_df["macro_ST-RAE"].idxmin()
]


# --------------------------------------------------
# 7. Find the best classification num_leaves value
# --------------------------------------------------
# HIGHER macro MCC is better

best_classification = results_df.loc[
    results_df["macro_MCC"].idxmax()
]


print("\nBest regression num_leaves:")
print(int(best_regression["num_leaves"]))
print("Macro ST-RAE:", best_regression["macro_ST-RAE"])


print("\nBest classification num_leaves:")
print(int(best_classification["num_leaves"]))
print("Macro MCC:", best_classification["macro_MCC"])


# ==================================================
# 8. REGRESSION PLOT
# ==================================================

plt.figure()

plt.plot(
    results_df["num_leaves"],
    results_df["macro_ST-RAE"],
    marker="o"
)

plt.xlabel("Number of leaves")
plt.ylabel("Macro ST-RAE")
plt.title("Tree Complexity vs Regression Performance")

plt.grid(True)

regression_plot_path = os.path.join(
    output_folder,
    "tree_complexity_regression.png"
)

plt.savefig(
    regression_plot_path,
    bbox_inches="tight"
)

plt.show()

print(f"Regression plot saved to: {regression_plot_path}")


# ==================================================
# 9. CLASSIFICATION PLOT
# ==================================================

plt.figure()

plt.plot(
    results_df["num_leaves"],
    results_df["macro_MCC"],
    marker="o"
)

plt.xlabel("Number of leaves")
plt.ylabel("Macro MCC")
plt.title("Tree Complexity vs Classification Performance")

plt.grid(True)

classification_plot_path = os.path.join(
    output_folder,
    "tree_complexity_classification.png"
)

plt.savefig(
    classification_plot_path,
    bbox_inches="tight"
)

plt.show()

print(f"Classification plot saved to: {classification_plot_path}")