import subprocess
import glob
import json
import os

import pandas as pd
import matplotlib.pyplot as plt


# --------------------------------------------------
# 1. Choose which learning rates we want to test, 0.03 is default LightGBM
# --------------------------------------------------


learning_rates = [0.01, 0.03, 0.05, 0.10]

results = []


# --------------------------------------------------
# 3. Create a folder for CSV files and plots
# -------------------------------------------------

output_folder = "experiments/results"
os.makedirs(output_folder, exist_ok=True)

# --------------------------------------------------
# 4. Test every learning rate (we might change the array)
# --------------------------------------------------

for learning_rate in learning_rates:

    print("\n-----------------------------------")
    print(f"Testing learning rate: {learning_rate}")
    print("-----------------------------------")


    # ==================================================
    # REGRESSION
    # ==================================================

    subprocess.run([
        "python",
        "finetune.py",
        "model=morgan_rdkit_lgbm",
        "task=regression",
        f"model.params.learning_rate={learning_rate}"
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
    # Each run contains validation metrics.
    # These we read and use for model selection 
    # --------------------------------------------------

    regression_metrics_path = os.path.join(
        regression_run,
        "validation_metrics.json"
    )

    with open(regression_metrics_path) as f:
        regression_metrics = json.load(f)


    # The main regression metric is macro ST-RAE.
    #
    # LOWER is better.

    macro_st_rae = regression_metrics["macro_ST-RAE"]


    # ==================================================
    # CLASSIFICATION
    # ==================================================

    # Now we run the same learning rate,
    # but on the classification task.

    subprocess.run([
        "python",
        "finetune.py",
        "model=morgan_rdkit_lgbm",
        "task=classification",
        f"model.params.learning_rate={learning_rate}"
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


    # The main classification metric is macro MCC.
    #
    # HIGHER is better.

    macro_mcc = classification_metrics["macro_MCC"]


    # ==================================================
    # SAVE THIS LEARNING RATE'S RESULTS
    # ==================================================
    # Example:
    #
    # {
    #     "learning_rate": 0.03,
    #     "macro_ST-RAE": 0.74,
    #     "macro_MCC": 0.10
    # }

    results.append({
        "learning_rate": learning_rate,
        "macro_ST-RAE": macro_st_rae,
        "macro_MCC": macro_mcc
    })


# --------------------------------------------------
# 5. Convert all results into a table
# --------------------------------------------------
# pandas DataFrame gives us a table like:
#
# learning_rate | macro_ST-RAE | macro_MCC
# -----------------------------------------
# 0.01          | ...          | ...
# 0.03          | ...          | ...
# 0.05          | ...          | ...
# 0.10          | ...          | ...

results_df = pd.DataFrame(results)


# --------------------------------------------------
# 6. Save the results as a CSV file
# --------------------------------------------------
# This makes it easy to:
# - open the results later
# - compare experiments
# - use the results in the report
# - make new plots without rerunning the models

csv_path = os.path.join(
    output_folder,
    "learning_rate_results.csv"
)

results_df.to_csv(csv_path, index=False)

print("\nFinal results:")
print(results_df)

print(f"\nResults saved to: {csv_path}")


# --------------------------------------------------
# 7. Find the best regression learning rate
# --------------------------------------------------
# For regression:
# LOWER macro ST-RAE is better.
#
# idxmin() finds the row with the smallest value.

best_regression = results_df.loc[
    results_df["macro_ST-RAE"].idxmin()
]


# --------------------------------------------------
# 8. Find the best classification learning rate
# --------------------------------------------------
# For classification:
# HIGHER macro MCC is better.
#
# idxmax() finds the row with the largest value.

best_classification = results_df.loc[
    results_df["macro_MCC"].idxmax()
]


print("\nBest regression learning rate:")
print(best_regression["learning_rate"])
print("Macro ST-RAE:", best_regression["macro_ST-RAE"])


print("\nBest classification learning rate:")
print(best_classification["learning_rate"])
print("Macro MCC:", best_classification["macro_MCC"])


# ==================================================
# 9. REGRESSION PLOT
# ==================================================
# This plot shows how regression performance changes
# when we change the learning rate.
#
# Remember:
# LOWER macro ST-RAE is better.

plt.figure()

plt.plot(
    results_df["learning_rate"],
    results_df["macro_ST-RAE"],
    marker="o"
)

plt.xlabel("Learning rate")
plt.ylabel("Macro ST-RAE")
plt.title("Learning rate vs Regression Performance")

plt.grid(True)

regression_plot_path = os.path.join(
    output_folder,
    "learning_rate_regression.png"
)

plt.savefig(
    regression_plot_path,
    bbox_inches="tight"
)

plt.show()

print(f"Regression plot saved to: {regression_plot_path}")


# ==================================================
# 10. CLASSIFICATION PLOT
# ==================================================
# This plot shows how classification performance changes
# when we change the learning rate.
#
# Remember:
# HIGHER macro MCC is better.

plt.figure()

plt.plot(
    results_df["learning_rate"],
    results_df["macro_MCC"],
    marker="o"
)

plt.xlabel("Learning rate")
plt.ylabel("Macro MCC")
plt.title("Learning rate vs Classification Performance")

plt.grid(True)

classification_plot_path = os.path.join(
    output_folder,
    "learning_rate_classification.png"
)

plt.savefig(
    classification_plot_path,
    bbox_inches="tight"
)

plt.show()

print(f"Classification plot saved to: {classification_plot_path}")