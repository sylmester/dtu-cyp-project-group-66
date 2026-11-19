# Boosting iterations testing

import subprocess
import glob
import json
import os
import pandas as pd
import matplotlib.pyplot as plt

# The n_estimators values to test and saving the results

n_estimators_values = [10, 50, 100, 250, 500, 1000, 5000]

results = []

# Output folder for the experiments

output_folder = "experiments/results"
os.makedirs(output_folder, exist_ok=True)

# We test every n_estimators value

for n_estimators in n_estimators_values:
    print(f"\n Testing n_estimators: {n_estimators}\n")

    subprocess.run(["python", 
                    "finetune.py", 
                    "model=morgan_rdkit_lgbm", 
                    "task=regression", 
                    f"model.params.n_estimators={n_estimators}"], check=True)

    # we find the newest regression run and they are timestamped

    regression_runs = glob.glob("runs/regression/morgan_rdkit_lgbm/*")
    regression_run = sorted(regression_runs)[-1]

    # reading the validation results

    regression_metrics_path = os.path.join(regression_run, "validation_metrics.json")
    with open(regression_metrics_path) as f:
        regression_metrics = json.load(f)

    # we are only interested in teh macro_ST-RAE metric where for regression, lower is better

    macro_st_rae = regression_metrics["macro_ST-RAE"]

    # We then do it for the classification taks

    subprocess.run(["python", 
                    "finetune.py", 
                    "model=morgan_rdkit_lgbm", 
                    "task=classification", 
                    f"model.params.n_estimators={n_estimators}"], check=True)

    # we find the newest classification run and they are timestamped

    classification_runs = glob.glob("runs/classification/morgan_rdkit_lgbm/*")
    classification_run = sorted(classification_runs)[-1]

    # reading the validation results

    classification_metrics_path = os.path.join(classification_run, "validation_metrics.json")
    with open(classification_metrics_path) as f:
        classification_metrics = json.load(f)


    # we are only interested in the macro_MCC metric where for classification, higher is better

    macro_mcc = classification_metrics["macro_MCC"]

    # we save the results
    results.append({"n_estimators": n_estimators,
                   "macro_ST-RAE": macro_st_rae,
                   "macro_MCC": macro_mcc})

# turn it into a table and save it in a csv

results_data = pd.DataFrame(results)
csv_path = os.path.join(output_folder, "boosting_iterations_results.csv")
results_data.to_csv(csv_path, index=False)

print("\n The final results are:")
print(results_data)

# We then find the best regression n_estimators and the best classification learning rate

best_reg = results_data.loc[results_data["macro_ST-RAE"].idxmin()]
best_class = results_data.loc[results_data["macro_MCC"].idxmax()]

print("\nThe Best regression n_estimators is:")
print(best_reg["n_estimators"])
print("macro_ST-RAE", best_reg["macro_ST-RAE"])

print("\nThe Best classification n_estimators is:")
print(best_class["n_estimators"])
print("macro_MCC", best_class["macro_MCC"])

# plotting it just so we can visualise it

plt.figure()
plt.plot(results_data["n_estimators"], results_data["macro_ST-RAE"], marker="o")
plt.xlabel("Tested n_estimators")
plt.ylabel("macro_ST-RAE")
plt.title("Boosting iterations against regression performance")
plt.grid(True)

reg_plot_path = os.path.join(output_folder, "boosting_iterations_reg.png")
plt.savefig(reg_plot_path, bbox_inches="tight")
plt.show()


plt.figure()
plt.plot(results_data["n_estimators"], results_data["macro_MCC"], marker="o")
plt.xlabel("Tested n_estimators")
plt.ylabel("macro_MCC")
plt.title("Boosting iterations against classification performance")
plt.grid(True)

class_plot_path = os.path.join(output_folder, "boosting_iterations_class.png")
plt.savefig(class_plot_path, bbox_inches="tight")
plt.show()
    