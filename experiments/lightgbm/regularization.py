# Regularization testing

import subprocess
import glob
import json
import os
import pandas as pd
import matplotlib.pyplot as plt

# The num lambda values to test and saving the results

reg_lambda_values = [0, 0.1, 0.5, 1, 5, 10, 20, 50, 75, 100, 200]

results = []

# Output folder for the experiments

output_folder = "experiments/results"
os.makedirs(output_folder, exist_ok=True)

# We test every lambda

for reg_lambda in reg_lambda_values:
    print(f"\n Testing reg_lambda: {reg_lambda}\n")

    subprocess.run(["python", 
                    "finetune.py", 
                    "model=morgan_rdkit_lgbm", 
                    "task=regression", 
                    f"model.params.reg_lambda={reg_lambda}"], check=True)

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
                    f"model.params.reg_lambda={reg_lambda}"], check=True)

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
    results.append({"reg_lambda": reg_lambda,
                   "macro_ST-RAE": macro_st_rae,
                   "macro_MCC": macro_mcc})

# turn it into a table and save it in a csv

results_data = pd.DataFrame(results)
csv_path = os.path.join(output_folder, "regularization_results.csv")
results_data.to_csv(csv_path, index=False)

print("\n The final results are:")
print(results_data)

# We then find the best regression reg_lambda and the best classification reg_lambda

best_reg = results_data.loc[results_data["macro_ST-RAE"].idxmin()]
best_class = results_data.loc[results_data["macro_MCC"].idxmax()]

print("\nThe Best regression reg_lambda is:")
print(best_reg["reg_lambda"])
print("macro_ST-RAE", best_reg["macro_ST-RAE"])

print("\nThe Best classification reg_lambda is:")
print(best_class["reg_lambda"])
print("macro_MCC", best_class["macro_MCC"])

# plotting it just so we can visualise it

plt.figure()
plt.plot(results_data["reg_lambda"], results_data["macro_ST-RAE"], marker="o")
plt.xlabel("Tested reg_lambda")
plt.ylabel("macro_ST-RAE")
plt.title("Regularization against regression performance")
plt.grid(True)

reg_plot_path = os.path.join(output_folder, "regularization_reg.png")
plt.savefig(reg_plot_path, bbox_inches="tight")
plt.show()


plt.figure()
plt.plot(results_data["reg_lambda"], results_data["macro_MCC"], marker="o")
plt.xlabel("Tested reg_lambda")
plt.ylabel("macro_MCC")
plt.title("Regularization against classification performance")
plt.grid(True)

class_plot_path = os.path.join(output_folder, "regularization_class.png")
plt.savefig(class_plot_path, bbox_inches="tight")
plt.show()
    