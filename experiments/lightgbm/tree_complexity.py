# Tree complexity testing

import subprocess
import glob
import json
import os
import pandas as pd
import matplotlib.pyplot as plt

# The num leaves values to test and saving the results

num_leaves_values = [2, 5, 7, 10, 31, 50, 100, 150]

results = []

# Output folder for the experiments

output_folder = "experiments/results"
os.makedirs(output_folder, exist_ok=True)

# We test every num leaves

for num_leaves in num_leaves_values:
    print(f"\n Testing num_leaves: {num_leaves}\n")

    subprocess.run(["python", 
                    "finetune.py", 
                    "model=morgan_rdkit_lgbm", 
                    "task=regression", 
                    f"model.params.num_leaves={num_leaves}"], check=True)

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
                    f"model.params.num_leaves={num_leaves}"], check=True)

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
    results.append({"num_leaves": num_leaves,
                   "macro_ST-RAE": macro_st_rae,
                   "macro_MCC": macro_mcc})

# turn it into a table and save it in a csv

results_data = pd.DataFrame(results)
csv_path = os.path.join(output_folder, "tree_complexity_results.csv")
results_data.to_csv(csv_path, index=False)

print("\n The final results are:")
print(results_data)

# We then find the best regression num_leaves and the best classification learning rate

best_reg = results_data.loc[results_data["macro_ST-RAE"].idxmin()]
best_class = results_data.loc[results_data["macro_MCC"].idxmax()]

print("\nThe Best regression num_leaves is:")
print(best_reg["num_leaves"])
print("macro_ST-RAE", best_reg["macro_ST-RAE"])

print("\nThe Best classification num_leaves is:")
print(best_class["num_leaves"])
print("macro_MCC", best_class["macro_MCC"])

# plotting it just so we can visualise it

plt.figure()
plt.plot(results_data["num_leaves"], results_data["macro_ST-RAE"], marker="o")
plt.xlabel("Tested num_leaves")
plt.ylabel("macro_ST-RAE")
plt.title("Tree complexity against regression performance")
plt.grid(True)

reg_plot_path = os.path.join(output_folder, "tree_complexity_reg.png")
plt.savefig(reg_plot_path, bbox_inches="tight")
plt.show()


plt.figure()
plt.plot(results_data["num_leaves"], results_data["macro_MCC"], marker="o")
plt.xlabel("Tested num_leaves")
plt.ylabel("macro_MCC")
plt.title("Tree complexity against classification performance")
plt.grid(True)

class_plot_path = os.path.join(output_folder, "tree_complexity_class.png")
plt.savefig(class_plot_path, bbox_inches="tight")
plt.show()
    