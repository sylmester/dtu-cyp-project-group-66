# Grid search for LightGBM

import subprocess
import glob
import json
import os
import pandas as pd


learning_rates = [0.01, 0.05, 0.15]
num_leaves_values = [2, 10]
n_estimators_values = [500, 5000]
reg_lambda_values = [0, 5, 50]

results = []

# Output folder for the experiments

output_folder = "experiments/results"
os.makedirs(output_folder, exist_ok=True)

# We test every cobination which will be: 3*2*2*3=36 combos

for learning_rate in learning_rates:
    for num_leaves in num_leaves_values:
        for n_estimators in n_estimators_values:
            for reg_lambda in reg_lambda_values:

                print(f"\n Testing: " 
                      f"learning_rate={learning_rate}, "
                      f"num_leaves={num_leaves}, "
                      f"n_estimators={n_estimators}, "
                      f"reg_lambda={reg_lambda}\n ")

                # reg run
                subprocess.run(["python", 
                                "finetune.py", 
                                "model=morgan_rdkit_lgbm", 
                                "task=regression", 
                                f"model.params.learning_rate={learning_rate}",
                                f"model.params.num_leaves={num_leaves}",
                                f"model.params.n_estimators={n_estimators}", 
                                f"model.params.reg_lambda={reg_lambda}"], 
                                check=True)
                
                regression_runs = glob.glob("runs/regression/morgan_rdkit_lgbm/*")
                regression_run = sorted(regression_runs)[-1]

                regression_metrics_path = os.path.join(regression_run, "validation_metrics.json")
                with open(regression_metrics_path) as f:
                    regression_metrics = json.load(f)

                macro_st_rae = regression_metrics["macro_ST-RAE"]


                # class run
                subprocess.run(["python", 
                                "finetune.py", 
                                "model=morgan_rdkit_lgbm", 
                                "task=classification", 
                                f"model.params.learning_rate={learning_rate}",
                                f"model.params.num_leaves={num_leaves}",
                                f"model.params.n_estimators={n_estimators}", 
                                f"model.params.reg_lambda={reg_lambda}"], 
                                check=True)
                
                classification_runs = glob.glob("runs/classification/morgan_rdkit_lgbm/*")
                classification_run = sorted(classification_runs)[-1]

                classification_metrics_path = os.path.join(classification_run, "validation_metrics.json")
                with open(classification_metrics_path) as f:
                    classification_metrics = json.load(f)

                macro_mcc = classification_metrics["macro_MCC"]


                # saving combo
                results.append({"learning_rate": learning_rate,
                                "num_leaves": num_leaves,
                                "n_estimators": n_estimators,
                                "reg_lambda": reg_lambda,
                                "macro_ST-RAE": macro_st_rae,
                                "macro_MCC": macro_mcc})

# sacig grid search results

results_data = pd.DataFrame(results)
csv_path = os.path.join(output_folder, "grid_search_results.csv")
results_data.to_csv(csv_path, index=False)

print("\n The final results of grid search combo are:")
print(results_data)

# the best reg and class combo

best_reg = results_data.loc[results_data["macro_ST-RAE"].idxmin()]
best_class = results_data.loc[results_data["macro_MCC"].idxmax()]

print("\nThe Best regression combo is:")
print(best_reg)

print("\nThe Best classification combo is:")
print(best_class)