import subprocess
import glob
import json
import os
import pandas as pd

output_folder = "experiments/atomjepa/results"
os.makedirs(output_folder, exist_ok=True)

results = []

# regression

subprocess.run(["python", 
                "finetune.py", 
                "model=atom_jepa", 
                "task=regression"], check=True)

regression_runs = glob.glob("runs/regression/atom_jepa/*")
regression_run = sorted(regression_runs)[-1]

regression_metrics_path = os.path.join(regression_run, "validation_metrics.json")
with open(regression_metrics_path) as f:
    regression_metrics = json.load(f)

macro_st_rae = regression_metrics["macro_ST-RAE"]

# classification

subprocess.run(["python", 
                "finetune.py", 
                "model=atom_jepa", 
                "task=classification"], check=True)

classification_runs = glob.glob("runs/classification/atom_jepa/*")
classification_run = sorted(classification_runs)[-1]

classification_metrics_path = os.path.join(classification_run, "validation_metrics.json")
with open(classification_metrics_path) as f:
    classification_metrics = json.load(f)

macro_mcc = classification_metrics["macro_MCC"]

# we save the results
results.append({"macro_ST-RAE": macro_st_rae,
                "macro_MCC": macro_mcc})

results_data = pd.DataFrame(results)
csv_path = os.path.join(output_folder, "atom_jepa_baseline_results.csv")
results_data.to_csv(csv_path, index=False)

print(results_data)







