# The final run based off of grid search results

import subprocess

print("\n Final regression config run\n")

subprocess.run(["python", 
                "finetune.py", 
                "model=morgan_rdkit_lgbm", 
                "task=regression", 
                "model.params.learning_rate=0.01",
                "model.params.num_leaves=10",
                "model.params.n_estimators=5000", 
                "model.params.reg_lambda=0"], 
                check=True)

print("\n Final classification config run\n")

subprocess.run(["python", 
                "finetune.py", 
                "model=morgan_rdkit_lgbm", 
                "task=classification", 
                "model.params.learning_rate=0.30",
                "model.params.num_leaves=2",
                "model.params.n_estimators=5000", 
                "model.params.reg_lambda=0"], 
                check=True)

print("\n Final config runs done\n")