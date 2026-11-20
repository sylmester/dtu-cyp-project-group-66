# The final run based off of grid search results

import subprocess

print("\n Final regression config run\n")

subprocess.run(["python", 
                "finetune.py", 
                "model=morgan_rdkit_lgbm", 
                "task=regression", 
                f"model.params.learning_rate=0.01",
                f"model.params.num_leaves=10",
                f"model.params.n_estimators=5000", 
                f"model.params.reg_lambda=0"], 
                check=True)

print("\n Final classification config run\n")

subprocess.run(["python", 
                "finetune.py", 
                "model=morgan_rdkit_lgbm", 
                "task=classification", 
                f"model.params.learning_rate=0.30",
                f"model.params.num_leaves=2",
                f"model.params.n_estimators=5000", 
                f"model.params.reg_lambda=0"], 
                check=True)

print("\n Final config runs done\n")