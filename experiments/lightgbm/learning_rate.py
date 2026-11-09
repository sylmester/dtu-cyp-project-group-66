# Learning rates

import subprocess

# Learning rates to test, might change this
learning_rates = [0.01, 0.03, 0.05, 0.10]

# Loop over them all
for learning_rate in learning_rates:

    print(f"\nTesting learning rate: {learning_rate}")

    # Run regression first
    subprocess.run([
        "python",
        "finetune.py",
        "model=morgan_rdkit_lgbm",
        "task=regression",
        f"model.params.learning_rate={learning_rate}"
    ])

    # Then run classification
    subprocess.run([
        "python",
        "finetune.py",
        "model=morgan_rdkit_lgbm",
        "task=classification",
        f"model.params.learning_rate={learning_rate}"
    ])