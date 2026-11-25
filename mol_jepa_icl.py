from tabicl import TabICLRegressor, TabICLClassifier
from transformers import AutoModel
import pandas as pd
import numpy as np
import torch
from prepare_data import REG, CLS
from sklearn.decomposition import PCA

from finetune import metrics

from omegaconf import OmegaConf

# We already have the splits
cfg = OmegaConf.load('conf/config.yaml')
task = cfg.task
targets = REG if task == 'regression' else CLS
train = pd.read_csv(f'data/{task}/train.csv')
val = pd.read_csv(f'data/{task}/val.csv')
test = pd.read_csv(f'data/{task}/test.csv')

# Run "mol_jepa_create_embeddings.py" to create embeddings once and then load them
ds_train = np.load(f"data/{task}/train_mol_jepa_emb.npz")
ds_val = np.load(f"data/{task}/val_mol_jepa_emb.npz")
ds_test = np.load(f"data/{task}/test_mol_jepa_emb.npz")

readout_input = "cls" # cls (512) | pred (6144))

# Flatten pred 12x512 to 6144
train_features = torch.tensor(ds_train[readout_input]).flatten(1)
val_features = torch.tensor(ds_val[readout_input]).flatten(1)
test_features = torch.tensor(ds_test[readout_input]).flatten(1)

# With 512 features the TabICL blows up memory usage...
# Use PCA to reduce embedding dimension.
pca = PCA(n_components=64).fit(train_features)

X_train_pca = pca.transform(train_features)
X_val_pca = pca.transform(val_features)
X_test_pca = pca.transform(test_features)


# TabICL Model.fit can take X, y
# X.shape = (n, 512) = n rows of embeddings of size 512 (With PCA: size=n_components)
# y. shape = (n) = n rows of 1 target value

#train_pred = np.zeros((len(train), len(targets)))
val_pred = np.zeros((len(val), len(targets)))
test_pred = np.zeros((len(test), len(targets)))

# This loop fits 2 or 4 different models. One for each target. It combines their predictions into one metric.
for j, target in enumerate(targets):

    print(f"Fitting and predicting target {j}: {target}..")
    train_mask = train[target].notna().to_numpy()   # train with all molecules that have this target
    
    X = X_train_pca[train_mask]              # [n_t, 512]
    y = train[target][train_mask].to_numpy() # [n_t]
    
    if task == "regression":
        model_t = TabICLRegressor(n_estimators=8, batch_size=8, random_state=123).fit(X, y)
        
        #train_pred[:, j] = model_t.predict(X_train_pca) # all train rows
        val_pred[:, j] = model_t.predict(X_val_pca) # predict on all val rows
        test_pred[:, j] = model_t.predict(X_test_pca) # all test rows
    else:
        model_t = TabICLClassifier(n_estimators=8, batch_size=8, random_state=123).fit(X, y)
        print("Classes: ", model_t.classes_)
        #train_pred[:, j] = model_t.predict_proba(X_train_pca)[:, 1]
        val_pred[:, j] = model_t.predict_proba(X_val_pca)[:, 1] # take the positive class probability
        test_pred[:, j] = model_t.predict_proba(X_test_pca)[:, 1]


# print(f"\n Training matrics:")
# print(metrics(train, train_pred, targets, task))

print(f"\n Validation matrics:")
print(metrics(val, val_pred, targets, task))

print(f"\n Test metrics: ")
print(metrics(test, test_pred, targets, task))