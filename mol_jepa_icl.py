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

X_train, X_val = np.asarray(train.SMILES), np.asarray(val.SMILES)
y_train, y_val = np.asarray(train[targets]), np.asarray(val[targets])

# Not all molecyles contains all 4 target values.
n = train[targets].notna()
print(f"Target values per molecyle: \n", n)

# Shapes
print("X_train: ", X_train[:2], X_train.shape)
print("X_val: ", X_val[:2], X_val.shape)
print("y_train: ", y_train[:2], y_train.shape)
print("y_val: ", y_val[:2], y_val.shape)


# 
model = AutoModel.from_pretrained("Flogrammer/Mol-JEPA", trust_remote_code=True)
model.eval()

# Generate embeddings from model one batch at a time to avoid running out of memory..
def embedding(smiles_list, batch_size=64):
    all_cls = []
    with torch.no_grad():
        for start in range(0, len(smiles_list), batch_size):
            print(f"Batch {start/batch_size} / {len(smiles_list) // batch_size}")
            batch = smiles_list[start:start + batch_size]
            out = model(batch)
            all_cls.append(out.cls)  # [batch, 512]
    return torch.cat(all_cls).numpy()  # [n_molecules, 512]


cls_train_features = embedding(X_train)
cls_val_features = embedding(X_val)

# With 512 features the TabICL blows up memory usage...
# Use PCA to reduce embedding dimension.
pca = PCA(n_components=64).fit(cls_train_features)
X_train_pca, X_val_pca = pca.transform(cls_train_features), pca.transform(cls_val_features)



# TabICL Model.fit can take X, y
# X.shape = (n, 512) = n rows of embeddings of size 512 (With PCA: size=n_components)
# y. shape = (n) = n rows of 1 target value

val_pred = np.zeros((len(val), len(targets)))

for j, target in enumerate(targets):

    print(f"Fitting and predicting target {j}: {target}..")
    train_mask = train[target].notna().to_numpy()   # train molecules that have this target
    #val_mask = val[target].notna().to_numpy()   # masking valid molecyles is done in metrics, just pass all rows

    
    X = X_train_pca[train_mask]              # [n_t, 512]
    y = train[target][train_mask].to_numpy()        # [n_t]
    
    if task == "regression":
        model_t = TabICLRegressor(n_estimators=8, batch_size=8, random_state=123).fit(X, y)
        val_pred[:, j] = model_t.predict(X_val_pca) # all val rows
    else:
        model_t = TabICLClassifier(n_estimators=8, batch_size=8, random_state=123).fit(X, y)
        print("Classes: ", model_t.classes_)
        val_pred[:, j] = model_t.predict_proba(X_val_pca)[:, 1] # take the positive class probability


print(metrics(val, val_pred, targets, task))