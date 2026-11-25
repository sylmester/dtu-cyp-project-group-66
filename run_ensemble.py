from pathlib import Path

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf

from finetune import metrics, predict
from models import MODELS
from models.gbm import LightGBMModel
from prepare_data import REG, CLS
import lightgbm as lgb

cfg = OmegaConf.load('conf/config.yaml')
task = cfg.task
data_dir = Path(cfg.data)
device = cfg.device
targets = REG if task == 'regression' else CLS
seed = cfg.seed

MORGAN_RDKIT_MLP_RUN = Path(f'runs/{task}/morgan_rdkit_mlp/2026-11-23_16-10-08') # reg 2026-11-23_16-10-08 | clas 2026-11-23_16-54-09
MOL_JEPA_RUN = Path(f'runs/{task}/mol_jepa/2026-11-23_14-19-52') # reg: 2026-11-23_14-19-52 | clas 2026-11-23_21-30-44
ATOM_JEPA_RUN = Path(f'runs/{task}/atom_jepa/2026-11-23_16-42-02')
LGBM_RUN = Path(f'runs/{task}/morgan_rdkit_lgbm/2026-11-25_17-36-57') # reg: 2026-11-25_17-36-57 | cls: 2026-11-25_17-37-16


# Funciton to get predictors for all models except lightGBM which was handled seperately in finetune.py...
def load_torch_predictor(run_dir, train):
    checkpoint = torch.load(run_dir / 'best.pt', map_location=device, weights_only=True)
    saved_cfg = OmegaConf.create(checkpoint['config'])

    model = MODELS[saved_cfg.model.type](saved_cfg.model, saved_cfg.readout, len(targets), data_dir, device)
    model.fit_preprocessing(train)
    model.load_state_dict(checkpoint['model'])

    mean = checkpoint['mean']  # (n_targets,)
    std = checkpoint['std']  # (n_targets,)

    def predict_frame(frame):
        loader = model.loader(frame, targets, saved_cfg.train.batch_size)
        return predict(model, loader, task, mean, std)  # predict function given in finetune.py 

    return predict_frame

# function to load predictions for lgbm
def load_lgbm_predictor(run_dir, model_config, seed):
    # 1st: load saved boosters
    boosters = []
    for target in targets:
        booster = lgb.Booster(model_file=str(run_dir / f'{target}.lgbm.txt'))
        boosters.append(booster)

    # 2nd: predict function. uses the loaded boosters to predict. (boosters take features which we use the model for)
    feature_model = LightGBMModel(model_config, data_dir, seed)
    def predict_frame(frame):
        features = feature_model.features(frame)  # (n_molecules, n_features)

        columns = []
        for booster in boosters:
            column = booster.predict(features)  # (n_molecules,)
            columns.append(column)

        return np.stack(columns, axis=1)  # (n_molecules, n_targets)

    return predict_frame


class EnsemblePredictor:
    def __init__(self, predict_functions):
        self.predict_functions = predict_functions

    def predict(self, frame):
        all_predictions = []
        for predict_function in self.predict_functions:
            predictions = predict_function(frame)  # (n_molecules, n_targets)
            all_predictions.append(predictions)
        stacked_predictions = np.stack(all_predictions)  # (n_models, n_molecules, n_targets)

        # Simply take the average prediction:
        return stacked_predictions.mean(axis=0)  # (n_molecules, n_targets)


train_all = pd.read_csv(data_dir / task / 'train.csv')
val_all = pd.read_csv(data_dir / task / 'val.csv')
test = pd.read_csv(data_dir / task / 'test.csv')

train = train_all[train_all[targets].notna().any(axis=1)].copy()
val = val_all[val_all[targets].notna().any(axis=1)].copy()

morgan_rdkit_mlp_predict = load_torch_predictor(MORGAN_RDKIT_MLP_RUN, train)
mol_jepa_predict = load_torch_predictor(MOL_JEPA_RUN, train)
atom_jepa_predict = load_torch_predictor(ATOM_JEPA_RUN, train)
# LGBM
lgbm_cfg = OmegaConf.load('conf/model/morgan_rdkit_lgbm.yaml')
lgbm_predict = load_lgbm_predictor(LGBM_RUN, lgbm_cfg.model, cfg.seed)

#ensemble = EnsemblePredictor([morgan_rdkit_mlp_predict])
#ensemble = EnsemblePredictor([lgbm_predict])
#ensemble = EnsemblePredictor([mol_jepa_predict])
#ensemble = EnsemblePredictor([atom_jepa_predict])
ensemble = EnsemblePredictor([morgan_rdkit_mlp_predict, mol_jepa_predict, lgbm_predict])
#ensemble = EnsemblePredictor([morgan_rdkit_mlp_predict, mol_jepa_predict, atom_jepa_predict])

validation_predictions = ensemble.predict(val)
print('Validation:', metrics(val, validation_predictions, targets, task))

test_predictions = ensemble.predict(test)
print('Test:', metrics(test, test_predictions, targets, task))