"""Pretrained Mol-JEPA encoder with a readout head from models/readouts.py."""
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel

from models.readouts import build_readout


class SMILES(Dataset):
    def __init__(self, frame, targets):
        self.frame = frame.reset_index(drop=True)
        self.y = frame.reindex(columns=targets).to_numpy(dtype=np.float32)

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, i):
        return self.frame.iloc[i].SMILES


class MolJEPAModel(nn.Module):
    def __init__(self, cfg, readout, n_targets, data_dir, device):
        super().__init__()
        self.cfg = cfg

        self.encoder = AutoModel.from_pretrained("Flogrammer/Mol-JEPA", trust_remote_code=True)
        self.encoder._set_gradient_checkpointing(not cfg.freeze_encoder)
        self.encoder.requires_grad_(not cfg.freeze_encoder)
        self.readout = build_readout(readout, 512, n_targets).to(self.encoder.device)

    @property
    def device(self):
        return self.encoder.device

    def train(self, mode=True):
        super().train(mode)
        if self.cfg.freeze_encoder:
            self.encoder.eval()
        return self

    def fit_preprocessing(self, train):
        pass

    def loader(self, frame, targets, batch_size, shuffle=False):
        return DataLoader(SMILES(frame, targets), batch_size=batch_size,
                          shuffle=shuffle, collate_fn=self.encoder.collate, num_workers=0)

    def param_groups(self):
        groups = [{'params': list(self.readout.parameters()), 'lr': self.cfg.readout_lr}]
        # if not self.cfg.freeze_encoder:
        #     groups.append({'params': list(self.encoder.parameters()), 'lr': self.cfg.encoder_lr})
        return groups

    def forward(self, batch):
        return self.readout(self.encoder(batch))  # Differentiable forward; do NOT use encoder.embed().
