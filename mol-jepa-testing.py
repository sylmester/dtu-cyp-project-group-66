from pathlib import Path

from models import MODELS
from transformers import AutoModel
from prepare_data import REG, CLS


model = AutoModel.from_pretrained("Flogrammer/Mol-JEPA", trust_remote_code=True)
model.eval()

smiles_list = ["Cn1cnc2n(C)c(=O)n(C)c(=O)c12", "CC(=O)Oc1ccccc1C(=O)O"]
out = model(smiles_list)
print(out)

print("Predicted embeddings shape:", out.predictions.shape)
print("CLS token shape:", out.cls.shape)
print("Latent embeddings shape:", out.embeddings.shape)
print("All 12 modality embeddings flattened", out.predictions.flatten(1).shape) # All 12 modality embeddings flattened


cfg = {
  "name": "mol_jepa",
  "type": "mol_jepa",
  "freeze_encoder": True,
  "readout_lr": 1.0e-3,
  "readout": {"type": "mlp", "hidden_dim": 512, "num_layers": 3, "dropout": 0.1},
  "data": "data"
}

from omegaconf import OmegaConf
cfg = OmegaConf.create(cfg)

task = 'regression'
targets = REG if task == 'regression' else CLS

data = Path(cfg["data"])

model = MODELS[cfg["type"]](cfg, cfg["readout"], len(targets), data, "cpu")
print(model)
