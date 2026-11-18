import numpy as np, pandas as pd, torch
from transformers import AutoModel

model = AutoModel.from_pretrained("Flogrammer/Mol-JEPA", trust_remote_code=True)
model.eval()

# Generate embeddings from model one batch at a time to avoid running out of memory..
def embedding(smiles_list, batch_size=64):

    cls_token, pred, emb = [], [], []

    with torch.no_grad():
        for start in range(0, len(smiles_list), batch_size):
            print(f"Batch {start/batch_size} / {len(smiles_list) // batch_size}")
            batch = smiles_list[start:start + batch_size]
            out = model(batch)

            cls_token.append(out.cls) # size 512
            pred.append(out.predictions) # size 12, 512
            emb.append(out.embeddings) # size 13, 512
    return [torch.cat(x).numpy() for x in (cls_token, pred, emb)]



for task in ['regression', 'classification']:
    for split in ['train', 'val', 'test']:
        smiles = pd.read_csv(f'data/{task}/{split}.csv').SMILES
        smiles = np.asarray(smiles) # pd -> np array
        cls_token, pred, emb = embedding(smiles)
        np.savez(f'data/{task}/{split}_mol_jepa_emb.npz', cls=cls_token, pred=pred, emb=emb)