# INFO on the MOL-JEPAA model

Instructions: https://huggingface.co/Flogrammer/Mol-JEPA
GitHub: https://github.com/Boehringer-Ingelheim/mol-jepa

Install in your env: `pip install torch torch-geometric transformers safetensors rdkit molfeat numpy`

Might need instead an earlier version of transformers: `pip install transformers<5`

To work with TabICL:
TabICL: `pip install tabicl`

DOCS: https://tabicl.readthedocs.io/en/latest/index.html

The model is intergrated into the finetune script so it can be run like the others when using the normal readouts.

Modify the mlp readout config to test different configs. The pretrained model is frozen, so we are basically fitting the readout layer from statisc embeddings which are the output form the pretrained model.

TabICL is the "Open Tabular Foundation Model" which is a pretrained model with a transformer architecture and it doesnt need any training. Instead we fit it to our data once and can then use it to predict.

Use it in `mol_jepa_icl.py`
