import os
import json
import pandas as pd
import torch
from kfold_classifier import load_embedding_file, filter_pairs, InteractionDataset, final_train_and_save

# === Paths ===
threshold = 400
base_dir = "/scratch/gpfs/jg9705/IW_code/results_kfold_final"
train_path = f"/scratch/gpfs/jg9705/IW_code/Model_Resources/splits/threshold{threshold}/train.tsv"
chem_fp_path = "/scratch/gpfs/jg9705/IW_code/Model_Resources/molecular_fingerprints.tsv"
prot_emb_path = "/scratch/gpfs/jg9705/IW_code/Model_Resources/encodings.tsv"
params_path = f"{base_dir}/threshold{threshold}/optuna_best_params_{threshold}.json"
out_dir = f"{base_dir}/threshold{threshold}"
model_path = os.path.join(out_dir, f"final_model_{threshold}.pt")

# === Load Data ===
train_df = pd.read_csv(train_path, sep="\t")
chem_lookup = load_embedding_file(chem_fp_path)
prot_lookup = load_embedding_file(prot_emb_path, embedding_col=1)
train_df = filter_pairs(train_df, chem_lookup, prot_lookup)
print(f"Training samples after filtering: {len(train_df)}")

input_dim = len(next(iter(chem_lookup.values()))) + len(next(iter(prot_lookup.values())))
full_dataset = InteractionDataset(train_df, chem_lookup, prot_lookup)

# === Load Best Params ===
with open(params_path, "r") as f:
    best_params = json.load(f)

# === Train and Save Final Model ===
model = final_train_and_save(full_dataset, input_dim, best_params, model_path, out_dir, threshold)
