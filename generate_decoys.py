import pandas as pd
import random

# Parameters
n_decoys_per_protein = 50  # adjust this as needed

# Load data
protein_df = pd.read_csv("/scratch/gpfs/jg9705/IW_data/gen_ideal_protein_chem_scores/unique_proteins.tsv", sep="\t")
chemical_df = pd.read_csv("/scratch/gpfs/jg9705/IW_data/Generate_Chemical_Embeddings/unique_chemicals.tsv", sep="\t")
pairs_df = pd.read_csv("/scratch/gpfs/jg9705/IW_data/gen_ideal_protein_chem_scores/9606.protein_chemical.links.v5.0.tsv", sep="\t")

# For fast lookup of known pairs
known_pairs = set(zip(pairs_df['protein'], pairs_df['chemical']))

# All available chemicals
all_chemicals = chemical_df['chemical'].tolist()

decoy_data = []

for protein in protein_df['protein']:
    decoys = set()
    attempts = 0
    max_attempts = 100 * n_decoys_per_protein  # prevent infinite loop

    while len(decoys) < n_decoys_per_protein and attempts < max_attempts:
        chem = random.choice(all_chemicals)
        if (protein, chem) not in known_pairs:
            decoys.add(chem)
        attempts += 1

    for chem in decoys:
        decoy_data.append({'protein': protein, 'chemical': chem, 'label': 0})  # label = 0 for decoy

# Convert to DataFrame and save
decoy_df = pd.DataFrame(decoy_data)
decoy_df.to_csv("/scratch/gpfs/jg9705/IW_data/gen_ideal_protein_chem_scores/decoy_pairs.tsv", sep="\t", index=False)
