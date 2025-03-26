import pandas as pd
import os

pairs_df = pd.read_csv("/scratch/gpfs/jg9705/IW_data/gen_ideal_protein_chem_scores/9606.protein_chemical.links.v5.0.tsv", sep="\t")

# Calculate the average score for each protein
average_scores = pairs_df.groupby("protein")["combined_score"].mean().reset_index()

# Save the average scores to a new TSV file
average_scores.to_csv("/scratch/gpfs/jg9705/IW_data/decoys/average_scores.tsv", sep="\t", index=False)