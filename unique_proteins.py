import pandas as pd

# Load the original TSV file
df = pd.read_csv("/scratch/gpfs/jg9705/IW_data/gen_ideal_protein_chem_scores/9606.protein_chemical.links.v5.0.tsv", sep="\t")

# Get the unique proteins
unique_proteins = df["protein"].drop_duplicates()

# Save to a new TSV file
unique_proteins.to_csv("unique_proteins.tsv", sep="\t", index=False, header=["protein"])
