import pandas as pd
import os

# Define file paths
chemical_protein_file = "/scratch/gpfs/jg9705/IW_data/gen_ideal_protein_chem_scores/9606.protein_chemical.links.v5.0.tsv"
large_chemicals_file = "/scratch/gpfs/jg9705/IW_data/gen_ideal_protein_chem_scores/chemicals.v5.0.tsv"
unique_chemicals_output = "/scratch/gpfs/jg9705/IW_data/Generate_Chemical_Embeddings/unique_chemicals.tsv"
filtered_chemicals_output = "/scratch/gpfs/jg9705/IW_data/Generate_Chemical_Embeddings/filtered_chemicals.tsv"

### **STEP 1: Check if Unique Chemicals File Exists**
if os.path.exists(unique_chemicals_output):
    print(f"Unique chemicals file '{unique_chemicals_output}' found. Loading existing file...")
    unique_chemicals = pd.read_csv(unique_chemicals_output, sep="\t")["chemical"].tolist()
else:
    print("Unique chemicals file not found. Extracting unique chemicals from dataset...")

    # Read chemical-protein dataset and extract unique chemicals
    df = pd.read_csv(chemical_protein_file, sep="\t", usecols=["chemical"])  # Keep only chemical column
    unique_chemicals = df["chemical"].drop_duplicates().reset_index(drop=True)

    print(f"Found {len(unique_chemicals)} unique chemicals.")

    print("Saving unique chemicals...")
    unique_chemicals.to_csv(unique_chemicals_output, sep="\t", index=False, header=True)

# Convert unique chemicals list to a set for fast lookup
unique_chemicals_set = set(unique_chemicals)

### **STEP 2: Filter Large Chemicals Dataset**
print("Reading large chemicals dataset in chunks...")

chunksize = 500000  # Process 500,000 rows at a time
output_mode = "w"  # Start with write mode (overwrite if file exists)

for chunk in pd.read_csv(large_chemicals_file, sep="\t", chunksize=chunksize):
    filtered_chunk = chunk[chunk["chemical"].isin(unique_chemicals_set)]

    if not filtered_chunk.empty:
        # Append to output file without keeping everything in memory
        filtered_chunk.to_csv(filtered_chemicals_output, sep="\t", mode=output_mode, index=False, header=(output_mode == "w"))
        output_mode = "a"  # Switch to append mode after the first chunk

print(f"Filtered dataset saved to {filtered_chemicals_output}")
