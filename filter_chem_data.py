import pandas as pd

# Define file paths
chemical_protein_file = "/scratch/gpfs/jg9705/IW_data/gen_ideal_protein_chem_scores/9606.protein_chemical.links.v5.0.tsv"  # Update this
large_chemicals_file = "/scratch/gpfs/jg9705/IW_data/gen_ideal_protein_chem_scores/chemicals.v5.0.tsv"   # Update this
unique_chemicals_output = "/scratch/gpfs/jg9705/IW_data/Generate_Chemical_Embeddings/unique_chemicals.tsv"
filtered_chemicals_output = "scratch/gpfs/jg9705/IW_data/Generate_Chemical_Embeddings/filtered_chemicals.tsv"

### **STEP 1: Extract Unique Chemicals from Chemical-Protein Pairs Dataset**
print("Reading chemical-protein dataset...")
df = pd.read_csv(chemical_protein_file, sep="\t", usecols=["SMILES_string"])  # Keep only chemical column

print("Removing duplicates...")
unique_chemicals = df["SMILES_string"].drop_duplicates().reset_index(drop=True)

print(f"Found {len(unique_chemicals)} unique chemicals.")

print("Saving unique chemicals...")
unique_chemicals.to_csv(unique_chemicals_output, sep="\t", index=False, header=True)

### **STEP 2: Filter Large Chemicals Dataset**
print("Loading unique chemicals into a set for fast lookup...")
unique_chemicals_set = set(unique_chemicals)

print("Reading large chemicals dataset in chunks...")
chunksize = 500000  # Process 500,000 rows at a time
filtered_chunks = []

for chunk in pd.read_csv(large_chemicals_file, sep="\t", chunksize=chunksize):
    filtered_chunk = chunk[chunk["SMILES_string"].isin(unique_chemicals_set)]
    filtered_chunks.append(filtered_chunk)

print("Concatenating filtered data...")
filtered_chemicals = pd.concat(filtered_chunks, ignore_index=True)

print(f"Filtered dataset contains {len(filtered_chemicals)} matching chemicals.")

print("Saving filtered chemicals...")
filtered_chemicals.to_csv(filtered_chemicals_output, sep="\t", index=False, header=True)

print("Processing complete. Filtered chemicals saved!")
