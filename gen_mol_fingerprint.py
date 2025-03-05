#!/usr/bin/env python

# A. Load necessary packages
import pandas as pd
import argparse
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
import torch
import sys
import numpy as np

def main():
    # B. Set up command-line arguments
    parser = argparse.ArgumentParser(
        description="Generate molecular fingerprints from SMILES strings in a TSV file."
    )
    parser.add_argument("input_file", help="Path to the input TSV file containing a 'SMILES_string' column")
    parser.add_argument("output_file", help="Path to the output file where embeddings will be saved")
    args = parser.parse_args()

    # 1. Load SMILES sequences from the input TSV file
    try:
        # Read the TSV file, handling potential formatting issues
        MTD = pd.read_csv(args.input_file, sep="\t", engine="python", on_bad_lines="skip")

        # Ensure 'SMILES_string' column exists
        if "SMILES_string" not in MTD.columns:
            print("Error: The input file must contain a 'SMILES_string' column.")
            print(f"Columns found: {MTD.columns.tolist()}")
            sys.exit(1)

        Chem_List = MTD["SMILES_string"].astype(str).tolist()

    except Exception as e:
        print(f"Error reading input file: {e}")
        sys.exit(1)

    # 2. Generate molecular fingerprints
    Chem_Embeddings_List = []

    for i, smiles in enumerate(Chem_List):
        try:
            Mol = Chem.MolFromSmiles(smiles)
            if Mol is None:
                print(f"Warning: Invalid SMILES '{smiles}' at row {i}, skipping.")
                continue

            morganfp = AllChem.GetMorganFingerprintAsBitVect(Mol, radius=2, nBits=2048)
            features = np.zeros((2048,))
            DataStructs.ConvertToNumpyArray(morganfp, features)
            feats = torch.from_numpy(features).squeeze().float()
            Chem_Embeddings_List.append(feats)

        except Exception as e:
            print(f"Error processing SMILES '{smiles}' at row {i}: {e}")

    # 3. Save embeddings to the output file
    try:
        with open(args.output_file, 'w') as file:
            for tensor in Chem_Embeddings_List:
                values = [str(val.item()) for val in tensor.view(-1)]
                file.write(' '.join(values) + '\n')

        print(f"Finished processing. Embeddings saved to {args.output_file}")

    except Exception as e:
        print(f"Error writing output file: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
