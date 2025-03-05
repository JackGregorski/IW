# Activate conda environment before running: conda activate sn-torch-env

# A. Load necessary packages
import pandas as pd
import argparse
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
import torch
import sys

def main():
    # B. Set up command-line arguments
    parser = argparse.ArgumentParser(
        description="Generate molecular fingerprints from SMILES strings in a CSV file."
    )
    parser.add_argument("input_file", help="Path to the input CSV file containing a 'SMILES' column")
    parser.add_argument("output_file", help="Path to the output file where embeddings will be saved")
    args = parser.parse_args()

    # 1. Load SMILES sequences from the input CSV file
    try:
        MTD = pd.read_csv(args.input_file)
        if "SMILES" not in MTD.columns:
            print("Error: The input file must contain a 'SMILES' column.")
            sys.exit(1)

        Chem_List = MTD['SMILES'].astype(str).tolist()
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
            features = np.zeros((1,))
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
