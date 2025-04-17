import pandas as pd
import argparse
import random
from collections import defaultdict

# Hardcoded path to all possible chemicals
CHEMICAL_DATASET_PATH = "/scratch/gpfs/jg9705/IW_data/Generate_Chemical_Embeddings/filtered_chemicals.tsv"

def filter_and_generate_negatives(input_file, base_output_file, threshold):
    random.seed(42)  # Reproducibility

    df = pd.read_csv(input_file, sep='\t')
    filtered_df = df[df['combined_score'] > threshold].copy()
    filtered_df['label'] = 1

    all_chemicals_df = pd.read_csv(CHEMICAL_DATASET_PATH, sep='\t')
    all_chemicals = set(all_chemicals_df['chemical'])

    protein_to_positives = defaultdict(set)
    for _, row in filtered_df.iterrows():
        protein_to_positives[row['protein']].add(row['chemical'])

    negative_rows = []
    for protein, pos_chems in protein_to_positives.items():
        num_positives = len(pos_chems)
        num_negatives = min(100, max(num_positives, 20))

        possible_negatives = list(all_chemicals - pos_chems)
        sampled = random.sample(possible_negatives, min(num_negatives, len(possible_negatives)))

        for chem in sampled:
            negative_rows.append({
                'chemical': chem,
                'protein': protein,
                'combined_score': 0,
                'label': 0
            })

    negatives_df = pd.DataFrame(negative_rows)
    combined_df = pd.concat([filtered_df, negatives_df], ignore_index=True)

    output_file = f"{base_output_file}_threshold{threshold}.tsv"
    combined_df.to_csv(output_file, sep='\t', index=False)
    print(f"Saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter by score and generate negatives for multiple thresholds.")
    parser.add_argument("input_file", help="Path to input .tsv file")
    parser.add_argument("output_base", help="Base name for output files")
    parser.add_argument("thresholds", help="Comma-separated list of thresholds (e.g., 400,500,600)")

    args = parser.parse_args()
    thresholds = [int(t.strip()) for t in args.thresholds.split(",")]

    for threshold in thresholds:
        filter_and_generate_negatives(args.input_file, args.output_base, threshold)
