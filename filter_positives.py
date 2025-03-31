import pandas as pd
import argparse
import random

# Hardcoded path to all possible chemicals
CHEMICAL_DATASET_PATH = "/scratch/gpfs/jg9705/IW_data/Generate_Chemical_Embeddings/filtered_chemicals.tsv"  # one column: chemical

def filter_and_generate_negatives(input_file, output_file, threshold):
    # Load input data
    df = pd.read_csv(input_file, sep='\t')

    # Filter based on threshold
    filtered_df = df[df['combined_score'] > threshold].copy()
    filtered_df['label'] = 1  # Positive label

    # Load all possible chemicals
    all_chemicals_df = pd.read_csv(CHEMICAL_DATASET_PATH, sep='\t')
    all_chemicals = set(all_chemicals_df['chemical'])
    chemical_list = list(all_chemicals)

    # Get positive (chemical, protein) pairs from filtered data
    positive_pairs = set(zip(filtered_df['chemical'], filtered_df['protein']))

    # Group filtered positives by protein
    protein_groups = filtered_df.groupby('protein')

    # Prepare list to collect negatives
    negative_rows = []

    # Generate negatives for each protein
    for protein, group in protein_groups:
        num_positives = len(group)
        num_negatives = min(100, max(num_positives, 20))

        used_chemicals = {chem for chem, prot in positive_pairs if prot == protein}
        possible_negatives = list(all_chemicals - used_chemicals)

        sampled = random.sample(possible_negatives, min(num_negatives, len(possible_negatives)))

        for chem in sampled:
            negative_rows.append({
                'chemical': chem,
                'protein': protein,
                'combined_score': 0,
                'label': 0  # Negative label
            })

    # Combine positives and negatives
    negatives_df = pd.DataFrame(negative_rows)
    combined_df = pd.concat([filtered_df, negatives_df], ignore_index=True)

    # Save output
    combined_df.to_csv(output_file, sep='\t', index=False)
    print(f"Saved filtered positives and generated negatives to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter by score and generate negatives with labels.")
    parser.add_argument("input_file", help="Path to the input .tsv file")
    parser.add_argument("output_file", help="Path for the output .tsv file")
    parser.add_argument("threshold", type=int, help="Minimum combined_score to keep")

    args = parser.parse_args()
    filter_and_generate_negatives(args.input_file, args.output_file, args.threshold)
