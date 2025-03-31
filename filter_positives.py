import pandas as pd
import argparse

def filter_by_score(input_file, output_file, x):
    # Read the TSV file
    df = pd.read_csv(input_file, sep='\t')

    # Filter rows with combined_score > x
    filtered_df = df[df['combined_score'] > x]

    # Save to output file
    filtered_df.to_csv(output_file, sep='\t', index=False)
    print(f"Filtered data saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter protein-chemical interactions by combined_score.")
    parser.add_argument("input_file", help="Path to the input .tsv file")
    parser.add_argument("output_file", help="Path for the output .tsv file")
    parser.add_argument("threshold", type=int, help="Minimum combined_score to keep")

    args = parser.parse_args()
    filter_by_score(args.input_file, args.output_file, args.threshold)
