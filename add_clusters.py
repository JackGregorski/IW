import pandas as pd
import argparse

def main():
    parser = argparse.ArgumentParser(description="Add protein cluster info to a TSV file and drop rows missing clusters.")
    parser.add_argument("input_file", help="Input .tsv file with a 'protein' column")
    parser.add_argument("lookup_file", help="Lookup .tsv file with 'protein' and 'protein_cluster' columns")
    parser.add_argument("output_file", help="Output .tsv file with cluster info added")
    args = parser.parse_args()

    # Load files
    df = pd.read_csv(args.input_file, sep='\t')
    lookup = pd.read_csv(args.lookup_file, sep='\t')

    # Merge and drop rows with missing clusters
    merged = df.merge(lookup, on="protein", how="left")
    merged = merged.dropna(subset=["protein_cluster"])

    # Save result
    merged.to_csv(args.output_file, sep='\t', index=False)
    print(f"✅ Saved merged file with clusters to: {args.output_file}")
    print(f"📉 Dropped {len(df) - len(merged)} rows without valid encodings")

if __name__ == "__main__":
    main()
