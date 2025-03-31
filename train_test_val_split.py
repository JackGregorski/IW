import pandas as pd
import argparse
import random
import os

def split_by_cluster(input_file, output_dir, train_frac=0.7, val_frac=0.15, seed=42):
    # Load full dataset
    df = pd.read_csv(input_file, sep='\t')
    df['protein_cluster'] = df['protein_cluster'].astype(int)

    # Get unique clusters
    clusters = df['protein_cluster'].unique().tolist()
    random.seed(seed)
    random.shuffle(clusters)

    # Compute split indices
    n_total = len(clusters)
    n_train = int(train_frac * n_total)
    n_val = int(val_frac * n_total)

    train_clusters = set(clusters[:n_train])
    val_clusters = set(clusters[n_train:n_train + n_val])
    test_clusters = set(clusters[n_train + n_val:])

    # Assign rows based on their cluster
    train_df = df[df['protein_cluster'].isin(train_clusters)]
    val_df = df[df['protein_cluster'].isin(val_clusters)]
    test_df = df[df['protein_cluster'].isin(test_clusters)]

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Save to separate files
    train_df.to_csv(os.path.join(output_dir, 'train.tsv'), sep='\t', index=False)
    val_df.to_csv(os.path.join(output_dir, 'val.tsv'), sep='\t', index=False)
    test_df.to_csv(os.path.join(output_dir, 'test.tsv'), sep='\t', index=False)

    print(f"Data split complete.\nTrain: {len(train_df)} rows\nVal: {len(val_df)} rows\nTest: {len(test_df)} rows")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split dataset by protein clusters into train/val/test.")
    parser.add_argument("input_file", help="Path to the input .tsv file with protein_cluster column")
    parser.add_argument("output_dir", help="Directory to save train.tsv, val.tsv, test.tsv")
    parser.add_argument("--train_frac", type=float, default=0.7, help="Fraction of clusters to use for training")
    parser.add_argument("--val_frac", type=float, default=0.15, help="Fraction of clusters to use for validation")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")

    args = parser.parse_args()
    split_by_cluster(args.input_file, args.output_dir, args.train_frac, args.val_frac, args.seed)
