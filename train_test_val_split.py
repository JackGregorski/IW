import pandas as pd
import argparse
import random
import os

def split_by_cluster(input_file, output_dir, test_frac=0.2, toy_frac=0.1, seed=42):
    # Load full dataset
    df = pd.read_csv(input_file, sep='\t')
    df['protein_cluster'] = df['protein_cluster'].astype(int)

    # Get unique clusters
    clusters = df['protein_cluster'].unique().tolist()
    random.seed(seed)
    random.shuffle(clusters)

    # Compute split
    n_test = int(test_frac * len(clusters))
    test_clusters = set(clusters[:n_test])
    train_clusters = set(clusters[n_test:])

    train_df = df[df['protein_cluster'].isin(train_clusters)].copy()
    test_df = df[df['protein_cluster'].isin(test_clusters)].copy()

    # Save full sets
    os.makedirs(output_dir, exist_ok=True)
    train_df.to_csv(os.path.join(output_dir, 'train.tsv'), sep='\t', index=False)
    test_df.to_csv(os.path.join(output_dir, 'test.tsv'), sep='\t', index=False)

    # Generate toy subsets (10%)
    train_toy = train_df.sample(frac=toy_frac, random_state=seed)
    test_toy = test_df.sample(frac=toy_frac, random_state=seed)
    train_toy.to_csv(os.path.join(output_dir, 'train_toy.tsv'), sep='\t', index=False)
    test_toy.to_csv(os.path.join(output_dir, 'test_toy.tsv'), sep='\t', index=False)

    # === Sanity Check Logging ===
    log_path = os.path.join(output_dir, 'sanity_check_log.txt')
    with open(log_path, 'w') as log:
        # Check cluster overlap
        overlap = train_clusters & test_clusters
        log.write(f"Cluster Overlap: {len(overlap)}\n")
        if overlap:
            log.write(f"Overlapping Clusters: {sorted(list(overlap))}\n")
        else:
            log.write("✅ No overlapping clusters between train and test.\n\n")

        # Positive/Negative ratios
        def label_stats(subset, name):
            pos = (subset['label'] == 1).sum()
            neg = (subset['label'] == 0).sum()
            ratio = pos / neg if neg > 0 else float('inf')
            log.write(f"{name} Set:\n")
            log.write(f"  Positives: {pos}, Negatives: {neg}, Ratio (pos/neg): {ratio:.2f}\n\n")

        label_stats(train_df, "Train")
        label_stats(test_df, "Test")

    print(f"✅ Data split complete.\nTrain: {len(train_df)} rows\nTest: {len(test_df)} rows")
    print(f"📝 Sanity check log saved to: {log_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split dataset by protein clusters into train/test, with toy sets and sanity checks.")
    parser.add_argument("input_file", help="Path to the input .tsv file with protein_cluster column")
    parser.add_argument("output_dir", help="Directory to save outputs")
    parser.add_argument("--test_frac", type=float, default=0.2, help="Fraction of clusters to use for test set")
    parser.add_argument("--toy_frac", type=float, default=0.1, help="Fraction of each set to save as toy set")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")

    args = parser.parse_args()
    split_by_cluster(args.input_file, args.output_dir, args.test_frac, args.toy_frac, args.seed)
