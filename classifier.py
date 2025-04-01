import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, roc_auc_score
import argparse

# Dataset class
class InteractionDataset(Dataset):
    def __init__(self, df, chem_lookup, prot_lookup):
        self.data = df
        self.chem_lookup = chem_lookup
        self.prot_lookup = prot_lookup

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        chem_fp = self.chem_lookup.get(row["chemical"])
        prot_emb = self.prot_lookup.get(row["protein"])
        if chem_fp is None or prot_emb is None:
            raise ValueError(f"Missing embedding for {row['chemical']} or {row['protein']}")
        x = torch.cat([chem_fp, prot_emb], dim=0)
        y = torch.tensor(row["label"], dtype=torch.float32)
        return x, y

# Simple binary classifier
class InteractionClassifier(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        return self.model(x).squeeze()

# Load embedding files
def load_embedding_file(path, has_header=False, embedding_col=1):
    lookup = {}
    with open(path, 'r') as f:
        if has_header:
            next(f)
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) <= embedding_col:
                continue
            key = parts[0]
            embedding_str = parts[embedding_col].strip()
            if embedding_str == "NA":
                continue
            try:
                vec = torch.tensor([float(x) for x in embedding_str.split()], dtype=torch.float32)
                lookup[key] = vec
            except ValueError as e:
                print(f"Skipping {key} due to conversion error: {e}")
    print(f"Loaded {len(lookup)} entries from {path}")
    return lookup

# Filter out rows with missing embeddings
def filter_pairs(df, chem_lookup, prot_lookup):
    original_len = len(df)
    valid_df = df[
        df["chemical"].isin(chem_lookup.keys()) &
        df["protein"].isin(prot_lookup.keys())
    ]
    dropped = original_len - len(valid_df)
    print(f"Filtered out {dropped} rows without valid embeddings.")
    return valid_df

# Training loop
def train_model(model, train_loader, val_loader, epochs=10, lr=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

        # Evaluation
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                logits = model(x)
                probs = torch.sigmoid(logits).cpu().numpy()
                preds = (probs > 0.5).astype(int)
                all_preds.extend(preds)
                all_labels.extend(y.numpy())

        acc = accuracy_score(all_labels, all_preds)
        auc = roc_auc_score(all_labels, all_preds)
        print(f"Epoch {epoch+1}: Val Acc = {acc:.4f}, AUC = {auc:.4f}")

# Main
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--val", required=True)
    parser.add_argument("--chem_fp", required=True)
    parser.add_argument("--prot_emb", required=True)
    args = parser.parse_args()

    # Load data
    train_df = pd.read_csv(args.train, sep="\t")
    val_df = pd.read_csv(args.val, sep="\t")

    chem_lookup = load_embedding_file(args.chem_fp, has_header=False, embedding_col=1)
    prot_lookup = load_embedding_file(args.prot_emb, has_header=False, embedding_col=2)

    # Check for missing keys
    print(f"Checking embedding coverage in train and val sets...")

    train_df = filter_pairs(train_df, chem_lookup, prot_lookup)
    val_df = filter_pairs(val_df, chem_lookup, prot_lookup)

    if train_df.empty:
        raise ValueError("Training dataset is empty after filtering.")
    if val_df.empty:
        raise ValueError("Validation dataset is empty after filtering.")

    # Get embedding dimension
    input_dim = len(next(iter(chem_lookup.values()))) + len(next(iter(prot_lookup.values())))
    print(f"Input dimension: {input_dim}")

    # Create datasets/loaders
    train_dataset = InteractionDataset(train_df, chem_lookup, prot_lookup)
    val_dataset = InteractionDataset(val_df, chem_lookup, prot_lookup)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64)

    # Train the model
    model = InteractionClassifier(input_dim)
    train_model(model, train_loader, val_loader)

if __name__ == "__main__":
    main()
