import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
import argparse
import matplotlib.pyplot as plt
import optuna
import os

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

# Dynamic classifier
class InteractionClassifier(nn.Module):
    def __init__(self, input_dim, hidden1, hidden2, dropout):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden2, 1),
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
    return lookup

# Filter pairs
def filter_pairs(df, chem_lookup, prot_lookup):
    return df[df["chemical"].isin(chem_lookup.keys()) & df["protein"].isin(prot_lookup.keys())]

# Training and evaluation

def train_eval_model(model, train_loader, val_loader, epochs, lr, device):
    model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for _ in range(epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device)
            logits = model(x)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_preds.extend(probs)
            all_labels.extend(y.numpy())

    auc = roc_auc_score(all_labels, all_preds)
    return auc

# Objective function for Optuna

def objective(trial, input_dim, train_dataset, val_dataset):
    hidden1 = trial.suggest_int("hidden1", 128, 1024)
    hidden2 = trial.suggest_int("hidden2", 64, hidden1)
    dropout = trial.suggest_float("dropout", 0.1, 0.5)
    lr = trial.suggest_loguniform("lr", 1e-5, 1e-2)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    epochs = trial.suggest_int("epochs", 5, 20)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    model = InteractionClassifier(input_dim, hidden1, hidden2, dropout)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return train_eval_model(model, train_loader, val_loader, epochs, lr, device)

# Final training and test evaluation

def final_train_and_evaluate(best_params, input_dim, train_dataset, test_dataset, out_dir):
    model = InteractionClassifier(input_dim, best_params['hidden1'], best_params['hidden2'], best_params['dropout'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader = DataLoader(train_dataset, batch_size=best_params['batch_size'], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=best_params['batch_size'])

    train_eval_model(model, train_loader, test_loader, best_params['epochs'], best_params['lr'], device)

    # Evaluation with ROC
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            logits = model(x)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_preds.extend(probs)
            all_labels.extend(y.numpy())

    fpr, tpr, _ = roc_curve(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_preds)

    plt.figure()
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {auc:.4f})')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Test ROC Curve')
    plt.legend(loc='lower right')
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(os.path.join(out_dir, "roc_curve.png"))

# Main

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--val", required=True)
    parser.add_argument("--test", required=True)
    parser.add_argument("--chem_fp", required=True)
    parser.add_argument("--prot_emb", required=True)
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()

    train_df = pd.read_csv(args.train, sep="\t")
    val_df = pd.read_csv(args.val, sep="\t")
    test_df = pd.read_csv(args.test, sep="\t")

    chem_lookup = load_embedding_file(args.chem_fp, has_header=False, embedding_col=1)
    prot_lookup = load_embedding_file(args.prot_emb, has_header=False, embedding_col=2)

    train_df = filter_pairs(train_df, chem_lookup, prot_lookup)
    val_df = filter_pairs(val_df, chem_lookup, prot_lookup)
    test_df = filter_pairs(test_df, chem_lookup, prot_lookup)

    input_dim = len(next(iter(chem_lookup.values()))) + len(next(iter(prot_lookup.values())))

    train_dataset = InteractionDataset(train_df, chem_lookup, prot_lookup)
    val_dataset = InteractionDataset(val_df, chem_lookup, prot_lookup)
    test_dataset = InteractionDataset(test_df, chem_lookup, prot_lookup)

    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, input_dim, train_dataset, val_dataset), n_trials=30)

    print("Best hyperparameters:", study.best_params)

    final_train_and_evaluate(study.best_params, input_dim, train_dataset, test_dataset, args.out_dir)

if __name__ == "__main__":
    main()
