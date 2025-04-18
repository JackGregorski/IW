# K-Fold Enhanced Classifier with Visualization and Final Evaluation

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset

from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
import optuna.visualization.matplotlib as optuna_vis

import matplotlib.pyplot as plt
import optuna
import os
import json
import argparse

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

class InteractionClassifier(nn.Module):
    def __init__(self, input_dim, hidden_sizes, dropout):
        super().__init__()
        layers = []
        for i, hidden in enumerate(hidden_sizes):
            in_dim = input_dim if i == 0 else hidden_sizes[i-1]
            layers.extend([
                nn.Linear(in_dim, hidden),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
        layers.append(nn.Linear(hidden_sizes[-1], 1))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x).squeeze()

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
            except ValueError:
                continue
    return lookup

def filter_pairs(df, chem_lookup, prot_lookup):
    return df[df["chemical"].isin(chem_lookup.keys()) & df["protein"].isin(prot_lookup.keys())]



def objective(trial, input_dim, dataset):
    num_layers = trial.suggest_int("num_hidden_layers", 1, 3)
    hidden_sizes = tuple(
        trial.suggest_int(f"n_units_layer_{i}", 64, 512, step=64)
        for i in range(num_layers)
    )
    dropout = trial.suggest_float("dropout", 0.2, 0.5)
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    epochs = trial.suggest_int("epochs", 3, 5)

    labels = dataset.data["label"].values
    groups = dataset.data["protein_cluster"].values
    kfold = GroupKFold(n_splits=3)
    aucs = []

    for train_idx, val_idx in kfold.split(np.zeros(len(labels)), labels, groups):
        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_subset, batch_size=batch_size, num_workers=4, pin_memory=True)

        model = InteractionClassifier(input_dim, hidden_sizes, dropout)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        auc = train_eval_model(model, train_loader, val_loader, epochs, lr, device)
        aucs.append(auc)

    trial.set_user_attr("best_params", {
        "hidden_sizes": hidden_sizes,
        "dropout": dropout,
        "lr": lr,
        "batch_size": batch_size,
        "epochs": epochs
    })
    return np.mean(aucs)

def train_eval_model(model, train_loader, val_loader, epochs, lr, device, early_stopping_patience=5, record_loss=False):
    model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    best_loss = float('inf')
    patience_counter = 0
    all_val_preds, all_val_labels = [], []
    train_curve, val_curve = [], []
    for epoch in range(epochs):
        model.train()
        train_losses = []
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses, val_preds, val_labels = [], [], []
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss = criterion(logits, y)
                val_losses.append(loss.item())
                val_preds.extend(torch.sigmoid(logits).cpu().numpy())
                val_labels.extend(y.cpu().numpy())
        train_curve.append(np.mean(train_losses))
        avg_val_loss = np.mean(val_losses)
        val_curve.append(avg_val_loss)


        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            all_val_preds, all_val_labels = val_preds, val_labels
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                break
        if not record_loss:
            try:
                return roc_auc_score(all_val_labels, all_val_preds)
            except ValueError:
                return 0.0  # Or np.nan if you prefer to skip in averaging
        else:
            try:
                auc = roc_auc_score(all_val_labels, all_val_preds)
            except ValueError:
                auc = 0.0
            return auc, train_curve, val_curve



def final_train_and_save(dataset, input_dim, best_params, model_path,out_dir):
    model = InteractionClassifier(input_dim, best_params["hidden_sizes"], best_params["dropout"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader = DataLoader(dataset, batch_size=best_params["batch_size"], shuffle=True,num_workers=4, pin_memory=True)
    auc, train_curve, val_curve = train_eval_model(model, loader, loader, best_params["epochs"], best_params["lr"], device, record_loss=True)
    plt.figure()
    plt.plot(train_curve, label="Train Loss")
    plt.plot(val_curve, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training vs Validation Loss")
    plt.savefig(os.path.join(out_dir, "loss_curve.png"))
    torch.save(model.state_dict(), model_path)
    print(f"Saved final model to {model_path}")
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--chem_fp", required=True)
    parser.add_argument("--prot_emb", required=True)
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    train_df = pd.read_csv(args.train, sep="\t")
    chem_lookup = load_embedding_file(args.chem_fp)
    prot_lookup = load_embedding_file(args.prot_emb, embedding_col=1)

    train_df = filter_pairs(train_df, chem_lookup, prot_lookup)

    # Use 50% of training set for tuning
    train_df_tune = train_df.sample(frac=0.5, random_state=42).reset_index(drop=True)

    input_dim = len(next(iter(chem_lookup.values()))) + len(next(iter(prot_lookup.values())))
    dataset_tune = InteractionDataset(train_df_tune, chem_lookup, prot_lookup)
    full_dataset = InteractionDataset(train_df, chem_lookup, prot_lookup)

    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, input_dim, dataset_tune), n_trials=10)

    best_params = study.best_trial.user_attrs["best_params"]
    print("Best hyperparameters:", best_params)

    model_path = os.path.join(args.out_dir, "final_model_400.pt")
    model = final_train_and_save(full_dataset, input_dim, best_params, model_path, args.out_dir)

if __name__ == "__main__":
    main()
