# K-Fold Enhanced Classifier with Visualization and Final Evaluation

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.metrics import (
    accuracy_score, roc_auc_score, roc_curve,
    precision_score, recall_score, f1_score,
    confusion_matrix, precision_recall_curve,
    brier_score_loss, log_loss
)
from sklearn.model_selection import StratifiedKFold
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
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

def train_eval_model(model, train_loader, val_loader, epochs, lr, device, early_stopping_patience=5):
    model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    best_loss = float('inf')
    patience_counter = 0
    all_val_preds, all_val_labels = [], []
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

        avg_val_loss = np.mean(val_losses)
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            all_val_preds, all_val_labels = val_preds, val_labels
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                break
    return roc_auc_score(all_val_labels, all_val_preds)

def evaluate_model(model, test_loader, out_dir):
    benchmark_path = os.path.join(out_dir, "benchmarks.json")
    benchmarks = {}
    if os.path.exists(benchmark_path):
        with open(benchmark_path) as f:
            benchmarks = json.load(f)

    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(next(model.parameters()).device)
            logits = model(x)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_preds.extend(probs)
            all_labels.extend(y.numpy())

    binary_preds = [1 if p >= 0.5 else 0 for p in all_preds]

    metrics = {
        "accuracy": accuracy_score(all_labels, binary_preds),
        "precision": precision_score(all_labels, binary_preds),
        "recall": recall_score(all_labels, binary_preds),
        "f1": f1_score(all_labels, binary_preds),
        "roc_auc": roc_auc_score(all_labels, all_preds),
        "brier_score": brier_score_loss(all_labels, all_preds),
        "log_loss": log_loss(all_labels, all_preds)
    }

    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(metrics, f, indent=4)

    fpr, tpr, _ = roc_curve(all_labels, all_preds)
    plt.figure()
    plt.plot(fpr, tpr, label=f"Model (AUC = {metrics['roc_auc']:.3f})")
    for name in ["dummy", "logreg"]:
        key = f"{name}_probs"
        if key in benchmarks:
            probs = np.array(benchmarks[key])
            fpr_b, tpr_b, _ = roc_curve(all_labels, probs)
            auc_b = roc_auc_score(all_labels, probs)
            plt.plot(fpr_b, tpr_b, linestyle="--", label=f"{name} (AUC = {auc_b:.3f})")
    plt.title("ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.savefig(os.path.join(out_dir, "roc_curve.png"))

    prec, rec, _ = precision_recall_curve(all_labels, all_preds)
    plt.figure()
    plt.plot(rec, prec, label="Model")
    baseline = np.mean(all_labels)
    plt.axhline(y=baseline, linestyle="--", color="gray", label=f"Baseline (rate = {baseline:.2f})")
    plt.title("Precision-Recall Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend()
    plt.savefig(os.path.join(out_dir, "precision_recall.png"))

    cm = confusion_matrix(all_labels, binary_preds)
    plt.figure()
    plt.imshow(cm, cmap='Blues')
    plt.title(f"Confusion Matrix\nAcc: {metrics['accuracy']:.3f} | F1: {metrics['f1']:.3f}")
    plt.colorbar()
    plt.xticks([0, 1], ["Pred 0", "Pred 1"])
    plt.yticks([0, 1], ["True 0", "True 1"])
    plt.savefig(os.path.join(out_dir, "confusion_matrix.png"))

def plot_dummy_baseline_pr(y_true, filename):
    positive_rate = np.mean(y_true)
    plt.figure()
    plt.axhline(y=positive_rate, color='gray', linestyle='--', label=f"Baseline (rate = {positive_rate:.2f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend()
    plt.title("Precision-Recall Curve Baseline")
    plt.savefig(filename)


def final_train_and_save(dataset, input_dim, best_params, model_path):
    model = InteractionClassifier(input_dim, best_params["hidden_sizes"], best_params["dropout"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader = DataLoader(dataset, batch_size=best_params["batch_size"], shuffle=True)
    train_eval_model(model, loader, loader, best_params["epochs"], best_params["lr"], device)
    torch.save(model.state_dict(), model_path)
    print(f"Saved final model to {model_path}")
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--test", required=True)
    parser.add_argument("--chem_fp", required=True)
    parser.add_argument("--prot_emb", required=True)
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    train_df = pd.read_csv(args.train, sep="\t")
    test_df = pd.read_csv(args.test, sep="\t")
    chem_lookup = load_embedding_file(args.chem_fp)
    prot_lookup = load_embedding_file(args.prot_emb, embedding_col=2)

    train_df = filter_pairs(train_df, chem_lookup, prot_lookup)
    test_df = filter_pairs(test_df, chem_lookup, prot_lookup)

    input_dim = len(next(iter(chem_lookup.values()))) + len(next(iter(prot_lookup.values())))
    dataset = InteractionDataset(train_df, chem_lookup, prot_lookup)
    test_dataset = InteractionDataset(test_df, chem_lookup, prot_lookup)

    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, input_dim, dataset), n_trials=30)
    best_params = study.best_trial.user_attrs["best_params"]
    print("Best hyperparameters:", best_params)

    model_path = os.path.join(args.out_dir, "final_model.pt")
    model = final_train_and_save(dataset, input_dim, best_params, model_path)

    test_loader = DataLoader(test_dataset, batch_size=best_params["batch_size"])
    evaluate_model(model, test_loader, args.out_dir)

    y_true = train_df['label'].values
    plot_dummy_baseline_pr(y_true, os.path.join(args.out_dir, "dummy_pr_baseline.png"))

if __name__ == "__main__":
    main()
