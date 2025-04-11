# Enhanced classifier with benchmarks, evaluation metrics, graphs, dynamic hidden layers, logging, early stopping, and calibration

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    accuracy_score, roc_auc_score, roc_curve,
    precision_score, recall_score, f1_score,
    confusion_matrix, precision_recall_curve,
    brier_score_loss, log_loss
)
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import calibration_curve
import argparse
import matplotlib.pyplot as plt
import optuna
import os
import json

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

# Load embedding file

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

# Train and evaluate

def train_eval_model(model, train_loader, val_loader, epochs, lr, device, early_stopping_patience=5):
    model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_loss = float('inf')
    patience_counter = 0

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
        val_losses = []
        all_preds, all_labels = [], []
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss = criterion(logits, y)
                val_losses.append(loss.item())
                probs = torch.sigmoid(logits).cpu().numpy()
                all_preds.extend(probs)
                all_labels.extend(y.cpu().numpy())

        avg_val_loss = np.mean(val_losses)
        print(f"Epoch {epoch+1}, Train Loss: {np.mean(train_losses):.4f}, Val Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print("Early stopping triggered.")
                break

    auc = roc_auc_score(all_labels, all_preds)
    return auc

# Objective

def objective(trial, input_dim, train_dataset, val_dataset):
    # Dynamically choose the number of hidden layers
    num_layers = trial.suggest_int("num_hidden_layers", 1, 4)

    # Dynamically choose the size of each layer
    hidden_layers = tuple(
        trial.suggest_int(f"n_units_layer_{i}", 64, 1024, step=64)
        for i in range(num_layers)
    )

    # Other hyperparameters
    dropout = trial.suggest_float("dropout", 0.1, 0.5)
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    epochs = trial.suggest_int("epochs", 5, 25)

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=4)

    # Model and training
    model = InteractionClassifier(input_dim, hidden_layers, dropout)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return train_eval_model(model, train_loader, val_loader, epochs, lr, device)

# Final evaluation

def evaluate_model(model, test_loader, out_dir):
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

    # Compute metrics
    metrics = {
        "accuracy": accuracy_score(all_labels, binary_preds),
        "precision": precision_score(all_labels, binary_preds),
        "recall": recall_score(all_labels, binary_preds),
        "f1": f1_score(all_labels, binary_preds),
        "roc_auc": roc_auc_score(all_labels, all_preds),
        "brier_score": brier_score_loss(all_labels, all_preds),
        "log_loss": log_loss(all_labels, all_preds)
    }

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(metrics, f, indent=4)

    # Load benchmark results if available
    benchmark_path = os.path.join(out_dir, "benchmarks.json")
    benchmarks = {}
    if os.path.exists(benchmark_path):
        print(f"Found benchmark file at {benchmark_path}")
        with open(benchmark_path) as f:
            benchmarks = json.load(f)
            print(f"Loaded benchmarks: {list(benchmarks.keys())}")
    else:
        print(f"Benchmark file not found at {benchmark_path}")

    # ROC Curve
    fpr, tpr, _ = roc_curve(all_labels, all_preds)
    plt.figure()
    plt.plot(fpr, tpr, label="Classifier (AUC = %.3f)" % metrics["roc_auc"])

    for name in ["dummy", "logreg"]:
        key = f"{name}_probs"
        if key in benchmarks:
            print(f"Plotting {name} benchmark")
            bench_probs = benchmarks[key]
            fpr_b, tpr_b, _ = roc_curve(all_labels, bench_probs)
            auc_b = roc_auc_score(all_labels, bench_probs)
            plt.plot(fpr_b, tpr_b, linestyle="--", label=f"{name} (AUC = {auc_b:.3f})")
        else:
            print(f"Missing {key} in benchmarks")

    plt.title("ROC Curve")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.legend()
    plt.savefig(os.path.join(out_dir, "roc_curve.png"))

    # Precision-Recall Curve
    prec, rec, _ = precision_recall_curve(all_labels, all_preds)
    plt.figure()
    plt.plot(rec, prec, label="Classifier")

    for name in ["dummy", "logreg"]:
        key = f"{name}_probs"
        if key in benchmarks:
            print(f"Plotting {name} benchmark for PR curve")
            bench_probs = benchmarks[key]
            prec_b, rec_b, _ = precision_recall_curve(all_labels, bench_probs)
            f1_b = f1_score(all_labels, [1 if p >= 0.5 else 0 for p in bench_probs])
            plt.plot(rec_b, prec_b, linestyle="--", label=f"{name} (F1 = {f1_b:.3f})")
        else:
            print(f"Missing {key} in benchmarks for PR curve")

    plt.title("Precision-Recall Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend()
    plt.savefig(os.path.join(out_dir, "precision_recall.png"))

    # Confusion matrix
    cm = confusion_matrix(all_labels, binary_preds)
    plt.figure()
    plt.imshow(cm, cmap='Blues')
    plt.title("Confusion Matrix\nAcc: %.3f | F1: %.3f" % (metrics["accuracy"], metrics["f1"]))
    plt.colorbar()
    plt.xticks([0, 1], ["Pred 0", "Pred 1"])
    plt.yticks([0, 1], ["True 0", "True 1"])
    plt.savefig(os.path.join(out_dir, "confusion_matrix.png"))

# Benchmarks

def run_benchmarks(train_df, test_df, chem_lookup, prot_lookup, out_dir):
    def extract_features(df):
        feats, labels = [], []
        for _, row in df.iterrows():
            chem = chem_lookup[row['chemical']]
            prot = prot_lookup[row['protein']]
            feats.append(torch.cat([chem, prot]).numpy())
            labels.append(row['label'])
        return np.array(feats), np.array(labels)

    X_train, y_train = extract_features(train_df)
    X_test, y_test = extract_features(test_df)

    results = {}

    for name, clf in {
        "dummy": DummyClassifier(strategy="most_frequent"),
        "logreg": LogisticRegression(max_iter=1000)
    }.items():
        print(f"Training {name} benchmark model...")
        clf.fit(X_train, y_train)
        probs = clf.predict_proba(X_test)[:, 1]
        preds = clf.predict(X_test)

        results[f"{name}_accuracy"] = accuracy_score(y_test, preds)
        results[f"{name}_f1"] = f1_score(y_test, preds)
        results[f"{name}_roc_auc"] = roc_auc_score(y_test, probs)
        results[f"{name}_probs"] = probs.tolist()

        print(f"Completed {name} benchmark with accuracy: {results[f'{name}_accuracy']:.4f}")

    benchmark_path = os.path.join(out_dir, "benchmarks.json")
    print(f"Saving benchmark results to {benchmark_path}")
    with open(benchmark_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Benchmark results saved with keys: {list(results.keys())}")


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

    chem_lookup = load_embedding_file(args.chem_fp)
    prot_lookup = load_embedding_file(args.prot_emb, embedding_col=2)

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

    num_layers = study.best_params["num_hidden_layers"]
    hidden_sizes = [study.best_params[f"n_units_layer_{i}"] for i in range(num_layers)]
    model = InteractionClassifier(input_dim, hidden_sizes, study.best_params["dropout"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    train_loader = DataLoader(train_dataset, batch_size=study.best_params['batch_size'], shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=study.best_params['batch_size'], num_workers=4)

    train_eval_model(model, train_loader, test_loader, study.best_params['epochs'], study.best_params['lr'], device)
    run_benchmarks(train_df, test_df, chem_lookup, prot_lookup, args.out_dir)
    evaluate_model(model, test_loader, args.out_dir)

if __name__ == "__main__":
    main()
