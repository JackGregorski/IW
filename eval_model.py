import os
import glob
import json
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

from torch.utils.data import DataLoader
import re
# Reuse your existing classes
from kfold_classifier import InteractionClassifier, InteractionDataset, load_embedding_file, filter_pairs

def evaluate_model(model, loader, device):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x)
            probs = torch.sigmoid(logits).cpu().numpy()
            preds.extend(probs)
            labels.extend(y.numpy())
    return np.array(preds), np.array(labels)

def plot_curves(y_true, y_pred, title, out_path_base):
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)
    pr_auc = average_precision_score(y_true, y_pred)

    plt.figure()
    plt.plot(fpr, tpr, label=f'ROC AUC = {roc_auc:.2f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC: {title}')
    plt.legend()
    plt.savefig(out_path_base + "_roc.png")
    plt.close()

    plt.figure()
    plt.plot(recall, precision, label=f'PR AUC = {pr_auc:.2f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall: {title}')
    plt.legend()
    plt.savefig(out_path_base + "_pr.png")
    plt.close()

def main():
    base_dir = "/scratch/gpfs/jg9705/IW_code"
    chem_fp_path = os.path.join(base_dir, "Model_Resources/molecular_fingerprints.tsv")
    prot_emb_path = os.path.join(base_dir, "Model_Resources/encodings.tsv")
    results_dir = os.path.join(base_dir, "results_kfold")
    test_base_dir = os.path.join(base_dir, "Model_Resources/splits")

    chem_lookup = load_embedding_file(chem_fp_path)
    prot_lookup = load_embedding_file(prot_emb_path, embedding_col=1)

    test_files = sorted(glob.glob(f"{test_base_dir}/threshold*/test.tsv"))
    model_files = sorted(glob.glob(f"{results_dir}/threshold*/final_model_*.pt"))


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for test_path in test_files:
        threshold = os.path.basename(os.path.dirname(test_path)).replace("threshold", "")
        test_df = pd.read_csv(test_path, sep="\t")
        test_df = filter_pairs(test_df, chem_lookup, prot_lookup)
        dataset = InteractionDataset(test_df, chem_lookup, prot_lookup)
        loader = DataLoader(dataset, batch_size=64, num_workers=4)

        X = []
        y = []
        for i in range(len(dataset)):
            x_i, y_i = dataset[i]
            X.append(x_i.numpy())
            y.append(y_i.item())
        X = np.stack(X)
        y = np.array(y)

        print(f"\n==== Test Set Threshold {threshold} ====")

        for model_path in model_files:
            model_name = os.path.splitext(os.path.basename(model_path))[0]


            # Extract threshold from model filename
            match = re.search(r"final_model_(\d+)\.pt", model_path)
            assert match, f"Cannot extract threshold from {model_path}"
            threshold = match.group(1)

            # Load saved hyperparameters
            threshold_dir = os.path.dirname(model_path)
            config_path = os.path.join(threshold_dir, "optuna_best_params_train.json")
            if not os.path.exists(config_path):
                print(f"❌ Config file missing for model {model_path}. Skipping.")
                continue

            with open(config_path) as f:
                config = json.load(f)

            model = InteractionClassifier(
                input_dim=X.shape[1],
                hidden_sizes=config["hidden_sizes"],
                dropout=config["dropout"],
                activation_name=config.get("activation", "relu")
            )
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.to(device)

            preds, labels = evaluate_model(model, loader, device)
            out_base = os.path.join(results_dir, f"{model_name}_on_test{threshold}")
            plot_curves(labels, preds, f"{model_name} on Test {threshold}", out_base)

        # Baseline models
        log_reg = LogisticRegression(max_iter=1000).fit(X, y)
        dummy = DummyClassifier(strategy='most_frequent').fit(X, y)

        for name, clf in [("logistic", log_reg), ("dummy", dummy)]:
            preds = clf.predict_proba(X)[:, 1]
            out_base = os.path.join(results_dir, f"{name}_on_test{threshold}")
            plot_curves(y, preds, f"{name.title()} on Test {threshold}", out_base)

if __name__ == "__main__":
    main()
