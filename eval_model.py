import os
import glob
import json
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, precision_score, recall_score, f1_score, brier_score_loss, log_loss,roc_auc_score
from torch.utils.data import DataLoader
import re
import seaborn as sns

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
    fpr, tpr, roc_thresholds = roc_curve(y_true, y_pred)
    precision, recall, pr_thresholds = precision_recall_curve(y_true, y_pred)
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

    plt.figure()
    plt.hist(y_pred, bins=50)
    plt.xlabel("Predicted Probability")
    plt.ylabel("Count")
    plt.title(f"Predicted Probability Histogram: {title}")
    plt.savefig(out_path_base + "_pred_hist.png")
    plt.close()

    plt.figure()
    plt.plot(roc_thresholds, tpr, label='TPR')
    plt.plot(roc_thresholds, fpr, label='FPR')
    plt.xlabel("Threshold")
    plt.ylabel("Rate")
    plt.title(f"TPR and FPR vs Threshold: {title}")
    plt.legend()
    plt.savefig(out_path_base + "_threshold_diagnostics.png")
    plt.close()

def main():
    base_dir = "/scratch/gpfs/jg9705/IW_code"
    chem_fp_path = os.path.join(base_dir, "Model_Resources/molecular_fingerprints.tsv")
    prot_emb_path = os.path.join(base_dir, "Model_Resources/encodings.tsv")
    results_dir = os.path.join(base_dir, "results_kfold_final")
    out_dir = os.path.join(results_dir, "final_figures")
    test_base_dir = os.path.join(base_dir, "Model_Resources/splits")

    chem_lookup = load_embedding_file(chem_fp_path)
    prot_lookup = load_embedding_file(prot_emb_path, embedding_col=1)

    test_files = sorted(glob.glob(f"{test_base_dir}/threshold*/test.tsv"))
    model_files = sorted(glob.glob(f"{results_dir}/threshold*/final_model_*.pt"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    metrics = {
        "Accuracy": {},
        "Precision": {},
        "Recall": {},
        "F1": {},
        "AUROC": {},
        "Brier Score": {},
        "Log Loss": {}
    }

    for test_path in test_files:
        test_threshold = os.path.basename(os.path.dirname(test_path)).replace("threshold", "")
        print(f"\n==== Starting evaluation for Test Set Threshold {test_threshold} ====")

        test_df = pd.read_csv(test_path, sep="\t")
        test_df = filter_pairs(test_df, chem_lookup, prot_lookup)
        test_dataset = InteractionDataset(test_df, chem_lookup, prot_lookup)

        X_test = []
        y_test = []
        for i in range(len(test_dataset)):
            x_i, y_i = test_dataset[i]
            X_test.append(x_i.numpy())
            y_test.append(y_i.item())
        X_test = np.stack(X_test)
        y_test = np.array(y_test)

        all_preds = {}

        for model_path in model_files:
            model_name = os.path.splitext(os.path.basename(model_path))[0]
            match = re.search(r"final_model_(\d+)\.pt", model_path)
            assert match, f"Cannot extract threshold from {model_path}"
            train_threshold = match.group(1)

            print(f"\n--- Evaluating model {model_name} (trained on threshold {train_threshold}) ---")

            threshold_dir = os.path.dirname(model_path)
            config_path = os.path.join(threshold_dir, f"optuna_best_params_{train_threshold}.json")
            print(f"Loading config from {config_path}")

            if not os.path.exists(config_path):
                print(f"Config file missing for model {model_path}. Skipping.")
                continue

            with open(config_path) as f:
                config = json.load(f)

            batch_size = config.get("batch_size", 64)
            print(f"Using batch size: {batch_size}")
            test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=4)

            model = InteractionClassifier(
                input_dim=X_test.shape[1],
                hidden_sizes=config["hidden_sizes"],
                dropout=config["dropout"],
                activation_name=config.get("activation", "relu")
            )
            model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
            model.to(device)
            print("Model loaded and moved to device.")

            print("Running evaluation...")
            preds, labels = evaluate_model(model, test_loader, device)
            print("Evaluation complete.")

            pred_labels = (preds >= 0.5).astype(int)

            metrics["Accuracy"][(train_threshold, test_threshold)] = np.mean(pred_labels == labels)
            metrics["Precision"][(train_threshold, test_threshold)] = precision_score(labels, pred_labels, zero_division=0)
            metrics["Recall"][(train_threshold, test_threshold)] = recall_score(labels, pred_labels, zero_division=0)
            metrics["F1"][(train_threshold, test_threshold)] = f1_score(labels, pred_labels, zero_division=0)
            metrics["AUROC"][(train_threshold, test_threshold)] = roc_auc_score(labels, preds)
            metrics["Brier Score"][(train_threshold, test_threshold)] = brier_score_loss(labels, preds)
            metrics["Log Loss"][(train_threshold, test_threshold)] = log_loss(labels, preds, labels=[0, 1])

            out_base = os.path.join(out_dir, f"{model_name}_on_test{test_threshold}")
            print("Plotting curves...")
            plot_curves(labels, preds, f"{model_name} on Test {test_threshold}", out_base)
            print("Curves plotted and saved.")

            all_preds[f"model_{train_threshold}"] = (labels, preds)

        # Baselines (logistic and dummy) are evaluated but not included in heatmaps
        print("Evaluating baselines (logistic, dummy)...")
        train_path = os.path.join(test_base_dir, f"threshold{test_threshold}", "train.tsv")
        train_df = pd.read_csv(train_path, sep="\t")
        train_df = filter_pairs(train_df, chem_lookup, prot_lookup)
        train_dataset = InteractionDataset(train_df, chem_lookup, prot_lookup)

        X_train = []
        y_train = []
        for i in range(len(train_dataset)):
            x_i, y_i = train_dataset[i]
            X_train.append(x_i.numpy())
            y_train.append(y_i.item())
        X_train = np.stack(X_train)
        y_train = np.array(y_train)

        log_reg = LogisticRegression(max_iter=1000).fit(X_train, y_train)
        dummy = DummyClassifier(strategy='most_frequent').fit(X_train, y_train)

        for name, clf in [("logistic", log_reg), ("dummy", dummy)]:
            preds = clf.predict_proba(X_test)[:, 1]
            out_base = os.path.join(results_dir, f"{name}_on_test{test_threshold}")
            plot_curves(y_test, preds, f"{name.title()} on Test {test_threshold}", out_base)
            all_preds[name] = (y_test, preds)

        print("Baseline models evaluated.")

        plt.figure()
        for name, (labels, preds) in all_preds.items():
            fpr, tpr, _ = roc_curve(labels, preds)
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f"{name} (AUC={roc_auc:.2f})")
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"Combined ROC - Test {test_threshold}")
        plt.legend()
        plt.savefig(os.path.join(results_dir, f"combined_ROC_on_test{test_threshold}.png"))
        plt.close()

        plt.figure()
        for name, (labels, preds) in all_preds.items():
            if name == "dummy":
                pos_ratio = np.mean(labels)
                plt.plot([0, 1], [pos_ratio, pos_ratio], '--', label=f"{name} (baseline @ {pos_ratio:.2f})")
            else:
                precision, recall, _ = precision_recall_curve(labels, preds)
                pr_auc = average_precision_score(labels, preds)
                plt.plot(recall, precision, label=f"{name} (AP={pr_auc:.2f})")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(f"Combined PR - Test {test_threshold}")
        plt.legend()
        plt.savefig(os.path.join(results_dir, f"combined_PR_on_test{test_threshold}.png"))
        plt.close()

        print(f"Combined plots for Test Threshold {test_threshold} saved.\n")

    out_heatmap_dir = os.path.join(results_dir, "heatmaps")
    os.makedirs(out_heatmap_dir, exist_ok=True)

    out_resource_dir = os.path.join(results_dir, "figure_resources")
    os.makedirs(out_resource_dir, exist_ok=True)

    models = sorted(set(train for (train, test) in metrics['Accuracy'].keys()))
    tests = sorted(set(test for (train, test) in metrics['Accuracy'].keys()))

    for metric_name, values in metrics.items():
        heatmap_data = np.zeros((len(models), len(tests)))
        for i, train_threshold in enumerate(models):
            for j, test_threshold in enumerate(tests):
                heatmap_data[i, j] = values.get((train_threshold, test_threshold), np.nan)

        df_heatmap = pd.DataFrame(heatmap_data, index=models, columns=tests)

        plt.figure(figsize=(10, 8))
        sns.heatmap(df_heatmap, annot=True, fmt=".3f", cmap="viridis")
        plt.title(f"{metric_name} Heatmap (Train vs Test Thresholds)")
        plt.xlabel("Test Threshold")
        plt.ylabel("Train Threshold")
        plt.tight_layout()
        plt.savefig(os.path.join(out_heatmap_dir, f"{metric_name.replace(' ', '_')}_heatmap.png"))
        plt.close()

        matrix_dict = df_heatmap.to_dict(orient="index")
        with open(os.path.join(out_resource_dir, f"{metric_name.replace(' ', '_')}_matrix.json"), "w") as f:
            json.dump(matrix_dict, f, indent=2)

if __name__ == "__main__":
    main()