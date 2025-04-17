def evaluate_model(model, test_loader, out_dir, logistic_regression_results=None):
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

    # Save metrics to JSON
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(metrics, f, indent=4)

    # --- ROC Curve ---
    fpr, tpr, _ = roc_curve(all_labels, all_preds)
    plt.figure()
    plt.plot(fpr, tpr, label=f"Model (AUC = {metrics['roc_auc']:.3f})")

    if logistic_regression_results:
        logreg_probs = logistic_regression_results["probs"]
        logreg_labels = logistic_regression_results["labels"]

        # ROC
        fpr_log, tpr_log, _ = roc_curve(logreg_labels, logreg_probs)
        auc_log = roc_auc_score(logreg_labels, logreg_probs)
        plt.plot(fpr_log, tpr_log, linestyle="--", label=f"LogReg (AUC = {auc_log:.3f})")


    plt.title("ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.savefig(os.path.join(out_dir, "roc_curve.png"))



    # --- Precision-Recall Curve ---
    prec, rec, _ = precision_recall_curve(all_labels, all_preds)
    plt.figure()
    plt.plot(rec, prec, label="Model")
    baseline = np.mean(all_labels)
    plt.axhline(y=baseline, linestyle="--", color="gray", label=f"Baseline (rate = {baseline:.2f})")

    if logistic_regression_results:
        # PR
        logreg_probs = logistic_regression_results["probs"]
        logreg_labels = logistic_regression_results["labels"]

        logreg_prec, logreg_rec, _ = precision_recall_curve(logreg_labels, logreg_probs)

        plt.plot(logreg_rec, logreg_prec, linestyle="--", label="LogReg")

    plt.title("Precision-Recall Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend()
    plt.savefig(os.path.join(out_dir, "precision_recall.png"))

    # --- Confusion Matrix ---
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

