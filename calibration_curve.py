import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
import os
from kfold_classifier import InteractionClassifier, load_embedding_file, InteractionDataset, filter_pairs
from torch.utils.data import DataLoader

# Paths (adjust if necessary)
base_dir = "/scratch/gpfs/jg9705/IW_code"
model_path = os.path.join(base_dir, "results_kfold_final/threshold600/final_model_train.pt")
config_path = os.path.join(base_dir, "results_kfold_final/threshold600/optuna_best_params_600.json")
test_path = os.path.join(base_dir, "Model_Resources/splits/threshold900/test.tsv")  # Best test set
chem_fp_path = os.path.join(base_dir, "Model_Resources/molecular_fingerprints.tsv")
prot_emb_path = os.path.join(base_dir, "Model_Resources/encodings.tsv")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load test data
chem_lookup = load_embedding_file(chem_fp_path)
prot_lookup = load_embedding_file(prot_emb_path, embedding_col=1)

test_df = pd.read_csv(test_path, sep="\t")
test_df = filter_pairs(test_df, chem_lookup, prot_lookup)
test_dataset = InteractionDataset(test_df, chem_lookup, prot_lookup)
test_loader = DataLoader(test_dataset, batch_size=64, num_workers=4)

# Load model
with open(config_path) as f:
    config = json.load(f)

input_dim = next(iter(test_loader))[0].shape[1]
model = InteractionClassifier(
    input_dim=input_dim,
    hidden_sizes=config["hidden_sizes"],
    dropout=config["dropout"],
    activation_name=config.get("activation", "relu")
)
model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
model.to(device)
model.eval()

# Get predictions
preds, labels = [], []
with torch.no_grad():
    for x, y in test_loader:
        x = x.to(device)
        logits = model(x)
        prob = torch.sigmoid(logits).cpu().numpy()
        preds.extend(prob)
        labels.extend(y.numpy())

preds = np.array(preds).flatten()
labels = np.array(labels)

# Generate calibration curve
fraction_of_positives, mean_predicted_value = calibration_curve(labels, preds, n_bins=10)

# Plot calibration curve
plt.figure(figsize=(8,6))
plt.plot(mean_predicted_value, fraction_of_positives, "s-", label="Threshold600 on Test900")
plt.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated")
plt.xlabel("Mean predicted probability")
plt.ylabel("Fraction of positives")
plt.title("Calibration Curve: Threshold600 on Test900")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig(os.path.join(base_dir, "results_kfold_final/final_figures/threshold600_calibration_curve.png"))
plt.close()

print("Calibration curve saved!")
