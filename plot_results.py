import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

paths = {
    "Baseline": r"F:\COMP576\final\culane_6k\baseline_20e\results.csv",
    "Previous Best": r"F:\COMP576\final\culane_6k\lane_opt_lightaug\results.csv",
    "Trial-7": r"F:\COMP576\final\final_ablation\trial_7\results.csv",
}

out_dir = r"F:\COMP576\final\figures"
os.makedirs(out_dir, exist_ok=True)

data = {}
for name, path in paths.items():
    data[name] = pd.read_csv(path)

plt.figure()
for name, df in data.items():
    plt.plot(df["epoch"], df["val/seg_loss"], label=name)

plt.xlabel("Epoch")
plt.ylabel("Validation Segmentation Loss")
plt.title("Validation Segmentation Loss over Epochs")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "figure_val_seg_loss_curve.png"), dpi=300)
plt.close()

models = []
stds = []

for name, df in data.items():
    models.append(name)
    stds.append(np.std(df["val/seg_loss"].values))

plt.figure()
plt.bar(models, stds)
plt.xlabel("Model")
plt.ylabel("Std of Validation Segmentation Loss")
plt.title("Validation Segmentation Loss Stability Comparison")
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "figure_val_seg_loss_std.png"), dpi=300)
plt.close()

print("Figures saved to:", out_dir)
