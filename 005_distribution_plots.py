#!/usr/bin/env python3
# extratrees_feature_distributions.py – v3
# Cosmetic‑only update: strip “Tigress”, angle x‑labels, and show visible tick marks.
"""
Visualise per‑feature value distributions of the test split as predicted by a
batch of ExtraTrees models. For every numeric feature the script pools all test
rows across up to ``N_SPLITS`` seeds, assigns the model’s *predicted class* to
each row, and draws one box plot per class label. Optional summary statistics
(mean, quartiles, etc.) are written to a text file.

Context
-------
* *prepare_svd_dataset_batch.py* generates multiple stratified train/test splits.
* *extratrees_from_prepared.py* trains one **ExtraTreesClassifier** per split
  and writes the pickled model to
  ``Prepared_SVD_g{GROUPING}/ExtraTrees_{TAG}/models``.
* The present script analyses these models’ **test‑set predictions** to
  highlight how each feature’s spread differs across predicted classes.

Workflow
--------
1. **Discovery** – For every seed ``1…N_SPLITS`` locate the estimator pickle
   and its corresponding test‑set CSV.
2. **Prediction** – Load the model, predict on the seed’s test matrix, attach
   the predicted class label to the feature matrix, and pool across all seeds.
3. **Plot generation** – Produce a box plot for every feature with class labels
   on the X‑axis (label prefix “Tigress” removed for readability). Tick marks
   are enlarged for clarity and X‑labels rotated 45° to avoid overlap.
4. **Summary statistics** – Optionally write a descriptive‑statistics table per
   feature × class to ``feature_distribution_summary.txt``.

Configuration
-------------
``GROUPING``
    0–3; must match the grouping level used in earlier preprocessing.
``USE_ADASYN``
    ``True`` if models were trained on the ADASYN‑balanced split; otherwise
    ``False`` expects RAW models.
``N_SPLITS``
    Maximum number of seed folders to include.
``CUSTOM_PALETTE``
    Five‑colour palette used for both bars and seaborn colour maps.
``BASE_FONT_SIZE`` / ``TITLE_FONT_SIZE``
    Global matplotlib font tweaks for publication‑grade figures.

Outputs
-------
* ``Evaluation_ExtraTrees_g{GROUPING}/distributions/{feature}_boxplot.png|.eps``
  – One pair of plots per feature.
* ``feature_distribution_summary.txt`` – Optional descriptive statistics.

Author
------
Sebastian Raubitzek
Date: 30 June 2025
"""

from pathlib import Path
import pickle, warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl

warnings.filterwarnings("ignore")

# ───────────── USER SETTINGS ─────────────
GROUPING   = 1
USE_ADASYN = True
N_SPLITS   = 100

CUSTOM_PALETTE = [
    "#E2DBBE",
    "#D5D6AA",
    "#9DBBAE",
    "#769FB6",
    "#188FA7",
]
BASE_FONT_SIZE  = 22
TITLE_FONT_SIZE = BASE_FONT_SIZE + 2
mpl.rcParams["xtick.labelsize"] = BASE_FONT_SIZE
mpl.rcParams["ytick.labelsize"] = BASE_FONT_SIZE
# ─────────────────────────────────────────

TAG        = "adasyn" if USE_ADASYN else "raw"
PREP_ROOT  = Path(f"Prepared_SVD_g{GROUPING}")
RUN_ROOT   = PREP_ROOT / f"ExtraTrees_{TAG}"
MODEL_DIR  = RUN_ROOT / "models"
OUT_DIR    = Path(f"Evaluation_ExtraTrees_g{GROUPING}") / "distributions"
OUT_DIR.mkdir(parents=True, exist_ok=True)

pooled_frames = []
label_order   = None

print(f"[INFO] generating pooled feature‑distribution data (tag={TAG})")

for seed in range(1, N_SPLITS + 1):
    model_path = MODEL_DIR / f"extratrees_seed{seed:04d}_{TAG}.pkl"
    data_dir   = PREP_ROOT / f"seed{seed:04d}_ADASYN" / "data"
    if not (model_path.exists() and data_dir.exists()):
        print(f"  • seed {seed:04d}: missing artefacts – skipped")
        continue

    model = pickle.load(open(model_path, "rb"))
    X_te = pd.read_csv(data_dir / "X_test.csv")
    y_pred = model.predict(X_te)

    if label_order is None:
        label_order = sorted(set(y_pred))
        label_order_nice = [
            lbl.replace("Tigress", "").replace("_", "\n") for lbl in label_order
        ]

    frame = X_te.copy()
    frame["PredictedClass"] = pd.Categorical(
        [lbl.replace("Tigress", "").replace("_", "\n") for lbl in y_pred],
        categories=label_order_nice,
        ordered=True,
    )
    pooled_frames.append(frame)
    print(f"  • seed {seed:04d}: collected {len(frame):,} rows")

if not pooled_frames:
    raise RuntimeError("No valid splits found – nothing to plot.")

combined_df = pd.concat(pooled_frames, ignore_index=True)
print(f"[INFO] pooled data shape: {combined_df.shape}")

sns.set_style("whitegrid")
for feat in combined_df.columns.drop("PredictedClass"):
    plt.figure(figsize=(8, 10))
    ax = sns.boxplot(
        data=combined_df,
        x="PredictedClass",
        y=feat,
        palette=CUSTOM_PALETTE,
        showfliers=True,
    )
    plt.xlabel("Predicted Class", fontsize=TITLE_FONT_SIZE)
    plt.ylabel(feat, fontsize=TITLE_FONT_SIZE)
    plt.xticks(rotation=45, ha="right")
    ax.tick_params(axis="x", length=6, width=1)   # visible tick marks
    plt.tight_layout()

    png_path = OUT_DIR / f"{feat}_boxplot.png"
    eps_path = OUT_DIR / f"{feat}_boxplot.eps"
    plt.savefig(png_path, dpi=300)
    plt.savefig(eps_path, format="eps")
    plt.clf()
    plt.close()
    print(f"[INFO] {feat:>35} → {png_path.name}")

# ───────────── optional summary statistics ─────────────

def generate_distribution_summary(df, group_col, out_file):
    """Write per‑class descriptive statistics of every feature to *out_file*."""
    with open(out_file, "w") as fh:
        for col in df.columns.drop(group_col):
            fh.write(f"Feature: {col}\n")
            fh.write(df.groupby(group_col)[col].describe().to_string())
            fh.write("\n" + "-" * 60 + "\n")

summary_file = OUT_DIR / "feature_distribution_summary.txt"
generate_distribution_summary(combined_df, "PredictedClass", summary_file)
print(f"[✓] distribution summary written to {summary_file}")
print(f"[✓] all plots stored under {OUT_DIR.resolve()}")
