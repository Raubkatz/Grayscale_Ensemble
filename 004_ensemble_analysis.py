#!/usr/bin/env python3
# evaluate_extratrees_runs.py – v7
# Cosmetic-only update: shorter labels, axis captions, tidy feature-importance Y-axis.
"""
Summarise the performance of a batch of **ExtraTreesClassifier** models trained
on SVD‑derived feature tables. The script aggregates per‑seed metrics, plots
feature‑importance rankings, and visualises mean row‑normalised confusion
matrices for train and test data.

Input folders
-------------
``Prepared_SVD_g{GROUPING}/ExtraTrees_{TAG}/``
    Folder produced by *extratrees_from_prepared.py* containing one pickled
    estimator per seed.
``Prepared_SVD_g{GROUPING}/seed####_ADASYN/data``
    Per‑seed data sub‑folder with the six CSV files written during dataset
    preparation (raw train/test and optional ADASYN‑balanced train split).

Workflow
--------
1. **Discovery** – Iterate through the first ``N_SPLITS`` seeds and locate the
   estimator pickle plus its associated data split.
2. **Feature‑importance capture** – Extract the raw and normalised feature
   importance vectors from each estimator; later compute mean ± SD.
3. **Metric evaluation** – For each model:
   * Evaluate on original train, ADASYN train (if present), and held‑out test
     splits.
   * Record accuracy, precision, recall, weighted F1.
   * Store row‑normalised confusion matrices for later averaging.
4. **Aggregation** – For every metric set compute mean and standard deviation
   across all processed seeds. Compute mean ± SD of feature‑importance values
   per feature.
5. **Visualisation** – Save the following plots to ``Evaluation_ExtraTrees_g{GROUPING}``:
   * Bar plot of top‑20 features (mean importance with SD error bars).
   * Heat maps of mean confusion matrices for test, raw‑train, and ADASYN‑train
     splits, using a custom sequential colour map.

Configuration
-------------
``GROUPING``
    0–3; must match the grouping level used during preprocessing.
``USE_ADASYN``
    ``True`` evaluates ADASYN‑balanced training predictions; ``False`` expects
    only raw train/test artefacts.
``N_SPLITS``
    Number of seed folders to scan (1…``N_SPLITS``).
``CUSTOM_PALETTE``
    Five‑colour palette used for feature‑importance bars and the confusion‑
    matrix colour map.

Outputs
--------
* ``feat_imp_meanSD_{TAG}.png / .eps`` – Feature‑importance bar plot.
* ``cm_*_mean_{TAG}.png / .eps`` – Mean confusion‑matrix heat maps.
* Console summary per processed seed.

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
import matplotlib.colors as mcolors
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,
)

warnings.filterwarnings("ignore")

# ─────────────── USER SETTINGS ───────────────
GROUPING   = 3          # 0…3 (must match preprocessing)
USE_ADASYN = True       # False → RAW run folder
N_SPLITS   = 100        # seeds 1…N
CUSTOM_PALETTE = [
    "#E2DBBE",
    "#D5D6AA",
    "#9DBBAE",
    "#769FB6",
    "#188FA7",
]
PALETTE = CUSTOM_PALETTE
CMAP = mcolors.LinearSegmentedColormap.from_list("custom_confmat", CUSTOM_PALETTE)
# ---------------------------------------------

TAG        = "adasyn" if USE_ADASYN else "raw"
PREP_ROOT  = Path(f"Prepared_SVD_g{GROUPING}")
RUN_ROOT   = PREP_ROOT / f"ExtraTrees_{TAG}"
MODEL_DIR  = RUN_ROOT / "models"
OUT_DIR    = Path(f"Evaluation_ExtraTrees_g{GROUPING}")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# collectors -----------------------------------------------------------------
fi_raw, fi_norm = [], []
mets_orig, mets_ada, mets_test = [], [], []
cm_orig_rel, cm_ada_rel, cm_test_rel = [], [], []
label_order = None        # fixed after first valid split
display_labels = None     # label_order with 'Tigress' removed
# ---------------------------------------------------------------------------

print(f"[INFO] scanning {MODEL_DIR.resolve()} (tag={TAG}, seeds 1…{N_SPLITS})")

for seed in range(1, N_SPLITS + 1):
    model_path = MODEL_DIR / f"extratrees_seed{seed:04d}_{TAG}.pkl"
    data_dir   = PREP_ROOT / f"seed{seed:04d}_ADASYN" / "data"
    if not (model_path.exists() and data_dir.exists()):
        print(f"  • seed {seed:04d}: missing artefacts – skipped")
        continue

    # ── load model & FI ----------------------------------------------------
    model = pickle.load(open(model_path, "rb"))
    fi = pd.Series(model.feature_importances_,
                   index=model.feature_names_in_,
                   name=f"seed{seed:04d}")
    fi_raw.append(fi)
    fi_norm.append(fi / fi.sum())

    # ── load splits --------------------------------------------------------
    X_to = pd.read_csv(data_dir / "X_train_raw.csv")
    y_to = pd.read_csv(data_dir / "y_train_raw.csv").iloc[:, 0]
    X_te = pd.read_csv(data_dir / "X_test.csv")
    y_te = pd.read_csv(data_dir / "y_test.csv").iloc[:, 0]

    ada_present = (data_dir / "X_train_adasyn.csv").exists()
    if ada_present:
        X_ta = pd.read_csv(data_dir / "X_train_adasyn.csv")
        y_ta = pd.read_csv(data_dir / "y_train_adasyn.csv").iloc[:, 0]

    # ── label order & display labels (once) --------------------------------
    if label_order is None:
        label_order = sorted(set(y_to) | set(y_te) |
                             (set(y_ta) if ada_present else set()))
        # cosmetic shortening only
        display_labels = [lbl.replace("Tigress", "") for lbl in label_order]

    # ── ORIGINAL-TRAIN -----------------------------------------------------
    y_pred = model.predict(X_to)
    mets_orig.append({
        "acc":  accuracy_score(y_to, y_pred),
        "prec": precision_score(y_to, y_pred, average="weighted", zero_division=0),
        "rec":  recall_score(y_to, y_pred, average="weighted", zero_division=0),
        "f1":   f1_score(y_to, y_pred, average="weighted", zero_division=0)
    })
    cm = confusion_matrix(y_to, y_pred, labels=label_order)
    cm_orig_rel.append(cm / cm.sum(axis=1, keepdims=True))

    # ── ADASYN-TRAIN -------------------------------------------------------
    if ada_present:
        y_pred = model.predict(X_ta)
        mets_ada.append({
            "acc":  accuracy_score(y_ta, y_pred),
            "prec": precision_score(y_ta, y_pred, average="weighted", zero_division=0),
            "rec":  recall_score(y_ta, y_pred, average="weighted", zero_division=0),
            "f1":   f1_score(y_ta, y_pred, average="weighted", zero_division=0)
        })
        cm = confusion_matrix(y_ta, y_pred, labels=label_order)
        cm_ada_rel.append(cm / cm.sum(axis=1, keepdims=True))

    # ── TEST ---------------------------------------------------------------
    y_pred = model.predict(X_te)
    mets_test.append({
        "acc":  accuracy_score(y_te, y_pred),
        "prec": precision_score(y_te, y_pred, average="weighted", zero_division=0),
        "rec":  recall_score(y_te, y_pred, average="weighted", zero_division=0),
        "f1":   f1_score(y_te, y_pred, average="weighted", zero_division=0)
    })
    cm = confusion_matrix(y_te, y_pred, labels=label_order)
    cm_test_rel.append(cm / cm.sum(axis=1, keepdims=True))

    print(f"  • seed {seed:04d}: processed")

# ── helper: mean ± SD -------------------------------------------------------

def mean_sd(mat_list):
    arr = np.stack(mat_list, axis=0)
    return arr.mean(axis=0), arr.std(axis=0)

def agg(df_list):
    if not df_list:
        return pd.DataFrame()
    df = pd.DataFrame(df_list)
    return pd.concat([df.mean().round(4).rename("mean"),
                      df.std().round(4).rename("sd")], axis=1)

# ── aggregate metrics -------------------------------------------------------
agg_orig = agg(mets_orig)
agg_ada  = agg(mets_ada)
agg_test = agg(mets_test)

fi_raw_df  = pd.concat(fi_raw,  axis=1).fillna(0)
fi_raw_mean, fi_raw_sd = fi_raw_df.mean(axis=1), fi_raw_df.std(axis=1)

cm_o_m, cm_o_s = mean_sd(cm_orig_rel)
cm_t_m, cm_t_s = mean_sd(cm_test_rel)
cm_a_m = cm_a_s = None
if cm_ada_rel:
    cm_a_m, cm_a_s = mean_sd(cm_ada_rel)

# ── plots -------------------------------------------------------------------
sns.set_style("whitegrid")
TOP = 20
top_imp = fi_raw_mean.sort_values(ascending=False).head(TOP)
top_sd  = fi_raw_sd.loc[top_imp.index]

# feature-importance ---------------------------------------------------------
plt.figure(figsize=(9, 7))
bar_palette = (CUSTOM_PALETTE * (TOP // len(CUSTOM_PALETTE) + 1))[:TOP]
sns.barplot(x=top_imp.values, y=top_imp.index, palette=bar_palette)
plt.errorbar(top_imp.values, range(len(top_imp)),
             xerr=top_sd.values, fmt="none", ecolor="black", capsize=2)
plt.title("ExtraTrees – feature importance (mean ± SD)")
plt.ylabel("")            # remove unwanted "None" label
plt.tight_layout()
plt.savefig(OUT_DIR / f"feat_imp_meanSD_{TAG}.png", dpi=300)
plt.savefig(OUT_DIR / f"feat_imp_meanSD_{TAG}.eps")
plt.close()

# ── confusion-matrices ------------------------------------------------------
def plot_cm(mat, title, fname):
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        mat, annot=True, fmt=".2f", cmap=CMAP,
        cbar_kws={"label": "row-norm. mean"},
        xticklabels=display_labels, yticklabels=display_labels,
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"{fname}.png", dpi=300)
    plt.savefig(OUT_DIR / f"{fname}.eps")
    plt.close()

plot_cm(cm_t_m, "TEST rel. confusion matrix",          f"cm_test_mean_{TAG}")
if cm_a_m is not None:
    plot_cm(cm_a_m, "TRAIN – ADASYN rel. confusion matrix",
            f"cm_train_adasyn_mean_{TAG}")
plot_cm(cm_o_m, "TRAIN – ORIGINAL rel. confusion matrix",
        f"cm_train_original_mean_{TAG}")

print(f"[✓] plots saved to {OUT_DIR}")
