#!/usr/bin/env python3
# extratrees_from_prepared.py
"""
Train a separate **ExtraTreesClassifier** for every prepared SVD split and log
per‑run metrics. Each split is produced by *prepare_svd_dataset_batch.py* and
lives under ``Prepared_SVD_g{GROUPING}/``.

Workflow
--------
1. **Split discovery** – Iterate over the first ``N_SPLITS`` seed folders.
2. **Data loading** – Depending on ``USE_ADASYN`` read either the raw or the
   ADASYN‑balanced training set plus the corresponding test set.
3. **Bayesian hyper‑parameter search** – Perform ``BAYES_ITERS`` iterations of
   :class:`skopt.BayesSearchCV` with 5‑fold cross‑validation using weighted F1
   as the optimisation target. The search space covers tree depth, feature
   sampling, and node‑split parameters.
4. **Evaluation** – Measure accuracy, precision, recall, F1, and the confusion
   matrix on the held‑out test set.
5. **Persistence** – Store the fitted estimator (``.pkl``) and a JSON‑encoded
   metrics file inside an experiment‑specific folder tree:

   ``Prepared_SVD_g{GROUPING}{extra_tag}/ExtraTrees_{TAG}/{models|reports}/``

Configuration
-------------
``GROUPING``
    0–3; must match the preprocessing setting of the metrics table.
``USE_ADASYN``
    ``True`` to use the oversampled training split, ``False`` for raw data.
``N_SPLITS``
    Number of seeds / prepared data folds to process.
``BAYES_ITERS``
    Bayesian optimisation budget per seed.

Outputs
-------
* ``models/extratrees_seed####_{TAG}.pkl`` – Pickled fitted estimator.
* ``reports/metrics_seed####_{TAG}.json`` – Hyper‑parameters and evaluation
  metrics.

Author
------
Date: 30 June 2025
"""

from pathlib import Path
import json
import pickle
import textwrap
import warnings
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
)
from skopt import BayesSearchCV
from skopt.space import Integer, Categorical

warnings.filterwarnings("ignore")

# ────────────────────────── USER PARAMS ─────────────────────────
GROUPING        = 1           # 0 … 3   (matches earlier preprocessing)
USE_ADASYN      = True        # False → X_train_original.csv
N_SPLITS        = 100         # number of prepared folds to process
BAYES_ITERS     = 50          # Bayesian optimisation iterations per seed
RANDOM_SEED0    = 137         # base seed (each run adds its own seed)
#extra_tag = "_fac11"
extra_tag = ""

# ----------------------------------------------------------------
PREP_ROOT = Path(f"Prepared_SVD_g{GROUPING}{extra_tag}")
TAG = "adasyn" if USE_ADASYN else "raw"

# hyper‑parameter search space (skopt syntax)
param_space = {
    "n_estimators": Integer(80, 200),
    "criterion": Categorical(["gini", "entropy"]),
    "max_features": Categorical(["log2", "sqrt", None]),
    "max_depth": Integer(10, 50),
    "min_samples_split": Integer(2, 10),
    "min_samples_leaf": Integer(1, 5),
}

# where to save artefacts
OUT_BASE = PREP_ROOT / f"ExtraTrees_{TAG}{extra_tag}"
(OUT_BASE / "models").mkdir(parents=True, exist_ok=True)
(OUT_BASE / "reports").mkdir(parents=True, exist_ok=True)

# ─────────────────────────── helpers ────────────────────────────

def cm_str(cm, labels):
    """Return a compact string representation of the confusion matrix."""
    head = "      " + " ".join(f"{l:>6}" for l in labels)
    rows = [f"{l:<6}" + " ".join(f"{n:6d}" for n in cm[i]) for i, l in enumerate(labels)]
    return head + "\n" + "\n".join(rows)


def one_run(seed: int) -> None:
    """Train and evaluate one ExtraTrees model for *seed*‑indexed split."""
    fold = PREP_ROOT / f"seed{seed:04d}_ADASYN" / "data"
    if not fold.exists():
        print(f"[WARN] fold for seed {seed} missing – skipped.")
        return

    # ---------------- load data ----------------
    X_train = pd.read_csv(fold / f"X_train_{TAG}.csv")
    y_train = pd.read_csv(fold / f"y_train_{TAG}.csv").iloc[:, 0]
    X_test  = pd.read_csv(fold / "X_test.csv")
    y_test  = pd.read_csv(fold / "y_test.csv").iloc[:, 0]

    # ---------------- hyper‑parameter search ----------------
    base_model = ExtraTreesClassifier(random_state=seed)
    search = BayesSearchCV(
        base_model, param_space, n_iter=BAYES_ITERS,
        cv=5, verbose=2, random_state=seed,
        n_jobs=-1,
        scoring="f1_weighted",  # weighted F1 balances class distribution
    )

    t0 = time.time()
    search.fit(X_train, y_train)
    hp_time = time.time() - t0

    model = search.best_estimator_
    best_par = search.best_params_
    best_cv  = search.best_score_

    # ---------------- evaluation ----------------
    t1 = time.time()
    y_pred = model.predict(X_test)
    eval_time = time.time() - t1

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted")
    rec  = recall_score(y_test, y_pred, average="weighted")
    f1   = f1_score(y_test, y_pred, average="weighted")
    cm   = confusion_matrix(y_test, y_pred)

    labels = sorted(np.unique(np.r_[y_train, y_test]))
    report = classification_report(y_test, y_pred, digits=4)

    # ---------------- save artefacts ----------------
    mod_path = OUT_BASE / "models" / f"extratrees_seed{seed:04d}_{TAG}.pkl"
    res_path = OUT_BASE / "reports" / f"metrics_seed{seed:04d}_{TAG}.json"

    with open(mod_path, "wb") as fh:
        pickle.dump(model, fh)

    json.dump(
        {
            "seed": seed,
            "grouping": GROUPING,
            "set": TAG,
            "best_params": best_par,
            "best_cv_score": best_cv,
            "hp_time_sec": hp_time,
            "eval_time_sec": eval_time,
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "confusion_matrix": cm.tolist(),
            "label_order": labels,
            "classification_report": report,
        },
        open(res_path, "w"),
        indent=2,
    )

    # ---------------- console summary ----------------
    print(
        textwrap.dedent(
            f"""
            ▶ seed {seed:04d} | {TAG.upper():5} | acc {acc:.3f} |
              best‑cv {best_cv:.3f} | hp {hp_time:.1f}s | eval {eval_time:.1f}s
            """
        ).strip()
    )

# ────────────────────────── main loop ──────────────────────────
for s in range(1, N_SPLITS + 1):
    one_run(s)

print(
    f"\n✓ Finished training ExtraTrees on all {N_SPLITS} prepared splits "
    f"(grouping={GROUPING}, set={TAG.upper()}).\n"
    f"Artefacts saved under: {OUT_BASE}"
)
