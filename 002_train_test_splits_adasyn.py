#!/usr/bin/env python3
# prepare_svd_dataset_batch.py
"""
Batch‑prepare repeated train/test splits for the *SVD complexity‑metrics* table,
optionally oversampling every class by *ADASYN* and writing all artefacts to
disk in a grouping‑aware directory tree.

Overview
--------
The script performs **up to 100** stratified splits (one per ``seed``) of the
metrics table produced by *svd_metrics_extended.py*. For every split it:

1. Reads the full feature matrix (``X_full``) and label vector (``y_full``)
   from the CSV defined in ``DATA_CSV``.
2. Creates a stratified train/test partition with ``TEST_SIZE`` percentage held
   out.
3. Applies **ADASYN** oversampling on the training data. Each class is grown to
   ``ADASYN_FACTOR × max(original class count)`` using ``ADASYN_NBRS`` nearest
   neighbours, ensuring a balanced label distribution without synthetically
   altering the evaluation set.
4. Writes six CSV artefacts into ``Prepared_SVD_g{GROUPING}`` (or a similarly
   tagged folder):

   * ``X_train_raw.csv`` / ``y_train_raw.csv``      – original training split.
   * ``X_train_adasyn.csv`` / ``y_train_adasyn.csv`` – post‑ADASYN training set.
   * ``X_test.csv`` / ``y_test.csv``                – untouched evaluation set.

Key constants
-------------
``GROUPING``
    Mirrors the grouping level used when generating the metrics table so that
    multiple granularities can coexist (>0 collapses labels).
``ADASYN_FACTOR``
    Target size relative to the majority class in the *training* subset.
``SEEDS``
    Inclusive range of integer seeds; the default (1–100) yields 100 distinct
    splits. Adjust upwards if more repetitions are desired.
``OUT_BASE``
    Root folder for the split‑specific folders. Existing folders are skipped so
    the script can be rerun incrementally.

Directory layout (example)
--------------------------
With ``GROUPING = 3`` and ``seed = 1`` the artefacts will be located below::

    Prepared_SVD_g3/
      └─ seed0001_ADASYN/
          └─ data/
              ├─ X_train_raw.csv
              ├─ y_train_raw.csv
              ├─ X_train_adasyn.csv
              ├─ y_train_adasyn.csv
              ├─ X_test.csv
              └─ y_test.csv

Version history
---------------
* **30 Jun 2025** – Documentation expanded, added detailed configuration notes
  and example folder layout.

Author
------
Sebastian Raubitzek
Date: 30 June 2025
"""

from pathlib import Path
import textwrap

import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import ADASYN

# ───────────────────── user parameters ──────────────────────
GROUPING       = 3               # <-- choose 0,1,2,3 …
DATA_CSV       = (
    f"results_ext_SVD_analysis/"
    f"final_processed_matrices_grouping{GROUPING}_extra_metrics.csv"
)
#add_tag = 'fac_11'
add_tag = ''
TARGET_COL     = "class"
TEST_SIZE      = 0.20
ADASYN_FACTOR  = 1.3               # every class → FACTOR × max_count g2 1.3 everythign else is 1
ADASYN_NBRS    = 5
SEEDS          = range(1, 101)    # generate 1‥1000 splits
OUT_BASE       = Path(f"Prepared_SVD_g{GROUPING}{add_tag}")

# ─────────────────────────────────────────────────────────────


def counts_pretty(series: pd.Series) -> str:
    """Return sorted class counts as multiline string."""
    return "\n".join(
        f"{lbl:>20}: {cnt}" for lbl, cnt in series.value_counts().sort_index().items()
    )


# 0 ─── load once
# ---------------------------------------------
# All splits share the same full dataset; load it a single time.
df = pd.read_csv(DATA_CSV)
X_full, y_full = df.drop(columns=[TARGET_COL]), df[TARGET_COL]
print(f"Loaded {len(df):,} rows • {X_full.shape[1]} features  (grouping={GROUPING})")

# 1 ─── iterate over seeds
# ---------------------------------------------
for seed in SEEDS:
    fold_dir = OUT_BASE / f"seed{seed:04d}_ADASYN" / "data"
    if fold_dir.exists():
        print(f"[skip] seed {seed} already done")
        continue
    fold_dir.mkdir(parents=True, exist_ok=True)

    # 1.1 split
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_full, y_full, test_size=TEST_SIZE, stratify=y_full, random_state=seed
    )

    # 1.2 ADASYN: every label → FACTOR × largest(original)
    max_count = y_tr.value_counts().max()
    target = int(ADASYN_FACTOR * max_count)
    strat = {lbl: target for lbl in y_tr.unique()}

    ada = ADASYN(
        sampling_strategy=strat,
        n_neighbors=ADASYN_NBRS,
        random_state=seed,
    )
    X_syn, y_syn = ada.fit_resample(X_tr, y_tr)

    # 1.3 save
    X_tr.to_csv(fold_dir / "X_train_raw.csv", index=False)
    y_tr.to_csv(fold_dir / "y_train_raw.csv", index=False, header=True)

    X_syn.to_csv(fold_dir / "X_train_adasyn.csv", index=False)
    y_syn.to_csv(fold_dir / "y_train_adasyn.csv", index=False, header=True)

    X_te.to_csv(fold_dir / "X_test.csv", index=False)
    y_te.to_csv(fold_dir / "y_test.csv", index=False, header=True)

    # 1.4 console report
    print(
        textwrap.dedent(
            f"""
            seed {seed:04d}  →  {fold_dir.parent.name}
              original rows   : {len(X_tr):>6}
              ADASYN rows     : {len(X_syn):>6}  (factor {ADASYN_FACTOR})
            ── class counts BEFORE ──
            {counts_pretty(y_tr)}
            ── class counts AFTER ──
            {counts_pretty(y_syn)}
            """
        ).rstrip()
    )

print("\n✓ All splits generated and stored under", OUT_BASE)
