"""
Process grayscale-image matrices stored as pickle files, compute a comprehensive
suite of singular‑value–based complexity metrics, and save the aggregated
results as a CSV file.

Overview
--------
This script searches *recursively* through the user‑defined ``DATA_DIR`` for
``.pkl`` files. Each pickle is expected to contain a two‑dimensional NumPy
array representing a grayscale image or any other real‑valued matrix. For each
matrix, the script performs the following steps:

1. **Label extraction** – A descriptive *class* label is derived from the file
   name, then optionally collapsed into broader categories controlled by the
   ``GROUPING`` constant.
2. **Spectrum calculation** – The singular‑value spectrum ``σ`` is obtained via
   :pyfunc:`numpy.linalg.svd`. If ``VALID`` is ``True``, values below
   ``THRESHOLD`` are discarded to mitigate numerical artefacts.
3. **Metric evaluation** – Twenty‑one metrics are computed:

   * *Baseline*: 6 legacy metrics retained for backward compatibility.
   * *Additional*: 10 metrics defined in ``func_add_metrics.py``.
   * *More*: 5 metrics defined in ``func_add_metrics_more.py``.

   The metrics capture entropy, energy distribution, rank surrogates, spectral
   flatness, higher‑order moments, and several information‑theoretic measures.
4. **Aggregation and export** – All per‑sample results are collected in a
   :class:`pandas.DataFrame` and written to ``OUT_DIR``.

Configuration
-------------
Adjust the constants below as needed:

* ``GROUPING``   – 0 keeps original labels; 1–3 increasingly coarse groupings.
* ``VALID``      – Drop singular values smaller than ``THRESHOLD`` when ``True``.
* ``DATA_DIR``   – Folder containing the input pickle files.
* ``OUT_DIR``    – Destination folder for the resulting CSV.
* ``THRESHOLD``  – Cut‑off used by the *valid* filter.

Version history
---------------
* **30 Jun 2025** – Added 15 new metrics, refined documentation, and clarified
  grouping rules.

Author
------
Date: 30 June 2025
"""

# ---------------------------------------------------------------------------
# Standard library imports
# ---------------------------------------------------------------------------
import os
import re
import pickle
from copy import deepcopy as dc

# ---------------------------------------------------------------------------
# Third‑party imports
# ---------------------------------------------------------------------------
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Local metric helpers – the import list mirrors earlier versions for
# backward compatibility. The helper modules now contain rich docstrings with
# literature references for each metric.
# ---------------------------------------------------------------------------
from func_SVD_metrics import (  # original six plus extras
    calculate_entropy,
    calculate_fisher_information,
    relative_decay,
    singular_value_energy,
    nuclear_norm,
    frobenius_norm,
    schatten_p_norm,
    effective_rank,
    participation_ratio,
    inverse_participation_ratio,
    spectral_flatness,
    renyi_entropy,
    tsallis_entropy,
    coefficient_of_variation,
    spectral_skewness,
    spectral_kurtosis,
)

# ---------------------------------------------------------------------------
# Configuration – modify these constants to tailor the run.
# ---------------------------------------------------------------------------
GROUPING = 3          # 0 = no grouping … 3 = most coarse
VALID = True          # Discard very small singular values when True
#DATA_DIR = "./image_data"
#DATA_DIR = "./image_data_add"
#DATA_DIR = "./new_data_set_pickles"
DATA_DIR = "./final_dataset_COSE"
OUT_DIR = "results_ext_SVD_analysis"
os.makedirs(OUT_DIR, exist_ok=True)

THRESHOLD = 1e-6      # Cut‑off for the VALID filter

# ---------------------------------------------------------------------------
# Processing loop – iterate over every pickle file in DATA_DIR.
# ---------------------------------------------------------------------------
records = []

for fname in os.listdir(DATA_DIR):
    path = os.path.join(DATA_DIR, fname)
    if not os.path.isfile(path):
        continue  # Skip sub‑directories and non‑files

    print(f"Processing {fname}")

    # -----------------------------------------------------------------------
    # 1. Load matrix from pickle
    # -----------------------------------------------------------------------
    with open(path, "rb") as fh:
        matrix = np.asarray(pickle.load(fh), dtype=float)

    # -----------------------------------------------------------------------
    # 2. Derive class / label string from file name and apply optional
    #    grouping rules. The mapping logic preserves historical behaviour.
    # -----------------------------------------------------------------------
    first_us, last_us = fname.find("_") + 1, fname.rfind("_")
    cls = (
        fname[first_us:last_us] if 0 < first_us < last_us else "Unknown"
    )
    cls = re.sub(r"O\d+$", "", cls)  # Strip trailing optimisation flags

    # Grouping rules ---------------------------------------------------------
    if GROUPING != 0:
        if GROUPING < 3:  # two‑level grouping
            if "tigress" in cls:
                if "encode" in cls:
                    cls = "TigressDataObfuscation"
                elif "Encode" in cls:
                    cls = "TigressDataObfuscation"
                elif "flatten" in cls or "split" in cls:
                    cls = "TigressCFGObfuscation"
                elif "Flatten" in cls or "Split" in cls:
                    cls = "TigressCFGObfuscation"
                elif "virtualize" in cls or "jit" in cls:
                    cls = "TigressDynamicObfuscation"
                elif "Virtualize" in cls or "Jit" in cls:
                    cls = "TigressDynamicObfuscation"
                elif "antitaint" in cls or "antialias" in cls:
                    cls = "TigressAntiAnalysis"
                elif "AntiTaint" in cls or "AntiAlias" in cls:
                    cls = "TigressAntiAnalysis"
            if GROUPING == 1 and "Tigress" in cls:
                # Collapse all Tigress variants into one bucket
                cls = "TigressObfuscation"
        else:  # fine‑grained mapping
            mapping = {
                "EncodeArithmetic": "TigressEncodeArithmetic",
                "EncodeLiterals":   "TigressEncodeLiterals",
                "encodeArithmetic": "TigressEncodeArithmetic",
                "encodeLiterals": "TigressEncodeLiterals",
                "encodearithmetic": "TigressEncodeArithmetic",
                "encodeliterals": "TigressEncodeLiterals",
                "flatten":          "TigressFlatten",
                "split":            "TigressSplit",
                "virtualize":       "TigressVirtualize",
                "jit":              "TigressJit",
                "Flatten": "TigressFlatten",
                "Split": "TigressSplit",
                "Virtualize": "TigressVirtualize",
                "Jit": "TigressJit",
                "AntiTaintAnalysis": "TigressAntiTaintAnalysis",
                "antitaintanalysis": "TigressAntiTaintAnalysis",
                "AntiAliasAnalysis": "TigressAntiAliasAnalysis",
                "antialiasanalysis": "TigressAntiAliasAnalysis",
            }
            for key, val in mapping.items():
                if key in cls:
                    cls = val
                    break
        # Collapse compiler‑generated classes into "NoObfuscation"
        if GROUPING >= 1 and any(x in cls for x in
                                 ("oslatest", "tinycc", "tendra", "clang")):
            cls = "NoObfuscation"
    else:
        # GROUPING == 0 – keep label as‑is (minus stray underscores)
        if cls[-1:] in "_-":
            cls = cls[:-1]

    # -----------------------------------------------------------------------
    # 3. Singular‑value spectrum (σ)
    # -----------------------------------------------------------------------
    sigma = np.linalg.svd(matrix, compute_uv=False)
    if VALID:
        sigma = dc(sigma[sigma > THRESHOLD])

    if sigma.size == 0:
        # All metrics undefined → skip sample
        print("  WARNING: no valid singular values, sample skipped.")
        continue

    # -----------------------------------------------------------------------
    # 4. Metric computation
    # -----------------------------------------------------------------------
    # Legacy six -------------------------------------------------------------
    svd_entropy = calculate_entropy(matrix, valid=VALID)
    singular_spectral_radius = sigma[0]
    svd_condition_number = sigma[0] / sigma[-1]
    svd_relative_decay = relative_decay(sigma)
    svd_energy = singular_value_energy(sigma)
    svd_fisher_info = calculate_fisher_information(matrix, valid=VALID)

    # Additional fifteen -----------------------------------------------------
    metric_dict = {
        # Basic length‑preserving helpers
        "nuclear_norm": nuclear_norm(sigma),
        "frobenius_norm": frobenius_norm(sigma),
        "schatten_p3_norm": schatten_p_norm(sigma, p=3),
        "effective_rank": effective_rank(sigma),
        "participation_ratio": participation_ratio(sigma),
        "inverse_participation_ratio": inverse_participation_ratio(sigma),
        "spectral_flatness": spectral_flatness(sigma),
        "renyi_entropy_alpha2": renyi_entropy(sigma, alpha=2),
        "tsallis_entropy_q2": tsallis_entropy(sigma, q=2),
        "coeff_variation": coefficient_of_variation(sigma),
        "spectral_skewness": spectral_skewness(sigma),
        "spectral_kurtosis": spectral_kurtosis(sigma),

        # Legacy metrics for continuity
        "svd_entropy": svd_entropy,
        "singular_spectral_radius": singular_spectral_radius,
        "svd_condition_number": svd_condition_number,
        "svd_relative_decay": svd_relative_decay,
        "svd_energy": svd_energy,
        "svd_fisher_info": svd_fisher_info,
        "class": cls,
    }

    records.append(metric_dict)

# ---------------------------------------------------------------------------
# 5. Assemble DataFrame & export
# ---------------------------------------------------------------------------

df = pd.DataFrame(records)
print(df.head())
outfile = f"./{OUT_DIR}/final_processed_matrices_grouping{GROUPING}_extra_metrics.csv"
df.to_csv(outfile, index=False)
print(f"\nSaved: {outfile}")
