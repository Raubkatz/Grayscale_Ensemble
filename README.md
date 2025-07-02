# Grayscale_Ensemble
Repository for the corresponding paper on grayscale-domain obfuscation identification

## Overview
This repository implements a complete pipeline that

1. **Extracts SVD-based complexity metrics** from images or other grayscale matrices.  
2. **Generates repeated, stratified train/test splits** and optionally balances the training data with ADASYN.  
3. **Trains an ExtraTrees ensemble** for every split with Bayesian hyper-parameter search.  
4. **Evaluates and visualises** model performance, feature importance, and per-class feature distributions.

The workflow is fully script-driven; no notebook interaction is required.

## Repository Structure

├── 001_build_svd_complexity_dataset.py # compute 18-metric SVD signature per pickle

├── 002_train_test_splits_adasyn.py # create N stratified splits + optional ADASYN

├── 003_train_extra_trees_ensemble.py # train ExtraTrees per split with BayesSearchCV

├── 004_ensemble_analysis.py # aggregate metrics, plot importance & confusion matrices

├── 005_distribution_plots.py # box-plot feature distributions per predicted class

├── func_SVD_metrics.py # metric helper library (imported by 001)

├── LICENSE # MIT license

└── README.md # this file


### Script Summaries
| Script | Purpose | Key Outputs |
|--------|---------|-------------|
| **001_build_svd_complexity_dataset.py** | Scans a directory of pickled 2-D arrays, applies grouping rules, computes 18 SVD metrics, and writes a single CSV. | `results_ext_SVD_analysis/final_processed_matrices_grouping*.csv` |
| **002_train_test_splits_adasyn.py** | Produces up to 1 000 stratified splits; saves raw and ADASYN-balanced train sets plus the held-out test set for each seed. | `Prepared_SVD_g*/seed####_ADASYN/data/*.csv` |
| **003_train_extra_trees_ensemble.py** | Runs Bayesian hyper-parameter search and fits one ExtraTreesClassifier per split. Saves pickled models and JSON metrics. | `Prepared_SVD_g*/ExtraTrees_{raw|adasyn}/models/*.pkl` and `reports/*.json` |
| **004_ensemble_analysis.py** | Aggregates per-seed metrics, plots top-20 feature importance with error bars, and saves mean row-normalised confusion matrices. | `Evaluation_ExtraTrees_g*/feat_imp_meanSD_*.png` / `.eps`, `cm_*_mean_*.png` |
| **005_distribution_plots.py** | Pools all test rows, assigns predicted class labels, and draws per-feature box plots plus an optional descriptive-stats text file. | `Evaluation_ExtraTrees_g*/distributions/*_boxplot.png` / `.eps`, `feature_distribution_summary.txt` |
| **func_SVD_metrics.py** | Contains the 18 metric implementations (6 baseline + 12 additional) used by script 001. | imported module |

### Prerequisites
The code is tested with **Python 3.9** and the package versions below:

numpy	2.2.6
pandas	2.2.3
scikit-learn	1.5.0
imbalanced-learn	0.13.0
scikit-optimize	0.10.2
matplotlib	3.10.3
seaborn	0.13.2

## Data

The data is compressed and collected in the corresponding archive.z*-zip files.
