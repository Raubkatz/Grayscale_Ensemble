# -*- coding: utf-8 -*-
"""
SVD‑based Complexity Metrics
============================
A concise, NumPy‑only collection of 18 scalar metrics for analysing the singular‑value
spectrum of a matrix.  Twelve modern descriptors are complemented by six legacy
wrappers preserved for backward compatibility.

* Author : Sebastian Raubitzek
* Date   : 2025‑06‑30

Public API
----------
Modern metrics
~~~~~~~~~~~~~~
- ``nuclear_norm``
- ``frobenius_norm``
- ``schatten_p_norm``  (p = 3 preset via *metric_dict*)
- ``effective_rank``
- ``participation_ratio``
- ``inverse_participation_ratio``
- ``spectral_flatness``
- ``renyi_entropy``      (α = 2 preset)
- ``tsallis_entropy``    (q = 2 preset)
- ``coefficient_of_variation``
- ``spectral_skewness``
- ``spectral_kurtosis``

Legacy metrics
~~~~~~~~~~~~~~
Retained for continuity with earlier codebases.
- ``svd_entropy`` (Shannon entropy of singular values)
- ``singular_spectral_radius`` (spectral radius)
- ``svd_condition_number`` (κ = σ_max / σ_min)
- ``svd_relative_decay`` (average first‑order decay)
- ``svd_energy`` (dominant energy ratio)
- ``svd_fisher_info`` (Fisher information measure)

All routines return a single ``float``.  Functions accepting ``sigma`` expect a
1‑D array of non‑negative singular values in any order.  Matrix‑level routines
(suffix *info* / *entropy*) accept a 2‑D NumPy array.  Numerical kernels are
unchanged from the original implementation; the module has merely been trimmed
to the metrics listed above and stripped of unused imports.
"""

# ---------------------------------------------------------------------------
# Imports (minimal)
# ---------------------------------------------------------------------------

import math
import random
from copy import deepcopy as dc

import numpy as np
from scipy.stats import entropy

# ---------------------------------------------------------------------------
# Internal helpers (shared)
# ---------------------------------------------------------------------------

def _to_numpy(arr):
    """Return *arr* as a 1‑D float64 NumPy array."""
    return np.asarray(arr, dtype=float).ravel()


def _normalise(svals, eps: float = 1e-12):
    """Return probability mass ``p_i = σ_i / Σσ_j`` with numerical safety."""
    total = np.sum(svals)
    if total < eps:
        raise ValueError("Sum of singular values is (near) zero.")
    return svals / total


def _safe_mean_std(arr, ddof: int = 0):
    """Return mean and std; raise if the distribution is degenerate."""
    mu = np.mean(arr)
    sigma = np.std(arr, ddof=ddof)
    if sigma == 0:
        raise ValueError("Standard deviation is zero.")
    return mu, sigma

# ---------------------------------------------------------------------------
# Modern metrics (1–12)
# ---------------------------------------------------------------------------

def nuclear_norm(singular_values):
    """Nuclear/trace norm ‖A‖_* = Σ σ_i."""
    s = _to_numpy(singular_values)
    return float(np.sum(s))


def frobenius_norm(singular_values):
    """Frobenius norm ‖A‖_F = √Σ σ_i²."""
    s = _to_numpy(singular_values)
    return float(np.sqrt(np.sum(s ** 2)))


def schatten_p_norm(singular_values, p: float = 3):
    """General Schatten‑p norm for any p ≥ 1."""
    if p < 1:
        raise ValueError("p must be ≥ 1.")
    s = _to_numpy(singular_values)
    return float(np.power(np.sum(s ** p), 1.0 / p))


def effective_rank(singular_values):
    """Effective rank r_eff = exp(−Σ p_i log p_i)."""
    s = _to_numpy(singular_values)
    p_i = _normalise(s)
    return float(np.exp(-np.sum(p_i * np.log(p_i))))


def participation_ratio(singular_values):
    """Participation ratio PR = (Σ σ_i²)² / Σ σ_i⁴."""
    s = _to_numpy(singular_values)
    return float(np.sum(s ** 2) ** 2 / np.sum(s ** 4))


def inverse_participation_ratio(singular_values):
    """Inverse participation ratio IPR = 1 / PR."""
    return 1.0 / participation_ratio(singular_values)


def spectral_flatness(singular_values):
    """Spectral flatness = geometric mean / arithmetic mean."""
    s = _to_numpy(singular_values)
    gm = np.exp(np.mean(np.log(s + 1e-20)))  # ε avoids log 0
    am = np.mean(s)
    return float(gm / am)


def renyi_entropy(singular_values, alpha: float = 2):
    """Rényi entropy of order α (α ≠ 1)."""
    if alpha == 1:
        raise ValueError("Use Shannon entropy for α = 1.")
    p_i = _normalise(_to_numpy(singular_values))
    return float(math.log(np.sum(p_i ** alpha)) / (1.0 - alpha))


def tsallis_entropy(singular_values, q: float = 2):
    """Tsallis entropy of order q (q ≠ 1)."""
    if q == 1:
        raise ValueError("Use Shannon entropy for q = 1.")
    p_i = _normalise(_to_numpy(singular_values))
    return float((1.0 - np.sum(p_i ** q)) / (q - 1.0))


def coefficient_of_variation(singular_values):
    """Coefficient of variation CV = σ / μ."""
    s = _to_numpy(singular_values)
    mu, sigma = _safe_mean_std(s)
    return float(sigma / mu)


def spectral_skewness(singular_values):
    """Skewness γ₁ of the singular‑value distribution."""
    s = _to_numpy(singular_values)
    mu, sigma = _safe_mean_std(s, ddof=0)
    return float(np.mean(((s - mu) / sigma) ** 3))


def spectral_kurtosis(singular_values):
    """Excess kurtosis γ₂ of the singular‑value distribution."""
    s = _to_numpy(singular_values)
    mu, sigma = _safe_mean_std(s, ddof=0)
    return float(np.mean(((s - mu) / sigma) ** 4) - 3.0)

# ---------------------------------------------------------------------------
# Support routines (used by legacy wrappers)
# ---------------------------------------------------------------------------

def singular_value_energy(singular_values, k: int = 3):
    """Energy ratio of the *k* dominant singular values."""
    s = _to_numpy(singular_values)
    dominant = np.sum(s[:k] ** 2)
    total = np.sum(s ** 2)
    return float(dominant / total)


def relative_decay(singular_values):
    """Average first‑order decay between consecutive singular values."""
    s = _to_numpy(singular_values)
    if s.size < 2:
        raise ValueError("At least two singular values are required.")
    return float(np.mean(s[:-1] - s[1:]))


def calculate_fisher_information(matrix, random_scramble: int = 0, *, make_square: bool = False, valid: bool = True):
    """Fisher information of the singular‑value spectrum.

    Parameters
    ----------
    matrix : ndarray
        Input matrix *(m × n)*.
    random_scramble : int, optional
        0 → no scrambling (default);
        1 → alternate row/column scrambling;
        2 → row scrambling only;
        3 → column scrambling only.
    make_square : bool, optional
        If *True*, premultiply/postmultiply to obtain a square matrix.
    valid : bool, optional
        If *True*, discard singular values below 1 e‑6 before normalisation.
    """
    A = np.asarray(matrix, dtype=float)

    # Optionally make *A* square (Gram matrix trick)
    if make_square:
        A = A @ A.T if A.shape[0] < A.shape[1] else A.T @ A

    def _scramble(mat):
        """In‑place row/column scrambling helper."""
        if random_scramble == 1:  # alternating
            idxs = np.random.choice(mat.shape[1 if i % 2 == 0 else 0], 2, replace=False)
            if i % 2 == 0:
                mat[:, idxs] = mat[:, idxs[::-1]]
            else:
                mat[idxs, :] = mat[idxs[::-1], :]
        elif random_scramble == 2:  # rows only
            idxs = np.random.choice(mat.shape[0], 2, replace=False)
            mat[idxs, :] = mat[idxs[::-1], :]
        elif random_scramble == 3:  # cols only
            idxs = np.random.choice(mat.shape[1], 2, replace=False)
            mat[:, idxs] = mat[:, idxs[::-1]]

    # Monte‑Carlo average if scrambling is requested
    if random_scramble:
        fis = []
        for i in range(1000):
            B = np.copy(A)
            for _ in range(random.randint(1, 100)):
                _scramble(B)
            fis.append(_single_fi(B, valid))
        return float(np.mean(fis))

    return _single_fi(A, valid)


def _single_fi(mat, valid):
    """Helper: Fisher information of a single matrix instance."""
    s = np.linalg.svd(mat, compute_uv=False)
    if valid:
        s = s[s > 1e-6]
    s = s / np.sum(s)  # normalise
    fi = (s[1:] - s[:-1]) ** 2 / s[:-1]
    return float(np.sum(fi))


def calculate_entropy(matrix, random_scramble: int = 0, *, make_square: bool = False, valid: bool = True):
    """Shannon entropy of the singular‑value spectrum (optionally scrambled)."""
    A = np.asarray(matrix, dtype=float)

    # Optionally make *A* square
    if make_square:
        A = A @ A.T if A.shape[0] < A.shape[1] else A.T @ A

    def _scramble(mat):
        if random_scramble == 1:
            idxs = np.random.choice(mat.shape[1 if i % 2 == 0 else 0], 2, replace=False)
            if i % 2 == 0:
                mat[:, idxs] = mat[:, idxs[::-1]]
            else:
                mat[idxs, :] = mat[idxs[::-1], :]
        elif random_scramble == 2:
            idxs = np.random.choice(mat.shape[0], 2, replace=False)
            mat[idxs, :] = mat[idxs[::-1], :]
        elif random_scramble == 3:
            idxs = np.random.choice(mat.shape[1], 2, replace=False)
            mat[:, idxs] = mat[:, idxs[::-1]]

    # Monte‑Carlo average if scrambling is requested
    if random_scramble:
        ents = []
        for i in range(1000):
            B = np.copy(A)
            for _ in range(random.randint(1, 100)):
                _scramble(B)
            ents.append(_single_entropy(B, valid))
        return float(np.mean(ents))

    return _single_entropy(A, valid)


def _single_entropy(mat, valid):
    s = np.linalg.svd(mat, compute_uv=False)
    if valid:
        s = s[s > 1e-6]
    return float(entropy(s))

# ---------------------------------------------------------------------------
# Legacy wrappers (13–18)
# ---------------------------------------------------------------------------

def svd_entropy(matrix, *args, **kwargs):
    """Alias of :func:`calculate_entropy` for backward compatibility."""
    return calculate_entropy(matrix, *args, **kwargs)


def singular_spectral_radius(singular_values):
    """Spectral radius ρ = max σ_i."""
    return float(np.max(_to_numpy(singular_values)))


def svd_condition_number(singular_values):
    """Condition number κ = σ_max / σ_min."""
    s = _to_numpy(singular_values)
    sigma_min = np.min(s[s > 1e-12])
    return float(np.max(s) / sigma_min)


def svd_relative_decay(singular_values):
    """Wrapper for :func:`relative_decay`."""
    return relative_decay(singular_values)


def svd_energy(singular_values, k: int = 3):
    """Wrapper for :func:`singular_value_energy`."""
    return singular_value_energy(singular_values, k=k)


def svd_fisher_info(matrix, *args, **kwargs):
    """Alias of :func:`calculate_fisher_information`."""
    return calculate_fisher_information(matrix, *args, **kwargs)

# Provide a benign placeholder so that metric_dict remains import‑clean even
# if *cls* is supplied from an external context at runtime.
cls = None  # type: ignore

