"""Core k-nearest-neighbors treatment-effect estimator.

Implements the method from:

    Garcia Bulle Bueno, Dahleh, and Hosoi (2026),
    "A k-nearest-neighbors based method for estimation of treatment effects
    in agriculture without treatment labels", Information Processing in Agriculture.

The idea: given two snapshots of an outcome variable (e.g. yield) measured over
a set of "treatable" units, we estimate the magnitude of a treatment effect
*without* knowing which units were treated. We compare the leave-one-out
k-nn prediction error in a baseline snapshot (no/all units treated) with the
error in a mixed snapshot (some units treated). Excess within-neighborhood
variance in the mixed snapshot is attributed to treatment heterogeneity.

Key equations (see Section 2 of the paper):

    E[(y_it - ybar_it)^2] = (beta^2 var(T_it) + var(eps)) (1 + 1/J)          (eq. 3)

    beta^2 = 1/(p(1-p)) * ( (E_t - E_0) / (1 + 1/J) )                        (eq. 6, p known)

    beta^2 >= 4 * ( (E_t - E_0) / (1 + 1/J) )                               (eq. 7, p unknown)

where E_0 and E_t are the mean leave-one-out squared k-nn errors in the
baseline and mixed snapshots, J is the number of neighbors (excluding self),
and p is the treated proportion.
"""

from __future__ import annotations

import numpy as np
from sklearn.neighbors import NearestNeighbors


def knn_loo_squared_errors(coords, values, n_neighbors):
    """Leave-one-out k-nn squared errors for every unit.

    For each unit i, predict its outcome as the mean outcome of its
    ``n_neighbors`` nearest neighbors *excluding itself*, and return the
    squared prediction error. This uses the algebraic leave-one-out trick:
    query ``n_neighbors + 1`` neighbors (which includes the point itself at
    distance 0), take the mean, and remove the self contribution.

    Parameters
    ----------
    coords : array-like, shape (n, d)
        Coordinates used to define neighborhoods (e.g. planar km, UTM meters).
    values : array-like, shape (n,)
        Outcome variable (e.g. yield).
    n_neighbors : int
        Number of neighbors J used for the estimate (excluding self).

    Returns
    -------
    numpy.ndarray, shape (n,)
        Squared leave-one-out prediction error for every unit.
    """
    coords = np.asarray(coords, dtype=float)
    values = np.asarray(values, dtype=float)

    n_query = n_neighbors + 1  # include self (distance 0)
    nn = NearestNeighbors(n_neighbors=n_query, algorithm="kd_tree")
    nn.fit(coords)
    _, idx = nn.kneighbors(coords, return_distance=True)

    y_knn = values[idx]                      # (n, n_neighbors + 1), includes self
    mean_incl_self = y_knn.mean(axis=1)
    # Remove self to obtain the leave-one-out neighbor mean.
    loo_pred = (n_query * mean_incl_self - values) / (n_query - 1)
    return (loo_pred - values) ** 2


def knn_loo_mse(coords, values, n_neighbors):
    """Mean leave-one-out k-nn squared error (the E[.] terms in the paper)."""
    return float(np.mean(knn_loo_squared_errors(coords, values, n_neighbors)))


def beta_from_errors(e_mixed, e_baseline, n_neighbors, p=None):
    """Estimate the treatment-effect magnitude |beta| from two error terms.

    Parameters
    ----------
    e_mixed : float
        Mean leave-one-out k-nn squared error in the mixed-treatment snapshot.
    e_baseline : float
        Mean leave-one-out k-nn squared error in the baseline snapshot.
    n_neighbors : int
        Number of neighbors J used (excluding self).
    p : float or None
        Treated proportion. If given, returns the exact estimate (eq. 6).
        If ``None``, returns the lower bound using p(1-p) <= 1/4 (eq. 7).

    Returns
    -------
    float
        Estimated |beta| (or a lower bound on it). Returns 0.0 when the excess
        error is non-positive (no detectable treatment-induced variation).
    """
    excess = (e_mixed - e_baseline) / (1.0 + 1.0 / n_neighbors)
    if excess <= 0:
        return 0.0
    if p is None:
        return float(np.sqrt(4.0 * excess))          # lower bound, p unknown
    return float(np.sqrt(excess / (p * (1.0 - p))))  # exact, p known


def estimate_beta(coords0, values0, coords_t, values_t, n_neighbors, p=None):
    """Full estimator: compute k-nn errors in both snapshots, then |beta|.

    ``coords0``/``values0`` describe the baseline snapshot (no/all treated),
    ``coords_t``/``values_t`` the mixed-treatment snapshot.
    """
    e0 = knn_loo_mse(coords0, values0, n_neighbors)
    et = knn_loo_mse(coords_t, values_t, n_neighbors)
    return beta_from_errors(et, e0, n_neighbors, p=p)
