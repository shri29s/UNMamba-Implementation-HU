from scipy.optimize import linear_sum_assignment
import numpy as np
from utils import to_np

def _as_endmember_matrix(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    if x.ndim != 2:
        raise ValueError(f"Expected a 2D endmember matrix, got shape {x.shape}")

    # Normalize to (K, C): one row per endmember, one column per spectral band.
    if x.shape[0] > x.shape[1]:
        x = x.T

    return x

# RMSE: Root Mean Square Error
# Reconstruction error, pixel-wise
# Lower is better
def rmse(hsi, x_hat):
    hsi = to_np(hsi).squeeze()
    x_hat = to_np(x_hat).squeeze()
    diff = hsi - x_hat
    return float(np.sqrt(np.mean(diff ** 2)))

# SAD: Spectral Angle Distance
# Endmember quality
# Lower is better
def sad(e_true, e_pred):
    e_true = _as_endmember_matrix(e_true)
    e_pred = _as_endmember_matrix(e_pred)

    if e_true.shape != e_pred.shape:
        raise ValueError(f"SAD shape mismatch: e_true={e_true.shape}, e_pred={e_pred.shape}")

    K, C = e_true.shape
    cost = np.zeros((K, K))

    for i in range(K):
        for j in range(K):
            num = np.dot(e_true[i], e_pred[j])
            denom = (np.linalg.norm(e_true[i]) * np.linalg.norm(e_pred[j]) + 1e-10)
            cost[i, j] = np.arccos(np.clip(num / denom, -1, 1))


    row_ind, col_ind = linear_sum_assignment(cost)
    sads = cost[row_ind, col_ind]
    return sads, float(np.mean(sads))

# SID: Spectral Information Divergence
# Lower is better
def sid(e_true, e_pred):
    e_true = _as_endmember_matrix(e_true)
    e_pred = _as_endmember_matrix(e_pred)

    if e_true.shape != e_pred.shape:
        raise ValueError(f"SID shape mismatch: e_true={e_true.shape}, e_pred={e_pred.shape}")

    eps = 1e-10
    K, C = e_true.shape

    # SID requires valid probability-like spectra: nonnegative and normalized.
    p = np.clip(e_true, a_min=0.0, a_max=None)
    q = np.clip(e_pred, a_min=0.0, a_max=None)
    p = p / (p.sum(axis=1, keepdims=True) + eps)
    q = q / (q.sum(axis=1, keepdims=True) + eps)
    p = np.clip(p, a_min=eps, a_max=None)
    q = np.clip(q, a_min=eps, a_max=None)

    cost = np.zeros((K, K))
    for i in range(K):
        for j in range(K):
            kl_pq = np.sum(p[:, i] * np.log(p[:, i] / q[:, j]))
            kl_qp = np.sum(q[:, j] * np.log(q[:, j] / p[:, i]))
            cost[i, j] = np.nan_to_num(kl_pq + kl_qp, nan=1e12, posinf=1e12, neginf=1e12)

    row_ind, col_ind = linear_sum_assignment(cost)
    sids = cost[row_ind, col_ind]
    return sids, float(np.mean(sids))

# RMSE on Abundance maps
def abundance_rmse(a_true, a_pred):
    if a_true.shape != a_pred.shape:
        if a_true.shape == a_pred.T.shape:
            a_pred = a_pred.T
        else:
            raise ValueError(f"Abundance RMSE shape mismatch: a_true={a_true.shape}, a_pred={a_pred.shape}")

    K = a_true.shape[1]
    cost = np.zeros((K, K))

    for i in range(K):
        for j in range(K):
            cost[i, j] = np.sqrt(np.mean((a_true[:, i] - a_pred[:, j]) ** 2))

    row_ind, col_ind = linear_sum_assignment(cost)
    per_em = cost[row_ind, col_ind]
    return per_em, float(np.mean(per_em))