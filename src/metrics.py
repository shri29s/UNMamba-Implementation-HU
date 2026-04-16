from scipy.optimize import linear_sum_assignment
import numpy as np
from utils import to_np

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
    C, K = e_true.shape
    cost = np.zeros((K, K))

    for i in range(K):
        for j in range(K):
            num = np.dot(e_true[:, i], e_pred[:, j])
            denom = (np.linalg.norm(e_true[:, i]) * np.linalg.norm(e_pred[:, j]) + 1e-10)
            cost[i, j] = np.arccos(np.clip(num / denom, -1, 1))


    row_ind, col_ind = linear_sum_assignment(cost)
    sads = cost[row_ind, col_ind]
    return sads, float(np.mean(sads))

# SID: Spectral Information Divergence
# Lower is better
def sid(e_true, e_pred):
    eps = 1e-10
    K = e_true.shape[1]

    p = e_true / (e_true.sum(axis=0, keepdims=True) + eps)
    q = e_pred / (e_pred.sum(axis=0, keepdims=True) + eps)

    cost = np.zeros((K, K))
    for i in range(K):
        for j in range(K):
            kl_pq = np.sum(p[:, i] * np.log((p[:, i] + eps) / q[:, j] + eps))
            kl_qp = np.sum(q[:, j] * np.log((q[:, j] + eps) / (p[:, i] + eps)))
            cost[i, j] = kl_pq + kl_qp

    row_ind, col_ind = linear_sum_assignment(cost)
    sids = cost[row_ind, col_ind]
    return sids, float(np.mean(sids))

# RMSE on Abundance maps
def abundance_rmse(a_true, a_pred):
    K = a_true.shape[1]
    cost = np.zeros((K, K))

    for i in range(K):
        for j in range(K):
            cost[i, j] = np.sqrt(np.mean((a_true[:, i] - a_pred[:, j]) ** 2))

    row_ind, col_ind = linear_sum_assignment(cost)
    per_em = cost[row_ind, col_ind]
    return per_em, float(np.mean(per_em))