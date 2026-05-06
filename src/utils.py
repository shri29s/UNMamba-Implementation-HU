import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
import numpy as np
import math
import os

def to_np(t) -> np.ndarray:
    return t.detach().cpu().numpy().astype(np.float64)

def get_permutation(e_true: np.ndarray, e_pred: np.ndarray) -> np.ndarray:
    # Find the optimal permutation to align predicted endmembers to ground truth.
    K = e_true.shape[0]
    cost = np.zeros((K, K))

    for i in range(K):
        for j in range(K):
            num = np.dot(e_true[i], e_pred[j])
            denom = np.linalg.norm(e_true[i]) * np.linalg.norm(e_pred[j]) + 1e-10
            cost[i, j] = np.arccos(np.clip(num / denom, -1, 1))

    _, col_ind = linear_sum_assignment(cost)
    return col_ind

def apply_permutation(col_ind: np.ndarray, endmems: np.ndarray, abundances: np.ndarray):
    # Reorder predicted endmembers and abundances to match ground truth order.
    return endmems[col_ind], abundances[col_ind]

def _as_endmember_matrix(endmembers: np.ndarray) -> np.ndarray:
    endmembers = np.asarray(endmembers)
    if endmembers.ndim != 2:
        raise ValueError(f"Expected a 2D endmember matrix, got shape {endmembers.shape}")

    # Accept either (K, C) or (C, K) and normalize to (K, C).
    if endmembers.shape[0] > endmembers.shape[1]:
        endmembers = endmembers.T

    return endmembers

def subplot_grid(n):
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    return rows, cols

def plot_results(results, dataset):
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    directory = os.path.join(results_dir, dataset.name)
    os.makedirs(directory, exist_ok=True)
    
    labels = dataset.labels if getattr(dataset, "labels", None) is not None else None
    has_gt_abundance = getattr(dataset, "gt_abundance", None) is not None
    has_gt_endmembers = getattr(dataset, "gt_endmembers", None) is not None

    abundance_map = results.get("final_M")
    if abundance_map is None or np.asarray(abundance_map).size == 0:
        if len(results.get("M_history", [])) == 0:
            raise ValueError("No abundance map available for plotting.")
        abundance_map = results["M_history"][-1]
    abundance_map = np.asarray(abundance_map).squeeze()

    if has_gt_endmembers:
        gt_endmembers = _as_endmember_matrix(dataset.gt_endmembers)
        endmems = results.get("final_E")
        if endmems is None or np.asarray(endmems).size == 0:
            if len(results.get("E_history", [])) == 0:
                raise ValueError("No endmember matrix available for plotting.")
            endmems = results["E_history"][-1]
        endmems = _as_endmember_matrix(endmems)
        col_ind = get_permutation(gt_endmembers, endmems)
        abundance_map = abundance_map[col_ind]

    n_materials = len(abundance_map)
    abundance_labels = labels if labels is not None else [f"Component {i + 1}" for i in range(n_materials)]

    # Plot abundance
    rows, cols = subplot_grid(n_materials)
    plt.figure(figsize=(10, 10), dpi=150)
    for i in range(n_materials):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(abundance_map[i], cmap="jet")
        plt.title(abundance_labels[i])
    plt.tight_layout()
    plt.savefig(os.path.join(directory, "abundance.png"))
    plt.close()

    if has_gt_endmembers:
        n_endmems = len(endmems)
        endmember_labels = labels if labels is not None else [f"Endmember {i + 1}" for i in range(n_endmems)]

        endmems, _ = apply_permutation(col_ind, endmems, abundance_map)

        # Plot endmembers
        rows, cols = subplot_grid(n_endmems)
        plt.figure(figsize=(18, 9), dpi=150)
        for i in range(n_endmems):
            plt.subplot(rows, cols, i + 1)
            plt.plot(endmems[i], color="blue", label="UNMamba")
            plt.plot(gt_endmembers[i], color="orange", linestyle="dashed", label="GT")
            plt.title(endmember_labels[i])
            plt.xlabel("Bands")
            plt.ylabel("Reflectance")
            plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(directory, "endmembers.png"))
        plt.close()

    # Plot loss graph
    plt.figure(figsize=(8, 5), dpi=150)
    plt.plot(results["epochs"], results["losses"])
    plt.title("Loss vs Epoch")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.tight_layout()
    plt.savefig(os.path.join(directory, "loss.png"))
    plt.close()

    return directory