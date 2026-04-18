import matplotlib.pyplot as plt
import numpy as np
import math
import os

def to_np(t) -> np.ndarray:
    return t.detach().cpu().numpy().astype(np.float64)

def subplot_grid(n):
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    return rows, cols

def plot_results(results, dataset):
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    directory = os.path.join(results_dir, dataset.name)
    os.makedirs(directory, exist_ok=True)
    
    map: np.ndarray = results["M_history"][-1]
    map = map.squeeze()

    n_materials = len(map)
    rows, cols = subplot_grid(n_materials)

    plt.figure(figsize=(10, 10), dpi=150)
    for i in range(n_materials):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(map[i], cmap="jet")
        plt.title(dataset.labels[i])
    
    plt.tight_layout()
    plt.savefig(os.path.join(directory, "abundance.png"))
    plt.close()

    endmems: np.ndarray = results["E_history"][-1]
    n_endmems = len(endmems)
    rows, cols = subplot_grid(n_endmems)

    plt.figure(figsize=(18, 9), dpi=150)
    for i in range(n_endmems):
        plt.subplot(rows, cols, i + 1)
        plt.plot(endmems[i], color="blue", label="UNMamba")
        plt.plot(dataset.gt_endmembers[i], color="orange", linestyle="dashed", label="GT")
        plt.title(dataset.labels[i])
        plt.xlabel("Bands")
        plt.ylabel("Reflectance")
        plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(directory, "endmembers.png"))
    plt.close()

    plt.figure(figsize=(8, 5), dpi=150)
    plt.plot(results["epochs"], results["losses"])
    plt.title("Loss vs Epoch")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.tight_layout()
    plt.savefig(os.path.join(directory, "loss.png"))
    plt.close()

    return directory