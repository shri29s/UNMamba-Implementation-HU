import numpy as np

def to_np(t) -> np.ndarray:
    return t.detach().cpu().numpy().astype(np.float64)