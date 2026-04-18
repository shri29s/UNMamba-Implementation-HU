from torch.utils.data import Dataset
import scipy.io as sio
import torch
import numpy as np

"""
Note:
1. HSI tensor shape: (C, H, W)
2. Ground-truth abundance (if available): (K, H, W)
3. Ground-truth endmembers (if available): (K, C)
"""

class JRDataset(Dataset):
    def __init__(self, hsi_path, gt_path = None, normalize = True):
        data = sio.loadmat(hsi_path)
        
        self.name = "JR"
        self.labels = np.array(["Tree", "Water", "Soil", "Road"])
        self.rows = int(data["nRow"].item())
        self.cols = int(data["nCol"].item())
        self.bands = int(data["nBand"].item())
        maxVal = float(data["maxValue"].item())
        
        n_pixels = self.rows * self.cols

        Y = data["R"].astype(np.float32)
        if Y.shape == (self.bands, n_pixels):
            Y_bc = Y
        elif Y.shape == (n_pixels, self.bands):
            Y_bc = Y.T
        else:
            raise ValueError(
                f"Unexpected R shape {Y.shape}. Expected ({self.bands}, {n_pixels}) or ({n_pixels}, {self.bands})."
            )

        hsi = Y_bc.reshape(self.bands, self.rows, self.cols)

        if normalize:
            hsi /= maxVal

        self.hsi = torch.from_numpy(hsi)
        self.gt_abundance = None
        self.gt_endmembers = None

        if gt_path is not None:
            gt = sio.loadmat(gt_path)
            A = gt["A"].astype(np.float32)
            if A.shape[1] == n_pixels:
                A_kn = A
            elif A.shape[0] == n_pixels:
                A_kn = A.T
            else:
                raise ValueError(
                    f"Unexpected A shape {A.shape}. Expected (K, {n_pixels}) or ({n_pixels}, K)."
                )

            num_endmembers = A_kn.shape[0]
            A = A_kn.reshape(num_endmembers, self.rows, self.cols)  # (K, H, W)

            M = gt["M"].astype(np.float32)
            if M.shape[0] == self.bands:
                M = M.T
            elif M.shape[1] == self.bands:
                M = M
            else:
                raise ValueError(
                    f"Unexpected M shape {M.shape}. Expected ({self.bands}, K) or (K, {self.bands})."
                )

            if M.shape[0] != num_endmembers:
                raise ValueError(
                    f"Endmember count mismatch between A ({num_endmembers}) and M ({M.shape[0]})."
                )

            self.gt_abundance = torch.from_numpy(A)
            self.gt_endmembers = torch.from_numpy(M)

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return {"hsi": self.hsi, "abundance": self.gt_abundance, "endmembers": self.gt_endmembers}
 