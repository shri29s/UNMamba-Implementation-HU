from torch.utils.data import Dataset
import scipy.io as sio
import torch
import numpy as np

class JRDataset(Dataset):
    def __init__(self, hsi_path, gt_path = None, normalize = True):
        data = sio.loadmat(hsi_path)
        
        self.rows = int(data["nRow"].item())
        self.cols = int(data["nCol"].item())
        self.bands = int(data["nBand"].item())
        maxVal = float(data["maxValue"].item())
        
        Y = data["R"].astype(np.float32)
        hsi = Y.reshape(-1, self.rows, self.cols)

        if normalize:
            hsi /= maxVal

        self.hsi = torch.from_numpy(hsi)
        self.gt_abundance = None
        self.gt_endmembers = None

        if gt_path is not None:
            gt = sio.loadmat(gt_path)
            A = gt["A"].astype(np.float32)
            A = A.reshape(-1, self.rows, self.cols) # Shape = (198, 100, 100) (K, H, W)

            M = gt["M"].astype(np.float32)
            M = M.T # Shape = (4, 198) (K, C)

            self.gt_abundance = torch.from_numpy(A)
            self.gt_endmembers = torch.from_numpy(M)

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return {"hsi": self.hsi, "abundance": self.gt_abundance, "endmembers": self.gt_endmembers}
 