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

class JasperDataset(Dataset):
    def __init__(self, hsi_path, normalize = True):
        data = sio.loadmat(hsi_path)
        
        self.name = "jasper"
        self.labels = np.array(["Tree", "Water", "Soil", "Road"])
        self.rows = int(data["H"].item()) if "H" in data else 100
        self.cols = int(data["W"].item()) if "W" in data else 100
        self.bands = int(data["C"].item()) if "C" in data else 198
        self.num_endmembers = int(data["P"].item()) if "P" in data else 4
        maxVal = float(data["Y"].max())
        
        hsi = data["Y"].astype(np.float32)
        if hsi.shape[0] != self.bands or hsi.shape[1] != self.rows or hsi.shape[2] != self.cols:
            raise ValueError(f"Expected HSI shape ({self.bands}, {self.rows}, {self.cols}), but got {hsi.shape}")

        if normalize:
            hsi /= maxVal

        self.hsi = torch.from_numpy(hsi)
        self.gt_abundance = torch.from_numpy(data["A"].astype(np.float32)) if "A" in data else None
        self.gt_endmembers = torch.from_numpy(data["M"].astype(np.float32)) if "M" in data else None

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        sample = {"hsi": self.hsi}
        if self.gt_abundance is not None:
            sample["abundance"] = self.gt_abundance
        if self.gt_endmembers is not None:
            sample["endmembers"] = self.gt_endmembers
        return sample
 
class CupriteDataset(Dataset):
    def __init__(self, hsi_path, normalize = True):
        super().__init__()
        data = sio.loadmat(hsi_path)
        self.name = "Cuprite"
        self.labels = np.array(["Alunite", "Kaolinite", "Montmorillonite", "Muscovite"])
        self.rows = int(data["H"].item()) if "H" in data else 512
        self.cols = int(data["W"].item()) if "W" in data else 614
        self.bands = int(data["C"].item()) if "C" in data else 188
        self.num_endmembers = int(data["P"].item()) if "P" in data else 4

        maxVal = float(data["X"].max())
        hsi = data["X"].astype(np.float32)
        if hsi.shape[0] != self.bands or hsi.shape[1] != self.rows or hsi.shape[2] != self.cols:
            raise ValueError(f"Expected HSI shape ({self.bands}, {self.rows}, {self.cols}), but got {hsi.shape}")

        if normalize:
            hsi /= maxVal

        self.hsi = torch.from_numpy(hsi)
        self.gt_abundance = torch.from_numpy(data["A"].astype(np.float32)) if "A" in data else None
        self.gt_endmembers = torch.from_numpy(data["M"].astype(np.float32)) if "M" in data else None

    def __len__(self):
        return 1
    
    def __getitem__(self, idx):
        sample = {"hsi": self.hsi}
        if self.gt_abundance is not None:
            sample["abundance"] = self.gt_abundance
        if self.gt_endmembers is not None:
            sample["endmembers"] = self.gt_endmembers
        return sample
    
class ChandrayaanDataset(Dataset):
    def __init__(self, hsi_path, normalize = True):
        super().__init__()
        data = sio.loadmat(hsi_path)
        self.name = "Chandrayaan"
        self.labels = None  # No labels available
        self.rows = int(data["H"].item()) if "H" in data else 2995
        self.cols = int(data["W"].item()) if "W" in data else 304
        self.bands = int(data["C"].item()) if "C" in data else 85
        self.num_endmembers = int(data["P"].item()) if "P" in data else 8

        maxVal = float(data["X"].max())
        hsi = data["X"].astype(np.float32)
        if hsi.shape[0] != self.bands or hsi.shape[1] != self.rows or hsi.shape[2] != self.cols:
            raise ValueError(f"Expected HSI shape ({self.bands}, {self.rows}, {self.cols}), but got {hsi.shape}")

        if normalize:
            hsi /= maxVal

        self.hsi = torch.from_numpy(hsi)
        self.gt_abundance = torch.from_numpy(data["A"].astype(np.float32)) if "A" in data else None
        self.gt_endmembers = torch.from_numpy(data["M"].astype(np.float32)) if "M" in data else None

    def __len__(self):
        return 1
    
    def __getitem__(self, idx):
        sample = {"hsi": self.hsi}
        if self.gt_abundance is not None:
            sample["abundance"] = self.gt_abundance
        if self.gt_endmembers is not None:
            sample["endmembers"] = self.gt_endmembers
        return sample

class PatchDatasetWrapper(Dataset):
    def __init__(self, dataset: Dataset, patch_size: int = 64, stride: int = 32):
        self.dataset = dataset
        self.patch_size = patch_size
        self.stride = stride
        self.name = dataset.name
        self.labels = dataset.labels
        self.bands = dataset.bands
        self.num_endmembers = dataset.num_endmembers
        self.gt_endmembers = getattr(dataset, "gt_endmembers", None)
        self.gt_abundance = getattr(dataset, "gt_abundance", None)
        
        self.full_hsi = dataset.hsi # (C, H, W)
        _, self.H, self.W = self.full_hsi.shape
        
        self.patches = []
        self._extract_patches()

    def _extract_patches(self):
        for y in range(0, self.H - self.patch_size + 1, self.stride):
            for x in range(0, self.W - self.patch_size + 1, self.stride):
                self.patches.append((y, x))
                
        # Handle edges if the image size is not perfectly divisible
        if (self.H - self.patch_size) % self.stride != 0:
            y = self.H - self.patch_size
            for x in range(0, self.W - self.patch_size + 1, self.stride):
                self.patches.append((y, x))
                
        if (self.W - self.patch_size) % self.stride != 0:
            x = self.W - self.patch_size
            for y in range(0, self.H - self.patch_size + 1, self.stride):
                self.patches.append((y, x))
                
        # Handle bottom right corner
        if (self.H - self.patch_size) % self.stride != 0 and (self.W - self.patch_size) % self.stride != 0:
            self.patches.append((self.H - self.patch_size, self.W - self.patch_size))

        # Remove duplicates
        self.patches = list(set(self.patches))

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        y, x = self.patches[idx]
        patch_hsi = self.full_hsi[:, y:y+self.patch_size, x:x+self.patch_size]
        
        sample = {
            "hsi": patch_hsi,
            "y": y,
            "x": x
        }
        
        if self.gt_abundance is not None:
            # GT Abundance is (K, H, W)
            sample["abundance"] = self.gt_abundance[:, y:y+self.patch_size, x:x+self.patch_size]
            
        if self.gt_endmembers is not None:
            sample["endmembers"] = self.gt_endmembers
            
        return sample