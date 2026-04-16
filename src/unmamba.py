"""
UNMamba: Cascaded Spatial-Spectral Mamba for Blind Hyperspectral Unmixing
Faithful implementation of Chen et al., IEEE GRSL 2025.

Key corrections vs the buggy version:
  1. SpatialBlock and SpectralBlock each return F^(i-1) + their correction,
     matching the residual definitions in Eq. 7 and Eq. 8.
  2. UNMambaBlock fusion follows Eq. 9 exactly:
       F^(i) = F^(i-1) + w_spa*F^(i)_spa + w_spe*F^(i)_spe
     The old code fed out_spa into the spectral block AND also added x
     a second time, causing double-counting of the spatial residual.
  3. SpectralBlock downsamples to H/16 x W/16 (not H/4 x W/4) matching
     the shape shown in Fig. 2 of the paper.
  4. EndmemberModule is randn-initialized as stated in the paper. The
     endmember loss (loss.py) is what drives convergence — without it
     endmembers stay as noise.
  5. einsum index fixed: 'rl,brn->bln' (L=bands, not 'bl').
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from mambapy.mamba import Mamba, MambaConfig


# ---------------------------------------------------------------------------
# Spatial Block  (Eq. 7)
# ---------------------------------------------------------------------------
class SpatialBlock(nn.Module):
    """
    T_spa1 = Mamba(Flatten(F^(i-1)))
    T_spa2 = Reshape(SiLU(LN(FC(T_spa1))))
    F^(i)_spa = F^(i-1) + T_spa2
    """
    def __init__(self, channels: int):
        super().__init__()
        cfg = MambaConfig(d_model=channels, n_layers=1)
        self.mamba = Mamba(cfg)
        self.fc    = nn.Linear(channels, channels)
        self.ln    = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        seq = x.flatten(2).permute(0, 2, 1)        # (B, H*W, C)
        t1  = self.mamba(seq)                       # (B, H*W, C)
        t2  = F.silu(self.ln(self.fc(t1)))          # (B, H*W, C)
        t2  = t2.permute(0, 2, 1).reshape(B, C, H, W)
        return x + t2                               # F^(i)_spa


# ---------------------------------------------------------------------------
# Spectral Block  (Eq. 8)
# ---------------------------------------------------------------------------
class SpectralBlock(nn.Module):
    """
    T_spe1 = Mamba(Flatten(Downsample(F^(i)_spa)))
    T_spe2 = Upsample(SiLU(LN(FC(T_spe1))))
    F^(i)_spe = F^(i)_spa + T_spe2

    Input is F^(i)_spa (output of the spatial block), not F^(i-1).
    Downsample factor of 16 matches Fig. 2 (H/16 x W/16 grid).
    """
    def __init__(self, channels: int, H: int, W: int, ds_factor: int = 16):
        super().__init__()
        ds_h = max(1, H // ds_factor)
        ds_w = max(1, W // ds_factor)
        cfg = MambaConfig(d_model=channels, n_layers=1)
        self.mamba = Mamba(cfg)
        self.fc    = nn.Linear(channels, channels)
        self.ln    = nn.LayerNorm(channels)
        self.down  = nn.AdaptiveAvgPool2d((ds_h, ds_w))
        self.up    = nn.Upsample(size=(H, W), mode="bilinear", align_corners=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        ds  = self.down(x)                                       # (B, C, ds_h, ds_w)
        seq = ds.flatten(2).permute(0, 2, 1)                     # (B, ds_h*ds_w, C)
        t1  = self.mamba(seq)                                    # (B, ds_h*ds_w, C)
        t2  = F.silu(self.ln(self.fc(t1)))                       # (B, ds_h*ds_w, C)
        t2  = t2.permute(0, 2, 1).reshape(B, C, *ds.shape[2:])  # (B, C, ds_h, ds_w)
        t2  = self.up(t2)                                        # (B, C, H, W)
        return x + t2                                            # F^(i)_spe


# ---------------------------------------------------------------------------
# UNMamba Block  (Eq. 9)
# ---------------------------------------------------------------------------
class UNMambaBlock(nn.Module):
    """
    F^(i) = F^(i-1) + w_spa * F^(i)_spa + w_spe * F^(i)_spe

    where:
      F^(i)_spa = SpatialBlock(F^(i-1))         — already contains residual
      F^(i)_spe = SpectralBlock(F^(i)_spa)      — already contains residual
    """
    def __init__(self, channels: int, H: int, W: int):
        super().__init__()
        self.spatial  = SpatialBlock(channels)
        self.spectral = SpectralBlock(channels, H, W)
        self.w_spa    = nn.Parameter(torch.tensor(0.5))
        self.w_spe    = nn.Parameter(torch.tensor(0.5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f_spa = self.spatial(x)          # F^(i)_spa: spatial block on F^(i-1)
        f_spe = self.spectral(f_spa)     # F^(i)_spe: spectral block on F^(i)_spa
        # Eq. 9
        return x + self.w_spa * f_spa + self.w_spe * f_spe


# ---------------------------------------------------------------------------
# Endmember Module  (Eq. 10)
# ---------------------------------------------------------------------------
class EndmemberModule(nn.Module):
    """
    E_hat^(i) = (1/K) * sum_j  w^(j) * Q_E^(j),  j in [i*K, (i+1)*K)

    Q_E: (R*K, L) trainable randn sequences
    w:   (R, K)   learnable weights (softmax-normalized)

    The endmember loss L_em (Eq. 11) pulls these sequences toward X_bar,
    providing the gradient signal that turns noise into real spectra.
    Without L_em, Q_E has no reason to represent meaningful spectra.
    """
    def __init__(self, num_endmembers: int, num_bands: int, K: int = 4):
        super().__init__()
        self.R, self.K = num_endmembers, K
        self.Q_E     = nn.Parameter(torch.randn(num_endmembers * K, num_bands))
        self.weights = nn.Parameter(torch.ones(num_endmembers, K) / K)

    def forward(self) -> torch.Tensor:
        """Returns E: (R, L)"""
        w    = torch.softmax(self.weights, dim=1)       # (R, K)
        seqs = self.Q_E.reshape(self.R, self.K, -1)     # (R, K, L)
        E    = (w.unsqueeze(-1) * seqs).sum(dim=1)      # (R, L)
        return E


# ---------------------------------------------------------------------------
# Full UNMamba model
# ---------------------------------------------------------------------------
class UNMamba(nn.Module):
    def __init__(
        self,
        num_bands:      int,
        num_endmembers: int,
        H:              int,
        W:              int,
        num_blocks:     int = 3,
        channels:       int = 64,
    ):
        super().__init__()

        # Eq. 6: Embedding = SiLU(GN(Conv(X)))
        self.embed = nn.Sequential(
            nn.Conv2d(num_bands, channels, kernel_size=1, bias=False),
            nn.GroupNorm(8, channels),
            nn.SiLU(),
        )

        self.blocks = nn.ModuleList([
            UNMambaBlock(channels, H, W) for _ in range(num_blocks)
        ])

        # Abundance head: Conv -> Softmax (enforces ANC + ASC)
        self.abundance_head = nn.Sequential(
            nn.Conv2d(channels, num_endmembers, kernel_size=1),
            nn.Softmax(dim=1),
        )

        self.endmember_module = EndmemberModule(num_endmembers, num_bands)

    def forward(self, x: torch.Tensor):
        """
        x:     (B, L, H, W) normalized HSI
        Returns:
          x_hat: (B, L, H, W)
          M:     (B, R, H, W)
          E:     (R, L)
        """
        feat = self.embed(x)
        for block in self.blocks:
            feat = block(feat)
        M = self.abundance_head(feat)       # (B, R, H, W)

        E = self.endmember_module()         # (R, L)

        # LMM: x_hat = E^T M
        B, R, H, W = M.shape
        M_flat = M.reshape(B, R, H * W)                     # (B, R, N)
        x_hat  = torch.einsum("rl,brn->bln", E, M_flat)     # (B, L, N)
        x_hat  = x_hat.reshape(B, -1, H, W)                 # (B, L, H, W)

        return x_hat, M, E