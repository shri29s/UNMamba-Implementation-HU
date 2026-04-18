import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from mambapy.mamba import Mamba, MambaConfig


# ---------------------------------------------------------------------------
# SumToOne (matches official)
# ---------------------------------------------------------------------------
class SumToOne(nn.Module):
    def __init__(self, scale=3.5):
        super().__init__()
        self.scale = scale

    def forward(self, x):
        return torch.softmax(self.scale * x, dim=1)


# ---------------------------------------------------------------------------
# Spatial Block — collapses B*H*W into one sequence (matches official)
# ---------------------------------------------------------------------------
class SpatialBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        cfg = MambaConfig(d_model=channels, n_layers=1)
        self.mamba = Mamba(cfg)
        self.proj = nn.Sequential(
            nn.Linear(channels, channels),
            nn.LayerNorm(channels),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        # Collapse batch+spatial into one long sequence (official behavior)
        seq = x.permute(0, 2, 3, 1).contiguous().view(1, B * H * W, C)
        out = self.mamba(seq)
        out = self.proj(out)
        out = out.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        return out + x


# ---------------------------------------------------------------------------
# Spectral Block — d_model is spatial size, sequence is channels (matches official)
# ---------------------------------------------------------------------------
class SpectralBlock(nn.Module):
    def __init__(self, channels: int, H: int, W: int, ds_factor: int = 4):
        super().__init__()
        self.ds = ds_factor
        ds_h = max(1, H // ds_factor)
        ds_w = max(1, W // ds_factor)
        d_model = ds_h * ds_w  # spatial positions are the model dim

        cfg = MambaConfig(d_model=d_model, n_layers=1)
        self.mamba = Mamba(cfg)
        self.proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.SiLU(),
        )
        self.up = nn.Upsample(size=(H, W), mode="bilinear", align_corners=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        # Downsample via unfolding + mean (matches official SpeMamba)
        x_unfolded = x.unfold(2, self.ds, self.ds).unfold(3, self.ds, self.ds)
        x_small = x_unfolded.mean(dim=(-1, -2))          # (B, C, H/ds, W/ds)
        _, _, sh, sw = x_small.shape

        # Sequence is along channels; spatial positions are model dim
        seq = x_small.view(B, C, sh * sw)                # (B, C, sh*sw)
        out = self.mamba(seq)                             # (B, C, sh*sw)
        out = self.proj(out)                              # (B, C, sh*sw)
        out = out.view(B, C, sh, sw)
        out = self.up(out)                                # (B, C, H, W)
        return out + x


# ---------------------------------------------------------------------------
# UNMamba Block
# ---------------------------------------------------------------------------
class UNMambaBlock(nn.Module):
    def __init__(self, channels: int, H: int, W: int, ds_factor: int = 4):
        super().__init__()
        self.spatial  = SpatialBlock(channels)
        self.spectral = SpectralBlock(channels, H, W, ds_factor)
        self.weights  = nn.Parameter(torch.ones(2) / 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f_spa = self.spatial(x)
        f_spe = self.spectral(f_spa)
        return x + self.weights[0] * f_spa + self.weights[1] * f_spe


# ---------------------------------------------------------------------------
# Endmember Module — matches official get_endmember() exactly
# ---------------------------------------------------------------------------
class EndmemberModule(nn.Module):
    def __init__(self, num_endmembers: int, num_bands: int, K: int = 30):
        super().__init__()
        self.num_endm = num_endmembers
        self.K = K
        self.num_bands = num_bands
        # Embedding: (R*K, L) — initialized with uniform distribution
        self.query_embed = nn.Embedding(num_endmembers * K, num_bands)
        # Raw (unscaled) weights, not softmaxed — matches official
        self.weights = nn.Parameter(torch.ones(num_endmembers, K))

    def forward(self) -> torch.Tensor:
        # Split embedding into (R, K, L)
        chunks = torch.chunk(self.query_embed.weight, self.num_endm, dim=0)
        seqs   = torch.stack(chunks)                              # (R, K, L)
        w      = self.weights.unsqueeze(-1).expand_as(seqs)      # (R, K, L)
        E      = (w * seqs).mean(dim=1)                          # (R, L)
        return E


# ---------------------------------------------------------------------------
# Full UNMamba
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
        scale:          float = 3.5,
        ds_factor:      int = 4,
        dropout:        float = 0.05,
        K:              int = 30,
    ):
        super().__init__()
        self.ds = ds_factor
        self.dropout = dropout

        self.embed = nn.Sequential(
            nn.Conv2d(num_bands, channels, kernel_size=1, bias=False),
            nn.GroupNorm(8, channels),
            nn.SiLU(),
        )

        self.blocks = nn.ModuleList([
            UNMambaBlock(channels, H, W, ds_factor) for _ in range(num_blocks)
        ])

        # Deeper abundance head matching official (hidden=128)
        self.abundance_head = nn.Sequential(
            nn.Conv2d(channels, 128, kernel_size=1),
            nn.GroupNorm(4, 128),
            nn.SiLU(),
            nn.Conv2d(128, num_endmembers, kernel_size=1),
            nn.BatchNorm2d(num_endmembers),
            SumToOne(scale=scale),
        )

        self.endmember_module = EndmemberModule(num_endmembers, num_bands, K)
        self._reset_parameters()

    def _reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, x: torch.Tensor):
        B, L, H, W = x.shape

        # Pad so H, W are divisible by ds
        pad_h = math.ceil(H / self.ds) * self.ds - H
        pad_w = math.ceil(W / self.ds) * self.ds - W
        x_pad = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")

        feat = self.embed(x_pad)
        for block in self.blocks:
            feat = block(feat)

        M = self.abundance_head(feat)[:, :, :H, :W]  # (B, R, H, W)

        if self.training:
            M = F.dropout2d(M, p=self.dropout)

        E = self.endmember_module()                   # (R, L)

        x_hat = torch.einsum("brhw,rl->blhw", M, E)  # (B, L, H, W)

        return x_hat + 1e-7, M, E