import torch
import torch.nn as nn
from mambapy.mamba import Mamba, MambaConfig

class SpatialBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        config = MambaConfig(d_model=channels, n_layers=1)
        self.mamba = Mamba(config)
        self.ln = nn.LayerNorm(channels)
        self.fc = nn.Linear(channels, channels)

    def forward(self, x):
        B, C, H, W = x.shape
        seq = x.flatten(2).permute(0, 2, 1)
        out = self.mamba(seq)

        out = self.fc(out)
        out = self.ln(out)
        out = torch.nn.functional.silu(out)
        out = out.permute(0, 2, 1).reshape(B, C, H, W)
        return out + x
    
class SpectralBlock(nn.Module):
    def __init__(self, channels, H, W):
        super().__init__()
        config = MambaConfig(d_model=channels, n_layers=1)
        self.mamba = Mamba(config)
        self.ln = nn.LayerNorm(channels)
        self.fc = nn.Linear(channels, channels)
        
        self.down = nn.AdaptiveAvgPool2d(output_size=(H // 4, W // 4))
        self.up = nn.Upsample(size=(H, W), mode="bilinear")

    def forward(self, x):
        B, C, H, W = x.shape
        ds = self.down(x)

        seq = ds.flatten(2).permute(0, 2, 1)
        out = self.mamba(seq)
        out = self.fc(out)
        out = self.ln(out)
        out = nn.functional.silu(out)

        out = out.permute(0, 2, 1).reshape(B, C, *ds.shape[2:])
        out = self.up(out)
        return x + out
    
class UNMambaBlock(nn.Module):
    def __init__(self, channels, H, W):
        super().__init__()
        self.spatial = SpatialBlock(channels)
        self.spectral = SpectralBlock(channels, H, W)
        self.w_spa = nn.Parameter(torch.tensor(0.5))
        self.w_spe = nn.Parameter(torch.tensor(0.5))

    def forward(self, x):
        out_spa = self.spatial(x)
        out_spe = self.spectral(out_spa)
        return x + self.w_spa * out_spa + self.w_spe * out_spe
    
class EndmemberModule(nn.Module):
    def __init__(self, num_endmembers, num_bands, k=4):
        super().__init__()
        self.R, self.K = num_endmembers, k

        self.seqs = nn.Parameter(torch.randn(self.R * self.K, num_bands))
        self.weights = nn.Parameter(torch.ones(self.R, self.K) / self.K)

    def forward(self):
        w = torch.softmax(self.weights, dim=1)
        seqs = self.seqs.reshape(self.R, self.K, -1)
        endmembers = (w.unsqueeze(-1) * seqs).sum(1)
        return endmembers
    
class UNMamba(nn.Module):
    def __init__(self, num_bands, num_endmembers, H, W, num_blocks=3, channels=64):
        super().__init__()

        self.embed = nn.Sequential(
            nn.Conv2d(num_bands, channels, 1),
            nn.GroupNorm(8, channels),
            nn.SiLU()
        )

        self.blocks = nn.ModuleList([
            UNMambaBlock(channels, H, W) for _ in range(num_blocks)
        ])

        self.abundance_head = nn.Sequential(
            nn.Conv2d(channels, num_endmembers, 1),
            nn.Softmax(dim=1)
        )

        self.endmember_module = EndmemberModule(num_endmembers, num_bands)

    def forward(self, x):
        feat = self.embed(x)
        for block in self.blocks:
            feat = block(feat)
        M = self.abundance_head(feat)

        E = self.endmember_module()

        B, R, H, W = M.shape
        M_flat = M.reshape(B, R, H * W)
        x_hat = torch.einsum("rl,brp->blp", E, M_flat)
        x_hat = x_hat.reshape(B, -1, H, W)

        return x_hat, M, E