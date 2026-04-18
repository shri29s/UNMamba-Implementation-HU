import torch
import torch.nn as nn
import torch.nn.functional as F


def mse_loss(x: torch.Tensor, x_hat: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(x_hat, x)


def sad_loss(x: torch.Tensor, x_hat: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """x, x_hat: (B, L, H, W)"""
    B, L, H, W = x.shape
    x_flat     = x.reshape(B, L, -1).permute(0, 2, 1)       # (B, N, L)
    x_hat_flat = x_hat.reshape(B, L, -1).permute(0, 2, 1)   # (B, N, L)

    dot        = (x_flat * x_hat_flat).sum(dim=-1)
    norm_x     = x_flat.norm(dim=-1).clamp(min=eps)
    norm_xhat  = x_hat_flat.norm(dim=-1).clamp(min=eps)

    cos_sim    = (dot / (norm_x * norm_xhat)).clamp(-1 + eps, 1 - eps)
    return torch.acos(cos_sim).mean()


def endmember_loss(E: torch.Tensor, x_bar: torch.Tensor) -> torch.Tensor:
    """
    E:     (R, L)
    x_bar: (L,)
    Matches official: MSE(hsi_mean, endm)
    """
    x_bar_exp = x_bar.unsqueeze(0).expand_as(E)   # (R, L)
    return F.mse_loss(x_bar_exp, E)


def sparsity_loss(M: torch.Tensor) -> torch.Tensor:
    """
    Official uses L0.5 quasi-norm along endmember dim, then averages.
    M: (B, R, H, W)
    """
    # norm over R dim (dim=1), result: (B, H, W)
    return torch.norm(M, p=0.5, dim=1).mean()


def unmixing_loss(
    x:      torch.Tensor,   # (B, L, H, W)
    x_hat:  torch.Tensor,   # (B, L, H, W)
    E:      torch.Tensor,   # (R, L)
    M:      torch.Tensor,   # (B, R, H, W)
    hsi_mean: torch.Tensor,
    alpha:  float = 0.001,  # endmember weight — official default
    beta:   float = 1e-6,   # sparsity weight  — official default
) -> torch.Tensor:
    l_mse    = mse_loss(x, x_hat)
    l_sad    = sad_loss(x, x_hat)
    l_em     = endmember_loss(E, hsi_mean)
    l_sparse = sparsity_loss(M)

    return l_mse + l_sad + alpha * l_em + beta * l_sparse