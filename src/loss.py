"""
Loss function for UNMamba (Chen et al., IEEE GRSL 2025).

Final loss (Eq. 13):
    L_final = L_rec + alpha * L_em + beta * L_sparse

where (Eq. 12):
    L_rec = L_mse + L_SAD

    L_mse = (1/HW) * sum_{h,w} || X(h,w) - X_hat(h,w) ||^2_2

    L_SAD = (1/HW) * sum_{h,w} arccos(
                X(h,w)^T X_hat(h,w) /
                (||X(h,w)||_2 * ||X_hat(h,w)||_2)
            )

and (Eq. 11):
    L_em = (1/R) * sum_i || E_hat^(i) - X_bar ||^2_2
    where X_bar is the mean spectrum of the input HSI.

    L_sparse = (1/HW) * sum_{h,w} || M(h,w) ||_1
             = mean(M)   [since M >= 0 already]

Paper hyperparameters: alpha=1, beta=0.0001

NOTE: Without L_em, the EndmemberModule gets NO gradient to pull its
randn sequences toward real spectra — this is why endmembers look like
noise when L_em is missing from the loss.
"""

import torch
import torch.nn.functional as F


def mse_loss(x: torch.Tensor, x_hat: torch.Tensor) -> torch.Tensor:
    """Pixel-wise MSE averaged over all pixels and bands."""
    return F.mse_loss(x_hat, x)


def sad_loss(x: torch.Tensor, x_hat: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Spectral Angle Distance loss averaged over all pixels.

    x, x_hat: (B, L, H, W)
    """
    B, L, H, W = x.shape
    x_flat     = x.reshape(B, L, -1).permute(0, 2, 1)      # (B, N, L)
    x_hat_flat = x_hat.reshape(B, L, -1).permute(0, 2, 1)  # (B, N, L)

    dot   = (x_flat * x_hat_flat).sum(dim=-1)               # (B, N)
    norm_x    = x_flat.norm(dim=-1).clamp(min=eps)
    norm_xhat = x_hat_flat.norm(dim=-1).clamp(min=eps)

    cos_sim = (dot / (norm_x * norm_xhat)).clamp(-1 + eps, 1 - eps)
    sad     = torch.acos(cos_sim)                           # (B, N)
    return sad.mean()


def endmember_loss(E: torch.Tensor, x_bar: torch.Tensor) -> torch.Tensor:
    """
    L_em = (1/R) * sum_i || E^(i) - x_bar ||^2_2   (Eq. 11)

    E:     (R, L)  endmember spectra
    x_bar: (L,)    mean spectrum of the input HSI

    This is THE key loss that pulls the random sequences in EndmemberModule
    toward real spectral distributions. Without it, endmembers are pure noise.
    """
    R = E.shape[0]
    x_bar = x_bar.unsqueeze(0).expand(R, -1)   # (R, L)
    return F.mse_loss(E, x_bar)                # averages over R and L


def sparsity_loss(M: torch.Tensor) -> torch.Tensor:
    """
    L_sparse = mean(M)
    Since M >= 0 (softmax output), this equals the mean L1 norm per pixel,
    encouraging sparse abundance maps.
    """
    return M.mean()


def unmixing_loss(
    x:      torch.Tensor,
    x_hat:  torch.Tensor,
    E:      torch.Tensor,
    M:      torch.Tensor,
    alpha:  float = 1.0,
    beta:   float = 1e-4,
) -> torch.Tensor:
    """
    Full UNMamba loss (Eq. 13):
        L = L_mse + L_SAD + alpha * L_em + beta * L_sparse

    Args:
        x:     (B, L, H, W)  input HSI
        x_hat: (B, L, H, W)  reconstructed HSI
        E:     (R, L)        endmember spectra from EndmemberModule
        M:     (B, R, H, W)  abundance maps
        alpha: weight for endmember loss (paper: 1.0)
        beta:  weight for sparsity loss  (paper: 0.0001)
    """
    # Mean spectrum of the input: average over batch and all spatial pixels
    # shape: (L,)
    x_bar = x.mean(dim=(0, 2, 3))

    l_mse    = mse_loss(x, x_hat)
    l_sad    = sad_loss(x, x_hat)
    l_em     = endmember_loss(E, x_bar)
    l_sparse = sparsity_loss(M)

    return l_mse + l_sad + alpha * l_em + beta * l_sparse