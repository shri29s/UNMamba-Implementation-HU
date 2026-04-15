import torch.nn as nn
import torch

def unmixing_loss(x, x_hat, E, M, alpha=1.0, beta=1e-4):
    mse = nn.functional.mse_loss(x_hat, x)
    cos = nn.functional.cosine_similarity(
        x_hat.flatten(2), x.flatten(2), dim=1
    )
    sad = torch.acos(cos.clamp(-1+1e-6, 1-1e-6)).mean()

    x_mean = x.mean(dim=[0, 2, 3])
    lem = ((E - x_mean) ** 2).mean()

    lsparse = M.abs().mean()
    return mse + sad + alpha * lem + beta * lsparse