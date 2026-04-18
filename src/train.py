import numpy as np
import torch
from torch.utils.data import DataLoader
from loader import JRDataset
from unmamba import UNMamba
from loss import unmixing_loss
import os

from utils import to_np
from metrics import rmse, sad, sid, abundance_rmse
   
dataset = JRDataset(
    hsi_path = "data/jasperRidge2_R198.mat",
    gt_path = "data/jasperRidge2_end4.mat",
    normalize = True
)

loader = DataLoader(dataset, batch_size = 1, shuffle = False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_bands = dataset.hsi.shape[0]
H = dataset.hsi.shape[1]
W = dataset.hsi.shape[2]
num_endmembers = dataset.gt_endmembers.shape[0] if dataset.gt_endmembers is not None else 4

# check
for batch in loader:
    print("HSI: ", batch["hsi"].shape)
    print("Abundance: ", batch["abundance"].shape if batch["abundance"] is not None else "NA")
    print("Endmembers: ", batch["endmembers"].shape if batch["endmembers"] is not None else "NA")

history_data = {
    "epochs": [],
    "losses": [],
    "M_history": [],
    "E_history": []
}

model = UNMamba(num_bands=num_bands, num_endmembers=num_endmembers, H=H, W=W).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)

batch = next(iter(loader))
hsi = batch["hsi"].to(device)

scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=45, gamma=0.9)
model.train()
for epoch in range(800):
    optimizer.zero_grad()
    x_hat, M, E = model(hsi)
    loss = unmixing_loss(hsi, x_hat, E, M)
    loss.backward()

    for p in model.query_embed.weight:
        p.data.clamp

    optimizer.step()
    scheduler.step()

    if (epoch + 1) % 15 == 0:
        print(f"Epoch {epoch + 1: 3d} | Loss {loss.item(): .5f}")

        history_data["epochs"].append(epoch + 1)
        history_data["losses"].append(loss.item())
        history_data["M_history"].append(M.detach().cpu().numpy())
        history_data["E_history"].append(E.detach().cpu().numpy())

# Evaluation metrics
model.eval()
torch.save(model.state_dict(), "model_JR.pth")
with torch.no_grad():
    x_hat, M, E = model(hsi)

hsi_np = to_np(hsi)
x_hat_np = to_np(x_hat)
M_np = to_np(M)
E_np = to_np(E)

gt = to_np(dataset.gt_endmembers).T  # (C, K)
a_true = dataset.gt_abundance
a_true = to_np(a_true).reshape(num_endmembers, -1).T  # (N, K)

rec_rmse = rmse(hsi, x_hat)
print(f"Reconstruction RMSE: {rec_rmse:.6f}")

e_pred = E_np.squeeze().T  # (C, K)
sad_per_em, mean_sad = sad(gt, e_pred)
print(f"SAD (per endmember): {np.round(np.degrees(sad_per_em), 4)} deg")
print(f"Mean SAD: {np.round(np.degrees(mean_sad), 4)} deg")

sid_per_em, mean_sid = sid(gt, e_pred)
print(f"SID (per endmember): {np.round(sid_per_em, 6)}")
print(f"Mean SID: {np.round(mean_sid, 6)}")

K, H, W = M_np.squeeze().shape # (K, H, W)
a_pred_flat = M_np.squeeze().reshape(K, -1).T
a_true_flat = a_true

ab_rmse_per_em, mean_ab_rmse = abundance_rmse(a_true=a_true_flat, a_pred=a_pred_flat)
print(f"Abundance RMSE (per endmember): {np.round(ab_rmse_per_em, 6)}")
print(f"Mean Abundance RMSE: {np.round(mean_ab_rmse, 6)}")

metrics = {
    "rec_rmse": rec_rmse,
    "sad_per_em": sad_per_em,
    "mean_sad": mean_sad, 
    "sid_per_em": sid_per_em,
    "mean_sid": mean_sid,
    "ab_rmse_per_em": ab_rmse_per_em,
    "mean_ab_rmse": mean_ab_rmse 
}

unmixing_results = {k: np.array(v) for k, v in history_data.items()}
unmixing_results["metrics"] = metrics
np.savez_compressed("unmixing_results.npz", **unmixing_results)