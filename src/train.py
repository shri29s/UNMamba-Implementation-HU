import numpy as np
import torch
from torch.utils.data import DataLoader
from .loader import JRDataset
from .unmamba import UNMamba
from .loss import unmixing_loss

from .utils import to_np
from .metrics import rmse, sad, sid, abundance_rmse
   
dataset = JRDataset(
    hsi_path = "../data/jasperRidge2_R198.mat",
    gt_path = "../data/jasperRidge2_end4.mat",
    normalize = True
)

loader = DataLoader(dataset, batch_size = 1, shuffle = False)

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

model = UNMamba(num_bands=198, num_endmembers=4, H=100, W=100).cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)

batch = next(iter(loader))
hsi = batch["hsi"].cuda()

scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=45, gamma=0.5)

for epoch in range(800):
    optimizer.zero_grad()
    x_hat, M, E = model(hsi)
    loss = unmixing_loss(hsi, x_hat, E, M)
    loss.backward()
    optimizer.step()
    scheduler.step()

    if (epoch + 1) % 15 == 0:
        print(f"Epoch {epoch: 3d} | Loss {loss.item(): .5f}")

        history_data["epochs"].append(epoch + 1)
        history_data["losses"].append(loss.item())
        history_data["M_history"].append(M.detach().cpu().numpy())
        history_data["E_history"].append(E.detach().cpu().numpy())

# Evaluation metrics
model.eval()
with torch.no_grad():
    x_hat, M, E = model(hsi)

hsi_np = to_np(hsi)
x_hat_np = to_np(x_hat)
M_np = to_np(M)
E_np = to_np(E)

gt = dataset.gt_endmembers.T # Shape (C, K)
a_true = dataset.gt_abundance
a_true = a_true.reshape(dataset.bands, -1) # Shape (K, H*W)

rec_rmse = rmse(hsi, x_hat)
print(f"Reconstruction RMSE: {rec_rmse:.6f}")

e_pred = E_np.squeeze()
sad_per_em, mean_sad = sad(gt, e_pred)
print(f"SAD (per endmember): {np.degrees(sad_per_em).round(4)} deg")
print(f"Mean SAD: {np.degrees(mean_sad).round(4)} deg")

sid_per_em, mean_sid = sid(gt, e_pred)
print(f"SID (per endmember): {sid_per_em.round(6)}")
print(f"Mean SID: {mean_sid.round(6)}")

K, H, W = M_np.squeeze().shape # (K, H, W)
a_pred_flat = M_np.squeeze().reshape(K, -1).T
a_true_flat = a_true.T

ab_rmse_per_em, mean_ab_rmse = abundance_rmse(a_true=a_true_flat, a_pred=a_pred_flat)
print(f"Abundance RMSE (per endmember): {ab_rmse_per_em.round(6)}")
print(f"Mean Abundance RMSE: {mean_ab_rmse.round(6)}")

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