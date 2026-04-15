import numpy as np
import torch
from torch.utils.data import DataLoader
from .loader import JRDataset
from .unmamba import UNMamba
from .loss import unmixing_loss
   
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

history_npz = {k: np.array(v) for k, v in history_data.items()}
np.savez_compressed("unmixing_results.npz", **history_npz)