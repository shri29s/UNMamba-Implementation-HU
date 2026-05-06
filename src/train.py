import numpy as np
import torch
import tqdm
import argparse
from torch.utils.data import DataLoader
from loader import JasperDataset, CupriteDataset, ChandrayaanDataset, PatchDatasetWrapper
from unmamba import UNMamba
from loss import unmixing_loss

from utils import to_np, plot_results
from metrics import rmse, sad, sid, abundance_rmse
   
def main(args):    
    if args.dataset == "jasper":
        dataset = JasperDataset(
            hsi_path = "data/jasper.mat",
            normalize = True
        )
    elif args.dataset == "cuprite":
        dataset = CupriteDataset(
            hsi_path = "data/cuprite.mat",
            normalize = True
        )
    elif args.dataset == "chandrayaan":
        dataset = ChandrayaanDataset(
            hsi_path = "data/chandrayaan.mat",
            normalize = True
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # We need to precompute the mean on the full unpatched image for the endmember loss
    full_hsi = dataset.hsi.to(device)
    hsi_mean = full_hsi.mean(dim=(0, 1, 2))  # (L,) — precompute once outside loop

    use_patching = args.patch_size > 0 and (args.patch_size < dataset.rows or args.patch_size < dataset.cols)
    if use_patching:
        patch_size = min(args.patch_size, dataset.rows, dataset.cols)
        train_dataset = PatchDatasetWrapper(dataset, patch_size=patch_size, stride=args.stride)
        H_model, W_model = patch_size, patch_size
        print(f"Using Patching: patch_size={patch_size}, stride={args.stride}, total patches={len(train_dataset)}")
    else:
        train_dataset = dataset
        H_model, W_model = dataset.rows, dataset.cols
        print("Using full image (no patching).")

    loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    num_bands = dataset.bands
    num_endmembers = dataset.num_endmembers

    history_data = {
        "epochs": [],
        "losses": [],
        "M_history": [],
        "E_history": []
    }

    model = UNMamba(num_bands=num_bands, num_endmembers=num_endmembers, H=H_model, W=W_model).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    has_gt_abundance = getattr(dataset, "gt_abundance", None) is not None
    has_gt_endmembers = getattr(dataset, "gt_endmembers", None) is not None

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=45, gamma=0.9)
    model.train()
    for epoch in tqdm.tqdm(range(1, args.epochs + 1), desc="Training"):
        model.train()
        epoch_loss = 0.0
        
        for batch in loader:
            hsi_batch = batch["hsi"].to(device)
            optimizer.zero_grad()
            
            x_hat, M, E = model(hsi_batch)
            loss = unmixing_loss(hsi_batch, x_hat, E, M, hsi_mean=hsi_mean, alpha=0.001, beta=1e-6)
            loss.backward()
    
            with torch.no_grad():
                for p in model.endmember_module.query_embed.weight:
                    p.data.clamp_(1e-7, 1.0)
    
            optimizer.step()
            epoch_loss += loss.item()

        scheduler.step()
        epoch_loss /= len(loader)

        if epoch % 50 == 0:
            print(f"\nEpoch {epoch:3d} | Loss {epoch_loss:.5f}")
            history_data["epochs"].append(epoch)
            history_data["losses"].append(epoch_loss)
            
            # For history visualization, we can just save E (global)
            history_data["E_history"].append(E.detach().cpu().numpy())
            # For M_history, if patching is used, tracking history of one batch is not very meaningful for the full image.
            # We'll omit M_history to save memory/avoid complexity, or just save the last batch's M.
            if not use_patching:
                history_data["M_history"].append(M.detach().cpu().numpy())

    # Evaluation metrics
    model.eval()
    with torch.no_grad():
        if not use_patching:
            hsi_full = full_hsi.unsqueeze(0)
            x_hat, M, E = model(hsi_full)
            x_hat_full = x_hat.squeeze(0)
            M_full = M.squeeze(0)
        else:
            # Inference stitching
            E = model.endmember_module()
            H, W = dataset.rows, dataset.cols
            
            M_sum = torch.zeros((num_endmembers, H, W), device=device)
            M_count = torch.zeros((1, H, W), device=device)
            
            # Using overlapping patches for inference stitching
            eval_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)
            for batch in eval_loader:
                hsi_batch = batch["hsi"].to(device)
                y_coords = batch["y"]
                x_coords = batch["x"]
                
                _, M_batch, _ = model(hsi_batch)
                
                for i in range(hsi_batch.size(0)):
                    y, x = y_coords[i], x_coords[i]
                    M_sum[:, y:y+patch_size, x:x+patch_size] += M_batch[i]
                    M_count[:, y:y+patch_size, x:x+patch_size] += 1
            
            M_full = M_sum / M_count
            
            # Reconstruct x_hat_full from M_full and E
            x_hat_full = torch.einsum("rhw,rl->lhw", M_full, E) + 1e-7
            hsi_full = full_hsi

    M_np = to_np(M_full)
    E_np = to_np(E)

    rec_rmse = rmse(hsi_full.unsqueeze(0), x_hat_full.unsqueeze(0))
    print(f"Reconstruction RMSE: {rec_rmse:.6f}")

    metrics = {
        "rec_rmse": rec_rmse,
    }

    if has_gt_endmembers:
        gt = to_np(dataset.gt_endmembers).T  # (C, K)
        e_pred = E_np.T  # (C, K)
        sad_per_em, mean_sad = sad(gt, e_pred)
        sid_per_em, mean_sid = sid(gt, e_pred)

        print(f"SAD (per endmember): {np.round(np.degrees(sad_per_em), 4)} deg")
        print(f"Mean SAD: {np.round(np.degrees(mean_sad), 4)} deg")
        print(f"SID (per endmember): {np.round(sid_per_em, 6)}")
        print(f"Mean SID: {np.round(mean_sid, 6)}")

        metrics.update({
            "sad_per_em": sad_per_em,
            "mean_sad": mean_sad,
            "sid_per_em": sid_per_em,
            "mean_sid": mean_sid,
        })

    if has_gt_abundance:
        a_true = dataset.gt_abundance
        a_true = to_np(a_true).reshape(num_endmembers, -1).T  # (N, K)
        K_val, H_val, W_val = M_np.shape # (K, H, W)
        a_pred_flat = M_np.reshape(K_val, -1).T

        ab_rmse_per_em, mean_ab_rmse = abundance_rmse(a_true=a_true, a_pred=a_pred_flat)
        print(f"Abundance RMSE (per endmember): {np.round(ab_rmse_per_em, 6)}")
        print(f"Mean Abundance RMSE: {np.round(mean_ab_rmse, 6)}")

        metrics.update({
            "ab_rmse_per_em": ab_rmse_per_em,
            "mean_ab_rmse": mean_ab_rmse,
        })

    unmixing_results = {k: np.array(v) for k, v in history_data.items()}
    unmixing_results["metrics"] = metrics
    unmixing_results["final_M"] = M_np
    unmixing_results["final_E"] = E_np
    np.savez_compressed(f"unmixing_results_{args.dataset}.npz", **unmixing_results)

    directory = plot_results(results=unmixing_results, dataset=dataset)
    print(f"\nPlots saved in: {directory}")

def get_args():
    parser = argparse.ArgumentParser(description="Train UNMamba for hyperspectral unmixing")
    parser.add_argument("--dataset", type=str, default="jasper", choices=["jasper", "cuprite", "chandrayaan"], help="Dataset to use")
    parser.add_argument("--epochs", type=int, default=800, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=3e-3, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for patching")
    parser.add_argument("--patch_size", type=int, default=64, help="Patch size for training. Set to 0 to disable.")
    parser.add_argument("--stride", type=int, default=32, help="Stride for patch extraction")
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    main(args)