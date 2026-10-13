import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from fourway_mri.losses import ssdu_loss
from fourway_mri.metrics import nmse, psnr, ssim
from fourway_mri.operators import apply_kspace_mask, multicoil_adjoint
from fourway_mri.sensitivity import estimate_sensitivities_from_acs, indices_to_torch_mask
from fourway_mri.ssdu_dataset import SSDUSliceDataset
from fourway_mri.ssdu_model_v4 import ScaleConstrainedSingleStepMoDL
from fourway_mri.torch_fft import ifft2c, rss_combine_torch


def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def true_global_grad_norm(model):
    total_sq = 0.0
    finite = True

    for p in model.parameters():
        if p.grad is not None:
            g = p.grad.detach()
            finite = finite and bool(torch.isfinite(g).all().item())
            total_sq += float(torch.sum(torch.abs(g) ** 2).item())

    return float(total_sq ** 0.5), bool(finite)


def make_loader(cfg, split, data_root, max_volumes, shuffle):
    ds = SSDUSliceDataset(
        split_csv=cfg["inputs"]["split_csv"],
        mask_indices_json=cfg["inputs"]["mask_indices_json"],
        split=split,
        data_root=data_root,
        max_volumes=max_volumes,
        max_slices_per_volume=cfg["data"]["max_slices_per_volume"],
    )

    return DataLoader(
        ds,
        batch_size=int(cfg["training"]["batch_size"]),
        shuffle=shuffle,
        num_workers=int(cfg["training"]["num_workers"]),
    )


def compute_reference_scale(kspace, percentile=99.0, eps=1e-8):
    with torch.no_grad():
        full_rss = rss_combine_torch(ifft2c(kspace), coil_dim=1)
        scale = float(np.percentile(full_rss.detach().cpu().numpy(), percentile) + eps)
    return scale, full_rss


def prepare_batch(batch, device, cfg):
    kspace_raw = batch["kspace"].to(device)

    theta_mask = batch["theta_mask"][0].to(device)
    lambda_rec_mask = batch["lambda_rec_mask"][0].to(device)
    theta_acs_indices = batch["theta_acs_indices"][0].tolist()

    # Reference scale is derived strictly from Theta_v.
    theta_kspace_raw = apply_kspace_mask(kspace_raw, theta_mask)

    reference_scale, _ = compute_reference_scale(
        theta_kspace_raw,
        percentile=float(cfg["normalization"]["reference_percentile"]),
        eps=float(cfg["normalization"]["epsilon"]),
    )

    # Retain the full acquired RSS only as the image-reference quantity.
    # It does not determine the normalization scale.
    with torch.no_grad():
        full_rss_raw = rss_combine_torch(
            ifft2c(kspace_raw),
            coil_dim=1,
        )

    kspace = kspace_raw / reference_scale
    full_rss = full_rss_raw / reference_scale

    width = int(batch["width"][0].item())
    theta_acs_mask = indices_to_torch_mask(
        width=width,
        indices=theta_acs_indices,
        device=device,
        dtype=kspace.real.dtype,
    )

    r_theta = batch["r_theta"].to(device).float()
    y_theta = apply_kspace_mask(kspace, theta_mask)

    sensitivities = estimate_sensitivities_from_acs(
        kspace[0],
        theta_acs_indices,
        body_mask_threshold=cfg["training"].get("body_mask_threshold", None),
        body_mask_softness=float(cfg["training"].get("body_mask_softness", 0.02)),
    )[None, ...].to(device)

    x0 = multicoil_adjoint(y_theta, sensitivities, theta_mask)
    input_scale = torch.sqrt(r_theta)

    return {
        "kspace": kspace,
        "full_rss": full_rss,
        "reference_scale": reference_scale,
        "y_theta": y_theta,
        "sensitivities": sensitivities,
        "theta_mask": theta_mask,
        "theta_acs_mask": theta_acs_mask,
        "lambda_rec_mask": lambda_rec_mask,
        "x0": x0,
        "input_scale": input_scale,
    }


def rss_from_sense_image(image, sensitivities):
    coil_images = sensitivities * image[:, None, :, :]
    return rss_combine_torch(coil_images, coil_dim=1)


def image_metrics(pred_rss, ref_rss):
    pred_np = pred_rss.detach().cpu().numpy()
    ref_np = ref_rss.detach().cpu().numpy()

    return {
        "nmse": nmse(ref_np, pred_np),
        "psnr": psnr(ref_np, pred_np, data_range=1.0),
        "ssim": ssim(ref_np, pred_np, data_range=1.0),
        "pred_p99": float(np.percentile(pred_np, 99.0)),
        "ref_p99": float(np.percentile(ref_np, 99.0)),
    }


def compute_v4_loss(out, prepared, cfg):
    loss_type = cfg["training"].get("loss_type", "relative_l1")
    lambda_acs = float(cfg["training"].get("lambda_acs", 1.0))

    loss_rec, diag_rec = ssdu_loss(
        prediction=out,
        target_kspace=prepared["kspace"],
        sensitivities=prepared["sensitivities"],
        lambda_rec_mask=prepared["lambda_rec_mask"],
        loss_type=loss_type,
    )

    loss_acs, diag_acs = ssdu_loss(
        prediction=out,
        target_kspace=prepared["kspace"],
        sensitivities=prepared["sensitivities"],
        lambda_rec_mask=prepared["theta_acs_mask"],
        loss_type=loss_type,
    )

    loss_total = loss_rec + lambda_acs * loss_acs
    return loss_total, loss_rec, loss_acs, diag_rec, diag_acs


def run_validation(model, loader, device, cfg):
    model.eval()
    rows = []

    with torch.no_grad():
        for batch in loader:
            prepared = prepare_batch(batch, device, cfg)

            out = model(
                x0=prepared["x0"],
                y_theta=prepared["y_theta"],
                sensitivities=prepared["sensitivities"],
                theta_mask=prepared["theta_mask"],
                input_scale=prepared["input_scale"],
            )

            loss_total, loss_rec, loss_acs, diag_rec, diag_acs = compute_v4_loss(
                out, prepared, cfg
            )

            out_rss = rss_from_sense_image(out, prepared["sensitivities"])
            x0_rss = rss_from_sense_image(prepared["x0"], prepared["sensitivities"])
            ref_rss = prepared["full_rss"]

            out_metrics = image_metrics(out_rss[0], ref_rss[0])
            x0_metrics = image_metrics(x0_rss[0], ref_rss[0])

            rows.append(
                {
                    "volume_id": batch["volume_id"][0],
                    "slice_idx": int(batch["slice_idx"][0].item()),
                    "val_loss_total": float(loss_total.detach().cpu().item()),
                    "val_loss_rec": float(loss_rec.detach().cpu().item()),
                    "val_loss_acs": float(loss_acs.detach().cpu().item()),
                    "out_nmse": out_metrics["nmse"],
                    "out_psnr": out_metrics["psnr"],
                    "out_ssim": out_metrics["ssim"],
                    "out_pred_p99": out_metrics["pred_p99"],
                    "out_ref_p99": out_metrics["ref_p99"],
                    "out_p99_ratio": float(out_metrics["pred_p99"] / (out_metrics["ref_p99"] + 1e-12)),
                    "x0_nmse": x0_metrics["nmse"],
                    "x0_psnr": x0_metrics["psnr"],
                    "x0_ssim": x0_metrics["ssim"],
                    "x0_pred_p99": x0_metrics["pred_p99"],
                    "x0_ref_p99": x0_metrics["ref_p99"],
                    "x0_p99_ratio": float(x0_metrics["pred_p99"] / (x0_metrics["ref_p99"] + 1e-12)),
                    "rec_target_l1": float(diag_rec["target_l1"].detach().cpu().item()),
                    "acs_target_l1": float(diag_acs["target_l1"].detach().cpu().item()),
                }
            )

    model.train()
    val_df = pd.DataFrame(rows)

    summary = {
        "val_loss_total_mean": float(val_df["val_loss_total"].mean()),
        "val_loss_rec_mean": float(val_df["val_loss_rec"].mean()),
        "val_loss_acs_mean": float(val_df["val_loss_acs"].mean()),
        "out_nmse_mean": float(val_df["out_nmse"].mean()),
        "out_psnr_mean": float(val_df["out_psnr"].mean()),
        "out_ssim_mean": float(val_df["out_ssim"].mean()),
        "out_p99_ratio_mean": float(val_df["out_p99_ratio"].mean()),
        "x0_nmse_mean": float(val_df["x0_nmse"].mean()),
        "x0_psnr_mean": float(val_df["x0_psnr"].mean()),
        "x0_ssim_mean": float(val_df["x0_ssim"].mean()),
        "x0_p99_ratio_mean": float(val_df["x0_p99_ratio"].mean()),
    }

    return val_df, summary


def save_checkpoint(path, model, optimizer, scheduler, epoch, best_val_rec_loss, epochs_without_improvement, cfg):
    payload = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": None if scheduler is None else scheduler.state_dict(),
        "epoch": epoch,
        "best_val_rec_loss": best_val_rec_loss,
        "epochs_without_improvement": epochs_without_improvement,
        "config": cfg,
    }
    torch.save(payload, path)


def load_checkpoint(path, model, optimizer, scheduler, device):
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    start_epoch = int(checkpoint["epoch"]) + 1
    best_val_rec_loss = float(checkpoint["best_val_rec_loss"])
    epochs_without_improvement = int(checkpoint.get("epochs_without_improvement", 0))

    return start_epoch, best_val_rec_loss, epochs_without_improvement


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/exp004_train_ssdu_v4_full.yaml")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--resume", default=None)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    torch.manual_seed(int(cfg["training"]["seed"]))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    output_dir = Path(cfg["outputs"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    train_loader = make_loader(
        cfg,
        split=cfg["data"]["train_split"],
        data_root=args.data_root,
        max_volumes=cfg["data"]["max_train_volumes"],
        shuffle=True,
    )

    val_loader = make_loader(
        cfg,
        split=cfg["data"]["val_split"],
        data_root=args.data_root,
        max_volumes=cfg["data"]["max_val_volumes"],
        shuffle=False,
    )

    model = ScaleConstrainedSingleStepMoDL(
        features=int(cfg["model"]["features"]),
        init_step_size=float(cfg["model"]["init_step_size"]),
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg["optimizer"]["lr"]))

    scheduler = None
    if cfg.get("scheduler", {}).get("enabled", False):
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=cfg["scheduler"].get("mode", "min"),
            factor=float(cfg["scheduler"].get("factor", 0.5)),
            patience=int(cfg["scheduler"].get("patience", 3)),
        )

    start_epoch = 0
    best_val_rec_loss = float("inf")
    epochs_without_improvement = 0

    if args.resume is not None:
        start_epoch, best_val_rec_loss, epochs_without_improvement = load_checkpoint(
            args.resume, model, optimizer, scheduler, device
        )

    accumulation_steps = int(cfg["training"].get("accumulation_steps", 1))
    clip_value = cfg["training"].get("gradient_clip_norm", None)
    patience = int(cfg["training"]["early_stopping_patience"])
    min_delta = float(cfg["training"].get("min_delta", 0.0))

    history_path = output_dir / "training_history.csv"
    train_trace_path = output_dir / "train_loss_trace.csv"

    history_rows = []
    train_rows = []

    if history_path.exists() and args.resume is not None:
        history_rows = pd.read_csv(history_path).to_dict("records")

    if train_trace_path.exists() and args.resume is not None:
        train_rows = pd.read_csv(train_trace_path).to_dict("records")

    for epoch in range(start_epoch, int(cfg["training"]["epochs"])):
        model.train()
        optimizer.zero_grad(set_to_none=True)

        epoch_losses_total = []
        epoch_losses_rec = []
        epoch_losses_acs = []

        pbar = tqdm(train_loader, desc=f"epoch {epoch}")

        for batch_idx, batch in enumerate(pbar):
            prepared = prepare_batch(batch, device, cfg)

            out = model(
                x0=prepared["x0"],
                y_theta=prepared["y_theta"],
                sensitivities=prepared["sensitivities"],
                theta_mask=prepared["theta_mask"],
                input_scale=prepared["input_scale"],
            )

            loss_total, loss_rec, loss_acs, diag_rec, diag_acs = compute_v4_loss(
                out, prepared, cfg
            )

            (loss_total / accumulation_steps).backward()

            should_step = (
                ((batch_idx + 1) % accumulation_steps == 0)
                or ((batch_idx + 1) == len(train_loader))
            )

            grad_before = None
            grad_after = None
            grad_finite = True

            if should_step:
                grad_before, finite_before = true_global_grad_norm(model)

                if clip_value is not None:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        max_norm=float(clip_value),
                    )

                grad_after, finite_after = true_global_grad_norm(model)
                grad_finite = bool(finite_before and finite_after)

                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            loss_total_value = float(loss_total.detach().cpu().item())
            loss_rec_value = float(loss_rec.detach().cpu().item())
            loss_acs_value = float(loss_acs.detach().cpu().item())

            epoch_losses_total.append(loss_total_value)
            epoch_losses_rec.append(loss_rec_value)
            epoch_losses_acs.append(loss_acs_value)

            train_rows.append(
                {
                    "epoch": epoch,
                    "batch_idx": batch_idx,
                    "loss_total": loss_total_value,
                    "loss_rec": loss_rec_value,
                    "loss_acs": loss_acs_value,
                    "lambda_acs": float(cfg["training"]["lambda_acs"]),
                    "grad_norm_before_clip_l2": grad_before,
                    "grad_norm_after_clip_l2": grad_after,
                    "grad_finite": bool(grad_finite),
                    "optimizer_step_taken": bool(should_step),
                    "volume_id": batch["volume_id"][0],
                    "slice_idx": int(batch["slice_idx"][0].item()),
                    "r_theta": float(batch["r_theta"][0].item()),
                }
            )

            pbar.set_postfix(
                total=loss_total_value,
                rec=loss_rec_value,
                acs=loss_acs_value,
            )

        val_df, val_summary = run_validation(model, val_loader, device, cfg)
        val_df.to_csv(output_dir / f"validation_metrics_epoch_{epoch:03d}.csv", index=False)

        val_rec = float(val_summary["val_loss_rec_mean"])

        if scheduler is not None:
            scheduler.step(val_rec)

        improved = val_rec < (best_val_rec_loss - min_delta)

        if improved:
            best_val_rec_loss = val_rec
            epochs_without_improvement = 0

            save_checkpoint(
                cfg["outputs"]["best_checkpoint"],
                model,
                optimizer,
                scheduler,
                epoch,
                best_val_rec_loss,
                epochs_without_improvement,
                cfg,
            )
        else:
            epochs_without_improvement += 1

        save_checkpoint(
            cfg["outputs"]["last_checkpoint"],
            model,
            optimizer,
            scheduler,
            epoch,
            best_val_rec_loss,
            epochs_without_improvement,
            cfg,
        )

        history_rows.append(
            {
                "epoch": epoch,
                "train_loss_total_mean": float(np.mean(epoch_losses_total)),
                "train_loss_rec_mean": float(np.mean(epoch_losses_rec)),
                "train_loss_acs_mean": float(np.mean(epoch_losses_acs)),
                "train_loss_total_max": float(np.max(epoch_losses_total)),
                "val_loss_total_mean": float(val_summary["val_loss_total_mean"]),
                "val_loss_rec_mean": val_rec,
                "val_loss_acs_mean": float(val_summary["val_loss_acs_mean"]),
                "best_val_rec_loss": float(best_val_rec_loss),
                "improved_by_val_rec_loss": bool(improved),
                "epochs_without_improvement": int(epochs_without_improvement),
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                **val_summary,
            }
        )

        pd.DataFrame(history_rows).to_csv(history_path, index=False)
        pd.DataFrame(train_rows).to_csv(train_trace_path, index=False)

        if epochs_without_improvement >= patience:
            break

    history_df = pd.DataFrame(history_rows)

    summary = {
        "experiment_id": cfg["experiment"]["id"],
        "device": str(device),
        "train_samples": int(len(train_loader.dataset)),
        "val_samples": int(len(val_loader.dataset)),
        "epochs_requested": int(cfg["training"]["epochs"]),
        "epochs_completed": int(len(history_df)),
        "best_val_rec_loss": float(best_val_rec_loss),
        "final_val_rec_loss": float(history_df["val_loss_rec_mean"].iloc[-1]),
        "final_val_acs_loss": float(history_df["val_loss_acs_mean"].iloc[-1]),
        "final_val_total_loss": float(history_df["val_loss_total_mean"].iloc[-1]),
        "final_train_loss_total_mean": float(history_df["train_loss_total_mean"].iloc[-1]),
        "train_loss": cfg["control"]["train_loss"],
        "checkpoint_selection_metric": cfg["control"]["checkpoint_selection_metric"],
        "scheduler_metric": cfg["control"]["scheduler_metric"],
        "early_stopping_metric": cfg["control"]["early_stopping_metric"],
        "acs_loss_used_for_backprop": bool(cfg["control"]["acs_loss_used_for_backprop"]),
        "acs_loss_used_for_checkpointing": bool(cfg["control"]["acs_loss_used_for_checkpointing"]),
        "image_metrics_used_for_checkpointing": bool(cfg["control"]["image_metrics_used_for_checkpointing"]),
        "image_metrics_used_for_scheduler": bool(cfg["control"]["image_metrics_used_for_scheduler"]),
        "image_metrics_used_for_early_stopping": bool(cfg["control"]["image_metrics_used_for_early_stopping"]),
        "lambda_risk_used": False,
        "lambda_hold_used": False,
        "best_checkpoint": cfg["outputs"]["best_checkpoint"],
        "last_checkpoint": cfg["outputs"]["last_checkpoint"],
        "outputs": {
            "training_history_csv": str(history_path),
            "train_loss_trace_csv": str(train_trace_path),
            "summary_json": str(output_dir / "exp004_v4_full_summary.json"),
        },
    }

    with open(output_dir / "exp004_v4_full_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
