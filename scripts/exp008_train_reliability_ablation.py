import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from fourway_mri.reliability_dataset import ReliabilityCacheDataset
from fourway_mri.reliability_model import ReliabilityUNetSmall, ReliabilityUNetWithPSFSkip


def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def seed_everything(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def apply_channel_ablation(x, zero_channels):
    if not zero_channels:
        return x

    x = x.clone()
    for ch in zero_channels:
        x[:, int(ch), :, :] = 0.0

    return x


def safe_pearson(y_true, y_pred):
    y_true = y_true.astype(np.float64)
    y_pred = y_pred.astype(np.float64)

    yt = y_true - y_true.mean()
    yp = y_pred - y_pred.mean()

    denom = np.sqrt(np.sum(yt**2) * np.sum(yp**2))
    if denom < 1e-12:
        return np.nan

    return float(np.sum(yt * yp) / denom)


def safe_spearman(y_true, y_pred):
    if np.std(y_true) < 1e-12 or np.std(y_pred) < 1e-12:
        return np.nan
    return float(spearmanr(y_true, y_pred).correlation)


def make_model(cfg, variant_cfg, device):
    if bool(variant_cfg["use_psf_skip"]):
        model = ReliabilityUNetWithPSFSkip(
            in_channels=int(cfg["model"]["in_channels"]),
            out_channels=int(cfg["model"]["out_channels"]),
            base_channels=int(cfg["model"]["base_channels"]),
            psf_channel_index=int(cfg["model"]["psf_channel_index"]),
            init_psf_weight=float(cfg["model"]["init_psf_weight"]),
            init_psf_bias=float(cfg["model"]["init_psf_bias"]),
        )
    else:
        model = ReliabilityUNetSmall(
            in_channels=int(cfg["model"]["in_channels"]),
            out_channels=int(cfg["model"]["out_channels"]),
            base_channels=int(cfg["model"]["base_channels"]),
        )

    return model.to(device)


def collect_predictions(model, loader, device, cfg, zero_channels, seed_offset=0):
    model.eval()

    max_metric_pixels = int(cfg["metrics"]["max_metric_pixels"])
    seed = int(cfg["metrics"]["random_seed"]) + int(seed_offset)

    total_abs = 0.0
    total_sq = 0.0
    total_n = 0

    sampled_true = []
    sampled_pred = []

    rng = np.random.default_rng(seed)
    per_sample = max(1, int(max_metric_pixels // max(1, len(loader.dataset))))

    sample_rows = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="eval", leave=False):
            x = batch["x"].to(device)
            y = batch["y"].to(device)

            x = apply_channel_ablation(x, zero_channels)

            pred = model(x)

            y_np = y.detach().cpu().numpy().astype(np.float32)
            p_np = pred.detach().cpu().numpy().astype(np.float32)

            y_flat = y_np.reshape(-1)
            p_flat = p_np.reshape(-1)

            diff = p_flat - y_flat

            total_abs += float(np.sum(np.abs(diff)))
            total_sq += float(np.sum(diff**2))
            total_n += int(y_flat.size)

            k = min(per_sample, y_flat.size)
            idx = rng.choice(y_flat.size, size=k, replace=False)

            sampled_true.append(y_flat[idx])
            sampled_pred.append(p_flat[idx])

            sample_rows.append(
                {
                    "sample_id": batch["sample_id"][0],
                    "volume_id": batch["volume_id"][0],
                    "slice_idx": int(batch["slice_idx"][0].item()),
                    "mae": float(np.mean(np.abs(diff))),
                    "mse": float(np.mean(diff**2)),
                    "pearson": safe_pearson(y_flat, p_flat),
                }
            )

    y_eval = np.concatenate(sampled_true)
    p_eval = np.concatenate(sampled_pred)

    return {
        "mae": float(total_abs / max(total_n, 1)),
        "mse": float(total_sq / max(total_n, 1)),
        "y_eval": y_eval,
        "p_eval": p_eval,
        "sample_rows": sample_rows,
    }


def evaluate_model(model, loader, split, device, cfg, zero_channels, seed_offset=0):
    pred_pack = collect_predictions(
        model=model,
        loader=loader,
        device=device,
        cfg=cfg,
        zero_channels=zero_channels,
        seed_offset=seed_offset,
    )

    y_eval = pred_pack["y_eval"]
    p_eval = pred_pack["p_eval"]

    top_fraction = float(cfg["metrics"]["top_risk_fraction"])
    threshold = np.quantile(y_eval, 1.0 - top_fraction)
    labels = (y_eval >= threshold).astype(np.int32)

    try:
        auroc = float(roc_auc_score(labels, p_eval))
    except ValueError:
        auroc = np.nan

    try:
        auprc = float(average_precision_score(labels, p_eval))
    except ValueError:
        auprc = np.nan

    summary = {
        "split": split,
        "mae": pred_pack["mae"],
        "mse": pred_pack["mse"],
        "pearson": safe_pearson(y_eval, p_eval),
        "spearman": safe_spearman(y_eval, p_eval),
        "top_fraction": top_fraction,
        "top_threshold": float(threshold),
        "top_risk_auroc": auroc,
        "top_risk_auprc": auprc,
        "num_eval_pixels_sampled": int(y_eval.size),
    }

    sample_df = pd.DataFrame(pred_pack["sample_rows"])
    sample_df["split"] = split

    return summary, sample_df


def save_checkpoint(path, model, optimizer, epoch, best_val_mae, epochs_without_improvement, cfg, variant_name, variant_cfg):
    payload = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": int(epoch),
        "best_val_mae": float(best_val_mae),
        "epochs_without_improvement": int(epochs_without_improvement),
        "config": cfg,
        "variant_name": variant_name,
        "variant_config": variant_cfg,
    }
    torch.save(payload, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/exp008_reliability_ablation_full.yaml")
    parser.add_argument("--variant", required=True)
    args = parser.parse_args()

    cfg = load_yaml(args.config)

    if args.variant not in cfg["ablation_variants"]:
        raise KeyError(
            f"Unknown variant {args.variant}. Available: {list(cfg['ablation_variants'].keys())}"
        )

    variant_name = args.variant
    variant_cfg = cfg["ablation_variants"][variant_name]
    zero_channels = [int(ch) for ch in variant_cfg.get("zero_channels", [])]

    seed_everything(int(cfg["training"]["seed"]))

    output_root = Path(cfg["outputs"]["output_root"])
    output_dir = output_root / variant_name
    output_dir.mkdir(parents=True, exist_ok=True)

    best_checkpoint = output_dir / "best_model.pt"
    last_checkpoint = output_dir / "last_model.pt"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = ReliabilityCacheDataset(cfg["inputs"]["cache_manifest"], split=cfg["data"]["train_split"])
    val_ds = ReliabilityCacheDataset(cfg["inputs"]["cache_manifest"], split=cfg["data"]["val_split"])
    test_ds = ReliabilityCacheDataset(cfg["inputs"]["cache_manifest"], split=cfg["data"]["test_split"])

    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg["training"]["batch_size"]),
        shuffle=True,
        num_workers=int(cfg["training"]["num_workers"]),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=int(cfg["training"]["num_workers"]),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=1,
        shuffle=False,
        num_workers=int(cfg["training"]["num_workers"]),
    )

    model = make_model(cfg, variant_cfg, device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(cfg["training"]["learning_rate"]),
        weight_decay=float(cfg["training"]["weight_decay"]),
    )

    huber_delta = float(cfg["training"]["huber_delta"])
    patience = int(cfg["training"]["early_stopping_patience"])
    min_delta = float(cfg["training"]["min_delta_mae"])

    best_val_mae = float("inf")
    epochs_without_improvement = 0
    history_rows = []

    for epoch in range(int(cfg["training"]["epochs"])):
        model.train()
        train_losses = []

        pbar = tqdm(train_loader, desc=f"{variant_name} epoch {epoch}")

        for batch in pbar:
            x = batch["x"].to(device)
            y = batch["y"].to(device)

            x = apply_channel_ablation(x, zero_channels)

            pred = model(x)

            loss = torch.nn.functional.huber_loss(
                pred,
                y,
                delta=huber_delta,
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            loss_value = float(loss.detach().cpu().item())
            train_losses.append(loss_value)
            pbar.set_postfix(loss=loss_value)

        val_summary, val_sample_df = evaluate_model(
            model,
            val_loader,
            split="calibration",
            device=device,
            cfg=cfg,
            zero_channels=zero_channels,
            seed_offset=epoch,
        )

        val_sample_df.to_csv(output_dir / f"calibration_sample_metrics_epoch_{epoch:03d}.csv", index=False)

        val_mae = float(val_summary["mae"])
        improved = val_mae < (best_val_mae - min_delta)

        if improved:
            best_val_mae = val_mae
            epochs_without_improvement = 0
            save_checkpoint(
                best_checkpoint,
                model,
                optimizer,
                epoch,
                best_val_mae,
                epochs_without_improvement,
                cfg,
                variant_name,
                variant_cfg,
            )
        else:
            epochs_without_improvement += 1

        save_checkpoint(
            last_checkpoint,
            model,
            optimizer,
            epoch,
            best_val_mae,
            epochs_without_improvement,
            cfg,
            variant_name,
            variant_cfg,
        )

        history_rows.append(
            {
                "epoch": epoch,
                "variant": variant_name,
                "train_huber_mean": float(np.mean(train_losses)),
                "train_huber_min": float(np.min(train_losses)),
                "train_huber_max": float(np.max(train_losses)),
                "val_mae": val_summary["mae"],
                "val_mse": val_summary["mse"],
                "val_pearson": val_summary["pearson"],
                "val_spearman": val_summary["spearman"],
                "val_top_risk_auroc": val_summary["top_risk_auroc"],
                "val_top_risk_auprc": val_summary["top_risk_auprc"],
                "best_val_mae": float(best_val_mae),
                "improved_by_val_mae": bool(improved),
                "epochs_without_improvement": int(epochs_without_improvement),
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
            }
        )

        pd.DataFrame(history_rows).to_csv(output_dir / "training_history.csv", index=False)

        if epochs_without_improvement >= patience:
            break

    best_model = make_model(cfg, variant_cfg, device)
    best_ckpt = torch.load(best_checkpoint, map_location=device)
    best_model.load_state_dict(best_ckpt["model_state_dict"])

    final_summaries = []
    final_sample_rows = []

    for split, loader in [
        ("train", train_loader),
        ("calibration", val_loader),
        ("test", test_loader),
    ]:
        split_summary, sample_df = evaluate_model(
            best_model,
            loader,
            split=split,
            device=device,
            cfg=cfg,
            zero_channels=zero_channels,
            seed_offset=1000 + len(split),
        )
        final_summaries.append(split_summary)
        final_sample_rows.append(sample_df)

    final_summary_df = pd.DataFrame(final_summaries)
    final_samples_df = pd.concat(final_sample_rows, ignore_index=True)

    final_summary_df.to_csv(output_dir / "final_split_metrics.csv", index=False)
    final_samples_df.to_csv(output_dir / "final_sample_metrics.csv", index=False)

    test_row = final_summary_df[final_summary_df["split"] == "test"].iloc[0].to_dict()
    baselines = cfg["baselines"]

    summary = {
        "experiment_id": cfg["experiment"]["id"],
        "variant": variant_name,
        "variant_description": variant_cfg["description"],
        "use_psf_skip": bool(variant_cfg["use_psf_skip"]),
        "zero_channels": zero_channels,
        "device": str(device),
        "train_samples": int(len(train_ds)),
        "calibration_samples": int(len(val_ds)),
        "test_samples": int(len(test_ds)),
        "epochs_completed": int(len(history_rows)),
        "best_epoch": int(best_ckpt["epoch"]),
        "checkpoint_selection_metric": "calibration_mae",
        "test_metrics_used_for_checkpointing": False,
        "test_metrics_used_for_training": False,
        "lambda_hold_used": False,
        "test_metrics": test_row,
        "baseline_gates": baselines,
        "beats_full_best_test_auprc": bool(test_row["top_risk_auprc"] > float(baselines["full_test_best_auprc"])),
        "beats_linear_6ch_test_auroc": bool(test_row["top_risk_auroc"] > float(baselines["full_test_linear_6ch_auroc"])),
        "beats_linear_6ch_test_spearman": bool(test_row["spearman"] > float(baselines["full_test_linear_6ch_spearman"])),
        "beats_linear_6ch_test_pearson": bool(test_row["pearson"] > float(baselines["full_test_linear_6ch_pearson"])),
        "outputs": {
            "training_history_csv": str(output_dir / "training_history.csv"),
            "final_split_metrics_csv": str(output_dir / "final_split_metrics.csv"),
            "final_sample_metrics_csv": str(output_dir / "final_sample_metrics.csv"),
            "best_checkpoint": str(best_checkpoint),
            "last_checkpoint": str(last_checkpoint),
            "summary_json": str(output_dir / "summary.json"),
        },
    }

    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
