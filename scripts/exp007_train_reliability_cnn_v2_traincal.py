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
from fourway_mri.reliability_model import ReliabilityUNetWithPSFSkip


def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def seed_everything(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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


def collect_predictions(model, loader, device, max_metric_pixels, seed):
    model.eval()

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


def evaluate_model(model, loader, split, device, cfg, seed_offset=0):
    max_metric_pixels = int(cfg["metrics"]["max_metric_pixels"])
    top_fraction = float(cfg["metrics"]["top_risk_fraction"])
    seed = int(cfg["metrics"]["random_seed"]) + int(seed_offset)

    pred_pack = collect_predictions(
        model=model,
        loader=loader,
        device=device,
        max_metric_pixels=max_metric_pixels,
        seed=seed,
    )

    y_eval = pred_pack["y_eval"]
    p_eval = pred_pack["p_eval"]

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


def save_checkpoint(path, model, optimizer, epoch, best_val_mae, epochs_without_improvement, cfg):
    payload = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": int(epoch),
        "best_val_mae": float(best_val_mae),
        "epochs_without_improvement": int(epochs_without_improvement),
        "config": cfg,
    }
    torch.save(payload, path)


def load_model_from_checkpoint(path, model, device):
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return checkpoint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/exp007_train_reliability_cnn_v2_pilot.yaml")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    seed_everything(int(cfg["training"]["seed"]))

    output_dir = Path(cfg["outputs"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = ReliabilityCacheDataset(
        cfg["inputs"]["cache_manifest"],
        split=cfg["data"]["train_split"],
    )
    val_ds = ReliabilityCacheDataset(
        cfg["inputs"]["cache_manifest"],
        split=cfg["data"]["val_split"],
    )

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

    model = ReliabilityUNetWithPSFSkip(
        in_channels=int(cfg["model"]["in_channels"]),
        out_channels=int(cfg["model"]["out_channels"]),
        base_channels=int(cfg["model"]["base_channels"]),
        psf_channel_index=int(cfg["model"]["psf_channel_index"]),
        init_psf_weight=float(cfg["model"]["init_psf_weight"]),
        init_psf_bias=float(cfg["model"]["init_psf_bias"]),
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(cfg["training"]["learning_rate"]),
        weight_decay=float(cfg["training"]["weight_decay"]),
    )

    loss_name = cfg["training"]["loss"]
    if loss_name != "huber":
        raise ValueError("Only huber loss is supported in this pilot.")

    huber_delta = float(cfg["training"]["huber_delta"])
    patience = int(cfg["training"]["early_stopping_patience"])
    min_delta = float(cfg["training"]["min_delta_mae"])

    history_rows = []

    best_val_mae = float("inf")
    epochs_without_improvement = 0

    for epoch in range(int(cfg["training"]["epochs"])):
        model.train()

        train_losses = []

        pbar = tqdm(train_loader, desc=f"epoch {epoch}")

        for batch in pbar:
            x = batch["x"].to(device)
            y = batch["y"].to(device)

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
            seed_offset=epoch,
        )

        val_sample_df.to_csv(output_dir / f"calibration_sample_metrics_epoch_{epoch:03d}.csv", index=False)

        val_mae = float(val_summary["mae"])
        improved = val_mae < (best_val_mae - min_delta)

        if improved:
            best_val_mae = val_mae
            epochs_without_improvement = 0
            save_checkpoint(
                cfg["outputs"]["best_checkpoint"],
                model,
                optimizer,
                epoch,
                best_val_mae,
                epochs_without_improvement,
                cfg,
            )
        else:
            epochs_without_improvement += 1

        save_checkpoint(
            cfg["outputs"]["last_checkpoint"],
            model,
            optimizer,
            epoch,
            best_val_mae,
            epochs_without_improvement,
            cfg,
        )

        history_rows.append(
            {
                "epoch": epoch,
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

    # Frozen development-only evaluation.
    # Test data deliberately remain unavailable at this stage.
    best_model = ReliabilityUNetWithPSFSkip(
        in_channels=int(cfg["model"]["in_channels"]),
        out_channels=int(cfg["model"]["out_channels"]),
        base_channels=int(cfg["model"]["base_channels"]),
        psf_channel_index=int(cfg["model"]["psf_channel_index"]),
        init_psf_weight=float(cfg["model"]["init_psf_weight"]),
        init_psf_bias=float(cfg["model"]["init_psf_bias"]),
    ).to(device)

    best_ckpt = load_model_from_checkpoint(
        cfg["outputs"]["best_checkpoint"],
        best_model,
        device,
    )

    final_summaries = []
    final_sample_rows = []

    for split, loader in [
        ("train", train_loader),
        ("calibration", val_loader),
    ]:
        split_summary, sample_df = evaluate_model(
            best_model,
            loader,
            split=split,
            device=device,
            cfg=cfg,
            seed_offset=1000 + len(split),
        )

        final_summaries.append(split_summary)
        final_sample_rows.append(sample_df)

    final_summary_df = pd.DataFrame(final_summaries)
    final_samples_df = pd.concat(
        final_sample_rows,
        ignore_index=True,
    )

    final_summary_df.to_csv(
        output_dir / "final_development_split_metrics.csv",
        index=False,
    )

    final_samples_df.to_csv(
        output_dir / "final_development_sample_metrics.csv",
        index=False,
    )

    summary = {
        "experiment_id": cfg["experiment"]["id"],
        "role": "primary_predefined_full_predictor",
        "device": str(device),
        "train_samples": int(len(train_ds)),
        "calibration_samples": int(len(val_ds)),
        "test_samples": 0,
        "epochs_completed": int(len(history_rows)),
        "best_epoch": int(best_ckpt["epoch"]),
        "best_val_mae": float(best_ckpt["best_val_mae"]),
        "checkpoint_selection_metric": "calibration_mae",
        "test_dataset_instantiated": False,
        "test_evaluation_performed": False,
        "test_metrics_used_for_checkpointing": False,
        "test_metrics_used_for_training": False,
        "lambda_hold_used": False,
        "outputs": {
            "training_history_csv":
                str(output_dir / "training_history.csv"),
            "final_development_split_metrics_csv":
                str(output_dir / "final_development_split_metrics.csv"),
            "final_development_sample_metrics_csv":
                str(output_dir / "final_development_sample_metrics.csv"),
            "best_checkpoint":
                cfg["outputs"]["best_checkpoint"],
            "last_checkpoint":
                cfg["outputs"]["last_checkpoint"],
            "summary_json":
                str(output_dir / "summary.json"),
        },
    }

    with open(
        output_dir / "summary.json",
        "w"
    ) as f:
        json.dump(
            summary,
            f,
            indent=2,
        )

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
