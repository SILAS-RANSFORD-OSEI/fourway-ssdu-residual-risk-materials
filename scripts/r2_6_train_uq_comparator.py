import argparse
import hashlib
import json
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from fourway_mri.reliability_dataset import ReliabilityCacheDataset
from fourway_mri.reliability_model import ReliabilityUNetSmall
from fourway_mri.r2_6_uq_model import (
    R26MCDropoutUNet,
    apply_a4_input,
)


def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def sha256_file(path, block=1024 * 1024):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(block), b""):
            h.update(chunk)
    return h.hexdigest()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def count_trainable_parameters(model):
    return int(
        sum(
            p.numel()
            for p in model.parameters()
            if p.requires_grad
        )
    )


def make_model(method, cfg, device):
    model_cfg = cfg["model"]

    if method == "mc_dropout":
        model = R26MCDropoutUNet(
            in_channels=int(model_cfg["in_channels"]),
            out_channels=int(model_cfg["out_channels"]),
            base_channels=int(model_cfg["base_channels"]),
            dropout_probability=float(
                model_cfg["dropout_probability"]
            ),
        )

    elif method == "ensemble":
        model = ReliabilityUNetSmall(
            in_channels=int(model_cfg["in_channels"]),
            out_channels=int(model_cfg["out_channels"]),
            base_channels=int(model_cfg["base_channels"]),
        )

    else:
        raise ValueError(
            f"Unknown method: {method}"
        )

    return model.to(device)


@torch.no_grad()
def deterministic_calibration_mae(model, loader, device):
    """
    Deterministic calibration MAE for checkpoint selection.

    The model remains in evaluation mode, including dropout layers.
    """
    model.eval()

    absolute_error_sum = 0.0
    pixel_count = 0

    for batch in loader:
        x = batch["x"].to(
            device,
            non_blocking=True,
        )

        y = batch["y"].to(
            device,
            non_blocking=True,
        )

        input_a4 = apply_a4_input(x)

        pred = model(input_a4)

        if pred.shape != y.shape:
            raise RuntimeError(
                f"Prediction/target shape mismatch: "
                f"{tuple(pred.shape)} vs {tuple(y.shape)}"
            )

        if not torch.isfinite(pred).all():
            raise RuntimeError(
                "Non-finite calibration prediction."
            )

        absolute_error_sum += float(
            torch.abs(pred - y).sum().cpu()
        )

        pixel_count += int(y.numel())

    if pixel_count == 0:
        raise RuntimeError(
            "Calibration loader contained zero pixels."
        )

    return absolute_error_sum / pixel_count


def save_checkpoint(
    path,
    model,
    optimizer,
    epoch,
    best_calibration_mae,
    epochs_without_improvement,
    method,
    member_seed,
    cfg,
):
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": int(epoch),
        "best_calibration_mae": float(
            best_calibration_mae
        ),
        "epochs_without_improvement": int(
            epochs_without_improvement
        ),
        "method": method,
        "member_seed": int(member_seed),
        "configuration": cfg,
        "lambda_hold_used": False,
        "test_accessed": False,
    }

    torch.save(
        checkpoint,
        path,
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--method",
        required=True,
        choices=[
            "mc_dropout",
            "ensemble",
        ],
    )

    parser.add_argument(
        "--config",
        required=True,
    )

    parser.add_argument(
        "--resolved-manifest",
        required=True,
    )

    parser.add_argument(
        "--member-seed",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--output-dir",
        required=True,
    )

    args = parser.parse_args()

    config_path = Path(
        args.config
    ).resolve()

    manifest_path = Path(
        args.resolved_manifest
    ).resolve()

    output_dir = Path(
        args.output_dir
    ).resolve()

    assert config_path.is_file()
    assert manifest_path.is_file()

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    cfg = load_yaml(
        config_path
    )

    # Resolve seed

    if args.method == "mc_dropout":

        seed = int(
            cfg["training"]["seed"]
        )

        if args.member_seed is not None:
            raise ValueError(
                "--member-seed is not used for mc_dropout."
            )

    else:

        if args.member_seed is None:
            raise ValueError(
                "--member-seed is required for ensemble."
            )

        allowed = [
            int(x)
            for x
            in cfg["ensemble"]["member_seeds"]
        ]

        seed = int(
            args.member_seed
        )

        if seed not in allowed:
            raise ValueError(
                f"Seed {seed} not in frozen member seeds {allowed}"
            )

    seed_everything(
        seed
    )

    # Device

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required for the frozen R2.6 training run."
        )

    device = torch.device(
        "cuda"
    )

    gpu_name = torch.cuda.get_device_name(
        0
    )

    # Development-only data

    train_split = cfg["data"]["train_split"]
    val_split = cfg["data"]["val_split"]

    if train_split != "train":
        raise RuntimeError(
            f"Unexpected train split: {train_split}"
        )

    if val_split != "calibration":
        raise RuntimeError(
            f"Unexpected validation split: {val_split}"
        )

    manifest_df = pd.read_csv(
        manifest_path
    )

    allowed_splits = set(
        manifest_df["split"].unique()
    )

    if allowed_splits != {
        "train",
        "calibration",
    }:
        raise RuntimeError(
            f"Development manifest contains unexpected splits: "
            f"{allowed_splits}"
        )

    train_ds = ReliabilityCacheDataset(
        str(manifest_path),
        split="train",
    )

    calibration_ds = ReliabilityCacheDataset(
        str(manifest_path),
        split="calibration",
    )

    train_generator = torch.Generator()
    train_generator.manual_seed(
        seed
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=int(
            cfg["training"]["batch_size"]
        ),
        shuffle=True,
        num_workers=int(
            cfg["training"]["num_workers"]
        ),
        pin_memory=True,
        generator=train_generator,
    )

    calibration_loader = DataLoader(
        calibration_ds,
        batch_size=1,
        shuffle=False,
        num_workers=int(
            cfg["training"]["num_workers"]
        ),
        pin_memory=True,
    )

    # Model / optimizer

    model = make_model(
        args.method,
        cfg,
        device,
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(
            cfg["training"]["learning_rate"]
        ),
        weight_decay=float(
            cfg["training"]["weight_decay"]
        ),
    )

    huber_delta = float(
        cfg["training"]["huber_delta"]
    )

    epochs_max = int(
        cfg["training"]["epochs"]
    )

    patience = int(
        cfg["training"]["early_stopping_patience"]
    )

    min_delta = float(
        cfg["training"]["min_delta_mae"]
    )

    # Record run provenance before training.

    run_start = datetime.now(
        timezone.utc
    ).isoformat()

    provenance = {
        "experiment": "R2.6",
        "method": args.method,
        "seed": seed,
        "run_start_utc": run_start,
        "device": str(device),
        "gpu": gpu_name,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "resolved_manifest_path": str(manifest_path),
        "resolved_manifest_sha256": sha256_file(
            manifest_path
        ),
        "train_samples": len(train_ds),
        "calibration_samples": len(
            calibration_ds
        ),
        "trainable_parameters": (
            count_trainable_parameters(model)
        ),
        "checkpoint_selection": (
            "deterministic calibration MAE; "
            "dropout disabled during checkpoint selection"
        ),
        "a4_zero_channels": [
            3,
            4,
            5,
        ],
        "lambda_hold_used": False,
        "test_accessed": False,
    }

    with open(
        output_dir / "run_start.json",
        "w",
    ) as f:
        json.dump(
            provenance,
            f,
            indent=2,
        )

    # Training

    best_checkpoint = (
        output_dir /
        "best_model.pt"
    )

    last_checkpoint = (
        output_dir /
        "last_model.pt"
    )

    history_path = (
        output_dir /
        "training_history.csv"
    )

    best_calibration_mae = float(
        "inf"
    )

    epochs_without_improvement = 0

    history_rows = []

    overall_start = time.time()

    for epoch in range(
        epochs_max
    ):

        model.train()

        train_losses = []

        epoch_start = time.time()

        pbar = tqdm(
            train_loader,
            desc=(
                f"{args.method} "
                f"seed={seed} "
                f"epoch={epoch:02d}"
            ),
        )

        for batch in pbar:

            x = batch["x"].to(
                device,
                non_blocking=True,
            )

            y = batch["y"].to(
                device,
                non_blocking=True,
            )

            input_a4 = apply_a4_input(
                x
            )

            pred = model(
                input_a4
            )

            if pred.shape != y.shape:
                raise RuntimeError(
                    f"Prediction/target shape mismatch: "
                    f"{tuple(pred.shape)} vs {tuple(y.shape)}"
                )

            loss = (
                torch.nn.functional.huber_loss(
                    pred,
                    y,
                    delta=huber_delta,
                )
            )

            if not torch.isfinite(
                loss
            ):
                raise RuntimeError(
                    "Non-finite training loss."
                )

            optimizer.zero_grad(
                set_to_none=True
            )

            loss.backward()

            optimizer.step()

            loss_value = float(
                loss.detach().cpu()
            )

            train_losses.append(
                loss_value
            )

            pbar.set_postfix(
                loss=f"{loss_value:.6f}"
            )

        # Deterministic calibration checkpoint selection

        calibration_mae = (
            deterministic_calibration_mae(
                model,
                calibration_loader,
                device,
            )
        )

        improved = (
            calibration_mae
            <
            best_calibration_mae
            -
            min_delta
        )

        if improved:

            best_calibration_mae = (
                calibration_mae
            )

            epochs_without_improvement = 0

            save_checkpoint(
                best_checkpoint,
                model,
                optimizer,
                epoch,
                best_calibration_mae,
                epochs_without_improvement,
                args.method,
                seed,
                cfg,
            )

        else:

            epochs_without_improvement += 1

        save_checkpoint(
            last_checkpoint,
            model,
            optimizer,
            epoch,
            best_calibration_mae,
            epochs_without_improvement,
            args.method,
            seed,
            cfg,
        )

        history_rows.append(
            {
                "epoch": int(epoch),
                "seed": int(seed),
                "method": args.method,
                "train_huber_mean": float(
                    np.mean(train_losses)
                ),
                "train_huber_min": float(
                    np.min(train_losses)
                ),
                "train_huber_max": float(
                    np.max(train_losses)
                ),
                "calibration_mae": float(
                    calibration_mae
                ),
                "best_calibration_mae": float(
                    best_calibration_mae
                ),
                "improved": bool(
                    improved
                ),
                "epochs_without_improvement": int(
                    epochs_without_improvement
                ),
                "learning_rate": float(
                    optimizer.param_groups[0]["lr"]
                ),
                "epoch_seconds": float(
                    time.time() - epoch_start
                ),
            }
        )

        pd.DataFrame(
            history_rows
        ).to_csv(
            history_path,
            index=False,
        )

        print(
            f"\nEpoch {epoch:02d}: "
            f"train_huber="
            f"{np.mean(train_losses):.8f} | "
            f"calibration_mae="
            f"{calibration_mae:.8f} | "
            f"best="
            f"{best_calibration_mae:.8f} | "
            f"improved={improved} | "
            f"patience="
            f"{epochs_without_improvement}/{patience}"
        )

        if (
            epochs_without_improvement
            >= patience
        ):
            print(
                "\nEarly stopping triggered."
            )
            break

    # Training summary

    if not best_checkpoint.is_file():
        raise RuntimeError(
            "Best checkpoint was never written."
        )

    total_seconds = float(
        time.time() -
        overall_start
    )

    best = torch.load(
        best_checkpoint,
        map_location="cpu",
    )

    summary = {
        **provenance,
        "run_end_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "epochs_completed": len(
            history_rows
        ),
        "best_epoch": int(
            best["epoch"]
        ),
        "best_calibration_mae": float(
            best["best_calibration_mae"]
        ),
        "best_checkpoint": str(
            best_checkpoint
        ),
        "best_checkpoint_sha256": (
            sha256_file(
                best_checkpoint
            )
        ),
        "last_checkpoint": str(
            last_checkpoint
        ),
        "last_checkpoint_sha256": (
            sha256_file(
                last_checkpoint
            )
        ),
        "training_history": str(
            history_path
        ),
        "training_history_sha256": (
            sha256_file(
                history_path
            )
        ),
        "elapsed_seconds": (
            total_seconds
        ),
        "lambda_hold_used": False,
        "test_accessed": False,
        "status": "COMPLETE_FROZEN_DEVELOPMENT",
    }

    with open(
        output_dir /
        "training_summary.json",
        "w",
    ) as f:

        json.dump(
            summary,
            f,
            indent=2,
        )

    print(
        "\n" +
        "=" * 110
    )

    print(
        "R2.6 DEVELOPMENT TRAINING COMPLETE"
    )

    print(
        "=" * 110
    )

    print(
        "Method                 :",
        args.method
    )

    print(
        "Seed                   :",
        seed
    )

    print(
        "Best epoch             :",
        summary["best_epoch"]
    )

    print(
        "Best calibration MAE   :",
        summary[
            "best_calibration_mae"
        ]
    )

    print(
        "Best checkpoint SHA256 :",
        summary[
            "best_checkpoint_sha256"
        ]
    )

    print(
        "Trainable parameters   :",
        summary[
            "trainable_parameters"
        ]
    )

    print(
        "Test accessed          : NO"
    )

    print(
        "Lambda_hold used       : NO"
    )


if __name__ == "__main__":
    main()
