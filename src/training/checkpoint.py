from __future__ import annotations

import os
from pathlib import Path

import torch


def atomic_torch_save(
    payload: dict,
    path: Path,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = path.with_name(
        path.name + ".tmp"
    )

    torch.save(
        payload,
        temporary,
    )

    os.replace(
        temporary,
        path,
    )


def load_checkpoint(
    path: Path,
    model,
    optimizer=None,
    scheduler=None,
    scaler=None,
) -> dict:
    checkpoint = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )

    if isinstance(checkpoint, dict) and "model_state" in checkpoint:
        model_state = checkpoint["model_state"]
    elif isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model_state = checkpoint["model_state_dict"]
    else:
        model_state = checkpoint

    model.load_state_dict(model_state)

    if (
        optimizer is not None
        and checkpoint.get(
            "optimizer_state"
        )
        is not None
    ):
        optimizer.load_state_dict(
            checkpoint[
                "optimizer_state"
            ]
        )

    if (
        scheduler is not None
        and checkpoint.get(
            "scheduler_state"
        )
        is not None
    ):
        scheduler.load_state_dict(
            checkpoint[
                "scheduler_state"
            ]
        )

    if (
        scaler is not None
        and checkpoint.get(
            "scaler_state"
        )
        is not None
    ):
        scaler.load_state_dict(
            checkpoint[
                "scaler_state"
            ]
        )

    return checkpoint if isinstance(checkpoint, dict) else {"model_state": checkpoint}
