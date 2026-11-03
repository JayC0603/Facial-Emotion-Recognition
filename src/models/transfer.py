from __future__ import annotations

from collections import OrderedDict

from torch import nn
from torchvision.models import (
    EfficientNet_B0_Weights,
    MobileNet_V3_Large_Weights,
    efficientnet_b0,
    mobilenet_v3_large,
)


EFFICIENTNET_B0_PRUNING = {
    "m2_effb0_p1_s6x3": {6: 3},
    "m2_effb0_p2_s6x2": {6: 2},
    "m2_effb0_p3_s5x2_s6x3": {5: 2, 6: 3},
    "m2_effb0_p4_s4x2_s5x2_s6x3": {4: 2, 5: 2, 6: 3},
    "m2_effb0_p3_s5x2": {5: 2},
    "m2_effb0_p3_s4x2": {4: 2},
    "m2_effb0_p3_s4x2_s5x2": {4: 2, 5: 2},
}


def build_mobilenet_v3_large(
    num_classes: int,
    pretrained: bool,
) -> nn.Module:
    weights = (
        MobileNet_V3_Large_Weights.DEFAULT
        if pretrained
        else None
    )

    model = mobilenet_v3_large(
        weights=weights
    )

    in_features = (
        model.classifier[-1]
        .in_features
    )

    model.classifier[-1] = (
        nn.Linear(
            in_features,
            num_classes,
        )
    )

    return model


def build_efficientnet_b0(
    num_classes: int,
    pretrained: bool,
    stage_repeats: dict[int, int] | None = None,
    stage_block_indices: dict[int, tuple[int, ...]] | None = None,
) -> nn.Module:
    weights = (
        EfficientNet_B0_Weights.DEFAULT
        if pretrained
        else None
    )

    model = efficientnet_b0(
        weights=weights
    )

    # Load full ImageNet state first. Prefix retention keeps transition blocks,
    # pretrained tensors, state keys and stochastic-depth probabilities intact.
    for stage_index, keep in (stage_repeats or {}).items():
        if stage_index not in (4, 5, 6):
            raise ValueError("Only B0 stages 4/5/6 may be depth-pruned.")
        stage = model.features[stage_index]
        if not 1 <= keep <= len(stage):
            raise ValueError("Repeat count must retain the transition block.")
        if any(not block.use_res_connect for block in list(stage)[keep:]):
            raise ValueError("Cannot remove a channel/stride transition block.")
        model.features[stage_index] = nn.Sequential(*list(stage)[:keep])

    # Select original same-shape repeat positions after the full pretrained
    # load. Original numeric keys and stochastic-depth probabilities survive.
    for stage_index, indices in (stage_block_indices or {}).items():
        if stage_index not in (5, 6) or stage_index in (stage_repeats or {}):
            raise ValueError("Selective retention is only for untrimmed stages5/6.")
        stage=model.features[stage_index]
        if (not indices or indices[0]!=0 or tuple(sorted(set(indices)))!=tuple(indices)
                or any(not isinstance(i,int) or not 0<=i<len(stage) for i in indices)):
            raise ValueError("Retain transition0 and sorted unique valid original indices.")
        removed=set(range(len(stage)))-set(indices)
        if any(not stage[i].use_res_connect for i in removed):
            raise ValueError("Cannot remove a channel/stride transition block.")
        model.features[stage_index]=nn.Sequential(OrderedDict((str(i),stage[i]) for i in indices))

    in_features = (
        model.classifier[-1]
        .in_features
    )

    model.classifier[-1] = (
        nn.Linear(
            in_features,
            num_classes,
        )
    )

    return model


def set_transfer_backbone_training_mode(model: nn.Module, training: bool) -> None:
    """Set backbone mode independently of the always-active classifier."""
    if not hasattr(model, "classifier"):
        raise TypeError("Transfer model must expose a classifier.")
    for name, module in model.named_children():
        if name != "classifier":
            module.train(training)


def set_transfer_backbone_trainable(
    model: nn.Module,
    trainable: bool,
) -> None:
    """
    Freeze/unfreeze transfer backbone while always keeping classifier trainable.
    """

    for parameter in model.parameters():
        parameter.requires_grad = (
            trainable
        )

    if not hasattr(
        model,
        "classifier",
    ):
        raise TypeError(
            "Unsupported transfer model: "
            "missing classifier."
        )

    for parameter in (
        model.classifier.parameters()
    ):
        parameter.requires_grad = True
