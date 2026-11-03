from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from torch import nn

from src.models.ablation import FERLiteRECAAblation, P61_ABLATIONS, P63_ABLATIONS

from src.models.ferlite import (
    FERLiteHFSALocalGlobal,
    FERLiteHybridConvECALocalGlobal,
    FERLiteMSCALocalGlobal,
    FERLitePartialECALocalGlobal,
    FERLiteRepECALocalGlobal,
    FERLiteLinearECALocalGlobal,
    FERLiteRECA,
    FERLiteRECALocalGlobal,
    FERLiteRECAStatisticalLocalGlobal,
    FERLiteRECASpatialAttentiveLocalGlobal,
    FERLiteRECALinearLocalGlobal,
    FERLiteResECALocalGlobal,
    FERLiteDSResECALocalGlobal,
)
from src.models.effb0_selective import EFFICIENTNET_B0_SELECTIVE
from src.models.transfer import (
    EFFICIENTNET_B0_PRUNING,
    build_efficientnet_b0,
    build_mobilenet_v3_large,
)
from src.models.notebook_models import NotebookEmotionCNN, NotebookEfficientNetB0


IMAGENET_MEAN = (
    0.485,
    0.456,
    0.406,
)

IMAGENET_STD = (
    0.229,
    0.224,
    0.225,
)


@dataclass(frozen=True)
class ModelSpec:
    name: str
    family: str
    transfer_learning: bool
    description: str


MODEL_SPECS = {
    "notebook_cnn_1m": ModelSpec(
        name="notebook_cnn_1m", family="custom_cnn", transfer_learning=False,
        description="Exact residual SE CNN from Notebook_CNN.ipynb.",
    ),
    "notebook_efficientnet_b0": ModelSpec(
        name="notebook_efficientnet_b0", family="transfer", transfer_learning=True,
        description="Exact grayscale EfficientNet-B0 from efficientnetb0.ipynb.",
    ),
    "m1_reca": ModelSpec(
        name="m1_reca",
        family="custom_cnn",
        transfer_learning=False,
        description=(
            "FERLite residual depthwise "
            "CNN with ECA."
        ),
    ),
    "m1_reca_lg": ModelSpec(
        name="m1_reca_lg",
        family="custom_cnn",
        transfer_learning=False,
        description=(
            "FERLite RECA with lightweight "
            "local-global head."
        ),
    ),
    "m1_reca_stat_lg": ModelSpec(
        name="m1_reca_stat_lg",
        family="custom_cnn",
        transfer_learning=False,
        description=(
            "Unchanged R4 backbone and mean pooling with non-learned "
            "global spatial standard deviation appended to the head."
        ),
    ),
    "m1_reca_sap_lg": ModelSpec(
        name="m1_reca_sap_lg",
        family="custom_cnn",
        transfer_learning=False,
        description=(
            "Unchanged R4 backbone and local-global classifier with "
            "spatial attentive pooling replacing only global average pooling."
        ),
    ),
    "m1_reca_linear_lg": ModelSpec(
        name="m1_reca_linear_lg",
        family="custom_cnn",
        transfer_learning=False,
        description=(
            "R4 RECA local-global backbone with only "
            "post-residual ReLU removed."
        ),
    ),
    "m1_partial_eca_lg": ModelSpec(
        name="m1_partial_eca_lg",
        family="custom_cnn",
        transfer_learning=False,
        description=(
            "R4 local-global CNN with quarter-channel dense 5x5 mixing "
            "only in stage 3's same-shape refinement block."
        ),
    ),
    "m1_res_eca_lg": ModelSpec(
        name="m1_res_eca_lg",
        family="custom_cnn",
        transfer_learning=False,
        description=(
            "Shallow full-convolution ResNet-style FER backbone "
            "with ECA and local-global head."
        ),
    ),
    "m1_dsres_eca_lg": ModelSpec(
        name="m1_dsres_eca_lg",
        family="custom_cnn",
        transfer_learning=False,
        description=(
            "Depthwise-pointwise residual counterpart of "
            "m1_res_eca_lg with identical stage topology."
        ),
    ),
    "m1_hybridconv_eca_lg": ModelSpec(
        name="m1_hybridconv_eca_lg",
        family="custom_cnn",
        transfer_learning=False,
        description=(
            "R4-preserving hybrid backbone with full-convolution "
            "bottleneck residuals only in late semantic stages."
        ),
    ),
    "m1_msca_lg": ModelSpec(
        name="m1_msca_lg",
        family="custom_cnn",
        transfer_learning=False,
        description=(
            "FERLite multi-scale depthwise CNN "
            "with coordinate attention and "
            "local-global head."
        ),
    ),
    "m1_hfsa_lg": ModelSpec(
        name="m1_hfsa_lg",
        family="custom_cnn",
        transfer_learning=False,
        description=(
            "FERLite hybrid fused CNN with "
            "ECA/spatial attention and mid-deep "
            "local-global fusion."
        ),
    ),
    "m1_rep_eca_lg": ModelSpec(
        name="m1_rep_eca_lg",
        family="custom_cnn",
        transfer_learning=False,
        description=(
            "FERLite RepConv early/mid backbone with "
            "late depthwise residuals, ECA, and "
            "local-global head."
        ),
    ),
    "m1_linear_eca_lg": ModelSpec(
        name="m1_linear_eca_lg",
        family="custom_cnn",
        transfer_learning=False,
        description=(
            "FERLite corrected linear-bottleneck backbone with "
            "downsample residuals, ECA, and local/max-global head."
        ),
    ),
    "m2_mobilenetv3_large": ModelSpec(
        name="m2_mobilenetv3_large",
        family="transfer",
        transfer_learning=True,
        description=(
            "ImageNet pretrained "
            "MobileNetV3-Large."
        ),
    ),
    "m2_efficientnet_b0": ModelSpec(
        name="m2_efficientnet_b0",
        family="transfer",
        transfer_learning=True,
        description=(
            "ImageNet pretrained "
            "EfficientNet-B0."
        ),
    ),
}


MODEL_SPECS.update({
    name: ModelSpec(
        name=name,
        family="custom_cnn",
        transfer_learning=False,
        description=config.description,
    )
    for name, config in {**P61_ABLATIONS, **P63_ABLATIONS}.items()
})

MODEL_SPECS["m1_reca_lateexp"] = ModelSpec(
    name="m1_reca_lateexp",
    family="custom_cnn",
    transfer_learning=False,
    description="Final M1: unchanged R4 topology with late expansion 2/2/3/3.",
)

MODEL_SPECS.update({
    name: ModelSpec(name=name, family="transfer", transfer_learning=True,
                    description=f"Pretrained B0 depth pruning; retained repeats {repeats}.")
    for name, repeats in EFFICIENTNET_B0_PRUNING.items()
})

MODEL_SPECS.update({
    name: ModelSpec(name=name,family="transfer",transfer_learning=True,
                    description=f"Clean P7 selective B0 original block retention: {config}.")
    for name,config in EFFICIENTNET_B0_SELECTIVE.items()
})

# Historical builders remain loadable by name, but are not offered for new
# production runs. The P6.1 A6 alias and canonical name share one builder.
ACTIVE_MODEL_NAMES = (
    "notebook_cnn_1m", "notebook_efficientnet_b0",
)

def list_models(*, include_legacy: bool = False) -> tuple[str, ...]:
    """List production models, optionally including archived experiments."""
    return tuple(MODEL_SPECS) if include_legacy else ACTIVE_MODEL_NAMES


def get_model_spec(
    name: str,
) -> ModelSpec:
    try:
        return MODEL_SPECS[name]
    except KeyError as error:
        raise KeyError(
            f"Unknown model '{name}'. "
            f"Available: {list_models()}"
        ) from error


def build_model(
    name: str,
    num_classes: int = 7,
    pretrained: bool = True,
) -> nn.Module:
    spec = get_model_spec(
        name
    )

    if name == "notebook_cnn_1m":
        return NotebookEmotionCNN(num_classes=num_classes)

    if name == "notebook_efficientnet_b0":
        return NotebookEfficientNetB0(num_classes=num_classes)

    if name in EFFICIENTNET_B0_SELECTIVE:
        return build_efficientnet_b0(num_classes,pretrained,**EFFICIENTNET_B0_SELECTIVE[name])

    if name in EFFICIENTNET_B0_PRUNING:
        return build_efficientnet_b0(num_classes, pretrained,
                                     EFFICIENTNET_B0_PRUNING[name])

    if name == "m1_reca_lateexp":
        return FERLiteRECAAblation(P61_ABLATIONS["m1_p61_a6_lateexp"], num_classes=num_classes)

    if name in P61_ABLATIONS:
        return FERLiteRECAAblation(P61_ABLATIONS[name], num_classes=num_classes)

    if name in P63_ABLATIONS:
        return FERLiteRECAAblation(P63_ABLATIONS[name], num_classes=num_classes)

    if name == "m1_reca":
        return FERLiteRECA(
            num_classes=num_classes
        )

    if name == "m1_reca_lg":
        return FERLiteRECALocalGlobal(
            num_classes=num_classes
        )

    if name == "m1_reca_stat_lg":
        return FERLiteRECAStatisticalLocalGlobal(
            num_classes=num_classes
        )

    if name == "m1_reca_sap_lg":
        return FERLiteRECASpatialAttentiveLocalGlobal(
            num_classes=num_classes
        )

    if name == "m1_partial_eca_lg":
        return FERLitePartialECALocalGlobal(
            num_classes=num_classes
        )

    if name == "m1_reca_linear_lg":
        return FERLiteRECALinearLocalGlobal(
            num_classes=num_classes
        )

    if name == "m1_res_eca_lg":
        return FERLiteResECALocalGlobal(
            num_classes=num_classes
        )

    if name == "m1_dsres_eca_lg":
        return FERLiteDSResECALocalGlobal(
            num_classes=num_classes
        )

    if name == "m1_hybridconv_eca_lg":
        return FERLiteHybridConvECALocalGlobal(
            num_classes=num_classes
        )

    if name == "m1_msca_lg":
        return FERLiteMSCALocalGlobal(
            num_classes=num_classes
        )

    if name == "m1_hfsa_lg":
        return FERLiteHFSALocalGlobal(
            num_classes=num_classes
        )

    if name == "m1_rep_eca_lg":
        return FERLiteRepECALocalGlobal(
            num_classes=num_classes
        )

    if name == "m1_linear_eca_lg":
        return FERLiteLinearECALocalGlobal(
            num_classes=num_classes
        )

    if name == "m2_mobilenetv3_large":
        return build_mobilenet_v3_large(
            num_classes=num_classes,
            pretrained=pretrained,
        )

    if name == "m2_efficientnet_b0":
        return build_efficientnet_b0(
            num_classes=num_classes,
            pretrained=pretrained,
        )

    raise RuntimeError(
        f"No builder for model: "
        f"{spec.name}"
    )


def count_parameters(
    model: nn.Module,
) -> tuple[int, int]:
    total = sum(
        parameter.numel()
        for parameter
        in model.parameters()
    )

    trainable = sum(
        parameter.numel()
        for parameter
        in model.parameters()
        if parameter.requires_grad
    )

    return total, trainable
