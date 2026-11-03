from __future__ import annotations

import torch
from torch import nn

from src.models.blocks import (
    BasicResidualBlock,
    BottleneckResidualBlock,
    ConvBNReLU,
    DepthwisePointwiseResidualBlock,
    CoordinateAttention,
    EfficientChannelAttention,
    FusedConvResidual,
    GlobalStdPool,
    InvertedDepthwiseResidual,
    LinearAddInvertedDepthwiseResidual,
    LinearBottleneckResidual,
    DownsampleLinearBottleneckResidual,
    MultiScaleInvertedDepthwiseResidual,
    PartialSpatialConv,
    RepConvBlock,
    SpatialAttentionGate,
    SpatialAttentivePool,
)


class FERLiteBackbone(nn.Module):
    """
    Lightweight FER backbone trained entirely from scratch.

    Explicitly preserves the lab-required CNN components:
    Conv -> BatchNorm -> ReLU -> Pooling -> Dropout.
    """

    output_channels = 192

    def __init__(
        self,
        feature_dropout: float = 0.10,
    ) -> None:
        super().__init__()

        self.stem = nn.Sequential(
            ConvBNReLU(
                3,
                24,
                kernel_size=3,
            ),
            nn.MaxPool2d(
                kernel_size=2,
                stride=2,
            ),
            ConvBNReLU(
                24,
                40,
                kernel_size=3,
            ),
        )

        self.stage1 = nn.Sequential(
            InvertedDepthwiseResidual(
                40,
                48,
                kernel_size=3,
            ),
            InvertedDepthwiseResidual(
                48,
                48,
                kernel_size=3,
            ),
        )

        self.stage2 = nn.Sequential(
            InvertedDepthwiseResidual(
                48,
                80,
                kernel_size=3,
                stride=2,
            ),
            InvertedDepthwiseResidual(
                80,
                80,
                kernel_size=3,
            ),
            EfficientChannelAttention(
                80
            ),
        )

        self.stage3 = nn.Sequential(
            InvertedDepthwiseResidual(
                80,
                128,
                kernel_size=5,
                stride=2,
            ),
            InvertedDepthwiseResidual(
                128,
                128,
                kernel_size=5,
            ),
            EfficientChannelAttention(
                128
            ),
        )

        self.stage4 = nn.Sequential(
            InvertedDepthwiseResidual(
                128,
                192,
                kernel_size=5,
                stride=2,
            ),
            InvertedDepthwiseResidual(
                192,
                192,
                kernel_size=5,
            ),
            EfficientChannelAttention(
                192
            ),
        )

        self.feature_dropout = (
            nn.Dropout2d(
                feature_dropout
            )
        )

        self._initialize_weights()

    def _initialize_weights(
        self,
    ) -> None:
        for module in self.modules():
            if isinstance(
                module,
                nn.Conv2d,
            ):
                nn.init.kaiming_normal_(
                    module.weight,
                    mode="fan_out",
                    nonlinearity="relu",
                )

            elif isinstance(
                module,
                nn.BatchNorm2d,
            ):
                nn.init.ones_(
                    module.weight
                )
                nn.init.zeros_(
                    module.bias
                )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)

        return self.feature_dropout(
            x
        )


class FERLiteRECA(nn.Module):
    """M1-A: residual + depthwise + ECA + global pooling."""

    def __init__(
        self,
        num_classes: int = 7,
        classifier_dropout: float = 0.35,
    ) -> None:
        super().__init__()

        self.backbone = FERLiteBackbone()

        self.pool = nn.AdaptiveAvgPool2d(
            1
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(
                classifier_dropout
            ),
            nn.Linear(
                self.backbone.output_channels,
                num_classes,
            ),
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        features = self.backbone(x)
        features = self.pool(features)

        return self.classifier(
            features
        )


class FERLiteRECALocalGlobal(nn.Module):
    """
    M1-B: same backbone as M1-A.

    Adds a lightweight 2x2 local representation beside global pooling.
    """

    def __init__(
        self,
        num_classes: int = 7,
        classifier_dropout: float = 0.35,
    ) -> None:
        super().__init__()

        self.backbone = FERLiteBackbone()

        channels = (
            self.backbone.output_channels
        )

        self.global_pool = (
            nn.AdaptiveAvgPool2d(1)
        )

        self.local_pool = (
            nn.AdaptiveAvgPool2d(
                (2, 2)
            )
        )

        classifier_features = (
            channels
            + channels * 4
        )

        self.classifier = nn.Sequential(
            nn.Dropout(
                classifier_dropout
            ),
            nn.Linear(
                classifier_features,
                num_classes,
            ),
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        features = self.backbone(x)

        global_features = (
            self.global_pool(features)
            .flatten(1)
        )

        local_features = (
            self.local_pool(features)
            .flatten(1)
        )

        fused = torch.cat(
            [
                global_features,
                local_features,
            ],
            dim=1,
        )

        return self.classifier(
            fused
        )


class FERLiteRECAStatisticalLocalGlobal(FERLiteRECALocalGlobal):
    """R4 head plus non-learned second-order spatial activation statistics.

    Retain the exact backbone and both original mean descriptors. Append
    population standard deviation per channel, widening only Linear 5C->6C.
    """

    def __init__(
        self,
        num_classes: int = 7,
        classifier_dropout: float = 0.35,
    ) -> None:
        if num_classes != 7:
            raise ValueError("m1_reca_stat_lg requires exactly 7 classes")
        super().__init__(
            num_classes=num_classes,
            classifier_dropout=classifier_dropout,
        )
        self.std_pool = GlobalStdPool()
        self.classifier[1] = nn.Linear(self.backbone.output_channels * 6, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        fused = torch.cat(
            [
                self.global_pool(features).flatten(1),
                self.local_pool(features).flatten(1),
                self.std_pool(features),
            ],
            dim=1,
        )
        return self.classifier(fused)


class FERLiteRECASpatialAttentiveLocalGlobal(FERLiteRECALocalGlobal):
    """R4 head-aggregation ablation with an input-dependent global descriptor.

    Reuse the exact R4 backbone, local average pooling, dropout, 960-feature
    classifier and global/local concatenation order. Only global pooling
    changes; SpatialAttentivePool already returns BC, so inherited flatten(1)
    preserves that descriptor.
    """

    def __init__(
        self,
        num_classes: int = 7,
        classifier_dropout: float = 0.35,
    ) -> None:
        if num_classes != 7:
            raise ValueError("m1_reca_sap_lg requires exactly 7 classes")
        super().__init__(
            num_classes=num_classes,
            classifier_dropout=classifier_dropout,
        )
        self.global_pool = SpatialAttentivePool(self.backbone.output_channels)


class FERLiteRECALinearBackbone(nn.Module):
    """
    Surgical R4 backbone variant.

    Everything is kept identical to ``FERLiteBackbone`` except the
    same-shape residual blocks use ``LinearAddInvertedDepthwiseResidual``.
    Stage-transition/channel-changing blocks retain the original R4 output
    ReLU, so this experiment isolates post-residual activation clipping.
    """

    output_channels = 192

    def __init__(
        self,
        feature_dropout: float = 0.10,
    ) -> None:
        super().__init__()

        self.stem = nn.Sequential(
            ConvBNReLU(
                3,
                24,
                kernel_size=3,
            ),
            nn.MaxPool2d(
                kernel_size=2,
                stride=2,
            ),
            ConvBNReLU(
                24,
                40,
                kernel_size=3,
            ),
        )

        self.stage1 = nn.Sequential(
            LinearAddInvertedDepthwiseResidual(
                40,
                48,
                kernel_size=3,
            ),
            LinearAddInvertedDepthwiseResidual(
                48,
                48,
                kernel_size=3,
            ),
        )

        self.stage2 = nn.Sequential(
            LinearAddInvertedDepthwiseResidual(
                48,
                80,
                kernel_size=3,
                stride=2,
            ),
            LinearAddInvertedDepthwiseResidual(
                80,
                80,
                kernel_size=3,
            ),
            EfficientChannelAttention(
                80
            ),
        )

        self.stage3 = nn.Sequential(
            LinearAddInvertedDepthwiseResidual(
                80,
                128,
                kernel_size=5,
                stride=2,
            ),
            LinearAddInvertedDepthwiseResidual(
                128,
                128,
                kernel_size=5,
            ),
            EfficientChannelAttention(
                128
            ),
        )

        self.stage4 = nn.Sequential(
            LinearAddInvertedDepthwiseResidual(
                128,
                192,
                kernel_size=5,
                stride=2,
            ),
            LinearAddInvertedDepthwiseResidual(
                192,
                192,
                kernel_size=5,
            ),
            EfficientChannelAttention(
                192
            ),
        )

        self.feature_dropout = nn.Dropout2d(
            feature_dropout
        )

        self._initialize_weights()

    def _initialize_weights(
        self,
    ) -> None:
        for module in self.modules():
            if isinstance(
                module,
                nn.Conv2d,
            ):
                nn.init.kaiming_normal_(
                    module.weight,
                    mode="fan_out",
                    nonlinearity="relu",
                )

            elif isinstance(
                module,
                nn.BatchNorm2d,
            ):
                nn.init.ones_(
                    module.weight
                )
                nn.init.zeros_(
                    module.bias
                )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)

        return self.feature_dropout(x)


class FERLiteRECALinearLocalGlobal(nn.Module):
    """
    R4 local-global classifier with only post-residual ReLU removed.

    Head, ECA placement, widths, kernels, pooling, dropout, and classifier are
    intentionally identical to ``FERLiteRECALocalGlobal``.
    """

    def __init__(
        self,
        num_classes: int = 7,
        classifier_dropout: float = 0.35,
    ) -> None:
        super().__init__()

        self.backbone = FERLiteRECALinearBackbone()

        channels = self.backbone.output_channels

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.local_pool = nn.AdaptiveAvgPool2d((2, 2))

        self.classifier = nn.Sequential(
            nn.Dropout(classifier_dropout),
            nn.Linear(
                channels * 5,
                num_classes,
            ),
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        features = self.backbone(x)

        global_features = (
            self.global_pool(features)
            .flatten(1)
        )

        local_features = (
            self.local_pool(features)
            .flatten(1)
        )

        fused = torch.cat(
            [
                global_features,
                local_features,
            ],
            dim=1,
        )

        return self.classifier(fused)


class FERLiteMSCABackbone(nn.Module):
    """
    M1-v2 backbone: multi-scale depthwise features + coordinate attention.

    The high-resolution early stages stay close to the proven RECA backbone.
    Multi-scale depthwise branches are introduced only in the later stages,
    where larger receptive fields are useful and computational cost is lower.
    """

    output_channels = 192

    def __init__(
        self,
        feature_dropout: float = 0.10,
    ) -> None:
        super().__init__()

        self.stem = nn.Sequential(
            ConvBNReLU(
                3,
                24,
                kernel_size=3,
            ),
            nn.MaxPool2d(
                kernel_size=2,
                stride=2,
            ),
            ConvBNReLU(
                24,
                40,
                kernel_size=3,
            ),
        )

        self.stage1 = nn.Sequential(
            InvertedDepthwiseResidual(
                40,
                48,
                kernel_size=3,
            ),
            InvertedDepthwiseResidual(
                48,
                48,
                kernel_size=3,
            ),
        )

        self.stage2 = nn.Sequential(
            InvertedDepthwiseResidual(
                48,
                80,
                kernel_size=3,
                stride=2,
            ),
            InvertedDepthwiseResidual(
                80,
                80,
                kernel_size=3,
            ),
            CoordinateAttention(80),
        )

        self.stage3 = nn.Sequential(
            MultiScaleInvertedDepthwiseResidual(
                80,
                128,
                kernel_sizes=(3, 5),
                stride=2,
            ),
            MultiScaleInvertedDepthwiseResidual(
                128,
                128,
                kernel_sizes=(3, 5),
            ),
            CoordinateAttention(128),
        )

        self.stage4 = nn.Sequential(
            MultiScaleInvertedDepthwiseResidual(
                128,
                192,
                kernel_sizes=(5, 7),
                stride=2,
            ),
            MultiScaleInvertedDepthwiseResidual(
                192,
                192,
                kernel_sizes=(5, 7),
            ),
            CoordinateAttention(192),
        )

        self.feature_dropout = nn.Dropout2d(
            feature_dropout
        )

        self._initialize_weights()

    def _initialize_weights(
        self,
    ) -> None:
        for module in self.modules():
            if isinstance(
                module,
                nn.Conv2d,
            ):
                nn.init.kaiming_normal_(
                    module.weight,
                    mode="fan_out",
                    nonlinearity="relu",
                )

                if module.bias is not None:
                    nn.init.zeros_(
                        module.bias
                    )

            elif isinstance(
                module,
                nn.BatchNorm2d,
            ):
                nn.init.ones_(
                    module.weight
                )
                nn.init.zeros_(
                    module.bias
                )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)

        return self.feature_dropout(x)


class FERLiteMSCALocalGlobal(nn.Module):
    """
    M1-v2: multi-scale DWConv + coordinate attention + local/global head.
    """

    def __init__(
        self,
        num_classes: int = 7,
        classifier_dropout: float = 0.35,
    ) -> None:
        super().__init__()

        self.backbone = FERLiteMSCABackbone()

        channels = self.backbone.output_channels

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.local_pool = nn.AdaptiveAvgPool2d((2, 2))

        self.classifier = nn.Sequential(
            nn.Dropout(classifier_dropout),
            nn.Linear(
                channels * 5,
                num_classes,
            ),
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        features = self.backbone(x)

        global_features = (
            self.global_pool(features)
            .flatten(1)
        )

        local_features = (
            self.local_pool(features)
            .flatten(1)
        )

        fused = torch.cat(
            [
                global_features,
                local_features,
            ],
            dim=1,
        )

        return self.classifier(fused)


class FERLiteHFSABackbone(nn.Module):
    """
    Hybrid Fused + Spatial Attention backbone for M1.

    The design keeps the proven early RECA stage, uses one fused dense
    transition before the depthwise-heavy stages, keeps ECA, adds a tiny
    spatial gate, and enlarges late receptive fields with dilation.
    """

    output_channels = 184
    mid_channels = 96

    def __init__(
        self,
        feature_dropout: float = 0.10,
    ) -> None:
        super().__init__()

        self.stem = nn.Sequential(
            ConvBNReLU(
                3,
                24,
                kernel_size=3,
            ),
            nn.MaxPool2d(
                kernel_size=2,
                stride=2,
            ),
            ConvBNReLU(
                24,
                40,
                kernel_size=3,
            ),
        )

        self.stage1 = nn.Sequential(
            InvertedDepthwiseResidual(
                40,
                48,
                kernel_size=3,
            ),
            InvertedDepthwiseResidual(
                48,
                48,
                kernel_size=3,
            ),
        )

        self.stage2 = nn.Sequential(
            FusedConvResidual(
                48,
                self.mid_channels,
                stride=2,
                expansion=1.5,
            ),
            InvertedDepthwiseResidual(
                self.mid_channels,
                self.mid_channels,
                kernel_size=3,
            ),
            EfficientChannelAttention(
                self.mid_channels
            ),
            SpatialAttentionGate(
                kernel_size=7
            ),
        )

        self.stage3 = nn.Sequential(
            InvertedDepthwiseResidual(
                self.mid_channels,
                144,
                kernel_size=3,
                stride=2,
            ),
            InvertedDepthwiseResidual(
                144,
                144,
                kernel_size=3,
                dilation=2,
            ),
            EfficientChannelAttention(144),
            SpatialAttentionGate(
                kernel_size=7
            ),
        )

        self.stage4 = nn.Sequential(
            InvertedDepthwiseResidual(
                144,
                self.output_channels,
                kernel_size=3,
                stride=2,
            ),
            InvertedDepthwiseResidual(
                self.output_channels,
                self.output_channels,
                kernel_size=3,
                dilation=2,
            ),
            EfficientChannelAttention(
                self.output_channels
            ),
            SpatialAttentionGate(
                kernel_size=7
            ),
        )

        self.feature_dropout = nn.Dropout2d(
            feature_dropout
        )

        self._initialize_weights()

    def _initialize_weights(
        self,
    ) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight,
                    mode="fan_out",
                    nonlinearity="relu",
                )

                if module.bias is not None:
                    nn.init.zeros_(
                        module.bias
                    )

            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(
                    module.weight
                )
                nn.init.zeros_(
                    module.bias
                )

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.stem(x)
        x = self.stage1(x)

        mid_features = self.stage2(x)

        x = self.stage3(mid_features)
        x = self.stage4(x)
        deep_features = self.feature_dropout(x)

        return mid_features, deep_features


class FERLiteHFSALocalGlobal(nn.Module):
    """
    M1-HFSA-LG: fused transition + ECA/spatial attention + mid/deep fusion.
    """

    def __init__(
        self,
        num_classes: int = 7,
        classifier_dropout: float = 0.35,
        mid_projection_channels: int = 40,
    ) -> None:
        super().__init__()

        self.backbone = FERLiteHFSABackbone()

        self.mid_projection = ConvBNReLU(
            self.backbone.mid_channels,
            mid_projection_channels,
            kernel_size=1,
        )

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.deep_local_pool = nn.AdaptiveAvgPool2d(
            (2, 2)
        )
        self.mid_local_pool = nn.AdaptiveAvgPool2d(
            (2, 2)
        )

        classifier_features = (
            self.backbone.output_channels
            + self.backbone.output_channels * 4
            + mid_projection_channels * 4
        )

        self.classifier = nn.Sequential(
            nn.Dropout(
                classifier_dropout
            ),
            nn.Linear(
                classifier_features,
                num_classes,
            ),
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        mid_features, deep_features = self.backbone(x)

        global_features = (
            self.global_pool(deep_features)
            .flatten(1)
        )

        deep_local_features = (
            self.deep_local_pool(deep_features)
            .flatten(1)
        )

        mid_local_features = (
            self.mid_local_pool(
                self.mid_projection(mid_features)
            )
            .flatten(1)
        )

        fused = torch.cat(
            [
                global_features,
                deep_local_features,
                mid_local_features,
            ],
            dim=1,
        )

        return self.classifier(fused)


class FERLiteRepECABackbone(nn.Module):
    """
    M1 Rep-ECA backbone.

    Dense RepConv blocks are used only in early/middle stages to improve
    joint spatial/channel feature learning during optimization. Later stages
    return to depthwise-pointwise residual blocks to preserve lightweight
    deployment. ECA is retained because it is already part of the strongest
    RECA baseline.
    """

    output_channels = 208

    def __init__(
        self,
        feature_dropout: float = 0.10,
    ) -> None:
        super().__init__()

        self.stem = nn.Sequential(
            ConvBNReLU(
                3,
                24,
                kernel_size=3,
            ),
            nn.MaxPool2d(
                kernel_size=2,
                stride=2,
            ),
            ConvBNReLU(
                24,
                40,
                kernel_size=3,
            ),
        )

        self.stage1 = nn.Sequential(
            RepConvBlock(
                40,
                56,
            ),
            RepConvBlock(
                56,
                56,
            ),
        )

        self.stage2 = nn.Sequential(
            RepConvBlock(
                56,
                96,
                stride=2,
            ),
            RepConvBlock(
                96,
                96,
            ),
            EfficientChannelAttention(96),
        )

        self.stage3 = nn.Sequential(
            InvertedDepthwiseResidual(
                96,
                144,
                kernel_size=3,
                stride=2,
            ),
            InvertedDepthwiseResidual(
                144,
                144,
                kernel_size=3,
            ),
            EfficientChannelAttention(144),
        )

        self.stage4 = nn.Sequential(
            InvertedDepthwiseResidual(
                144,
                self.output_channels,
                kernel_size=5,
                stride=2,
            ),
            InvertedDepthwiseResidual(
                self.output_channels,
                self.output_channels,
                kernel_size=5,
            ),
            EfficientChannelAttention(
                self.output_channels
            ),
        )

        self.feature_dropout = nn.Dropout2d(
            feature_dropout
        )

        self._initialize_weights()

    def _initialize_weights(
        self,
    ) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight,
                    mode="fan_out",
                    nonlinearity="relu",
                )

                if module.bias is not None:
                    nn.init.zeros_(
                        module.bias
                    )

            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(
                    module.weight
                )
                nn.init.zeros_(
                    module.bias
                )

    def switch_to_deploy(
        self,
    ) -> "FERLiteRepECABackbone":
        for module in list(self.modules()):
            if isinstance(module, RepConvBlock):
                module.switch_to_deploy()

        return self

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)

        return self.feature_dropout(x)


class FERLiteRepECALocalGlobal(nn.Module):
    """
    M1 Rep-ECA-LG: training-time RepConv + ECA + local/global head.

    Rep branches can be fused after training via ``switch_to_deploy`` without
    changing the mathematical output in eval mode apart from floating-point
    roundoff.
    """

    def __init__(
        self,
        num_classes: int = 7,
        classifier_dropout: float = 0.35,
    ) -> None:
        super().__init__()

        self.backbone = FERLiteRepECABackbone()

        channels = self.backbone.output_channels

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.local_pool = nn.AdaptiveAvgPool2d((2, 2))

        self.classifier = nn.Sequential(
            nn.Dropout(
                classifier_dropout
            ),
            nn.Linear(
                channels * 5,
                num_classes,
            ),
        )

    def switch_to_deploy(
        self,
    ) -> "FERLiteRepECALocalGlobal":
        self.backbone.switch_to_deploy()
        return self

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        features = self.backbone(x)

        global_features = (
            self.global_pool(features)
            .flatten(1)
        )

        local_features = (
            self.local_pool(features)
            .flatten(1)
        )

        fused = torch.cat(
            [
                global_features,
                local_features,
            ],
            dim=1,
        )

        return self.classifier(fused)


class FERLiteLinearECABackbone(nn.Module):
    """
    Corrected mobile-style FER backbone.

    Compared with the RECA baseline this backbone fixes four structural
    mismatches while keeping the successful channel schedule and ECA idea:

    1. learned stride-2 stem instead of a hard MaxPool transition;
    2. true linear bottlenecks (no ReLU after the narrow projection);
    3. residual information paths across stride-2 stage transitions;
    4. larger expansion only at lower spatial resolutions, where it is cheap.
    """

    output_channels = 192

    def __init__(
        self,
        feature_dropout: float = 0.10,
    ) -> None:
        super().__init__()

        # Keep the first high-resolution feature extraction, but make the
        # 160->80 reduction learnable instead of MaxPool-based.
        self.stem = nn.Sequential(
            ConvBNReLU(
                3,
                24,
                kernel_size=3,
            ),
            ConvBNReLU(
                24,
                40,
                kernel_size=3,
                stride=2,
            ),
        )

        # High-resolution stage: modest expansion to limit compute.
        self.stage1 = nn.Sequential(
            LinearBottleneckResidual(
                40,
                48,
                kernel_size=3,
                expansion=2.0,
            ),
            LinearBottleneckResidual(
                48,
                48,
                kernel_size=3,
                expansion=2.0,
            ),
        )

        self.stage2 = nn.Sequential(
            DownsampleLinearBottleneckResidual(
                48,
                80,
                kernel_size=3,
                expansion=2.0,
            ),
            LinearBottleneckResidual(
                80,
                80,
                kernel_size=3,
                expansion=2.0,
            ),
            EfficientChannelAttention(80),
        )

        # Later low-resolution stages can afford a wider expansion space.
        self.stage3 = nn.Sequential(
            DownsampleLinearBottleneckResidual(
                80,
                128,
                kernel_size=5,
                expansion=3.0,
            ),
            LinearBottleneckResidual(
                128,
                128,
                kernel_size=5,
                expansion=3.0,
            ),
            EfficientChannelAttention(128),
        )

        self.stage4 = nn.Sequential(
            DownsampleLinearBottleneckResidual(
                128,
                self.output_channels,
                kernel_size=5,
                expansion=3.0,
            ),
            LinearBottleneckResidual(
                self.output_channels,
                self.output_channels,
                kernel_size=5,
                expansion=3.0,
            ),
            EfficientChannelAttention(
                self.output_channels
            ),
        )

        self.feature_dropout = nn.Dropout2d(
            feature_dropout
        )

        self._initialize_weights()

    def _initialize_weights(
        self,
    ) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight,
                    mode="fan_out",
                    nonlinearity="relu",
                )
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        return self.feature_dropout(x)


class FERLiteLinearECALocalGlobal(nn.Module):
    """
    M1 Linear-ECA-LG candidate.

    The 2x2 local-average descriptor is retained because it preserves coarse
    facial layout. The old global-average descriptor is replaced by global
    max pooling: global average is already the mean of the four equally sized
    2x2 local-average bins for the final 10x10 map and is therefore redundant.
    Global max adds a complementary peak-response statistic at exactly the
    same classifier dimensionality as the original head.
    """

    def __init__(
        self,
        num_classes: int = 7,
        classifier_dropout: float = 0.35,
    ) -> None:
        super().__init__()

        self.backbone = FERLiteLinearECABackbone()
        channels = self.backbone.output_channels

        self.local_pool = nn.AdaptiveAvgPool2d(
            (2, 2)
        )
        self.global_pool = nn.AdaptiveMaxPool2d(
            1
        )

        self.classifier = nn.Sequential(
            nn.Dropout(
                classifier_dropout
            ),
            nn.Linear(
                channels * 5,
                num_classes,
            ),
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        features = self.backbone(x)

        local_features = (
            self.local_pool(features)
            .flatten(1)
        )
        global_features = (
            self.global_pool(features)
            .flatten(1)
        )

        fused = torch.cat(
            [
                local_features,
                global_features,
            ],
            dim=1,
        )

        return self.classifier(fused)


class _FERLiteResidualFamilyBackbone(nn.Module):
    """
    Shared topology for the two controlled residual-family candidates.

    The experiment keeps stage widths, downsampling, ECA placement, dropout,
    and classifier input contract identical.  The only intended difference is
    the residual branch primitive supplied through ``block_type``:

    - ``BasicResidualBlock``: two full 3x3 convolutions;
    - ``DepthwisePointwiseResidualBlock``: each 3x3 is factorized into
      depthwise 3x3 + pointwise 1x1.

    Capacity is deliberately concentrated after the first downsampling stages
    to avoid spending most compute on the 80x80 feature map.
    """

    output_channels = 120

    def __init__(
        self,
        block_type: type[nn.Module],
        feature_dropout: float = 0.10,
        zero_init_residual: bool = True,
    ) -> None:
        super().__init__()

        self.stem = nn.Sequential(
            ConvBNReLU(
                3,
                24,
                kernel_size=3,
            ),
            nn.MaxPool2d(
                kernel_size=2,
                stride=2,
            ),
            ConvBNReLU(
                24,
                32,
                kernel_size=3,
            ),
        )

        # 80x80: only one residual block.  Keep early compute conservative.
        self.stage1 = nn.Sequential(
            block_type(
                32,
                40,
                stride=1,
            ),
        )

        # 40x40: first semantic stage and first channel-attention gate.
        self.stage2 = nn.Sequential(
            block_type(
                40,
                64,
                stride=2,
            ),
            EfficientChannelAttention(64),
        )

        # 20x20: add one same-shape residual refinement block.
        self.stage3 = nn.Sequential(
            block_type(
                64,
                88,
                stride=2,
            ),
            block_type(
                88,
                88,
                stride=1,
            ),
            EfficientChannelAttention(88),
        )

        # 10x10: richest stage, still below the 1M-parameter project budget.
        self.stage4 = nn.Sequential(
            block_type(
                88,
                self.output_channels,
                stride=2,
            ),
            block_type(
                self.output_channels,
                self.output_channels,
                stride=1,
            ),
            EfficientChannelAttention(
                self.output_channels
            ),
        )

        self.feature_dropout = nn.Dropout2d(
            feature_dropout
        )

        self._initialize_weights(
            zero_init_residual=zero_init_residual
        )

    def _initialize_weights(
        self,
        zero_init_residual: bool,
    ) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight,
                    mode="fan_out",
                    nonlinearity="relu",
                )
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

        # Standard residual-network initialization trick: start each residual
        # branch close to zero so the block initially behaves like its
        # shortcut.  Apply it identically to both candidate families.
        if zero_init_residual:
            for module in self.modules():
                if isinstance(
                    module,
                    (
                        BasicResidualBlock,
    BottleneckResidualBlock,
                        DepthwisePointwiseResidualBlock,
                    ),
                ):
                    nn.init.zeros_(
                        module.final_bn.weight
                    )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        return self.feature_dropout(x)


class FERLiteResECABackbone(
    _FERLiteResidualFamilyBackbone
):
    """Full-convolution shallow ResNet-ECA FER backbone."""

    def __init__(
        self,
        feature_dropout: float = 0.10,
    ) -> None:
        super().__init__(
            block_type=BasicResidualBlock,
            feature_dropout=feature_dropout,
        )


class FERLiteDSResECABackbone(
    _FERLiteResidualFamilyBackbone
):
    """Depthwise-pointwise version of the same residual topology."""

    def __init__(
        self,
        feature_dropout: float = 0.10,
    ) -> None:
        super().__init__(
            block_type=(
                DepthwisePointwiseResidualBlock
            ),
            feature_dropout=feature_dropout,
        )


class _FERLiteResidualLocalGlobal(nn.Module):
    """Shared R4-compatible local/global average-pooling classifier head."""

    def __init__(
        self,
        backbone: nn.Module,
        num_classes: int = 7,
        classifier_dropout: float = 0.35,
    ) -> None:
        super().__init__()

        self.backbone = backbone
        channels = self.backbone.output_channels

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.local_pool = nn.AdaptiveAvgPool2d(
            (2, 2)
        )

        self.classifier = nn.Sequential(
            nn.Dropout(classifier_dropout),
            nn.Linear(
                channels * 5,
                num_classes,
            ),
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        features = self.backbone(x)

        global_features = (
            self.global_pool(features)
            .flatten(1)
        )
        local_features = (
            self.local_pool(features)
            .flatten(1)
        )

        fused = torch.cat(
            [
                global_features,
                local_features,
            ],
            dim=1,
        )

        return self.classifier(fused)


class FERLitePartialBackbone(FERLiteBackbone):
    """R4 with one partial spatial operator in stage 3 refinement.

    Keep expanded width 256 and the original 5x5 receptive field. Only 64
    hidden channels use dense mixing; the other 192 retain depthwise mixing.
    The inherited block preserves BN/ReLU, pointwise projection, identity
    shortcut and post-add ReLU, as do every stage transition and ECA gate.
    """

    def __init__(self, feature_dropout: float = 0.10) -> None:
        super().__init__(feature_dropout=feature_dropout)
        spatial = PartialSpatialConv(
            channels=256,
            dense_channels=64,
            kernel_size=5,
        )
        # Initialize only the replacement with R4's Conv2d policy; the
        # inherited modules already received their usual initialization.
        for module in spatial.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_out", nonlinearity="relu"
                )
        self.stage3[1].block[1][0] = spatial


class FERLitePartialECALocalGlobal(_FERLiteResidualLocalGlobal):
    """One-operator partial-mixing ablation of R4, trained from scratch."""

    def __init__(
        self,
        num_classes: int = 7,
        classifier_dropout: float = 0.35,
    ) -> None:
        if num_classes != 7:
            raise ValueError("m1_partial_eca_lg requires exactly 7 classes")
        super().__init__(
            backbone=FERLitePartialBackbone(),
            num_classes=num_classes,
            classifier_dropout=classifier_dropout,
        )


class FERLiteResECALocalGlobal(
    _FERLiteResidualLocalGlobal
):
    """
    M1 Res-ECA-LG.

    Shallow custom ResNet using full 3x3 convolutions, ECA at the lower
    resolutions, and the already-proven local/global average-pooling head.
    """

    def __init__(
        self,
        num_classes: int = 7,
        classifier_dropout: float = 0.35,
    ) -> None:
        super().__init__(
            backbone=FERLiteResECABackbone(),
            num_classes=num_classes,
            classifier_dropout=(
                classifier_dropout
            ),
        )


class FERLiteDSResECALocalGlobal(
    _FERLiteResidualLocalGlobal
):
    """
    M1 DSRes-ECA-LG.

    Controlled parameter-efficient counterpart of Res-ECA-LG.  It uses the
    same stem, stage widths, shortcuts, ECA placement, pooling, and classifier;
    residual-branch full 3x3 convolutions are the only component replaced by
    depthwise 3x3 + pointwise 1x1 factorization.
    """

    def __init__(
        self,
        num_classes: int = 7,
        classifier_dropout: float = 0.35,
    ) -> None:
        super().__init__(
            backbone=FERLiteDSResECABackbone(),
            num_classes=num_classes,
            classifier_dropout=(
                classifier_dropout
            ),
        )


class FERLiteHybridConvBackbone(nn.Module):
    """
    R4-preserving hybrid backbone with richer late semantic mixing.

    Stages 1-2 are copied from the strongest R4 backbone.  Only stages 3-4
    replace depthwise-dominated inverted residuals with bottleneck residual
    blocks whose central 3x3 convolution is a *full* convolution.  This keeps
    the successful early inductive bias and local/global head while spending
    extra capacity only at 20x20 and 10x10.
    """

    output_channels = 192

    def __init__(
        self,
        feature_dropout: float = 0.10,
    ) -> None:
        super().__init__()

        # Exact R4 stem.
        self.stem = nn.Sequential(
            ConvBNReLU(3, 24, kernel_size=3),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ConvBNReLU(24, 40, kernel_size=3),
        )

        # Exact R4 early stage.
        self.stage1 = nn.Sequential(
            InvertedDepthwiseResidual(40, 48, kernel_size=3),
            InvertedDepthwiseResidual(48, 48, kernel_size=3),
        )

        # Exact R4 mid stage and ECA.
        self.stage2 = nn.Sequential(
            InvertedDepthwiseResidual(
                48, 80, kernel_size=3, stride=2
            ),
            InvertedDepthwiseResidual(80, 80, kernel_size=3),
            EfficientChannelAttention(80),
        )

        # 20x20 semantic stage: full 3x3 mixing at reduced width.
        self.stage3 = nn.Sequential(
            BottleneckResidualBlock(
                80, 128, bottleneck_channels=64, stride=2
            ),
            BottleneckResidualBlock(
                128, 128, bottleneck_channels=64, stride=1
            ),
            EfficientChannelAttention(128),
        )

        # 10x10 semantic stage: wider bottleneck, still far cheaper than
        # two full 192->192 3x3 convolutions.
        self.stage4 = nn.Sequential(
            BottleneckResidualBlock(
                128, 192, bottleneck_channels=96, stride=2
            ),
            BottleneckResidualBlock(
                192, 192, bottleneck_channels=96, stride=1
            ),
            EfficientChannelAttention(192),
        )

        self.feature_dropout = nn.Dropout2d(feature_dropout)
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight,
                    mode="fan_out",
                    nonlinearity="relu",
                )
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        return self.feature_dropout(x)


class FERLiteHybridConvECALocalGlobal(nn.Module):
    """
    Hybrid R4/ResNet candidate.

    Keeps the proven R4 stem, early mobile blocks, ECA placement, and
    local/global average-pooling head.  Only the 20x20 and 10x10 semantic
    stages use full-convolution bottleneck residual blocks.
    """

    def __init__(
        self,
        num_classes: int = 7,
        classifier_dropout: float = 0.35,
    ) -> None:
        super().__init__()

        self.backbone = FERLiteHybridConvBackbone()
        channels = self.backbone.output_channels

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.local_pool = nn.AdaptiveAvgPool2d((2, 2))

        self.classifier = nn.Sequential(
            nn.Dropout(classifier_dropout),
            nn.Linear(channels * 5, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        global_features = self.global_pool(features).flatten(1)
        local_features = self.local_pool(features).flatten(1)
        return self.classifier(
            torch.cat([global_features, local_features], dim=1)
        )
