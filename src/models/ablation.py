"""Frozen, controlled R4 configurations for architecture ablation screening."""

from dataclasses import dataclass

import torch
from torch import nn

from src.models.blocks import InvertedDepthwiseResidual
from src.models.ferlite import FERLiteBackbone


@dataclass(frozen=True)
class RECAAblationConfig:
    description: str
    eca_stages: tuple[int, ...] = (2, 3, 4)
    late_expansion: float = 2.0
    extra_stage4_blocks: int = 0
    local_grid: int = 2
    include_global: bool = True
    stage3_expansion: float | None = None
    stage4_expansion: float | None = None


P61_ABLATIONS = {
    "m1_p61_a0_r4": RECAAblationConfig("Exact R4 architecture control"),
    "m1_p61_a1_noeca": RECAAblationConfig("Remove all ECA", eca_stages=()),
    "m1_p61_a2_eca34": RECAAblationConfig("ECA only in stages 3/4", eca_stages=(3, 4)),
    "m1_p61_a3_eca4": RECAAblationConfig("ECA only in stage 4", eca_stages=(4,)),
    "m1_p61_a4_localonly": RECAAblationConfig("Local 2x2 only", include_global=False),
    "m1_p61_a5_local3": RECAAblationConfig("Local 3x3 plus global mean", local_grid=3),
    "m1_p61_a6_lateexp": RECAAblationConfig("Expansion 3 in stages 3/4", late_expansion=3.0),
    "m1_p61_a7_stage4deep": RECAAblationConfig("One extra stage 4 identity block", extra_stage4_blocks=1),
}


P63_ABLATIONS = {
    "m1_p63_b1_e32": RECAAblationConfig(
        "Stage3/4 expansion 3/2", stage3_expansion=3.0, stage4_expansion=2.0,
    ),
    "m1_p63_b2_e23": RECAAblationConfig(
        "Stage3/4 expansion 2/3", stage3_expansion=2.0, stage4_expansion=3.0,
    ),
    "m1_p63_b4_e43": RECAAblationConfig(
        "Stage3/4 expansion 4/3", stage3_expansion=4.0, stage4_expansion=3.0,
    ),
}


class FERLiteAblationBackbone(FERLiteBackbone):
    """Reuse R4 and modify only the factor specified by the frozen config."""

    def __init__(self, config: RECAAblationConfig) -> None:
        super().__init__()
        for stage in (2, 3, 4):
            if stage not in config.eca_stages:
                getattr(self, f"stage{stage}")[2] = nn.Identity()

        new_blocks = []
        for stage, input_channels, output_channels in ((3, 80, 128), (4, 128, 192)):
            override = getattr(config, f"stage{stage}_expansion")
            expansion = config.late_expansion if override is None else override
            if expansion != 2.0:
                sequence = getattr(self, f"stage{stage}")
                for index, cin, stride in ((0, input_channels, 2), (1, output_channels, 1)):
                    block = InvertedDepthwiseResidual(
                        cin, output_channels, kernel_size=5, stride=stride,
                        expansion=expansion,
                    )
                    sequence[index] = block
                    new_blocks.append(block)

        if config.extra_stage4_blocks:
            blocks = list(self.stage4.children())
            gate = blocks.pop()
            for _ in range(config.extra_stage4_blocks):
                block = InvertedDepthwiseResidual(192, 192, kernel_size=5, expansion=2.0)
                blocks.append(block)
                new_blocks.append(block)
            self.stage4 = nn.Sequential(*blocks, gate)

        # Match R4's policy only for newly introduced blocks. Retain already
        # initialized shared modules, including stem and unchanged stages.
        for block in new_blocks:
            for module in block.modules():
                if isinstance(module, nn.Conv2d):
                    nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                elif isinstance(module, nn.BatchNorm2d):
                    nn.init.ones_(module.weight)
                    nn.init.zeros_(module.bias)


class FERLiteRECAAblation(nn.Module):
    """Configurable R4 mean-pooling head; no new pooling or attention primitive."""

    def __init__(self, config: RECAAblationConfig, num_classes: int = 7) -> None:
        super().__init__()
        if num_classes != 7:
            raise ValueError("P6.1 FER ablations require exactly 7 classes")
        self.include_global = config.include_global
        self.backbone = FERLiteAblationBackbone(config)
        self.global_pool = nn.AdaptiveAvgPool2d(1) if self.include_global else nn.Identity()
        self.local_pool = nn.AdaptiveAvgPool2d((config.local_grid, config.local_grid))
        features = self.backbone.output_channels * (config.local_grid**2 + int(self.include_global))
        self.classifier = nn.Sequential(nn.Dropout(0.35), nn.Linear(features, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        if torch.onnx.is_in_onnx_export() and self.local_pool.output_size == (3, 3):
            # Legacy ONNX cannot lower non-divisible adaptive pooling. Unroll
            # the nine exact adaptive bins in the exported graph instead.
            # These loops enumerate output regions, never samples or pixels;
            # eager training still uses the original AdaptiveAvgPool2d.
            height, width = features.shape[-2:]
            rows = []
            for row in range(3):
                cells = []
                for column in range(3):
                    region = features[
                        :, :,
                        row * height // 3: ((row + 1) * height + 2) // 3,
                        column * width // 3: ((column + 1) * width + 2) // 3,
                    ]
                    cells.append(region.mean(dim=(2, 3), keepdim=True))
                rows.append(torch.cat(cells, dim=3))
            local = torch.cat(rows, dim=2).flatten(1)
        else:
            local = self.local_pool(features).flatten(1)
        if self.include_global:
            global_features = self.global_pool(features).flatten(1)
            local = torch.cat([global_features, local], dim=1)
        return self.classifier(local)
