from __future__ import annotations

import math

import torch
from torch import nn


def make_divisible(
    value: float,
    divisor: int = 8,
) -> int:
    return max(
        divisor,
        int(value + divisor / 2)
        // divisor
        * divisor,
    )


class ConvBNReLU(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        groups: int = 1,
        dilation: int = 1,
    ) -> None:
        padding = dilation * (kernel_size // 2)

        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class EfficientChannelAttention(nn.Module):
    """Lightweight ECA channel attention."""

    def __init__(
        self,
        channels: int,
        gamma: int = 2,
        bias: int = 1,
    ) -> None:
        super().__init__()

        kernel_size = int(
            abs(
                (
                    math.log2(channels)
                    + bias
                )
                / gamma
            )
        )

        if kernel_size % 2 == 0:
            kernel_size += 1

        kernel_size = max(
            kernel_size,
            3,
        )

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.conv = nn.Conv1d(
            1,
            1,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias=False,
        )

        self.sigmoid = nn.Sigmoid()

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        attention = self.pool(x)

        attention = attention.squeeze(-1)
        attention = attention.transpose(-1, -2)

        attention = self.conv(attention)

        attention = attention.transpose(-1, -2)
        attention = attention.unsqueeze(-1)

        attention = self.sigmoid(
            attention
        )

        return x * attention


class GlobalStdPool(nn.Module):
    """Population spatial standard deviation, BCHW -> BC, without parameters."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=(2, 3), keepdim=True)
        centered = x - mean
        variance = centered.square().mean(dim=(2, 3))
        return torch.sqrt(variance + 1e-6)


class SpatialAttentivePool(nn.Module):
    """Input-dependent spatial aggregation from BCHW to BC.

    A single shared spatial weight map (B, 1, H*W) broadcasts across the
    flattened feature channels (B, C, H*W). Softmax normalizes over spatial
    locations with PyTorch's numerically stable implementation.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.score = nn.Conv2d(channels, 1, kernel_size=1, bias=True)

    def attention_weights(self, x: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.score(x).flatten(2), dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = self.attention_weights(x)
        return (x.flatten(2) * weights).sum(dim=-1)


class InvertedDepthwiseResidual(nn.Module):
    """
    1x1 expansion -> depthwise convolution -> 1x1 projection.

    Residual is used only when shape is unchanged.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        expansion: float = 2.0,
        dilation: int = 1,
    ) -> None:
        super().__init__()

        hidden_channels = make_divisible(
            in_channels * expansion
        )

        self.use_residual = (
            stride == 1
            and in_channels == out_channels
        )

        layers: list[nn.Module] = []

        if hidden_channels != in_channels:
            layers.append(
                ConvBNReLU(
                    in_channels,
                    hidden_channels,
                    kernel_size=1,
                )
            )

        layers.extend([
            ConvBNReLU(
                hidden_channels,
                hidden_channels,
                kernel_size=kernel_size,
                stride=stride,
                groups=hidden_channels,
                dilation=dilation,
            ),
            nn.Conv2d(
                hidden_channels,
                out_channels,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(
                out_channels
            ),
        ])

        self.block = nn.Sequential(
            *layers
        )

        self.activation = nn.ReLU(
            inplace=True
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        output = self.block(x)

        if self.use_residual:
            output = output + x

        return self.activation(
            output
        )


class PartialSpatialConv(nn.Module):
    """Dense spatial mixing on a subset; depthwise mixing on the remainder.

    Both paths use the same kernel and preserve channel order. This replaces
    only a spatial convolution: the caller retains its existing BN/ReLU and
    subsequent pointwise fusion. Static slices avoid in-place writes and
    support dynamic-batch ONNX export.
    """

    def __init__(
        self,
        channels: int,
        dense_channels: int,
        kernel_size: int = 5,
    ) -> None:
        super().__init__()
        if not 0 < dense_channels < channels:
            raise ValueError("dense_channels must be between 0 and channels")
        if kernel_size < 1 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be positive and odd")

        self.dense_channels = dense_channels
        remaining_channels = channels - dense_channels
        self.dense = nn.Conv2d(
            dense_channels,
            dense_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias=False,
        )
        self.depthwise = nn.Conv2d(
            remaining_channels,
            remaining_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=remaining_channels,
            bias=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            [
                self.dense(x[:, :self.dense_channels]),
                self.depthwise(x[:, self.dense_channels:]),
            ],
            dim=1,
        )


class LinearAddInvertedDepthwiseResidual(nn.Module):
    """
    R4-compatible inverted depthwise residual with a linear residual output.

    This block intentionally changes only one behavior from
    ``InvertedDepthwiseResidual``:

    - when a true identity residual is present (stride=1 and channels match),
      the projected branch is added to the shortcut and returned *without* a
      post-add ReLU;
    - when no residual is available (channel change or stride change), the
      original R4 output ReLU is retained.

    The convolution/BN structure and parameter count are therefore identical
    to the R4 block. This isolates the effect of repeatedly clipping the
    residual bottleneck representation without changing stage widths,
    downsampling, ECA, pooling, or classifier capacity.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        expansion: float = 2.0,
        dilation: int = 1,
    ) -> None:
        super().__init__()

        hidden_channels = make_divisible(
            in_channels * expansion
        )

        self.use_residual = (
            stride == 1
            and in_channels == out_channels
        )

        layers: list[nn.Module] = []

        if hidden_channels != in_channels:
            layers.append(
                ConvBNReLU(
                    in_channels,
                    hidden_channels,
                    kernel_size=1,
                )
            )

        layers.extend([
            ConvBNReLU(
                hidden_channels,
                hidden_channels,
                kernel_size=kernel_size,
                stride=stride,
                groups=hidden_channels,
                dilation=dilation,
            ),
            nn.Conv2d(
                hidden_channels,
                out_channels,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(
                out_channels
            ),
        ])

        self.block = nn.Sequential(
            *layers
        )

        self.activation = nn.ReLU(
            inplace=True
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        output = self.block(x)

        if self.use_residual:
            return output + x

        return self.activation(output)


class CoordinateAttention(nn.Module):
    """
    Lightweight coordinate attention.

    Unlike channel-only attention, this preserves directional positional
    information by pooling independently along height and width. This is
    useful for FER because discriminative regions (eyes, brows, mouth) are
    spatially structured.
    """

    def __init__(
        self,
        channels: int,
        reduction: int = 24,
        min_channels: int = 8,
    ) -> None:
        super().__init__()

        reduced_channels = max(
            min_channels,
            make_divisible(
                channels / reduction,
                divisor=8,
            ),
        )

        self.reduce = nn.Sequential(
            nn.Conv2d(
                channels,
                reduced_channels,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(
                reduced_channels
            ),
            nn.ReLU(inplace=True),
        )

        self.attention_h = nn.Conv2d(
            reduced_channels,
            channels,
            kernel_size=1,
            bias=True,
        )

        self.attention_w = nn.Conv2d(
            reduced_channels,
            channels,
            kernel_size=1,
            bias=True,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        height = int(x.shape[2])
        width = int(x.shape[3])

        pooled_h = x.mean(
            dim=3,
            keepdim=True,
        )

        pooled_w = (
            x.mean(
                dim=2,
                keepdim=True,
            )
            .transpose(2, 3)
        )

        merged = torch.cat(
            [pooled_h, pooled_w],
            dim=2,
        )

        merged = self.reduce(
            merged
        )

        encoded_h, encoded_w = (
            torch.split(
                merged,
                [height, width],
                dim=2,
            )
        )

        encoded_w = encoded_w.transpose(
            2,
            3,
        )

        weight_h = torch.sigmoid(
            self.attention_h(
                encoded_h
            )
        )

        weight_w = torch.sigmoid(
            self.attention_w(
                encoded_w
            )
        )

        return x * weight_h * weight_w


class MultiScaleInvertedDepthwiseResidual(nn.Module):
    """
    Pointwise expansion -> parallel depthwise kernels -> pointwise projection.

    The parallel depthwise branches add multi-scale receptive fields at very
    low parameter cost, while the final 1x1 projection mixes information from
    all branches. Residual is used only when the tensor shape is unchanged.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_sizes: tuple[int, ...] = (3, 5),
        stride: int = 1,
        expansion: float = 2.0,
    ) -> None:
        super().__init__()

        if not kernel_sizes:
            raise ValueError(
                "kernel_sizes must not be empty."
            )

        if any(
            kernel_size % 2 == 0
            or kernel_size < 3
            for kernel_size in kernel_sizes
        ):
            raise ValueError(
                "kernel_sizes must contain odd values >= 3."
            )

        branch_count = len(
            kernel_sizes
        )

        hidden_channels = make_divisible(
            in_channels * expansion,
            divisor=branch_count * 8,
        )

        branch_channels = (
            hidden_channels
            // branch_count
        )

        self.use_residual = (
            stride == 1
            and in_channels == out_channels
        )

        self.expand = (
            ConvBNReLU(
                in_channels,
                hidden_channels,
                kernel_size=1,
            )
            if hidden_channels != in_channels
            else nn.Identity()
        )

        self.branches = nn.ModuleList([
            ConvBNReLU(
                branch_channels,
                branch_channels,
                kernel_size=kernel_size,
                stride=stride,
                groups=branch_channels,
            )
            for kernel_size in kernel_sizes
        ])

        self.project = nn.Sequential(
            nn.Conv2d(
                hidden_channels,
                out_channels,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(
                out_channels
            ),
        )

        self.activation = nn.ReLU(
            inplace=True
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        expanded = self.expand(x)

        chunks = torch.chunk(
            expanded,
            len(self.branches),
            dim=1,
        )

        mixed = torch.cat(
            [
                branch(chunk)
                for branch, chunk in zip(
                    self.branches,
                    chunks,
                )
            ],
            dim=1,
        )

        output = self.project(
            mixed
        )

        if self.use_residual:
            output = output + x

        return self.activation(
            output
        )


class FusedConvResidual(nn.Module):
    """
    Fused spatial/channel block for the early-to-middle transition.

    A dense 3x3 convolution performs spatial and channel mixing together,
    then a 1x1 projection controls output width. Residual is used only when
    input and output shapes match.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        expansion: float = 1.5,
    ) -> None:
        super().__init__()

        hidden_channels = make_divisible(
            in_channels * expansion
        )

        self.use_residual = (
            stride == 1
            and in_channels == out_channels
        )

        self.block = nn.Sequential(
            ConvBNReLU(
                in_channels,
                hidden_channels,
                kernel_size=3,
                stride=stride,
            ),
            nn.Conv2d(
                hidden_channels,
                out_channels,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(
                out_channels
            ),
        )

        self.activation = nn.ReLU(
            inplace=True
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        output = self.block(x)

        if self.use_residual:
            output = output + x

        return self.activation(output)


class SpatialAttentionGate(nn.Module):
    """Tiny spatial gate from channel-average and channel-max maps."""

    def __init__(
        self,
        kernel_size: int = 7,
    ) -> None:
        super().__init__()

        if kernel_size < 3 or kernel_size % 2 == 0:
            raise ValueError(
                "kernel_size must be odd and >= 3"
            )

        self.conv = nn.Conv2d(
            2,
            1,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias=False,
        )
        self.sigmoid = nn.Sigmoid()

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        average_map = x.mean(
            dim=1,
            keepdim=True,
        )
        maximum_map = x.amax(
            dim=1,
            keepdim=True,
        )

        attention = torch.cat(
            [average_map, maximum_map],
            dim=1,
        )
        attention = self.sigmoid(
            self.conv(attention)
        )

        return x * attention


class RepConvBlock(nn.Module):
    """
    RepVGG-style training-time multi-branch convolution.

    Training graph:
        3x3 Conv-BN + 1x1 Conv-BN + optional identity BN -> ReLU

    Deployment graph:
        a single equivalent 3x3 Conv -> ReLU

    The block intentionally uses dense convolution (groups=1) in the early
    and middle stages to strengthen joint spatial/channel feature learning.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        deploy: bool = False,
    ) -> None:
        super().__init__()

        if stride not in (1, 2):
            raise ValueError(
                "RepConvBlock stride must be 1 or 2."
            )

        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.stride = int(stride)
        self.deploy = bool(deploy)

        self.activation = nn.ReLU(inplace=True)

        if self.deploy:
            self.reparam = nn.Conv2d(
                self.in_channels,
                self.out_channels,
                kernel_size=3,
                stride=self.stride,
                padding=1,
                bias=True,
            )
            self.branch_3x3 = None
            self.branch_1x1 = None
            self.identity_bn = None
        else:
            self.reparam = None

            self.branch_3x3 = nn.Sequential(
                nn.Conv2d(
                    self.in_channels,
                    self.out_channels,
                    kernel_size=3,
                    stride=self.stride,
                    padding=1,
                    bias=False,
                ),
                nn.BatchNorm2d(
                    self.out_channels
                ),
            )

            self.branch_1x1 = nn.Sequential(
                nn.Conv2d(
                    self.in_channels,
                    self.out_channels,
                    kernel_size=1,
                    stride=self.stride,
                    padding=0,
                    bias=False,
                ),
                nn.BatchNorm2d(
                    self.out_channels
                ),
            )

            self.identity_bn = (
                nn.BatchNorm2d(
                    self.in_channels
                )
                if self.stride == 1
                and self.in_channels == self.out_channels
                else None
            )

    @staticmethod
    def _fuse_conv_bn(
        branch: nn.Sequential,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        conv = branch[0]
        bn = branch[1]

        if not isinstance(conv, nn.Conv2d):
            raise TypeError(
                "Expected Conv2d in RepConvBlock branch."
            )

        if not isinstance(bn, nn.BatchNorm2d):
            raise TypeError(
                "Expected BatchNorm2d in RepConvBlock branch."
            )

        kernel = conv.weight

        if conv.bias is None:
            conv_bias = torch.zeros(
                conv.out_channels,
                device=kernel.device,
                dtype=kernel.dtype,
            )
        else:
            conv_bias = conv.bias

        std = torch.sqrt(
            bn.running_var + bn.eps
        )
        scale = bn.weight / std

        fused_kernel = kernel * scale.reshape(
            -1,
            1,
            1,
            1,
        )

        fused_bias = (
            bn.bias
            + (conv_bias - bn.running_mean)
            * scale
        )

        return fused_kernel, fused_bias

    def _fuse_identity_bn(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.identity_bn is None:
            reference = self.branch_3x3[0].weight
            return (
                torch.zeros_like(reference),
                torch.zeros(
                    self.out_channels,
                    device=reference.device,
                    dtype=reference.dtype,
                ),
            )

        bn = self.identity_bn
        reference = self.branch_3x3[0].weight

        identity_kernel = torch.zeros_like(
            reference
        )

        channel_indices = torch.arange(
            self.out_channels,
            device=reference.device,
        )

        identity_kernel[
            channel_indices,
            channel_indices,
            1,
            1,
        ] = 1.0

        std = torch.sqrt(
            bn.running_var + bn.eps
        )
        scale = bn.weight / std

        fused_kernel = (
            identity_kernel
            * scale.reshape(-1, 1, 1, 1)
        )

        fused_bias = (
            bn.bias
            - bn.running_mean * scale
        )

        return fused_kernel, fused_bias

    @staticmethod
    def _pad_1x1_to_3x3(
        kernel: torch.Tensor,
    ) -> torch.Tensor:
        return torch.nn.functional.pad(
            kernel,
            [1, 1, 1, 1],
        )

    def get_equivalent_kernel_bias(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.deploy:
            if self.reparam is None:
                raise RuntimeError(
                    "Deploy RepConvBlock has no reparameterized conv."
                )
            return self.reparam.weight, self.reparam.bias

        if self.branch_3x3 is None or self.branch_1x1 is None:
            raise RuntimeError(
                "Training RepConvBlock branches are missing."
            )

        kernel_3x3, bias_3x3 = self._fuse_conv_bn(
            self.branch_3x3
        )
        kernel_1x1, bias_1x1 = self._fuse_conv_bn(
            self.branch_1x1
        )
        kernel_identity, bias_identity = (
            self._fuse_identity_bn()
        )

        kernel = (
            kernel_3x3
            + self._pad_1x1_to_3x3(
                kernel_1x1
            )
            + kernel_identity
        )
        bias = (
            bias_3x3
            + bias_1x1
            + bias_identity
        )

        return kernel, bias

    def switch_to_deploy(
        self,
    ) -> "RepConvBlock":
        if self.deploy:
            return self

        kernel, bias = (
            self.get_equivalent_kernel_bias()
        )

        reparam = nn.Conv2d(
            self.in_channels,
            self.out_channels,
            kernel_size=3,
            stride=self.stride,
            padding=1,
            bias=True,
        ).to(
            device=kernel.device,
            dtype=kernel.dtype,
        )

        with torch.no_grad():
            reparam.weight.copy_(kernel)
            reparam.bias.copy_(bias)

        self.reparam = reparam
        self.branch_3x3 = None
        self.branch_1x1 = None
        self.identity_bn = None
        self.deploy = True

        return self

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        if self.deploy:
            if self.reparam is None:
                raise RuntimeError(
                    "Deploy RepConvBlock is not initialized."
                )

            return self.activation(
                self.reparam(x)
            )

        if self.branch_3x3 is None or self.branch_1x1 is None:
            raise RuntimeError(
                "Training RepConvBlock branches are missing."
            )

        output = (
            self.branch_3x3(x)
            + self.branch_1x1(x)
        )

        if self.identity_bn is not None:
            output = output + self.identity_bn(x)

        return self.activation(output)


class LinearBottleneckResidual(nn.Module):
    """
    MobileNetV2-style inverted residual with a true linear bottleneck.

    Expansion and depthwise stages use ReLU, while the narrow projection
    deliberately has no activation. When the tensor shape is unchanged the
    shortcut is also added without a post-addition non-linearity. This keeps
    the bottleneck representation from being clipped repeatedly.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        expansion: float = 2.0,
    ) -> None:
        super().__init__()

        hidden_channels = make_divisible(
            in_channels * expansion
        )

        self.use_residual = (
            stride == 1
            and in_channels == out_channels
        )

        self.expand = (
            ConvBNReLU(
                in_channels,
                hidden_channels,
                kernel_size=1,
            )
            if hidden_channels != in_channels
            else nn.Identity()
        )

        self.depthwise = ConvBNReLU(
            hidden_channels,
            hidden_channels,
            kernel_size=kernel_size,
            stride=stride,
            groups=hidden_channels,
        )

        self.project = nn.Sequential(
            nn.Conv2d(
                hidden_channels,
                out_channels,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(
                out_channels
            ),
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        output = self.expand(x)
        output = self.depthwise(output)
        output = self.project(output)

        if self.use_residual:
            output = output + x

        return output


class DownsampleLinearBottleneckResidual(nn.Module):
    """
    Information-preserving stage transition.

    The main path is a stride-2 linear inverted residual. A cheap average-pool
    plus 1x1 projection carries low-frequency information directly across the
    resolution change, avoiding a complete break in the residual information
    path at every downsampling boundary.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        expansion: float = 2.0,
    ) -> None:
        super().__init__()

        self.main = LinearBottleneckResidual(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=2,
            expansion=expansion,
        )

        self.shortcut = nn.Sequential(
            nn.AvgPool2d(
                kernel_size=2,
                stride=2,
            ),
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(
                out_channels
            ),
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return self.main(x) + self.shortcut(x)


class BasicResidualBlock(nn.Module):
    """
    Small ResNet-style basic block for scratch FER training.

    Uses two full 3x3 convolutions so spatial and cross-channel mixing happen
    jointly.  A 1x1 projection is used only when the spatial size or channel
    count changes.  The final branch BN is exposed as ``final_bn`` so the
    backbone can optionally zero-initialize it and start each residual block
    close to an identity mapping.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
    ) -> None:
        super().__init__()

        self.conv1 = ConvBNReLU(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
        )

        self.final_bn = self.conv2[-1]

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

        self.activation = nn.ReLU(inplace=True)

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        output = self.conv1(x)
        output = self.conv2(output)
        output = output + self.shortcut(x)
        return self.activation(output)


class DepthwisePointwiseResidualBlock(nn.Module):
    """
    Depthwise-pointwise counterpart of ``BasicResidualBlock``.

    The topology, stage widths, shortcut policy, ECA placement, and head can
    therefore be kept identical to the full-convolution ResNet candidate.
    Only each residual-branch 3x3 spatial convolution is factorized into a
    depthwise 3x3 followed by a pointwise 1x1 convolution.

    The second separable unit ends with a linear pointwise projection before
    residual addition, matching the usual residual-block convention that the
    branch output is not activated immediately before the sum.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
    ) -> None:
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                groups=in_channels,
                bias=False,
            ),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                groups=out_channels,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
        )

        self.final_bn = self.conv2[-1]

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

        self.activation = nn.ReLU(inplace=True)

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        output = self.conv1(x)
        output = self.conv2(output)
        output = output + self.shortcut(x)
        return self.activation(output)


class BottleneckResidualBlock(nn.Module):
    """
    Lightweight full-convolution bottleneck residual block.

    The expensive 3x3 operation is performed after a 1x1 channel reduction,
    so the block preserves true spatial+cross-channel mixing without paying
    for a full 3x3 convolution at the outer channel width.  This is intended
    for the semantic 20x20/10x10 stages where the residual-family ablation
    showed full convolutions were materially stronger than depthwise-pointwise
    factorization, while the successful R4 early topology is preserved.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bottleneck_channels: int,
        stride: int = 1,
    ) -> None:
        super().__init__()

        if bottleneck_channels <= 0:
            raise ValueError("bottleneck_channels must be positive")

        self.reduce = ConvBNReLU(
            in_channels,
            bottleneck_channels,
            kernel_size=1,
        )

        self.spatial = ConvBNReLU(
            bottleneck_channels,
            bottleneck_channels,
            kernel_size=3,
            stride=stride,
        )

        self.expand = nn.Sequential(
            nn.Conv2d(
                bottleneck_channels,
                out_channels,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
        )

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

        self.activation = nn.ReLU(inplace=True)

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        output = self.reduce(x)
        output = self.spatial(output)
        output = self.expand(output)
        output = output + self.shortcut(x)
        return self.activation(output)
