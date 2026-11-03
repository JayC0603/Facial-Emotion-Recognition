"""Exact inference architectures from the submitted training notebooks."""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models import efficientnet_b0


class SE(nn.Module):
    def __init__(self, channels: int, reduction: int = 8) -> None:
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.fc1 = nn.Conv2d(channels, hidden, 1)
        self.fc2 = nn.Conv2d(hidden, channels, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        scale = F.adaptive_avg_pool2d(inputs, 1)
        return inputs * torch.sigmoid(self.fc2(F.relu(self.fc1(scale), inplace=True)))


class ResBlock(nn.Module):
    def __init__(self, cin: int, cout: int, stride: int = 1, drop: float = 0.0) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)
        self.se = SE(cout)
        self.drop = nn.Dropout2d(drop) if drop > 0 else nn.Identity()
        self.short = None
        if stride != 1 or cin != cout:
            self.short = nn.Sequential(
                nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout)
            )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        identity = inputs if self.short is None else self.short(inputs)
        output = F.relu(self.bn1(self.conv1(inputs)), inplace=True)
        output = self.drop(output)
        output = self.se(self.bn2(self.conv2(output)))
        return F.relu(output + identity, inplace=True)


class NotebookEmotionCNN(nn.Module):
    def __init__(self, num_classes: int = 7, stem: int = 24,
                 widths: tuple[int, ...] = (32, 48, 88, 136),
                 drop2d: float = 0.03, drop_fc: float = 0.2) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, stem, 3, 1, 1, bias=False), nn.BatchNorm2d(stem),
            nn.ReLU(inplace=True),
        )
        layers: list[nn.Module] = []
        cin = stem
        for index, width in enumerate(widths):
            drop = drop2d * (index + 1)
            layers += [ResBlock(cin, width, 2, drop), ResBlock(width, width, 1, drop)]
            cin = width
        self.stages = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(drop_fc),
            nn.Linear(cin, num_classes),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.head(self.stages(self.stem(inputs)))


class NotebookEfficientNetB0(nn.Module):
    def __init__(self, num_classes: int = 7, drop: float = 0.3,
                 drop_path: float = 0.1, size: int = 160) -> None:
        super().__init__()
        net = efficientnet_b0(weights=None, stochastic_depth_prob=drop_path)
        old = net.features[0][0]
        net.features[0][0] = nn.Conv2d(
            1, old.out_channels, old.kernel_size, old.stride, old.padding, bias=False
        )
        net.classifier = nn.Sequential(
            nn.Dropout(drop), nn.Linear(net.classifier[1].in_features, num_classes)
        )
        self.net, self.size = net, size

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        inputs = inputs * 0.5 + 0.5
        inputs = (inputs - 0.449) / 0.226
        if inputs.shape[-1] != self.size:
            inputs = F.interpolate(
                inputs, size=(self.size, self.size), mode="bilinear", align_corners=False
            )
        return self.net(inputs)
