import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, in_planes, planes, norm_layer=nn.InstanceNorm2d, stride=1, dilation=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, dilation=dilation,
                               padding=dilation, stride=stride, bias=False)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, dilation=dilation,
                               padding=dilation, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.norm1 = norm_layer(planes)
        self.norm2 = norm_layer(planes)

        if stride == 1 and in_planes == planes:
            self.downsample = None
        else:
            self.norm3 = norm_layer(planes)
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride),
                self.norm3,
            )

    def forward(self, x):
        identity = x
        x = self.relu(self.norm1(self.conv1(x)))
        x = self.relu(self.norm2(self.conv2(x)))
        if self.downsample is not None:
            identity = self.downsample(identity)
        return self.relu(x + identity)


class Backbone(nn.Module):
    """Lightweight ResNet-style feature extractor used by Corr-NMRF."""

    def __init__(self, output_dim=128, norm_layer=nn.InstanceNorm2d):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.norm1 = norm_layer(64)
        self.relu1 = nn.ReLU(inplace=True)

        self.in_planes = 64
        self.layer1 = self._make_layer(64, stride=1, norm_layer=norm_layer)
        self.layer2 = self._make_layer(96, stride=2, norm_layer=norm_layer)
        self.layer3 = self._make_layer(128, stride=1, norm_layer=norm_layer)
        self.conv2 = nn.Conv2d(128, output_dim, 1, 1, 0)
        self.output_dim = output_dim
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
            if m.weight is not None:
                nn.init.constant_(m.weight, 1.0)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, dim, stride=1, dilation=1, norm_layer=nn.InstanceNorm2d):
        layers = (
            ResidualBlock(self.in_planes, dim, norm_layer=norm_layer, stride=stride, dilation=dilation),
            ResidualBlock(dim, dim, norm_layer=norm_layer, stride=1, dilation=dilation),
        )
        self.in_planes = dim
        return nn.Sequential(*layers)

    def forward(self, x):
        x = 2 * (x / 255.0) - 1.0
        x = self.relu1(self.norm1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.conv2(x)
        return [x, F.avg_pool2d(x, kernel_size=2, stride=2)]


def create_backbone(cfg):
    if cfg.BACKBONE.MODEL_TYPE != "resnet":
        raise ValueError("This Corr-NMRF release keeps only the ResNet backbone used by whu_stereo_resnet.yaml")

    if cfg.BACKBONE.NORM_FN == "instance":
        norm_layer = nn.InstanceNorm2d
    elif cfg.BACKBONE.NORM_FN == "batch":
        norm_layer = nn.BatchNorm2d
    else:
        raise ValueError(f"Invalid backbone normalization type: {cfg.BACKBONE.NORM_FN}")

    return Backbone(cfg.BACKBONE.OUT_CHANNELS, norm_layer)
