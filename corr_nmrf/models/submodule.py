from torch import nn


def groupwise_correlation(fea1, fea2, num_groups):
    B, C, H, W = fea1.shape
    assert C % num_groups == 0
    channels_per_group = C // num_groups
    cost = (fea1 * fea2).view([B, num_groups, channels_per_group, H, W]).mean(dim=2)
    assert cost.shape == (B, num_groups, H, W)
    return cost


def build_correlation_volume(refimg_fea, targetimg_fea, maxdisp, num_groups, mindisp=0):
    B, C, H, W = refimg_fea.shape
    num_disp = maxdisp - mindisp
    if num_disp <= 0:
        raise ValueError(f"Invalid disparity range: mindisp={mindisp}, maxdisp={maxdisp}")
    volume = refimg_fea.new_zeros([B, num_groups, num_disp, H, W])
    for d in range(mindisp, maxdisp):
        idx = d - mindisp
        if d > 0:
            volume[:, :, idx, :, d:] = groupwise_correlation(
                refimg_fea[:, :, :, d:],
                targetimg_fea[:, :, :, :-d],
                num_groups,
            )
        elif d < 0:
            shift = -d
            volume[:, :, idx, :, :-shift] = groupwise_correlation(
                refimg_fea[:, :, :, :-shift],
                targetimg_fea[:, :, :, shift:],
                num_groups,
            )
        else:
            volume[:, :, idx, :, :] = groupwise_correlation(refimg_fea, targetimg_fea, num_groups)
    volume = volume.contiguous()
    return volume


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride, downsample, pad, dilation):
        super().__init__()

        self.conv1 = nn.Sequential(nn.Conv2d(inplanes, planes, 3, stride, padding=dilation if dilation > 1 else pad, dilation=dilation), nn.ReLU(inplace=True))
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, padding=dilation if dilation > 1 else pad, dilation=dilation)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)

        if self.downsample is not None:
            x = self.downsample(x)

        out += x

        return out