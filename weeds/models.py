"""Lightweight CNN architectures for edge weed classification."""
import torch
import torch.nn as nn
from torchvision import models


class DSConv(nn.Module):
    """Depthwise-separable conv block (MobileNet-style) with optional SE attention."""

    def __init__(self, cin, cout, stride=1, se=True):
        super().__init__()
        self.dw = nn.Sequential(
            nn.Conv2d(cin, cin, 3, stride, 1, groups=cin, bias=False),
            nn.BatchNorm2d(cin), nn.Hardswish(inplace=True))
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Conv2d(cin, max(8, cin // 4), 1), nn.ReLU(inplace=True),
            nn.Conv2d(max(8, cin // 4), cin, 1), nn.Hardsigmoid(inplace=True)) if se else None
        self.pw = nn.Sequential(
            nn.Conv2d(cin, cout, 1, bias=False), nn.BatchNorm2d(cout), nn.Hardswish(inplace=True))
        self.residual = stride == 1 and cin == cout

    def forward(self, x):
        y = self.dw(x)
        if self.se is not None:
            y = y * self.se(y)
        y = self.pw(y)
        return x + y if self.residual else y


class WeedNetLite(nn.Module):
    """Custom ~0.3M-parameter CNN designed from scratch for real-time edge inference."""

    def __init__(self, num_classes, width=1.0, dropout=0.2):
        super().__init__()
        c = lambda n: max(8, int(n * width))
        cfg = [  # (out_channels, stride)
            (c(32), 1), (c(64), 2), (c(64), 1), (c(128), 2), (c(128), 1),
            (c(256), 2), (c(256), 1), (c(256), 1), (c(512), 2)]
        layers = [nn.Sequential(nn.Conv2d(3, c(32), 3, 2, 1, bias=False),
                                nn.BatchNorm2d(c(32)), nn.Hardswish(inplace=True))]
        cin = c(32)
        for cout, s in cfg:
            layers.append(DSConv(cin, cout, s))
            cin = cout
        self.features = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                  nn.Dropout(dropout), nn.Linear(cin, num_classes))
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.head(self.features(x))


def _replace_head(model, attr_path, num_classes):
    parent = model
    *path, last = attr_path.split(".")
    for p in path:
        parent = getattr(parent, p)
    old = getattr(parent, last) if not last.isdigit() else parent[int(last)]
    new = nn.Linear(old.in_features, num_classes)
    if last.isdigit():
        parent[int(last)] = new
    else:
        setattr(parent, last, new)
    return model


ARCHS = ["weednet_lite", "mobilenet_v3_small", "mobilenet_v3_large",
         "efficientnet_b0", "shufflenet_v2", "squeezenet"]


def build_model(arch, num_classes, pretrained=True):
    w = "DEFAULT" if pretrained else None
    if arch == "weednet_lite":
        return WeedNetLite(num_classes)
    if arch == "mobilenet_v3_small":
        return _replace_head(models.mobilenet_v3_small(weights=w), "classifier.3", num_classes)
    if arch == "mobilenet_v3_large":
        return _replace_head(models.mobilenet_v3_large(weights=w), "classifier.3", num_classes)
    if arch == "efficientnet_b0":
        return _replace_head(models.efficientnet_b0(weights=w), "classifier.1", num_classes)
    if arch == "shufflenet_v2":
        return _replace_head(models.shufflenet_v2_x1_0(weights=w), "fc", num_classes)
    if arch == "squeezenet":
        m = models.squeezenet1_1(weights=w)
        m.classifier[1] = nn.Conv2d(512, num_classes, 1)
        return m
    raise ValueError(f"Unknown arch {arch}. Choose from {ARCHS}")


def count_params(model):
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    for a in ARCHS:
        m = build_model(a, 9, pretrained=False)
        print(f"{a:22s} {count_params(m)/1e6:6.2f}M params  out={tuple(m(torch.randn(1,3,224,224)).shape)}")
