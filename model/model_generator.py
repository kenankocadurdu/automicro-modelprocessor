import torch
import torch.nn as nn
import logging
from torchvision.models import resnet18, mobilenet_v3_small
from torchvision import models
from model.model_arch import Module

class generator:
    def __init__(self, name, num_class, image_size, run_name=None):
        self.name = name
        self.num_class = num_class
        self.image_size = image_size
        self.run_name = run_name or f"{name}_experiment"

        if self.name == "ResNet18":
            self.model = ResNet18(self.num_class)
        elif self.name == "MobileNetV3Small":
            self.model = MobileNetV3Small(self.num_class)
        elif self.name == "ResNet18_SE":
            self.model = ResNet18_SE(self.num_class)
        elif self.name == "ResNet18_CBAM":
            self.model = ResNet18_CBAM(self.num_class)
        elif self.name == "ResNet18_ECA":
            self.model = ResNet18_ECA(self.num_class)
        else:
            msg = f"Model '{self.name}' not found."
            logging.error(msg)
            raise ValueError(msg)

def available_models():
    return ["ResNet18", "MobileNetV3Small", "ResNet18_SE", "ResNet18_CBAM", "ResNet18_ECA"]


class ResNet18(Module):
    def __init__(self, num_classes=1):
        super().__init__()
        self.backbone = resnet18(weights=None)
        self.backbone.fc = torch.nn.Linear(self.backbone.fc.in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)

class MobileNetV3Small(Module):
    def __init__(self, num_classes=1):
        super().__init__()
        self.backbone = mobilenet_v3_small(weights=None)
        self.backbone.classifier[-1] = torch.nn.Linear(self.backbone.classifier[-1].in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)

########################################
class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y
########################################
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.shared_mlp = nn.Sequential(
            nn.Linear(in_planes, in_planes // ratio),
            nn.ReLU(),
            nn.Linear(in_planes // ratio, in_planes)
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        avg = torch.mean(x, dim=(2, 3)).view(b, c)
        max_ = torch.amax(x, dim=(2, 3)).view(b, c)
        out = self.shared_mlp(avg) + self.shared_mlp(max_)
        scale = torch.sigmoid(out).view(b, c, 1, 1)
        return x * scale

class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        scale = torch.sigmoid(self.conv(x_cat))
        return x * scale

class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention()

    def forward(self, x):
        x = self.ca(x)
        x = self.sa(x)
        return x
########################################
class ECABlock(nn.Module):
    def __init__(self, channel, k_size=3):
        super(ECABlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, 1, c)
        y = self.conv(y)
        y = self.sigmoid(y).view(b, c, 1, 1)
        return x * y
########################################
class ResNet18_SE(Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = models.resnet18(pretrained=True)
        self.backbone.layer3[1].relu = nn.Sequential(
            self.backbone.layer3[1].relu,
            SEBlock(256)
        )
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)

class ResNet18_CBAM(Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = models.resnet18(pretrained=True)
        self.backbone.layer3[1].relu = nn.Sequential(
            self.backbone.layer3[1].relu,
            CBAM(256)
        )
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)

class ResNet18_ECA(Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = models.resnet18(pretrained=True)
        self.backbone.layer3[1].relu = nn.Sequential(
            self.backbone.layer3[1].relu,
            ECABlock(256)
        )
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)
    
