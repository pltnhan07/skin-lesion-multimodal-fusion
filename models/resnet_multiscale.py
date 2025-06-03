import torch
import torch.nn as nn
from torchvision import models

class ResNetMultiScale(nn.Module):
    """
    ResNet-50 backbone for multi-scale feature extraction.

    Extracts features from three scales: layer2, layer3, layer4.
    """
    def __init__(self, pretrained=True):
        super().__init__()
        base = models.resnet50(pretrained=pretrained)
        self.conv1 = base.conv1
        self.bn1 = base.bn1
        self.relu = base.relu
        self.maxpool = base.maxpool
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4

    def forward_scales(self, x):
        """
        Args:
            x (Tensor): Input image tensor of shape (B, 3, H, W)
        Returns:
            List[Tensor]: Feature maps from layer2, layer3, and layer4
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x1 = self.layer1(x)   # Not returned, but you can if needed
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        return [x2, x3, x4]
