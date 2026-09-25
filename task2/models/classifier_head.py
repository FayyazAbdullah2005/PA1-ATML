import torch
import torch.nn as nn
from task2.models.backbone import PACSResNet18Backbone, freeze_bn_running_stats

class PACSClassifierHead(nn.Module):
    def __init__(self, in_features=512, num_classes=7):
        super().__init__()
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, feat):
        return self.fc(feat)

class PACSResNet18(nn.Module):
    """
    Combined PACS model: ResNet-18 backbone + 7-class linear classifier head.
    Returns (feat, logits).
    """
    def __init__(self, num_classes=7):
        super().__init__()
        self.backbone = PACSResNet18Backbone()
        self.classifier = PACSClassifierHead(in_features=512, num_classes=num_classes)

    def forward(self, x):
        feat = self.backbone(x)
        logits = self.classifier(feat)
        return feat, logits

    def train(self, mode=True):
        super().train(mode)
        if mode:
            freeze_bn_running_stats(self.backbone)
