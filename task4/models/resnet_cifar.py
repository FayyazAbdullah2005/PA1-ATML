import torch
import torch.nn as nn
import torchvision.models as tv_models

class CIFARResNet18(nn.Module):
    """
    CIFAR-adapted ResNet-18:
    - Replace 7x7 stride-2 conv with 3x3 stride-1 conv
    - Remove max-pooling layer (replace with Identity)
    - 10-class output head
    """
    def __init__(self, num_classes=10):
        super().__init__()
        base = tv_models.resnet18(weights=None)
        
        # Modify for CIFAR-10 (32x32 images)
        base.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        base.maxpool = nn.Identity()
        base.fc = nn.Linear(512, num_classes)
        
        self.backbone = base
        
    def forward(self, x, return_feature=False, mixup_after_layer2=False):
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        
        if mixup_after_layer2:
            # We assume x is split into two halves or we just mix within the batch
            # Actually, PROSER says "split mini-batch into two equal parts... second half for manifold mixup"
            # It's easier if we just return x here if requested, or do the mixup externally.
            return x

        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        x = self.backbone.avgpool(x)
        feat = torch.flatten(x, 1)
        
        logits = self.backbone.fc(feat)
        
        if return_feature:
            return feat, logits
        return logits
        
    def forward_from_layer3(self, x, return_feature=False):
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        x = self.backbone.avgpool(x)
        feat = torch.flatten(x, 1)
        logits = self.backbone.fc(feat)
        if return_feature:
            return feat, logits
        return logits
