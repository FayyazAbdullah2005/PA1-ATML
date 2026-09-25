import torch
import torch.nn as nn
import torchvision.models as tv_models

def freeze_bn_running_stats(module):
    """
    Per assignment instruction:
    'freeze all BatchNorm running means and variances at their pretrained ImageNet values.
     The BatchNorm scale and bias parameters (gamma and beta) remain trainable.
     In PyTorch, after calling model.train(), place only the BatchNorm modules in evaluation
     mode so their running statistics are not updated; do not place the complete model in evaluation mode.'
    """
    for m in module.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()

class PACSResNet18Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        base = tv_models.resnet18(weights=tv_models.ResNet18_Weights.IMAGENET1K_V1)
        self.conv1 = base.conv1
        self.bn1 = base.bn1
        self.relu = base.relu
        self.maxpool = base.maxpool
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4
        self.avgpool = base.avgpool
        
        # Initial BatchNorm statistics freeze
        freeze_bn_running_stats(self)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        feat = torch.flatten(self.avgpool(x), 1) # 512-d feature representation
        return feat

    def train(self, mode=True):
        super().train(mode)
        if mode:
            freeze_bn_running_stats(self)
