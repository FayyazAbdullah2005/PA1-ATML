import torch.nn as nn

class VanillaMethod:
    """
    Standard Closed-Set Classifier training.
    """
    def __init__(self, model):
        self.model = model
        self.criterion = nn.CrossEntropyLoss()

    def compute_loss(self, x, y):
        logits = self.model(x)
        return self.criterion(logits, y)
