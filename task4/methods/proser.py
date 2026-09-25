import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class PROSERMethod:
    """
    PROSER: Learning Placeholders for Open-Set Recognition.
    """
    def __init__(self, model, num_known_classes=10, num_dummy_classes=5, beta=1.0, gamma=0.1):
        self.model = model
        self.num_known_classes = num_known_classes
        self.num_dummy_classes = num_dummy_classes
        self.beta = beta
        self.gamma = gamma
        self.criterion = nn.CrossEntropyLoss()
        
        # Add dummy classifier to model if not already present
        if not hasattr(model.backbone, 'dummy_fc'):
            # Same input features as main FC
            in_features = model.backbone.fc.in_features
            model.backbone.dummy_fc = nn.Linear(in_features, num_dummy_classes).to(next(model.parameters()).device)
            # Randomly initialized as per PROSER protocol
            nn.init.xavier_uniform_(model.backbone.dummy_fc.weight)
            if model.backbone.dummy_fc.bias is not None:
                nn.init.zeros_(model.backbone.dummy_fc.bias)

    def _get_mixup_lambda(self, batch_size):
        # Beta(2, 2) distribution
        lam = np.random.beta(2.0, 2.0)
        return float(lam)

    def compute_loss(self, x, y):
        # Split mini-batch into two equal parts
        half_len = x.size(0) // 2
        
        x_cls = x[:half_len]
        y_cls = y[:half_len]
        
        x_mix = x[half_len:]
        y_mix = y[half_len:]
        
        device = x.device
        
        # 1. Classifier-Placeholder Loss (using first half)
        # Forward pass normally
        feat_cls, logits_cls = self.model(x_cls, return_feature=True)
        dummy_logits_cls = self.model.backbone.dummy_fc(feat_cls)
        
        # Cross entropy on known classes
        loss_cls = self.criterion(logits_cls, y_cls)
        
        # Dummy loss: we want max dummy logit to be higher than all remaining known logits
        max_dummy, _ = torch.max(dummy_logits_cls, dim=1, keepdim=True)
        
        # Combine known logits and max dummy logit
        combined_logits = torch.cat((logits_cls, max_dummy), dim=1) # Shape: (B, 11)
        
        # Mask out the correct known class by setting it to -inf
        batch_indices = torch.arange(half_len, device=device)
        combined_logits[batch_indices, y_cls] = -1e9
        
        # Target for dummy loss is the max dummy logit index (which is at index num_known_classes)
        dummy_targets = torch.full((half_len,), self.num_known_classes, dtype=torch.long, device=device)
        loss_dummy_cls = self.criterion(combined_logits, dummy_targets)
        
        # 2. Data-Placeholder Loss via Manifold Mixup (using second half)
        # Get features up to layer2
        feat_pre = self.model(x_mix, mixup_after_layer2=True)
        
        # Mixup
        lam = self._get_mixup_lambda(x_mix.size(0))
        indices = torch.randperm(x_mix.size(0), device=device)
        feat_mixed = lam * feat_pre + (1 - lam) * feat_pre[indices]
        
        # Forward from layer3
        feat_post, logits_mix = self.model.forward_from_layer3(feat_mixed, return_feature=True)
        dummy_logits_mix = self.model.backbone.dummy_fc(feat_post)
        
        # Combined output for mixed data: shape (B, 15)
        # We want it to be classified as one of the dummy classes.
        # But wait, original code sets targets for mixed features to dummy class target.
        # Original code does: 
        # prehalfoutput = torch.cat((latter2blockclf1(net,mixed_embeddings),latter2blockclf2(net,mixed_embeddings)),1)
        # loss1 = criterion(prehalfoutput, target=10)
        
        combined_logits_mix = torch.cat((logits_mix, dummy_logits_mix), dim=1) # Shape: (B, 15)
        # PROSER's code just targets the first dummy class (or treats all dummies as single class?)
        # Let's target it to index `num_known_classes` (the first dummy class). 
        # Wait, if `combined_logits_mix` has 15 classes, setting target to 10 means we push it to the first dummy class.
        # Yes, original code uses target = args.known_class
        mix_targets = torch.full((x_mix.size(0),), self.num_known_classes, dtype=torch.long, device=device)
        
        loss_dummy_mix = self.criterion(combined_logits_mix, mix_targets)
        
        # Total loss
        # PROSER code: loss = 0.01 * loss_mix + lamda1 * loss_cls + lamda2 * loss_dummy_cls
        # In the prompt: beta=1, gamma=0.1
        # beta is for classifier-placeholder, gamma is for data-placeholder
        total_loss = loss_cls + self.beta * loss_dummy_cls + self.gamma * loss_dummy_mix
        
        return total_loss
