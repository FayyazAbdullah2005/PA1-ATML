import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import torchvision.models as tv_models
import open_clip
import numpy as np
from tqdm import tqdm
from task1.data_utils import NORM_CONFIGS, STL10_CLASSES, set_seed

class ResNet50Extractor(nn.Module):
    def __init__(self):
        super().__init__()
        weights = tv_models.ResNet50_Weights.IMAGENET1K_V2
        base = tv_models.resnet50(weights=weights)
        self.backbone = nn.Sequential(*list(base.children())[:-1]) # GAP output
        self.norm = NORM_CONFIGS['resnet50']
        for p in self.parameters():
            p.requires_grad = False
        self.eval()

    def forward(self, x):
        # x is [B, 3, 224, 224] in [0, 1]
        x_norm = self.norm(x)
        feat = self.backbone(x_norm)
        return torch.flatten(feat, 1) # 2048-d

class ViTB16Extractor(nn.Module):
    def __init__(self):
        super().__init__()
        weights = tv_models.ViT_B_16_Weights.IMAGENET1K_V1
        self.model = tv_models.vit_b_16(weights=weights)
        self.norm = NORM_CONFIGS['vit_b_16']
        for p in self.parameters():
            p.requires_grad = False
        self.eval()

    def forward(self, x):
        # x is [B, 3, 224, 224] in [0, 1]
        x_norm = self.norm(x)
        # Process input to tokens
        x_tok = self.model._process_input(x_norm)
        n = x_tok.shape[0]
        batch_class_token = self.model.class_token.expand(n, -1, -1)
        x_cat = torch.cat([batch_class_token, x_tok], dim=1)
        x_enc = self.model.encoder(x_cat)
        return x_enc[:, 0] # 768-d CLS token

class OpenCLIPExtractor(nn.Module):
    def __init__(self, device='cpu'):
        super().__init__()
        self.model, _, _ = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
        self.norm = NORM_CONFIGS['clip']
        self.tokenizer = open_clip.get_tokenizer('ViT-B-32')
        self.device = device
        self.model.to(device)
        for p in self.parameters():
            p.requires_grad = False
        self.eval()

    def forward(self, x):
        # x is [B, 3, 224, 224] in [0, 1]
        x_norm = self.norm(x)
        feat = self.model.encode_image(x_norm)
        return feat / feat.norm(dim=-1, keepdim=True) # 512-d normalized

    def get_text_classifier(self, classes=STL10_CLASSES):
        prompts = [f"a photo of a {c}" for c in classes]
        tokens = self.tokenizer(prompts).to(self.device)
        with torch.no_grad():
            text_features = self.model.encode_text(tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features # [10, 512]

class LinearHead(nn.Module):
    def __init__(self, in_features, num_classes=10):
        super().__init__()
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.fc(x)

def extract_features(extractor, dataloader, device):
    extractor.eval()
    extractor.to(device)
    feats = []
    labels = []
    with torch.no_grad():
        for imgs, targets in tqdm(dataloader, desc="Extracting features", leave=False):
            imgs = imgs.to(device)
            f = extractor(imgs)
            feats.append(f.cpu())
            labels.append(targets)
    return torch.cat(feats, dim=0), torch.cat(labels, dim=0)

def train_linear_head(train_feats, train_labels, val_feats, val_labels, num_classes=10, max_epochs=50, patience=5, lr=1e-3, wd=1e-4, seed=6304, device='cpu'):
    set_seed(seed)
    in_dim = train_feats.shape[1]
    head = LinearHead(in_dim, num_classes).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=wd)
    criterion = nn.CrossEntropyLoss()
    
    train_loader = DataLoader(TensorDataset(train_feats, train_labels), batch_size=128, shuffle=True)
    
    best_val_acc = -1.0
    best_weights = None
    patience_counter = 0
    
    for epoch in range(max_epochs):
        head.train()
        for x_b, y_b in train_loader:
            x_b, y_b = x_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            logits = head(x_b)
            loss = criterion(logits, y_b)
            loss.backward()
            optimizer.step()
            
        # Validation
        head.eval()
        with torch.no_grad():
            val_logits = head(val_feats.to(device))
            val_preds = val_logits.argmax(dim=-1).cpu()
            val_acc = (val_preds == val_labels).float().mean().item()
            
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_weights = {k: v.cpu().clone() for k, v in head.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break
                
    head.load_state_dict(best_weights)
    head.eval()
    return head, best_val_acc
