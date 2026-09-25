import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from task4.data.cifar10 import get_cifar10_datasets
from task4.data.cifar100_unknowns import get_cifar100_unknowns
from task4.models.resnet_cifar import CIFARResNet18
from task4.methods.vanilla import VanillaMethod
from task4.methods.proser import PROSERMethod
from torch.optim.lr_scheduler import CosineAnnealingLR

def train_osr_model(config_name, cfg, device):
    pacs_data = get_cifar10_datasets(use_randaug=cfg['use_randaug'])
    train_loader = DataLoader(pacs_data['train'], batch_size=cfg['batch_size'], shuffle=True, drop_last=True)
    val_loader = DataLoader(pacs_data['val'], batch_size=128, shuffle=False)
    
    model = CIFARResNet18(num_classes=10).to(device)
    
    method_name = cfg['method']
    if method_name == "proser":
        # Load vanilla checkpoint first
        model.load_state_dict(torch.load("checkpoints/task4_vanilla.pt", map_location=device), strict=False)
        method = PROSERMethod(model, beta=cfg.get('beta', 1.0), gamma=cfg.get('gamma', 0.1))
    else:
        # Vanilla or GCSC
        method = VanillaMethod(model)
        
    optimizer = torch.optim.SGD(
        model.parameters(), 
        lr=cfg['learning_rate'], 
        momentum=cfg['momentum'], 
        weight_decay=cfg['weight_decay']
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg['epochs'])
    
    best_acc = 0.0
    
    for epoch in range(1, cfg['epochs'] + 1):
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            
            loss = method.compute_loss(x, y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            # For simplicity, calculate accuracy on closed-set without dummy predictions during training
            if method_name == "proser":
                logits = model(x)
            else:
                logits = model(x)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
            
        scheduler.step()
        
        # Validation (Closed Set Accuracy)
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                preds = logits.argmax(dim=1)
                val_correct += (preds == y).sum().item()
                val_total += y.size(0)
                
        val_acc = 100.0 * val_correct / val_total
        print(f"[{config_name}] Epoch {epoch}/{cfg['epochs']} - Loss: {train_loss/len(train_loader):.4f} - Train Acc: {100.0*correct/total:.2f}% - Val Acc: {val_acc:.2f}%")
        
        if val_acc > best_acc:
            best_acc = val_acc
            os.makedirs("checkpoints", exist_ok=True)
            torch.save(model.state_dict(), f"checkpoints/task4_{config_name}.pt")
            
    print(f"Finished {config_name}. Best Val Acc: {best_acc:.2f}%")
    model.load_state_dict(torch.load(f"checkpoints/task4_{config_name}.pt", map_location=device))
    return model
