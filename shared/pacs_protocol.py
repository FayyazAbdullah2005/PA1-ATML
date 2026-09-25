import os
import json
import numpy as np
import torch
from torchvision.datasets import ImageFolder
from sklearn.model_selection import train_test_split

from common.seed import set_seed, SEED
from shared.pacs import get_pacs_transforms, SubsetImageFolder

def get_pacs_datasets(pacs_root='data/PACS', split_file='shared/splits/pacs_sketch_seed6304.json', seed=SEED):
    """
    Sets up the official PACS datasets for Tasks 2 and 3:
    - Photo, Art Painting, Cartoon: Stratified 80/20 train/validation splits (seed 6304)
    - Sketch: Target domain. In Task 2, unlabeled images available during adaptation.
    """
    set_seed(seed)
    train_transform, eval_transform = get_pacs_transforms()
    
    source_domains = ['photo', 'art_painting', 'cartoon']
    target_domain = 'sketch'
    
    datasets = {
        'source_train': {},
        'source_val': {},
        'target_adapt': None,
        'target_eval': None
    }
    
    # Check if split file exists, else generate deterministically
    splits_record = None
    if os.path.exists(split_file):
        with open(split_file, 'r') as f:
            splits_record = json.load(f)
    elif os.path.exists('data/splits/pacs_sketch_seed6304.json'):
        with open('data/splits/pacs_sketch_seed6304.json', 'r') as f:
            splits_record = json.load(f)
            
    if splits_record is None:
        splits_record = {'seed': seed, 'sources': {}}
        for domain in source_domains:
            domain_dir = os.path.join(pacs_root, domain)
            raw_dataset = ImageFolder(domain_dir)
            targets = np.array(raw_dataset.targets)
            
            train_idx, val_idx = train_test_split(
                np.arange(len(targets)),
                test_size=0.2,
                stratify=targets,
                random_state=seed
            )
            splits_record['sources'][domain] = {
                'train_idx': train_idx.tolist(),
                'val_idx': val_idx.tolist()
            }
        os.makedirs(os.path.dirname(split_file), exist_ok=True)
        with open(split_file, 'w') as f:
            json.dump(splits_record, f, indent=2)
            
    # Also ensure shared/splits/ copy exists
    os.makedirs(os.path.dirname(split_file), exist_ok=True)
    if not os.path.exists(split_file):
        with open(split_file, 'w') as f:
            json.dump(splits_record, f, indent=2)

    for domain in source_domains:
        domain_dir = os.path.join(pacs_root, domain)
        raw_dataset = ImageFolder(domain_dir)
        train_idx = splits_record['sources'][domain]['train_idx']
        val_idx = splits_record['sources'][domain]['val_idx']
        
        datasets['source_train'][domain] = SubsetImageFolder(raw_dataset, train_idx, transform=train_transform)
        datasets['source_val'][domain] = SubsetImageFolder(raw_dataset, val_idx, transform=eval_transform)
        
    sketch_dir = os.path.join(pacs_root, target_domain)
    sketch_raw = ImageFolder(sketch_dir)
    datasets['target_adapt'] = SubsetImageFolder(sketch_raw, np.arange(len(sketch_raw)), transform=train_transform)
    datasets['target_eval'] = SubsetImageFolder(sketch_raw, np.arange(len(sketch_raw)), transform=eval_transform)
    
    return datasets

class BalancedDomainBatchSampler:
    """
    Yields batches of 8 Photo + 8 Art + 8 Cartoon (24 source total)
    and 24 target examples, cycling loaders when necessary.
    Target class labels are strictly masked during training/adaptation.
    """
    def __init__(self, loader_p, loader_a, loader_c, loader_t=None):
        self.loader_p = loader_p
        self.loader_a = loader_a
        self.loader_c = loader_c
        self.loader_t = loader_t
        self.num_batches = max(len(loader_p), len(loader_a), len(loader_c))

    def __iter__(self):
        iter_p = iter(self.loader_p)
        iter_a = iter(self.loader_a)
        iter_c = iter(self.loader_c)
        iter_t = iter(self.loader_t) if self.loader_t is not None else None
        
        for _ in range(self.num_batches):
            try:
                x_p, y_p = next(iter_p)
            except StopIteration:
                iter_p = iter(self.loader_p)
                x_p, y_p = next(iter_p)
                
            try:
                x_a, y_a = next(iter_a)
            except StopIteration:
                iter_a = iter(self.loader_a)
                x_a, y_a = next(iter_a)
                
            try:
                x_c, y_c = next(iter_c)
            except StopIteration:
                iter_c = iter(self.loader_c)
                x_c, y_c = next(iter_c)
                
            x_src = torch.cat([x_p, x_a, x_c], dim=0) # [24, 3, 224, 224]
            y_src = torch.cat([y_p, y_a, y_c], dim=0) # [24]
            
            if iter_t is not None:
                try:
                    x_t, _ = next(iter_t) # Target class labels strictly ignored
                except StopIteration:
                    iter_t = iter(self.loader_t)
                    x_t, _ = next(iter_t)
                yield x_src, y_src, x_t
            else:
                yield x_src, y_src
                
    def __len__(self):
        return self.num_batches
