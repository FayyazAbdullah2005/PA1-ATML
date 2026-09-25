import os
import io
import json
import random
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, Subset, DataLoader
import torchvision.transforms as T
from sklearn.model_selection import train_test_split

SEED = 6304

def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

STL10_CLASSES = [
    'airplane', 'bird', 'car', 'cat', 'deer',
    'dog', 'frog', 'horse', 'monkey', 'truck'
]

NORM_CONFIGS = {
    'resnet50': T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    'vit_b_16': T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    'clip': T.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
}

class FastSTL10(Dataset):
    def __init__(self, parquet_path, transform=None):
        self.df = pd.read_parquet(parquet_path)
        self.transform = transform
        self.labels = self.df['label'].values.astype(int)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_bytes = row['image']['bytes']
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        label = int(row['label'])
        if self.transform:
            img = self.transform(img)
        return img, label

def get_stl10_datasets(data_dir='./data/stl10_fast', seed=SEED):
    set_seed(seed)
    
    base_transform = T.Compose([
        T.Resize((224, 224), interpolation=T.InterpolationMode.BILINEAR),
        T.ToTensor()
    ])
    
    train_path = os.path.join(data_dir, 'train.parquet')
    test_path = os.path.join(data_dir, 'test.parquet')
    
    train_dataset = FastSTL10(train_path, transform=base_transform)
    test_dataset = FastSTL10(test_path, transform=base_transform)
    
    train_targets = train_dataset.labels
    test_targets = test_dataset.labels
    
    # Stratified 80/20 train/val split
    train_idx, val_idx = train_test_split(
        np.arange(len(train_targets)),
        test_size=0.2,
        stratify=train_targets,
        random_state=seed
    )
    
    # Stratified class-balanced 500 test images (50 per class)
    test_500_idx, _ = train_test_split(
        np.arange(len(test_targets)),
        train_size=500,
        stratify=test_targets,
        random_state=seed
    )
    
    os.makedirs('./data/splits', exist_ok=True)
    split_info = {
        'seed': seed,
        'train_indices': train_idx.tolist(),
        'val_indices': val_idx.tolist(),
        'test_500_indices': test_500_idx.tolist()
    }
    with open('./data/splits/stl10_splits_seed6304.json', 'w') as f:
        json.dump(split_info, f, indent=2)
        
    train_sub = Subset(train_dataset, train_idx)
    val_sub = Subset(train_dataset, val_idx)
    test_500_sub = Subset(test_dataset, test_500_idx)
    
    print(f"STL-10 splits ready: Train={len(train_sub)}, Val={len(val_sub)}, Test 500={len(test_500_sub)}")
    return train_sub, val_sub, test_500_sub, test_dataset
