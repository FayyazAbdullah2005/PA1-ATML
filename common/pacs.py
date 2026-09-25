import os
import torch
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
import torchvision.transforms as T

PACS_CLASSES = ['dog', 'elephant', 'giraffe', 'guitar', 'horse', 'house', 'person']
PACS_DOMAINS = ['photo', 'art_painting', 'cartoon', 'sketch']

def get_pacs_transforms():
    """
    Standard PACS transforms per assignment specification:
    - Training: Resize 256x256 -> RandomCrop 224x224 + RandomHorizontalFlip -> ImageNet Normalization
    - Validation/Evaluation: Resize 256x256 -> CenterCrop 224x224 -> ImageNet Normalization
    """
    train_transform = T.Compose([
        T.Resize((256, 256)),
        T.RandomCrop((224, 224)),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    eval_transform = T.Compose([
        T.Resize((256, 256)),
        T.CenterCrop((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, eval_transform

class SubsetImageFolder(Dataset):
    """
    Wraps an ImageFolder dataset with a specific subset of sample indices.
    """
    def __init__(self, base_dataset, indices, transform=None):
        self.base_dataset = base_dataset
        self.indices = indices
        self.transform = transform
        self.targets = [base_dataset.targets[i] for i in indices]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        path, target = self.base_dataset.samples[self.indices[idx]]
        img = self.base_dataset.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        return img, target
