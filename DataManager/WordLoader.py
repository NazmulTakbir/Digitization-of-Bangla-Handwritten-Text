import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader

from .WordDataset import WordDataset

data_transform = A.Compose([        
        ToTensorV2(),
    ])

def get_word_loader(img_dir, label_file_path, batch_size=256, num_workers=1, shuffle=True):
    word_dataset = WordDataset(img_dir, label_file_path, transform=data_transform)
    word_loader = DataLoader(word_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return word_loader, len(word_dataset)