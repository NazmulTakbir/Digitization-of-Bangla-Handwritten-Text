import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import ConcatDataset, DataLoader

from .WordDataset import WordDataset

data_transform = A.Compose([        
        ToTensorV2(),
    ])

def get_word_loader(img_dirs, label_file_paths, batch_size=256, num_workers=1, shuffle=True):
    if isinstance(img_dirs, str):
        img_dirs = [img_dirs]
    if isinstance(label_file_paths, str):
        label_file_paths = [label_file_paths]

    assert len(img_dirs) == len(label_file_paths), "Number of image directories and label files must be equal"

    datasets = []
    for img_dir, label_file_path in zip(img_dirs, label_file_paths):
        datasets.append(WordDataset(img_dir, label_file_path, transform=data_transform))

    merged_dataset = ConcatDataset(datasets)
    word_loader = DataLoader(merged_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return word_loader, len(word_loader)