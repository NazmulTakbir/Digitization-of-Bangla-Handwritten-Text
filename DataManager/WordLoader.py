import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import ConcatDataset, DataLoader
from .SyntheticWordDataset import SyntheticWordDataset
from .WordDataset import WordDataset

data_transform_aug = A.Compose([ 
        A.Rotate(limit=10, p=0.5),
        A.Blur(blur_limit=3, p=0.25),
        A.OpticalDistortion(p=0.25),
        A.GridDistortion(p=0.25),
        A.ElasticTransform(alpha=0.5, sigma=0, alpha_affine=0, p=0.25),
        A.GaussNoise(var_limit=(120.0, 135.0), mean=0, always_apply=False, p=0.25),       
        ToTensorV2(),
    ])

data_transform_aug_synthetic = A.Compose([ 
        A.Rotate(limit=10, p=0.5),
        A.Blur(blur_limit=3, p=0.5),
        A.OpticalDistortion(p=0.5),
        A.GridDistortion(p=0.5),
        A.ElasticTransform(alpha=0.5, sigma=0, alpha_affine=0, p=0.5),
        A.GaussNoise(var_limit=(120.0, 135.0), mean=0, always_apply=False, p=0.5),       
        ToTensorV2(),
    ])

data_transform_no_aug = A.Compose([ 
        ToTensorV2(),
    ])

def get_word_loader(img_dirs, label_file_paths, augment=True, batch_size=256, num_workers=1, shuffle=True, use_synthetic=False):
    if isinstance(img_dirs, str):
        img_dirs = [img_dirs]
    if isinstance(label_file_paths, str):
        label_file_paths = [label_file_paths]

    assert len(img_dirs) == len(label_file_paths), "Number of image directories and label files must be equal"

    datasets = []
    for img_dir, label_file_path in zip(img_dirs, label_file_paths):
        if augment:
            datasets.append(WordDataset(img_dir, label_file_path, transform=data_transform_aug))
        else:
            datasets.append(WordDataset(img_dir, label_file_path, transform=data_transform_no_aug))

    if use_synthetic:
        if augment:
            datasets.append(SyntheticWordDataset(transform=data_transform_aug_synthetic))
        else:
            datasets.append(SyntheticWordDataset(transform=data_transform_no_aug))


    merged_dataset = ConcatDataset(datasets)
    word_loader = DataLoader(merged_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return word_loader, len(word_loader)