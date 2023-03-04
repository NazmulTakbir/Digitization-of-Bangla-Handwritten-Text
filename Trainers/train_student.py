import json

from Models.Student.Student import Student
from DataManager.WordLoader import get_word_loader

with open('Graphemes/Extracted/graphemes_bw_bnhtrd_syn.json', 'r') as f:
    graphemes_dict = json.load(f)

teacher_data = {
    'teacher_type': 'ResNet18',
    'saved_path': '/content/drive/MyDrive/ML-Project-Files/SavedModels/teacher_ResNet18_085.pt',
    'img_dir': '/content/ML_Project/Datasets/SyntheticCharacters/train'
}

student  = Student(graphemes_dict, teacher_data=teacher_data, variant='syntheticdata')

train_datasets = [
        {
            'img_dir': 'Datasets/Bn-HTRd/train',
            'label_file_path': 'Datasets/Bn-HTRd/train/labels.csv',
        },
        {
            'img_dir': 'Datasets/Bn-HTRd/val',
            'label_file_path': 'Datasets/Bn-HTRd/val/labels.csv'
        },
        {
            'img_dir': 'Datasets/SyntheticWords/all',
            'label_file_path': 'Datasets/SyntheticWords/all/labels.csv',
            'virtual_size': 25000,
            'synthetic': True
        },
    ]

val_datasets = [
        {
            'img_dir': 'Datasets/BanglaWriting/train',
            'label_file_path': 'Datasets/BanglaWriting/train/labels.csv'
        },
        {
            'img_dir': 'Datasets/BanglaWriting/val',
            'label_file_path': 'Datasets/BanglaWriting/val/labels.csv'
        },
    ]

train_loader, train_size = get_word_loader(train_datasets, augment=True)
val_loader, val_size = get_word_loader(val_datasets, augment=False)

save_path = '/content/drive/MyDrive/ML-Project-Files/SavedModels'

student.train(train_loader, val_loader, save_path, epochs=50)