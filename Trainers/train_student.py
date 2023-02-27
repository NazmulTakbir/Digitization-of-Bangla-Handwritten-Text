import json

from Models.Student.Student import Student
from DataManager.WordLoader import get_word_loader

with open('Graphemes/Extracted/graphemes_bw_bnhtrd_syn.json', 'r') as f:
    graphemes_dict = json.load(f)

student  = Student(graphemes_dict)

train_imgs = ['Datasets/Bn-HTRd/train', 'Datasets/Bn-HTRd/val']
train_labels = ['Datasets/Bn-HTRd/train/labels.csv', 'Datasets/Bn-HTRd/val/labels.csv']

val_imgs = ['Datasets/BanglaWriting/train', 'Datasets/BanglaWriting/val']
val_labels = ['Datasets/BanglaWriting/train/labels.csv', 'Datasets/BanglaWriting/val/labels.csv']

train_loader, train_size = get_word_loader(train_imgs, train_labels, batch_size=2)
val_loader, val_size = get_word_loader(val_imgs, val_labels, batch_size=2)

student.train(train_loader, val_loader)