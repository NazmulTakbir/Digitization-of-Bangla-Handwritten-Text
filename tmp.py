import json

from Models.Student.Student import Student
from DataManager.WordLoader import get_word_loader

# from DataManager.WordLoader import get_word_loader

with open('Graphemes/Extracted/graphemes_BanglaWriting.json', 'r') as f:
    graphemes_dict = json.load(f)

student  = Student(graphemes_dict)

train_loader, train_size = get_word_loader('Datasets/BanglaWriting/train', 
                                           'Datasets/BanglaWriting/train/labels.csv',
                                           batch_size=2)

student.train(train_loader)