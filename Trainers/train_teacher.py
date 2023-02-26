from DataManager.SyntheticCharacterLoader import SyntheticCharacterLoader
from Models.Teacher.Teacher import Teacher

teacher = Teacher('BasicConv', 178)

train_loader = SyntheticCharacterLoader('../Datasets/SyntheticCharacters/train')
val_loader = SyntheticCharacterLoader('../Datasets/SyntheticCharacters/val', batch_size=1024)

teacher.train(train_loader, val_loader, n_epochs=100, lr=0.001, verbose_freq=5)
logits = teacher.get_logits(val_loader)
print(logits.shape)