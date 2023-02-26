import torch
from .BasicConv import BasicConv
import torch.nn as nn
import torch.optim as optim
import time
import numpy as np
class Teacher():

    def __init__(self, teacher_type, n_classes):
        self.teacher_type = teacher_type
        self.n_classes = n_classes
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        if teacher_type == 'BasicConv':
            self.model =  BasicConv(n_classes)
        elif teacher_type == 'ResNet18':
            pass
        else:
            raise ValueError('Teacher type not supported')

        self.model.to(self.device)

    def train(self, train_loader, val_loader, n_epochs=10, lr=0.001, verbose_freq=1):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        start_time = time.time()
        
        self.model.train()
        for epoch in range(1, n_epochs+1):
            running_loss = 0.0
            for images, labels in train_loader:
                images, labels = images.to(self.device), labels.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
            
            if epoch % verbose_freq == 0:
                self.print_train_metrics(train_loader, val_loader, epoch, n_epochs, start_time, running_loss)

        print("Training finished.")

    def print_train_metrics(self, train_loader, val_loader, epoch, n_epochs, start_time, running_loss):
        epoch_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch}/{n_epochs}, Loss: {epoch_loss:.2f}")
        self.test('Train', train_loader)
        self.test('Validation', val_loader)
        print(f"Time elapsed: {round(time.time() - start_time)} seconds\n")

    def test(self, test_name, test_loader):
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print(f"{test_name} set accuracy: {100 * correct / total:.2f}%")

    def get_logits(self, data_loader):
        self.model.eval()
        logits = []
        with torch.no_grad():
            for images, labels in data_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                logits.append(outputs.cpu().numpy())
        return np.concatenate(logits)

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))

    def stack_predictions(self, batch):
        self.model.eval()