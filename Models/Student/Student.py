import torch
import torch.optim as optim

from CRNN import CRNN


class Student:

    def __init__(self, extractor_type='VGG'):

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.model = CRNN(extractor_type, 10)
        self.model.to(self.device)

    def train(self, train_loader, epochs=30, lr = 0.0003):
        self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr, weight_decay=1e-05)
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, T_0=15, T_mult=1, eta_min=0.0001)

        for epoch in range(1,epochs+1):
            self.model.train()

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))