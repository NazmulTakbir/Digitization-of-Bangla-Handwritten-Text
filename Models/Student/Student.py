import torch
import torch.optim as optim
from tqdm import tqdm
from torch import nn
import numpy as np
import os

from Graphemes.extract_graphemes import decode_prediction, decode_label, words_to_labels
from .CRNN import CRNN
from Metrics.metrics import recognition_metrics, accuracy_metrics, print_metric
class Student:

    def __init__(self, graphemes_dict, extractor_type='VGG', teacher=None):

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        if teacher is None:
            self.student_type = extractor_type + '_hasteacher'
        else:
            self.student_type = extractor_type + '_noteacher'
        self.graphemes_dict = graphemes_dict
        self.inv_graphemes_dict = {v: k for k, v in graphemes_dict.items()}
        self.n_classes = len(graphemes_dict)+1
        self.model = CRNN(extractor_type, self.n_classes)
        self.model.to(self.device)
        self.teacher = teacher
        self.ctc_loss = torch.nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True).to(self.device)

    def init_training(self, lr, save_dir):
        self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr, weight_decay=1e-05)
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, T_0=15, T_mult=1, eta_min=0.0001)
        self.best_wrr = 0
        self.best_epoch = 0
        self.save_dir = save_dir

    def init_epoch(self, epoch, train=True):
        if train:
            self.model.train()
        else:
            self.model.eval()
        self.y_true = []
        self.y_pred = []
        self.decoded_preds = []
        self.decoded_labels = []
        self.batch_loss = 0
        self.alpha = 0.5
        self.t = 2
        self.epoch = epoch

    def add_teacher_loss(self, loss, logits, labels):
        probs = torch.nn.functional.log_softmax(logits/self.t , dim=2)
        teacher_probs = self.teacher.get_stacked_probs(labels).cuda()
        ty = nn.KLDivLoss()(probs , teacher_probs)
        loss = ty * (self.t*self.t * 2.0 + self.alpha) + loss * (1.-self.alpha)
        return loss
    
    def forward(self, images, words=None):
        images = images.to(self.device).float() / 255.
        logits = self.model(images)
        probs = torch.nn.functional.log_softmax(logits , dim=2)
        self.batch_size = images.size(0)

        if words is not None:
            labels, label_lengths = words_to_labels(words, self.graphemes_dict)
            probs_size = torch.tensor([probs.size(0)] * self.batch_size, dtype=torch.long).to(self.device)

            loss = self.ctc_loss(probs, labels, probs_size, label_lengths)
            if self.teacher is not None:
                loss = self.add_teacher_loss(loss, logits, labels)
            self.batch_loss += loss.item()  

            return probs, labels, loss
        
        return probs

    def train(self, train_loader, val_loader, save_dir, epochs=30, lr = 0.0003):
        self.init_training(lr, save_dir)

        for epoch in range(1, epochs+1):
            self.init_epoch(epoch)
            print("Training Epoch: ", str(epoch)+"/"+str(epochs))
            for images, words in tqdm(train_loader):
                probs, labels, loss = self.forward(images, words)
                              
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                self.save_mini_batch_results(probs, labels)
        
            self.scheduler.step()
            self.print_stats('Train')
            self.validate(epoch, val_loader)
            print("_"*75)
    
    def validate(self, epoch, val_loader):
        self.init_epoch(epoch, train=False)
        with torch.no_grad():
            print("Validating:")
            for images, words in tqdm(val_loader):
                probs, labels, loss = self.forward(images, words)
                self.save_mini_batch_results(probs, labels)
            self.print_stats('Validation')

    def save_mini_batch_results(self, probs, labels):
        _, preds = probs.max(2)
        preds = preds.transpose(1, 0).contiguous().detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()

        for pred, label in zip(preds, labels):
            grapheme_id_list, grapheme = decode_prediction(pred, self.inv_graphemes_dict)

            self.decoded_preds.append(grapheme)
            self.decoded_labels.append(decode_label(label, self.inv_graphemes_dict))

            min_len = min(len(grapheme_id_list), len(label))
            self.y_true.extend(grapheme_id_list[:min_len])
            self.y_pred.extend(list(label)[:min_len])

    def print_stats(self, data_set):
        print_metric(f"{data_set} loss", self.batch_loss/self.batch_size)
        wrr = recognition_metrics(self.decoded_preds, self.decoded_labels, final_action='both')['wrr']
        accuracy_metrics(self.y_true, self.y_pred, self.n_classes, final_action='print',
                         target_names=[v for _, v in self.inv_graphemes_dict.items()])
        self.print_samples()

        if data_set == 'Validation' and wrr > self.best_wrr:
            self.best_wrr = wrr
            self.best_epoch = self.epoch
            self.save_best_model()

    def save_best_model(self):
        prev = os.path.join(self.save_dir, f"student_{self.student_type}_{str(self.best_epoch).zfill(3)}.pt") 
        if os.path.exists(prev):
            os.remove(prev)
        self.save_model(os.path.join(self.save_dir, f"student_{self.student_type}_{str(self.best_epoch).zfill(3)}.pt"))

    def print_samples(self, sample_size=5):
        total = len(self.decoded_preds)
        sample = np.random.choice(total, sample_size, replace=False)
        print("Actual :: Predicted")
        for i in sample:
            print(f"{self.decoded_labels[i]} :: {self.decoded_preds[i]}", end="  |||  ")
        print("")

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)