import torch
import torch.optim as optim
from tqdm import tqdm
from torch import nn

from Graphemes.utils import normalize_word, ads_grapheme_extraction
from .CRNN import CRNN
from Metrics.metrics import recognition_metrics, accuracy_metrics, print_metric
class Student:

    def __init__(self, graphemes_dict, extractor_type='VGG', teacher=None):

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.graphemes_dict = graphemes_dict
        self.inv_graphemes_dict = {v: k for k, v in graphemes_dict.items()}
        self.n_classes = len(graphemes_dict)+1
        self.model = CRNN(extractor_type, self.n_classes)
        self.model.to(self.device)
        self.teacher = teacher
        self.ctc_loss = torch.nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True).to(self.device)

    def words_to_labels(self, words):
        labels = []
        lengths = []
        maxlen = 0
        for word in words:
            word = normalize_word(word)
            label = []
            for grapheme in ads_grapheme_extraction(word):
                label.append(self.graphemes_dict[grapheme])
            labels.append(label)
            lengths.append(len(label))
            maxlen = max(len(label), maxlen)

        # pad all labels to the same length - maxlen of current batch
        for i in range(len(labels)):
            labels[i] = labels[i] + [0]*(maxlen-len(labels[i]))
        
        labels = torch.tensor(labels, dtype=torch.long).to(self.device)
        lengths = torch.tensor(lengths, dtype=torch.long).to(self.device)

        return labels, lengths

    def init_training(self, lr):
        self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr, weight_decay=1e-05)
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, T_0=15, T_mult=1, eta_min=0.0001)
    
    def init_train_loop(self, epoch):
        self.model.train()
        self.y_true = []
        self.y_pred = []
        self.pred_ = []
        self.label_ = []
        self.decoded_preds = []
        self.decoded_labels = []
        self.total_wer = 0
        self.batch_loss = 0
        self.alpha = 0.5
        self.t = 2

    def add_teacher_loss(self, loss, logits, labels):
        probs = torch.nn.functional.log_softmax(logits/self.t , dim=2)
        teacher_probs = self.teacher.get_stacked_probs(labels).cuda()
        ty = nn.KLDivLoss()(probs , teacher_probs)
        loss = ty * (self.t*self.t * 2.0 + self.alpha) + loss * (1.-self.alpha)
        return loss

    def decode_prediction(self, pred):
        grapheme_list = []
        grapheme_id_list = []

        for i in range(len(pred)):
            if pred[i] != 0 and (i == 0 or pred[i] != pred[i-1]):
                grapheme_list.append(self.inv_graphemes_dict[pred[i]])
                grapheme_id_list.append(pred[i])

        return grapheme_id_list, ''.join(grapheme_list)
    
    def decode_label(self, label):
        decoded = []
        for i in range(len(label)):
            if label[i] != 0:
                decoded.append(self.inv_graphemes_dict[label[i]])
        return ''.join(decoded)
    
    def train(self, train_loader, epochs=30, lr = 0.0003):
        self.init_training(lr)

        for epoch in range(epochs):
            self.init_train_loop(epoch)
            for images, words in tqdm(train_loader):
                images = images.to(self.device).float() / 255.
                labels, label_lengths = self.words_to_labels(words)
                self.batch_size = images.size(0)
                
                logits = self.model(images)
                probs = torch.nn.functional.log_softmax(logits , dim=2)
                probs_size = torch.tensor([probs.size(0)] * self.batch_size, dtype=torch.long).to(self.device)

                loss = self.ctc_loss(probs, labels, probs_size, label_lengths)
                if self.teacher is not None:
                    loss = self.add_teacher_loss(loss, logits, labels)
                
                self.batch_loss += loss.item()                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                self.save_mini_batch_results(probs, labels)
        
            self.scheduler.step()
            self.print_epoch_stats(epoch)
    
    def save_mini_batch_results(self, probs, labels):
        _, preds = probs.max(2)
        preds = preds.transpose(1, 0).contiguous().detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()

        for pred, label in zip(preds, labels):
            grapheme_id_list, grapheme = self.decode_prediction(pred)

            self.decoded_preds.append(grapheme)
            self.decoded_labels.append(self.decode_label(label))

            min_len = min(len(grapheme_id_list), len(label))
            self.y_true.extend(grapheme_id_list[:min_len])
            self.y_pred.extend(list(label)[:min_len])

    def print_epoch_stats(self, epoch):
        print(f"Epoch: {epoch}:")
        print_metric("Training loss", self.batch_loss/self.batch_size)
        recognition_metrics(self.decoded_preds, self.decoded_labels, final_action='print')
        accuracy_metrics(self.y_true, self.y_pred, self.n_classes, final_action='print',
                         target_names=[v for _, v in self.inv_graphemes_dict.items()])
        print("_"*75)

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))