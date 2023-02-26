import os
import cv2
import numpy as np
from torch.utils import data

class WordDataset(data.Dataset):
    def __init__(self, img_dir, label_file_path, inp_h=32, inp_w=128, transform=None):
        """
            label_file is a csv file with the following format:
            filename1,word1
            filename2,word2
            ....
            filenameN,wordN
            All filenames should be have the same length
            No header should be present
            Each file is an image of a word
        """
        self.inp_h = inp_h
        self.inp_w = inp_w
        self.transform = transform

        label_file = open(label_file_path, "r").readlines()
        img_names = [line.split(",")[0] for line in label_file]
        img_paths = [os.path.join(img_dir, img_name) for img_name in img_names]
        words = [line[len(img_names[0])+1:].strip() for line in label_file]
        self.data = list(zip(img_paths, words))
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = cv2.imread(self.data[idx][0])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        img_h, img_w = image.shape
        image = cv2.resize(image, (0,0), fx=self.inp_w/img_w, fy=self.inp_h/img_h, interpolation=cv2.INTER_CUBIC)
        image = np.reshape(image, (self.inp_h, self.inp_w, 1))

        if self.transform is not None:
            image = self.transform(image = image)["image"]
            return image, self.data[idx][1]
        else:
            image = image.transpose(2, 0, 1)
            
        return image, self.data[idx][1]