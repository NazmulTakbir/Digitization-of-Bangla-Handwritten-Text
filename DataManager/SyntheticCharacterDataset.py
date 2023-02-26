from torch.utils.data import Dataset
import cv2
import os
import numpy as np
class SyntheticCharacterDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.inp_h = 32
        self.inp_w = 32
        self.transform = transform
        self.img_names = sorted(os.listdir(img_dir))

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        image = cv2.imread(os.path.join(self.img_dir, img_name))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        img_h, img_w = image.shape
        image = cv2.resize(image, (0,0), fx=self.inp_w/img_w, fy=self.inp_h/img_h, interpolation=cv2.INTER_CUBIC)
        image = np.reshape(image, (self.inp_h, self.inp_w, 1))

        if self.transform is not None:
            image = self.transform(image = image)["image"]
        else:
            image = image.transpose(2, 0, 1)

        label = int(img_name.split('_')[0])-1
    
        return image/255.0, label