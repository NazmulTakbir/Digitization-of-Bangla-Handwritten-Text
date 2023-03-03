from torch.utils import data
from Synthesizers.Words.generate import word_generator
import numpy as np
from ImageProcessing.image_processing import adjust_contrast_grey

class SyntheticWordDataset(data.Dataset):
    def __init__(self, inp_h=32, inp_w=128, transform=None):
        self.inp_h = inp_h
        self.inp_w = inp_w
        self.transform = transform
        
    def __len__(self):
        # increasing this number increases the proportion synthetic words in the dataset
        return 100000

    def __getitem__(self, idx):
        image, word = word_generator(self.inp_h, self.inp_w)
        image = np.array(image)
        image = adjust_contrast_grey(image)

        if self.transform is not None:
            image = self.transform(image = image)["image"]
        else:
            image = image.transpose(2, 0, 1)
            
        return image, word