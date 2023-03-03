from torch.utils import data
from Synthesizers.Words.generate import word_generator
import numpy as np
from ImageProcessing.image_processing import adjust_contrast_grey
from Graphemes.utils import skip_chars

class SyntheticWordDataset(data.Dataset):
    def __init__(self, inp_h=32, inp_w=128, transform=None):
        self.inp_h = inp_h
        self.inp_w = inp_w
        self.transform = transform
        
    def __len__(self):
        # increasing this number increases the proportion synthetic words in the dataset
        return 25000

    def __getitem__(self, idx):
        while True:
            image, word = word_generator(self.inp_h, self.inp_w)
            if not any((c in skip_chars) for c in word):
                break
        
        image = np.array(image)
        image = adjust_contrast_grey(image)

        if self.transform is not None:
            image = self.transform(image = image)["image"]
        else:
            image = image.transpose(2, 0, 1)
            
        return image, word