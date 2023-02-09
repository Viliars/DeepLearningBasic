import os
import string
from itertools import groupby

import cv2
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm

CHARS = string.digits + string.ascii_lowercase
CHAR_LABEL = {CHARS[idx]: idx for idx in range(len(CHARS))}

class OCRDataset(torch.utils.data.Dataset):
    
    def __init__(self, data_path, mode="train"):
        images, labels = [], []
        self.mode = mode

        for name in os.listdir(data_path):
            images.append(os.path.join(data_path, name))
            labels.append(name.split(".")[0])

        inp_train, inp_test, label_train, label_test = train_test_split(images, \
                                                    labels, test_size=0.2, random_state=123)
                 
        self.inp_train = inp_train
        self.inp_test = inp_test
        self.label_train = label_train
        self.label_test = label_test
                 
            
    def to_tensor(self, label):
        res = []
        for s in label:
            res.append(CHAR_LABEL[s])
        res = torch.LongTensor(res)
        return res

    def __len__(self):
        return len(self.inp_train) if self.mode == "train" else len(self.inp_test)

    def __getitem__(self, idx):
        if self.mode == "train":
            img = self.inp_train[idx]
            label = self.to_tensor(self.label_train[idx])
        elif self.mode == "test":
            img = self.inp_test[idx]
            label = self.to_tensor(self.label_test[idx])
                 
        image = cv2.imread(img, 0).astype(np.float32)
        image = cv2.resize(image, (100, 32))
        image = torch.FloatTensor(image)[..., None]
        
        image = image.permute(2, 0, 1)
        return image, label
