import random
import matplotlib.pyplot as plt
import numpy as np
import torch
import os

from data_loader import *
from model import *
from train import *

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

DEVICE = "cuda"

DATA_PATH = "~/ml-hdd/laba-dataset/samples"

CLASSES = 37
LR = 1e-3

EPOCHS = 100
BATCH_SIZE = 64
SAVE_PATH = "./experiment"

def main():
    
    if not os.path.exists(SAVE_PATH):
        os.mkdir(SAVE_PATH)
    
    train_dataset = OCRDataset(data_path=DATA_PATH, mode="train")
    test_dataset = OCRDataset(data_path=DATA_PATH, mode="test")
    
    model = OCRModel(num_classes=CLASSES)
    model.to(DEVICE)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = torch.nn.CTCLoss(reduction="sum", zero_infinity=True, blank=36)
    criterion.to(DEVICE)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    train_losses, test_losses, train_cers, test_cers = train(train_loader, \
                                                            test_loader, model, optimizer, criterion, EPOCHS, SAVE_PATH)
    #write results
    with open(os.path.join(SAVE_PATH, "train_losses.txt"), 'w') as f:
        for l in train_losses:
            l = l.item()
            f.write(str(l) + ' ')
    with open(os.path.join(SAVE_PATH, "test_losses.txt"), 'w') as f:
        for l in test_losses:
            l = l.item()
            f.write(str(l) + ' ')
    with open(os.path.join(SAVE_PATH, "train_cers.txt"), 'w') as f:
        for l in train_cers:
            l = l.item()
            f.write(str(l) + ' ')
    with open(os.path.join(SAVE_PATH, "test_cers.txt"), 'w') as f:
        for l in test_cers:
            l = l.item()
            f.write(str(l) + ' ')
    
if __name__ == "__main__":
    main()
