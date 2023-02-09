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

test_dataset = OCRDataset(data_path=DATA_PATH, mode="test")

model = OCRModel(num_classes=CLASSES)
model.to(DEVICE)

weights = torch.load("./experiment/best.pth")
model.load_state_dict(weights)
model.eval()

for img in test_dataset:
    x, y = img
    x = torch.unsqueeze(x.to(DEVICE), 0)
    out = model(x)
    out = torch.argmax(out.cpu().detach(), dim=2).permute(1, 0).numpy()
    w = label_to_string(out[0], is_prediction=True)
    target = label_to_string(y.numpy())
    if w != target:
        print(w, target)
