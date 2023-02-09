import torch
import torch.nn as nn
import torch.nn.functional as F

class OCRModel(torch.nn.Module):
    def __init__(self, num_classes):
        super(OCRModel, self).__init__()
        
        #feature extractor
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(2, 2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(2, 2),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=(2, 1)),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.MaxPool2d(kernel_size=(2, 1)),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=2, stride=1, padding=0),
        )
        
        self.lin = nn.Linear(512, 64)
        self.lstm = nn.LSTM(64, 256, num_layers=1, bidirectional=True)
        self.fc = nn.Linear(256*2, num_classes)

    def forward(self, x):
        x = self.layer1(x)
        
        b, ch, h, w = x.size()
        x = x.view(b, ch * h, w)
        x = x.permute(2, 0, 1)
        
        x = self.lin(x)
        x, _ = self.lstm(x)
        x = self.fc(x)
        return F.log_softmax(x, dim=2)
