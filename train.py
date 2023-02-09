import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchmetrics import CharErrorRate
from tqdm.auto import tqdm
import os

DEVICE = "cuda"
SAVE_FREQ = 10

CHARS = string.digits + string.ascii_lowercase
LABEL_CHAR = { idx: CHARS[idx] for idx in range(len(CHARS)) }

def label_to_string(label, pred=False):
    res = []
    for s in label:
        if s in LABEL_CHAR.keys():
            res.append(LABEL_CHAR[s])
        else:
            res.append("-")
            
    if pred:
        res = [s for s, _ in groupby(res) if s != '-']
    return res


def train_epoch(model, train_loader, criterion, optimizer):

    model.train()
    total_loss = 0.0
    n = 0

    predicted_words = []
    target_words = []

    for batch in tqdm(iter(train_loader)):
        inputs, labels = batch
        inputs = inputs.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()
        output = model(inputs)
        
        input_lengths = torch.LongTensor([output.size(0)] * output.size(1))
        target_lengths = torch.LongTensor([5 for _ in range(output.size(1))])

        loss = criterion(output, labels, input_lengths, target_lengths)
        loss.backward()
        optimizer.step()

        predicted = torch.argmax(output.detach().cpu(), dim=2).permute(1, 0).numpy()
        target = labels.cpu().detach().numpy()
        
        for i in range(predicted.shape[0]):
            code = label_to_string(predicted[i], pred=True)
            predicted_words.append(code)
            
        for i in range(target.shape[0]):
            a = label_to_string(target[i])
            target_words.append(a)

        total_loss += loss
        n += inputs.size(0)

    total_loss /= n
    return total_loss, predicted_words, target_words


def eval_epoch(model, val_loader, criterion):

    model.eval()
    
    total_loss = 0.0
    n = 0

    predicted_words = []
    target_words = []

    for batch in tqdm(iter(val_loader)):
        inputs, labels = batch
        inputs = inputs.to(DEVICE)
        labels = labels.to(DEVICE)

        with torch.no_grad():
            output = model(inputs)

            input_lengths = torch.LongTensor([output.size(0)] * output.size(1))
            target_lengths = torch.LongTensor([5 for _ in range(output.size(1))])

            loss = criterion(output, labels, input_lengths, target_lengths)

            predicted_batch = torch.argmax(output.cpu().detach(), dim=2).permute(1, 0).numpy()
            target_batch = labels.cpu().detach().numpy()

            for w in predicted_batch:
                predicted_words.append(label_to_string(w, pred=True))
            
            for w in target_batch:
                target_words.append(label_to_string(w))

            total_loss += loss
            n += inputs.size(0)

    return total_loss / n, predicted_words, target_words


def train(train_loader, test_loader, model, 
          optimizer, criterion, epochs, save_path):

    train_losses = []
    test_losses = []

    train_cers = []
    test_cers = []

    cer = CharErrorRate()

    best_model = None
    best_cer = 1000

    for epoch in range(epochs):
        train_loss, train_predicted, train_target = train_epoch(model, train_loader, criterion, optimizer)
        test_loss, test_predicted, test_target = eval_epoch(model, test_loader, criterion)
        
        train_cer = cer(train_predicted, train_target).numpy()
        test_cer = cer(test_predicted, test_target).numpy()

        train_losses.append(train_loss)
        test_losses.append(test_loss)

        train_cers.append(train_cer)
        test_cers.append(test_cer)

        if test_cer < best_cer:
            best_model = model
            best_cer = test_cer
        
        if epoch % SAVE_FREQ == 0:
            torch.save(model.state_dict(), os.path.join(save_path, "model_{}.pth".format(epoch)))
            
        print("Epoch {}, train loss: {}, train  CER: {} , test loss: {}, test CER: {}".format(epoch, \
                                                                        train_loss, train_cer, test_loss, test_cer))
        
    torch.save(best_model.state_dict(), os.path.join(save_path, "best.pth"))

    return train_losses, test_losses, train_cers, test_cers
