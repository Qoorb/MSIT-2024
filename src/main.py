from collections import OrderedDict

import pandas as pd

import torch
from torch import nn
from torch.utils.data import DataLoader

from dataset import EEGDataset
from models import Autoencoder
from models.processing import train, validate


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Project running on device: ", device)

config = {
    'input_size': 1,
    'hidden_size': 2,
    'layers': 3,
    'kernels': [3,3,3],
    'channels': [128,256,512],
    'strides': [2,2,2],
    'use_dropout': True
}

model = Autoencoder(**config).to(device)

train_data = EEGDataset.load_dataset('./data/train_dataset_EEG.pkl')
valid_data = EEGDataset.load_dataset('./data/test_dataset_EEG.pkl')

train_loader = DataLoader(train_data, batch_size=10, shuffle=True)
valid_loader = DataLoader(valid_data, batch_size=10, shuffle=True)

criterion = nn.MSELoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001 * 10)

log = OrderedDict()
log['train_loss'] = []
log['R2_train'] = []
log['validation_loss'] = []
log['R2_validation'] = []

epoch = 10
for i in range(epoch):
    curr_lr = float(optimizer.param_groups[0]["lr"])

    train_loss, r2_train = train(model, train_loader, criterion, optimizer, device)
    valid_loss, r2_valid = validate(model, valid_loader, criterion, optimizer, device)

    print(f"[Epoch: {i+1}/{epoch}]\n[Train loss: {train_loss:.4f}][R2 train: {r2_train:.4f}][Validation loss: {valid_loss:.4f}][R2 valid: {r2_valid:.4f}][lr: {curr_lr:.4f}]")

    log['train_loss'].append(train_loss)
    log['R2_train'].append(r2_train.detach().numpy())
    log['validation_loss'].append(valid_loss)
    log['R2_validation'].append(r2_valid.detach().numpy())

    pd.DataFrame(log).to_csv('./model/log.csv',index=False)

torch.save(model, './model/autoencoder')
