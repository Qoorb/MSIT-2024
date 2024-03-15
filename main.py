from collections import OrderedDict
from datetime import datetime

import pandas as pd

import torch
from torch import nn
from torch.utils.data import DataLoader

from dataset import EEGDataset
from model import Autoencoder


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Project running on device: ", device)

config = {
    'input_size': 63,
    'hidden_size': 1,
    'layers': 3,
    'kernels': [3,3,3],
    'channels': [128,256,512],
    'strides': [2,2,2],
    'use_dropout': False
}

model = Autoencoder(**config).to(device)

data = EEGDataset.load_dataset("./data/dataset_EEG.pkl")
data_train = DataLoader(data, batch_size=10, shuffle=True)

criterion = nn.MSELoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

log = OrderedDict()
log['loss'] = []

print(f'[training started][time:{datetime.now()}]')

num_epochs = 5
for epoch in range(num_epochs):
    running_loss = 0.0

    for data in data_train:
        # (batch_size, channels, length)
        inputs = data.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, inputs)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    print(f'[Epoch:{epoch+1}][Loss:{running_loss / len(data_train):.2f}]')

    log['loss'].append(round(running_loss / len(data_train), 2))
    pd.DataFrame(log).to_csv('./model/log.csv',index=False)
    
    torch.save(model, './model/autoencoder')

print(f'[training ended][time:{datetime.now()}]')
