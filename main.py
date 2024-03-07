import torch
from torch import nn
from torch.utils.data import DataLoader

from model import Autoencoder
from dataset import EEGDataset


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Project running on device: ", device)

model = Autoencoder().to(device)

data = EEGDataset.load_dataset("./data/dataset_EEG.pkl")
data_train = DataLoader(data, batch_size=4, shuffle=True)

criterion = nn.MSELoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

num_epochs = 1000
for epoch in range(num_epochs):
    running_loss = 0.0
    for data in data_train:
        inputs = data.to(device)

        optimizer.zero_grad()
        
        outputs = model(inputs)
        loss = criterion(outputs, inputs)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    print(f'Epoch {epoch+1}, Loss: {running_loss / len(data_train)}')

print('training ended')
