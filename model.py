import torch
from torch import nn
from torch.utils.data import DataLoader

from dataset import EEGDataset


class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(63 * 1000, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU()
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 63 * 1000),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.encoder(x)
        x = self.decoder(x)
        x = x.view(x.size(0), 63, 1000)
        
        return x


if __name__ == '__main__':

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = Autoencoder().to(device)

    data = EEGDataset.load_dataset("./data/dataset_EEG.pkl")
    data_train = DataLoader(data, batch_size=4, shuffle=True)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 10
    for epoch in range(num_epochs):
        running_loss = 0.0
        for data in data_train:
            inputs = data

            optimizer.zero_grad()

            outputs = model(inputs)

            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f'Epoch {epoch+1}, Loss: {running_loss / len(data_train)}')

    print('training ended')