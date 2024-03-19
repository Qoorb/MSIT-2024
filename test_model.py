from torch import nn

from torchsummary import summary


class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(63 * 1000, 9000),
            nn.ELU(),
            nn.Linear(9000, 1000),
            nn.ELU(),
            nn.Linear(1000, 100),
            nn.ELU()
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(100, 1000),
            nn.ELU(),
            nn.Linear(1000, 9000),
            nn.ELU(),
            nn.Linear(9000, 63 * 1000),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.encoder(x)
        x = self.decoder(x)
        x = x.view(x.size(0), 63, 1000)
        
        return x


class Conv_autoencoder(nn.Module):
    def __init__(self) -> None:
        super(Conv_autoencoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv1d(63, 128, 3, 2, padding=1),
            nn.ELU(),
            nn.Conv1d(128, 256, 3, 2, padding=1),
            nn.ELU(),
            nn.Conv1d(256, 512, 3, 2, padding=1),
            nn.ELU(),
            nn.Conv1d(512, 1024, 5, 5),
            nn.ELU(),
            nn.Conv1d(1024, 2048, 5, 5),
            nn.ELU(),
            nn.Conv1d(2048, 4096, 5, 5)
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(4096, 2048, 5, 5),
            nn.ELU(),
            nn.ConvTranspose1d(2048, 1024, 5, 5),
            nn.ELU(),
            nn.ConvTranspose1d(1024, 512, 5, 5),
            nn.ELU(),
            nn.ConvTranspose1d(512, 256, 3, 2, padding=1, output_padding=1),
            nn.ELU(),
            nn.ConvTranspose1d(256, 128, 3, 2, padding=1, output_padding=1),
            nn.ELU(),
            nn.ConvTranspose1d(128, 63, 3, 2, padding=1, output_padding=1)
        )
    
    def forward(self, x):
        return self.decoder(self.encoder(x))
        
    

if __name__ == '__main__':
    summary(Conv_autoencoder(), (63, 1000), batch_size=10)
