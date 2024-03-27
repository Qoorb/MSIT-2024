import torch
from torch import nn

from torchsummary import summary


class Encoder(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, 
                 layers: int, kernels: list[int],
                 channels: list[int], strides: list[int],
                 use_dropout: bool = True) -> None:
        super(Encoder, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.use_dropout = use_dropout

        self.layers = layers
        self.kernels = kernels
        self.channels = channels
        self.strides = strides
        self.conv = self._create_layers()

        self.fc_dim = 512000
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(self.fc_dim, self.hidden_size)

    def _create_layers(self) -> nn.Sequential:
        conv_layers = nn.Sequential()

        for i in range(self.layers):
            if i == 0:
                conv_layers.append(nn.Conv2d(in_channels=1,
                                             out_channels=self.channels[i],
                                             kernel_size=self.kernels[i],
                                             stride=self.strides[i],
                                             padding=1))
            else:
                conv_layers.append(nn.Conv2d(in_channels=self.channels[i-1],
                                             out_channels=self.channels[i],
                                             kernel_size=self.kernels[i],
                                             stride=self.strides[i], 
                                             padding=1))
            
            conv_layers.append(nn.BatchNorm2d(self.channels[i]))

            conv_layers.append(nn.ELU())

            if self.use_dropout:
                conv_layers.append(nn.Dropout2d(0.1))

        return conv_layers
    
    def forward(self, x) -> torch.Tensor:
        x = self.conv(x)
        x = self.flatten(x)

        return self.linear(x)
    

class Decoder(nn.Module):
    
    def __init__(self, hidden_size: int, output_size: int,
                 layers: int, kernels: list[int],
                 channels: list[int], strides: list[int],
                 use_dropout: bool = True) -> None:
        super(Decoder, self).__init__()

        self.fc_dim = 512000
        self.hidden_size = hidden_size

        self.use_dropout = use_dropout
        
        self.layers = layers
        self.kernels = kernels
        self.channels = channels[::-1] # flip the channel dimensions
        self.strides = strides
        
        self.linear = nn.Linear(self.hidden_size, self.fc_dim)
        self.conv =  self._create_layers()

        self.output = nn.Conv2d(self.channels[-1], output_size, kernel_size=1, stride=1)

    def _create_layers(self) -> nn.Sequential:
        conv_layers = nn.Sequential()

        for i in range(self.layers):
            
            if i == 0: conv_layers.append(nn.ConvTranspose2d(self.channels[i],
                                               self.channels[i],
                                               kernel_size=self.kernels[i],
                                               stride=self.strides[i],
                                               padding=1,
                                               output_padding=1))
            
            else: conv_layers.append(nn.ConvTranspose2d(self.channels[i-1], 
                                               self.channels[i],
                                               kernel_size=self.kernels[i],
                                               stride=self.strides[i],
                                               padding=1,
                                               output_padding=1))

            if i != self.layers - 1:
                conv_layers.append(nn.BatchNorm2d(self.channels[i]))

            conv_layers.append(nn.ELU())

            if self.use_dropout:
                conv_layers.append(nn.Dropout2d(0.1))

        return conv_layers
        
    def forward(self, x) -> torch.Tensor:
        x = self.linear(x)
        x = x.view(x.size(0), 512, 8, 125) # hardcoded
        x = self.conv(x)
        
        return self.output(x)


class Autoencoder(nn.Module):
    
    def __init__(self, input_size: int, hidden_size: int,
                 layers: int, kernels: list[int],
                 channels: list[int], strides: list[int],
                 use_dropout: bool = True) -> None:
        super(Autoencoder, self).__init__()
        
        self.params = [layers, kernels, channels, strides, use_dropout]

        self.encoder = Encoder(input_size, hidden_size, *self.params)
        self.decoder = Decoder(hidden_size, input_size, *self.params)
        
    def forward(self, x) -> torch.Tensor:
        return self.decoder(self.encoder(x))


if __name__ == '__main__':
    summary(Autoencoder(1, 1, 3, [3,3,3], [128,256,512], [2,2,2]), (1, 63, 1000), batch_size=10)