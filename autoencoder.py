from torch import nn

from torchsummary import summary


'''
TODO:
    fix dimensional losses?
'''

# Model parameters:
LAYERS = 3
KERNELS = [3, 4, 3]
CHANNELS = [128, 256, 512]
STRIDES = [2, 2, 2]
LINEAR_DIM = 64000

class Encoder(nn.Module):

    def __init__(self, input_size: int, hidden_size: int) -> None:
        super(Encoder, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.layers = LAYERS
        self.kernels = KERNELS
        self.channels = CHANNELS
        self.strides = STRIDES
        self.conv = self._create_layers()

        self.fc_dim = LINEAR_DIM
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(self.fc_dim, self.hidden_size)

    def _create_layers(self):
        conv_layers = nn.Sequential()

        for i in range(self.layers):
            if i == 0:
                conv_layers.append(nn.Conv1d(in_channels=self.input_size,
                                             out_channels=self.channels[i],
                                             kernel_size=self.kernels[i],
                                             stride=self.strides[i], 
                                             padding=1))
            else:
                conv_layers.append(nn.Conv1d(in_channels=self.channels[i-1],
                                             out_channels=self.channels[i],
                                             kernel_size=self.kernels[i],
                                             stride=self.strides[i], 
                                             padding=1))
            
            conv_layers.append(nn.ReLU())

        return conv_layers
    
    def forward(self, x):
        x = self.conv(x)
        x = self.flatten(x)

        return self.linear(x)
    

class Decoder(nn.Module):
    
    def __init__(self, hidden_size: int, output_size: int):
        super(Decoder, self).__init__()

        self.fc_dim = LINEAR_DIM
        self.hidden_size = hidden_size
        
        self.layers = LAYERS
        self.kernels = KERNELS
        self.channels = CHANNELS[::-1] # flip the channel dimensions
        self.strides = STRIDES
        
        self.linear = nn.Linear(self.hidden_size, self.fc_dim)
        self.conv =  self._create_layers()

        self.output = nn.Conv1d(self.channels[-1], output_size, kernel_size=1, stride=1, padding=1)

    def _create_layers(self):
        conv_layers = nn.Sequential()

        for i in range(self.layers):
            
            if i == 0: conv_layers.append(nn.ConvTranspose1d(self.channels[i],
                                               self.channels[i],
                                               kernel_size=self.kernels[i],
                                               stride=self.strides[i],
                                               padding=1))
            
            else: conv_layers.append(nn.ConvTranspose1d(self.channels[i-1], 
                                               self.channels[i],
                                               kernel_size=self.kernels[i],
                                               stride=self.strides[i],
                                               padding=1,
                                               output_padding=1))

            conv_layers.append(nn.ReLU())

        return conv_layers
        
    def forward(self, x):
        x = self.linear(x)
        x = x.view(x.size(0), 512, 125)
        x = self.conv(x)
        
        return self.output(x)


class AutoEncoder(nn.Module):
    
    def __init__(self):
        super(AutoEncoder, self).__init__()
        
        self.encoder = Encoder(input_size=63, hidden_size=2)
        self.decoder = Decoder(hidden_size=2, output_size=63)
        
    def forward(self, x):
        return self.decoder(self.encoder(x))


if __name__ == '__main__':
    summary(AutoEncoder(), (63, 1000), batch_size=10)