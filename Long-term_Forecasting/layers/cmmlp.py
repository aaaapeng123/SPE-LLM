import torch
import torch.nn as nn
class ChannelMixMLP(nn.Module):
    '''
    Multilayer perceptron with Channel Mixing for high dimension sequential data encoding/decoding
    '''
    def __init__(self,
                 f_in,
                 f_out,
                 hidden_dim=256,
                 hidden_layers=2,
                 dropout=0.1,
                 activation='tanh'):
        super(ChannelMixMLP, self).__init__()
        self.f_in = f_in
        self.f_out = f_out
        self.hidden_dim = hidden_dim
        self.hidden_layers = hidden_layers
        self.dropout = dropout
        # Activation selection
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        else:
            raise NotImplementedError
        # Input linear layer
        self.time_mixer = nn.Linear(self.f_in, self.hidden_dim)  # Time mixing (sequence dimension)
        # Channel mixing layers (mixing across features)
        self.channel_mixer = nn.Linear(self.hidden_dim, self.hidden_dim)  # Channel mixing (feature dimension)
        # Building the hidden layers
        layers = [self.time_mixer, self.activation, nn.Dropout(self.dropout)]
        for i in range(self.hidden_layers - 1):
            # Add a sequence of time mixer and channel mixer layers
            layers += [
                self.time_mixer,  # Channel mixing layer
                self.activation,
                nn.Dropout(self.dropout)
            ]
        # Final output layer
        layers += [nn.Linear(self.hidden_dim, self.f_out)]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        # x: B x S x f_in  -> Input is batch x sequence x features

        # Apply the time and channel mixing sequentially
        y = self.layers(x)  # Mixing along both the time (sequence) and channel (feature) dimensions

        return y
