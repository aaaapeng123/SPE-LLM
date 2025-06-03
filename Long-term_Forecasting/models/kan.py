
import torch
import torch.nn as nn
class LearnableActivationFunction(nn.Module):
    """
    A learnable activation function that applies a transformation with trainable parameters.
    """
    def __init__(self, hidden_dim):
        super(LearnableActivationFunction, self).__init__()
        # Parameters to adjust the shape of the activation function
        self.scale = nn.Parameter(torch.ones(hidden_dim))
        self.shift = nn.Parameter(torch.zeros(hidden_dim))
        self.nonlinearity = nn.Tanh()  # Base nonlinearity can be adjusted

    def forward(self, x):
        # Learnable scaling and shifting of the activation function
        return self.nonlinearity(x * self.scale + self.shift)

class KnowledgeAttentionLayerKAN(nn.Module):
    '''
    Modified Knowledge Attention Layer with learnable edge-based activation functions.
    '''

    def __init__(self, hidden_dim, dropout=0.1):
        super(KnowledgeAttentionLayerKAN, self).__init__()
        self.hidden_dim = hidden_dim
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.edge_activation = LearnableActivationFunction(hidden_dim)

    def forward(self, x):
        # x: B x S x hidden_dim
        Q = self.edge_activation(self.query(x))  # Apply learnable activation to queries
        K = self.edge_activation(self.key(x))    # Apply learnable activation to keys
        V = self.value(x)                         # No activation on values for attention computation

        # Compute attention scores with edge-based learnable activation
        attn_weights = torch.softmax(torch.bmm(Q, K.transpose(1, 2)) / (self.hidden_dim ** 0.5), dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Weighted sum of values
        output = torch.bmm(attn_weights, V)  # B x S x hidden_dim
        return output + x  # Add residual connection

class KAN(nn.Module):
    '''
    Knowledge-Aware Network with learnable edge activations.
    '''
    def __init__(self,
                 f_in,
                 f_out,
                 hidden_dim=256,
                 hidden_layers=2,
                 dropout=0.1):
        super(KAN, self).__init__()
        self.f_in = f_in
        self.f_out = f_out
        self.hidden_dim = hidden_dim
        self.hidden_layers = hidden_layers
        self.dropout = dropout

        # Initial linear layer to project input to hidden dimension
        self.input_proj = nn.Linear(self.f_in, self.hidden_dim)
        self.edge_activation = LearnableActivationFunction(self.hidden_dim)

        # Knowledge-aware attention layers with learnable activations on edges
        self.kan_layers = nn.ModuleList([
            KnowledgeAttentionLayerKAN(self.hidden_dim, dropout=self.dropout)
            for _ in range(self.hidden_layers)
        ])

        # Final linear layer to project hidden representation to output dimension
        self.output_proj = nn.Linear(self.hidden_dim, self.f_out)

    def forward(self, x):
        # x: B x S x f_in
        x = self.input_proj(x)  # B x S x hidden_dim
        x = self.edge_activation(x)  # Learnable activation function

        # Pass through multiple KAN layers with edge-based learnable activations
        for layer in self.kan_layers:
            x = layer(x)

        # Project back to output dimension
        y = self.output_proj(x)  # B x S x f_out
        return y
