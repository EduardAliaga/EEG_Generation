import torch
import torch.nn as nn


class Linear_Model(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Linear_Model, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 2*hidden_size)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(2*hidden_size, hidden_size)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(hidden_size, output_size)
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        out = self.relu3(out)
        out = self.fc4(out)
        return out


class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(RNNModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size) 
        out, _ = self.rnn(x, h0)  
        out = self.fc(out[:, -1, :])  
        return out
    
class RNODE(nn.Module):
    def __init__(self, input_dim):
        super(RNODE, self).__init__()
        self.input_dim = input_dim

        # Parameters to learn
        self.in_mat = nn.Parameter(torch.randn(input_dim, input_dim))
        self.W = nn.Parameter(torch.randn(input_dim, input_dim))
        self.theta = 1 # Scalar parameter per input dimension
        self.tau = 10  # Scalar parameter

        # Activation function
        self.sigma = nn.Sigmoid()

    def forward(self, x, u, dt):
        dxdt = -x / self.tau + self.W @ self.sigma(x + self.theta) + self.in_mat @ u
        x = x + dxdt * dt
        return x
    
class CombinedModel(nn.Module):
    def __init__(self, rnode, feedforward, input_dim):
        super(CombinedModel, self).__init__()
        self.rnode = rnode
        self.feedforward = feedforward
        self.input_dim = input_dim

    def forward(self, u, dt):
        batch_size, channels, seq_len = u.shape
        outputs = torch.zeros(batch_size, channels, seq_len)
        membrane_potentials = torch.zeros(batch_size, channels, seq_len)
        for b in range(batch_size):
            x = torch.zeros(channels)
            for t in range(seq_len):
                x = self.rnode(x, u[b, :, t], dt)
                outputs[b, :, t] = self.feedforward(x)
                membrane_potentials[b, :, t] = x
        return outputs, membrane_potentials