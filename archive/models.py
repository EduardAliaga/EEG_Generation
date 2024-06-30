import torch
import torch.nn as nn
import numpy as np



class Measurement_Model(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Measurement_Model, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 2*hidden_size)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(2*hidden_size, hidden_size)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(hidden_size, output_size)
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x)
        out = self.fc4(x)
        return out
    
class EEGModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(EEGModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

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
        self.theta = nn.Parameter(torch.tensor(1.0, requires_grad=True)) # Scalar parameter per input dimension
        self.tau = nn.Parameter(torch.tensor(100.0, requires_grad=True))  # Scalar parameter

        # Activation function
        self.sigma = nn.Sigmoid() #change sigmoid

    def forward(self, x, u, dt):
        dxdt = -x / self.tau + self.W @ self.sigma(x*self.theta) + self.in_mat @ u
        x = x + dxdt * dt
        return x
    
class CombinedModel(nn.Module):
    def __init__(self, rnode, feedforward, input_dim):
        super(CombinedModel, self).__init__()
        self.rnode = rnode
        self.feedforward = feedforward
        self.input_dim = input_dim
    def forward(self, u, dt):
        inquiry_size, channels, seq_len = u.shape
        outputs = torch.zeros(inquiry_size, channels, seq_len)
        membrane_potentials = torch.zeros(inquiry_size, channels, seq_len)
        #initialize x
        x = torch.zeros(channels) 
        for i_inquiry in range(inquiry_size):
            for t in range(seq_len):
                x = self.rnode(x, u[i_inquiry, :, t], dt)
                outputs[i_inquiry, :, t] = self.feedforward(x)
                membrane_potentials[i_inquiry, :, t] = x
        return outputs, membrane_potentials
    