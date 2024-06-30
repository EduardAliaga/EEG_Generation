import sys
sys.path.insert(0, '../')
import data.get_raw_data as grd
from models_src.models import RNODE, Linear_Model
from data.data_processing import read_subject_data, compute_stimuli
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

class CombinedModel(nn.Module):
    def __init__(self, rnode, feedforward, input_dim):
        super(CombinedModel, self).__init__()
        self.rnode = rnode
        self.feedforward = feedforward
        self.input_dim = input_dim

    def forward(self, u, dt):
        channels, seq_len = u.shape
        outputs = torch.zeros(channels, seq_len)
        membrane_potentials = torch.zeros(channels, seq_len)
        x = torch.zeros(channels)
        for t in range(seq_len):
            x = self.rnode(x, u[:, t], dt)
            outputs[:, t] = self.feedforward(x)
            membrane_potentials[:, t] = x
        return outputs, membrane_potentials

# Define the model dimensions
input_dim = 3
hidden_dim = 256

# Initialize models
rnode = RNODE(input_dim)
feedforward = Linear_Model(input_dim, hidden_dim, input_dim)
model = CombinedModel(rnode, feedforward, input_dim)

# Loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Load and preprocess data
inquiries, timings, labels = read_subject_data("/Users/aliag/Desktop/BciPy/bcipy/EEG_generation", [1, 4, 5])
total_stimulis = compute_stimuli(timings, labels, 396)

# Reshape the data to (3, 396 * 100)
input_stimuli_signals = torch.tensor(total_stimulis, dtype=torch.float32).reshape(3, -1)
time_series_eeg_data = torch.tensor(inquiries, dtype=torch.float32).reshape(3, -1)

# Print shapes
print(input_stimuli_signals.shape)
print(time_series_eeg_data.shape)

# Train-test split
train_inputs, test_inputs, train_targets, test_targets = train_test_split(input_stimuli_signals.T, time_series_eeg_data.T, test_size=0.2, random_state=42)
train_inputs = train_inputs.T
test_inputs = test_inputs.T
train_targets = train_targets.T
test_targets = test_targets.T

# Normalize the data
mean = train_inputs.mean(dim=1, keepdim=True)
std = train_inputs.std(dim=1, keepdim=True)
train_inputs = (train_inputs - mean) / std
test_inputs = (test_inputs - mean) / std

num_epochs = 10
dt = 1 / 3000

for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()

    # Forward pass
    output, membrane_potentials = model(train_inputs, dt)
    loss = criterion(output, train_targets)

    # Backward pass and optimization
    loss.backward()
    optimizer.step()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}')

# Evaluate on the test set
model.eval()
with torch.no_grad():
    test_output, test_membrane_potentials = model(test_inputs, dt)
    test_loss = criterion(test_output, test_targets)
    print(f'Test Loss: {test_loss.item()}')

# Plotting results for one sample from the test set
plt.figure(figsize=(15, 10))
for channel in range(3):
    plt.subplot(6, 1, channel + 1)
    plt.plot(test_targets[channel].numpy(), label='True EEG')
    plt.plot(test_output[channel].numpy(), label='Predicted EEG')
    plt.title(f'Channel {channel + 1} - EEG')
    plt.legend()

for channel in range(3):
    plt.subplot(6, 1, channel + 4)
    plt.plot(test_membrane_potentials[channel].numpy(), label='Membrane Potential')
    plt.title(f'Channel {channel + 1} - Membrane Potential')
    plt.legend()

plt.tight_layout()
plt.show()
