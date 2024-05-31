import sys
sys.path.insert(0, '../')
import data.get_raw_data as grd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

class RNODE(nn.Module):
    def __init__(self, input_dim):
        super(RNODE, self).__init__()
        self.input_dim = input_dim

        # Parameters to learn
        self.in_mat = nn.Parameter(torch.randn(input_dim, input_dim) * 0.1)
        self.W = nn.Parameter(torch.randn(input_dim, input_dim) * 0.1)
        self.theta = nn.Parameter(torch.randn(input_dim) * 0.1)  # Scalar parameter per input dimension
        self.tau = nn.Parameter(torch.tensor(1.0))  # Scalar parameter

        # Activation function
        self.sigma = nn.Sigmoid()

    def forward(self, x, u, dt):
        dxdt = -x / self.tau + self.W @ self.sigma(x + self.theta) + self.in_mat @ u
        x = x + dxdt * dt
        return x

class FeedForwardNetwork(nn.Module):
    def __init__(self, input_dim):
        super(FeedForwardNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, input_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

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

# Initialize models
rnode = RNODE(input_dim)
feedforward = FeedForwardNetwork(input_dim)
model = CombinedModel(rnode, feedforward, input_dim)

# Loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

raw_inquiries = grd.process_eeg_and_triggers(
    eeg_file_path='/Users/aliag/Desktop/Data/S002/S002_Matrix_Calibration_Thu_18_May_2023_12hr43min40sec_-0400/raw_data_2.csv',
    triggers_file_path='/Users/aliag/Desktop/Data/S002/S002_Matrix_Calibration_Thu_18_May_2023_12hr43min40sec_-0400/triggers_2.txt'
)

inquiries_per_channels = []
stimulis = []
for inquiry in raw_inquiries:
    inquiries_per_channels.append(np.array(inquiry[['Cz', 'O1', 'O2']]))
    stm = np.array(inquiry['stimuli_signal']) + 10
    stimulis.append([stm, stm, stm])

stimulis = np.array(stimulis[10:90])
inquiries_per_channels = np.array(inquiries_per_channels[10:90])

# Reshape the data to (3, 3030 * 80)
stimulis = stimulis.transpose(1, 0, 2).reshape(3, -1)
inquiries_per_channels = inquiries_per_channels.transpose(1, 0, 2).reshape(3, -1)

# Convert to torch tensors
input_stimuli_signals = torch.tensor(stimulis, dtype=torch.float32)
time_series_eeg_data = torch.tensor(inquiries_per_channels, dtype=torch.float32)

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
dt = 0.01

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
