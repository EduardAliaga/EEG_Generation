import sys
sys.path.insert(0, '../')
import data.get_raw_data as grd
from models_src.models import RNODE, Linear_Model, CombinedModel
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

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

stimulis = stimulis[10:90]
inquiries_per_channels = inquiries_per_channels[10:90]

# Assuming you have your input stimuli signals and time series EEG data as torch tensors
input_stimuli_signals = torch.tensor(stimulis, dtype=torch.float32)  # Example shape (100, 3, 3030)
time_series_eeg_data = torch.tensor(inquiries_per_channels, dtype=torch.float32).reshape(80, 3, 3030)  # Example shape (100, 3, 3030)
print(input_stimuli_signals.shape)
print(time_series_eeg_data.shape)

# Train-test split
train_inputs, test_inputs, train_targets, test_targets = train_test_split(input_stimuli_signals, time_series_eeg_data, test_size=0.2, random_state=42)

num_epochs = 10
dt = 0.01

for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()

    # Forward pass
    output, membrane_potentials = model(train_inputs, dt)
    loss = criterion(membrane_potentials, train_targets)

    # Backward pass and optimization
    loss.backward()
    optimizer.step()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}')

# Evaluate on the test set
model.eval()
with torch.no_grad():
    test_output, test_membrane_potentials = model(test_inputs, dt)
    test_loss = criterion(test_membrane_potentials, test_targets)
    print(f'Test Loss: {test_loss.item()}')

# Plotting results for one sample from the test set
sample_idx = 0  # Index of the sample to plot
plt.figure(figsize=(15, 10))
for channel in range(3):
    plt.subplot(6, 1, channel + 1)
    plt.plot(test_targets[sample_idx, channel].numpy(), label='True EEG')
    plt.plot(test_output[sample_idx, channel].numpy(), label='Predicted EEG')
    plt.title(f'Channel {channel + 1} - EEG')
    plt.legend()

for channel in range(3):
    plt.subplot(6, 1, channel + 4)
    plt.plot(test_membrane_potentials[sample_idx, channel].numpy(), label='Membrane Potential')
    plt.title(f'Channel {channel + 1} - Membrane Potential')
    plt.legend()

plt.tight_layout()
plt.show()





