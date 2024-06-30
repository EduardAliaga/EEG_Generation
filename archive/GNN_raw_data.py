import sys
sys.path.insert(0, '../')
import data.get_raw_data as grd
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.conv3(x, edge_index)
        return x

# Initialize data
num_samples = 100
num_nodes = 3
num_timesteps = 3030

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
X = torch.tensor(stimulis, dtype=torch.float32)  # Shape (80, 3, 3030)
Y = torch.tensor(inquiries_per_channels, dtype=torch.float32).reshape(80, 3, 3030)  # Shape (80, 3, 3030)

# Train-test split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Initial adjacency matrix (learnable)
A = torch.rand((num_nodes, num_nodes), requires_grad=True)

# Edge index (assuming fully connected graph initially)
edge_index = torch.tensor([[0, 1, 2, 0, 1, 2],
                           [1, 2, 0, 2, 0, 1]], dtype=torch.long)

# Initialize model
model = GCN(num_timesteps, 64, num_timesteps)
optimizer = torch.optim.Adam(list(model.parameters()) + [A], lr=0.01)

# Training loop
num_epochs = 1000
train_losses = []
test_losses = []

for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    
    # Forward pass (train)
    out_train = model(X_train.view(-1, num_nodes, num_timesteps), edge_index)
    
    # Compute loss (train)
    loss_train = F.mse_loss(out_train, Y_train.view(-1, num_nodes, num_timesteps))
    
    # Backward pass and optimization
    loss_train.backward()
    optimizer.step()
    
    # Save train loss
    train_losses.append(loss_train.item())
    
    # Evaluate on test set
    model.eval()
    with torch.no_grad():
        out_test = model(X_test.view(-1, num_nodes, num_timesteps), edge_index)
        loss_test = F.mse_loss(out_test, Y_test.view(-1, num_nodes, num_timesteps))
        test_losses.append(loss_test.item())
    
    print(f'Epoch {epoch+1}, Train Loss: {loss_train.item()}, Test Loss: {loss_test.item()}')

"""# Plot the train and test losses
plt.figure(figsize=(10, 5))
plt.plot(range(num_epochs), train_losses, label='Train Loss')
plt.plot(range(num_epochs), test_losses, label='Test Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Train and Test Loss Over Epochs')
plt.show()"""

# Plot predictions vs. real data for the test set
# Take the first sample from the test set for visualization
sample_index = 0
real_data = Y_test[sample_index].detach().numpy()
predicted_data = out_test[sample_index].view(num_nodes, num_timesteps).detach().numpy()

plt.figure(figsize=(15, 5))
for channel in range(num_nodes):
    plt.subplot(num_nodes, 1, channel + 1)
    plt.plot(real_data[channel, :], label='Real Data')
    plt.plot(predicted_data[channel, :], label='Predicted Data')
    plt.xlabel('Time Step')
    plt.ylabel('Signal Value')
    plt.legend()
    plt.title(f'Channel {channel + 1}')

plt.tight_layout()
plt.show()

print("Learned adjacency matrix:", A.detach().numpy())



