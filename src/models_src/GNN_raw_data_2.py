import sys
sys.path.insert(0, '../')
import data.get_raw_data as grd
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data, DataLoader
from torch_geometric.utils import add_self_loops
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

class GraphEncoder(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, latent_channels):
        super(GraphEncoder, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, latent_channels)

    def forward(self, x, edge_index, edge_weight=None):
        x = F.relu(self.conv1(x, edge_index, edge_weight))
        x = self.conv2(x, edge_index, edge_weight)
        return x

class GraphDecoder(torch.nn.Module):
    def __init__(self, latent_channels, hidden_channels, out_channels):
        super(GraphDecoder, self).__init__()
        self.conv1 = GCNConv(latent_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, z, edge_index, edge_weight=None):
        z = F.relu(self.conv1(z, edge_index, edge_weight))
        z = self.conv2(z, edge_index, edge_weight)
        return z

class GraphAutoencoder(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, latent_channels, out_channels, num_edges):
        super(GraphAutoencoder, self).__init__()
        self.encoder = GraphEncoder(in_channels, hidden_channels, latent_channels)
        self.decoder = GraphDecoder(latent_channels, hidden_channels, out_channels)
        self.edge_weight = torch.nn.Parameter(torch.randn(num_edges + num_nodes))  # Learnable edge weights

    def forward(self, x, edge_index):
        edge_index, edge_weight = add_self_loops(edge_index, num_nodes=x.size(0))
        z = self.encoder(x, edge_index, edge_weight)
        recon_x = self.decoder(z, edge_index, edge_weight)
        return recon_x

# Constants
num_nodes = 3
num_samples = 80
time_series_length = 3030

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

# Prepare input signals and EEG signals
input_signals = torch.tensor(np.array(stimulis), dtype=torch.float32).reshape(80, 3, 3030)
eeg_signals = torch.tensor(np.array(inquiries_per_channels), dtype=torch.float32).reshape(80, 3, 3030)  # Shape (80, 3, 3030)

# Fully connected graph (with self-loops)
edge_index = torch.tensor([[0, 0, 1, 1, 2, 2, 0, 1, 0, 2, 1, 2],
                           [1, 2, 0, 2, 0, 1, 0, 0, 1, 1, 2, 2]], dtype=torch.long)

# Combine stimuli signals and EEG signals
combined_signals = torch.cat((input_signals, eeg_signals), dim=1)  # Shape (80, 6, 3030)

# Create DataLoader
data_list = []
for i in range(num_samples):
    x = combined_signals[i].t()  # Transpose to shape (3030, 6)
    data_list.append(Data(x=x, edge_index=edge_index))

# Train-test split
train_data, test_data = train_test_split(data_list, test_size=0.2, random_state=42)
train_loader = DataLoader(train_data, batch_size=1, shuffle=True)
test_loader = DataLoader(test_data, batch_size=1)

# Define the model, optimizer, and loss function
in_channels = combined_signals.size(1)  # Number of nodes
hidden_channels = 30
latent_channels = 64
out_channels = combined_signals.size(1)  # Number of nodes

model = GraphAutoencoder(in_channels, hidden_channels, latent_channels, out_channels, edge_index.size(1))
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Lambda value for regularization
lambda_reg = 0.01

# Training function
def train(model, loader, optimizer, num_epochs=100, lambda_reg=0.01):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for data in loader:
            optimizer.zero_grad()
            recon_x = model(data.x, data.edge_index)
            mse_loss = F.mse_loss(recon_x, data.x)
            regularization_loss = lambda_reg * torch.norm(model.edge_weight, p=2)
            loss = mse_loss + regularization_loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f'Epoch {epoch+1}, Loss: {total_loss/len(loader)}, Reg Loss: {regularization_loss.item()}')
        print(model.edge_weight)

# Train the model
train(model, train_loader, optimizer, num_epochs=10, lambda_reg=lambda_reg)

# Evaluate the model and plot the results for one test sample
model.eval()
with torch.no_grad():
    for i, data in enumerate(test_loader):
        recon_x = model(data.x, data.edge_index)
        recon_x = recon_x.cpu().numpy()
        data_x = data.x.cpu().numpy()
        # Plot the original and reconstructed signals for the first test batch
        if i == 0:
            plt.figure(figsize=(15, 5))
            for j in range(3,6):
                plt.subplot(3, 3, j+1)
                plt.plot(data_x[:, j], label='Original')
                plt.plot(recon_x[:, j], label='Reconstructed')
                plt.title(f'Node {j+1}')
                plt.xlabel('Time')
                plt.ylabel('Signal')
                plt.legend()
            plt.tight_layout()
            plt.show()
            break






















