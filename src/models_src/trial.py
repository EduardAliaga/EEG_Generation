import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Define the model
class Model(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Model, self).__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.theta = nn.Parameter(torch.randn(2, 2))  # Adjusted to match the shape for multiplication
        self.H_e = nn.Parameter(torch.randn(1))
        self.tau_e = nn.Parameter(torch.randn(1))
        self.H_i = nn.Parameter(torch.randn(1))
        self.tau_i = nn.Parameter(torch.randn(1))
        self.lambda_1 = nn.Parameter(torch.randn(1))
        self.lambda_2 = nn.Parameter(torch.randn(1))
        self.lambda_3 = nn.Parameter(torch.randn(1))
        self.lambda_4 = nn.Parameter(torch.randn(1))
        self.C_f = nn.Parameter(torch.randn(2, 2))
        self.C_l = nn.Parameter(torch.randn(2, 2))
        self.C_u = nn.Parameter(torch.randn(2, 2))
        self.C_b = nn.Parameter(torch.randn(2, 2))
        self.linear = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, u, dt):
        def sigmoid(x):
            return 1 / (1 + torch.exp(-x))

        batch_size, seq_len, _, _ = x.size()
        outputs = []
        h = None  # hidden state for LSTM

        for t in range(seq_len):
            xt = x[:, t, :, :]
            ut = u[:, t, :, :]

            xt_theta = torch.matmul(xt[:, 0, :], self.theta)  # Apply theta to the appropriate dimension
            ut_theta = torch.matmul(ut.squeeze(-1), self.theta)  # Apply theta to the appropriate dimension and keep the dimension

            x0 = xt[:, 0, :] + dt * (xt[:, 5, :] - xt[:, 6, :])
            x1 = xt[:, 1, :] + dt * xt[:, 4, :]
            x2 = xt[:, 2, :] + dt * xt[:, 5, :]
            x3 = xt[:, 3, :] + dt * xt[:, 6, :]
            x4 = xt[:, 4, :].T + dt * (self.H_e / self.tau_e * (torch.matmul(self.C_f + self.C_l + self.lambda_1 * torch.eye(2), sigmoid(xt_theta).T) + torch.matmul(self.C_u, ut_theta.T)) - 2 * xt[:, 4, :].T / self.tau_e - xt[:, 1, :].T / self.tau_e ** 2)
            x5 = xt[:, 5, :].T + dt * (self.H_e / self.tau_e * (torch.matmul(self.C_b + self.C_l, sigmoid(xt_theta).T) + self.lambda_2 * sigmoid(xt[:, 3, :]).T) - 2 * xt[:, 5, :].T / self.tau_e - xt[:, 2, :].T / self.tau_e ** 2)
            x6 = xt[:, 6, :].T + dt * (self.H_i / self.tau_i * self.lambda_4 * sigmoid(xt[:, 7, :]).T - 2 * xt[:, 6, :].T / self.tau_i - xt[:, 3, :].T / self.tau_i ** 2)
            x7 = xt[:, 7, :] + dt * xt[:, 8, :]
            x8 = xt[:, 8, :].T + dt * (self.H_e / self.tau_e * (torch.matmul(self.C_b + self.C_l + self.lambda_3 * torch.eye(2), sigmoid(xt_theta).T)) - 2 * xt[:, 8, :].T / self.tau_e - xt[:, 7, :].T / self.tau_e ** 2)
            x_h_0 = xt[:, 9, :]
            x_h_1 = xt[:, 10, :]

            output = torch.stack([x0, x1, x2, x3, x4.T, x5.T, x6.T, x7, x8.T, x_h_0, x_h_1], dim=1)
            output = output.view(output.size(0), -1)  # Flatten the tensor to fit the linear layer
            lstm_out, h = self.lstm(output.unsqueeze(1), h)  # Unsqueeze to add the sequence dimension
            lstm_out = lstm_out.squeeze(1)  # Remove the sequence dimension
            output = self.linear(lstm_out)
            output = output.view(output.size(0), 2, 1)  # Reshape back to (2, 1)
            outputs.append(output)

        outputs = torch.stack(outputs, dim=1)  # Stack along the time dimension
        return outputs

# Load your data
data_file = 'synthetic_data.npy'
data = np.load(data_file, allow_pickle=True).item()
stimuli = data['stimuli']
membrane_potentials = data['membrane_potentials']
measurements_noisy = data['measurements_noisy']

# Reshape the arrays
membrane_potentials = np.array(membrane_potentials).reshape(3000, 11, 2)
measurements_noisy = measurements_noisy.reshape(3000, 2, 1)
stimuli = stimuli.reshape(3000, 2, 1)

# Convert to PyTorch tensors
membrane_potentials = torch.from_numpy(membrane_potentials).float().unsqueeze(0)  # Add batch dimension
measurements_noisy = torch.from_numpy(measurements_noisy).float().unsqueeze(0)  # Add batch dimension
stimuli = torch.from_numpy(stimuli).float().unsqueeze(0)  # Add batch dimension

# Split the data into training and testing sets
train_idx = int(0.8 * membrane_potentials.shape[1])
x_train, x_test = membrane_potentials[:, :train_idx, :, :], membrane_potentials[:, train_idx:, :, :]
y_train, y_test = measurements_noisy[:, :train_idx, :, :], measurements_noisy[:, train_idx:, :, :]
u_train, u_test = stimuli[:, :train_idx, :, :], stimuli[:, train_idx:, :, :]

# Create the model
model = Model(input_dim=22, hidden_dim=50, output_dim=2)  # Changed input_dim to match the concatenated size
dt = 0.01

# Training parameters
learning_rate = 0.001
num_epochs = 20

# Loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
train_losses = []
test_losses = []

for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(x_train, u_train, dt)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()

    train_losses.append(loss.item())

    model.eval()
    with torch.no_grad():
        test_outputs = model(x_test, u_test, dt)
        test_loss = criterion(test_outputs, y_test)
        test_losses.append(test_loss.item())

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {loss.item():.4f}, Test Loss: {test_loss.item():.4f}')

# Plotting the results
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Train Loss')
plt.plot(test_losses, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Testing Loss')
plt.show()

model.eval()
with torch.no_grad():
    predicted_train = model(x_train, u_train, dt).squeeze().cpu().numpy()
    predicted_test = model(x_test, u_test, dt).squeeze().cpu().numpy()
    y_train = y_train.squeeze().cpu().numpy()
    y_test = y_test.squeeze().cpu().numpy()

plt.figure(figsize=(12, 6))

# Plot for training data
plt.subplot(1, 2, 1)
plt.plot(y_train.flatten(), label='Ground Truth')
plt.plot(predicted_train.flatten(), label='Predicted')
plt.title('Training Data')
plt.xlabel('Time Step')
plt.ylabel('Value')
plt.legend()

# Plot for testing data
plt.subplot(1, 2, 2)
plt.plot(y_test.flatten(), label='Ground Truth')
plt.plot(predicted_test.flatten(), label='Predicted')
plt.title('Testing Data')
plt.xlabel('Time Step')
plt.ylabel('Value')
plt.legend()

plt.tight_layout()
plt.show()
