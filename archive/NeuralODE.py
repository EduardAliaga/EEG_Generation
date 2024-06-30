import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from autograd import jacobian
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def read_subject_data(data_path, sensors):
    inquiries = []
    for sensor in sensors:
        inquiry = pd.read_csv(f'{data_path}/data_{sensor}.csv')
        inquiry = np.array(inquiry)
        print(len(inquiry))
        inquiries.append(np.transpose(inquiry))
    timings = np.array(pd.read_csv(f'{data_path}/timing.csv'))
    labels = np.array(pd.read_csv(f'{data_path}/labels.csv'))
    return inquiries, timings, labels

def compute_stimuli(timings, labels, length):
    sensors_stimulis = []
    for sample in range(0,len(timings)):
        stimuli = np.zeros(length)
        for time_step in range(0, len(timings[0])):
            start = timings[sample][time_step]
            end = start + 10
            if labels[sample][time_step] == 1:
                stimuli[start:end] = 1
            else:
                stimuli[start:end] = 0.5
        sensors_stimulis.append([stimuli, stimuli, stimuli])
    return sensors_stimulis

# Function to flatten and concatenate parameters for Kalman filter
def flatten_params(neural_ode):
    return np.concatenate([neural_ode.W.detach().numpy().flatten(), [neural_ode.theta.item()], [neural_ode.tau.item()]])

# Function to unflatten and set parameters from Kalman filter
def unflatten_params(neural_ode, params):
    W_size = hidden_size * hidden_size
    neural_ode.W.data = torch.tensor(params[:W_size].reshape(hidden_size, hidden_size), dtype=torch.float32)
    neural_ode.theta.data = torch.tensor(params[W_size], dtype=torch.float32)
    neural_ode.tau.data = torch.tensor(params[W_size + 1], dtype=torch.float32)

def evaluate_model(neural_ode, eeg_model, input_signal, dt, timesteps):
    neural_ode.eval()
    eeg_model.eval()
    
    with torch.no_grad():
        u = torch.tensor(input_signal.T, dtype=torch.float32)
        x0 = torch.zeros(neural_ode.hidden_size)
        membrane_potential = compute_membrane_potential(neural_ode, u, x0, dt, timesteps)
        eeg_output = eeg_model(membrane_potential)
        eeg_output = eeg_output.view(3, 396)
        return eeg_output.numpy(), membrane_potential.numpy()


class NeuralODE(nn.Module):
    def __init__(self, tau, theta, input_size, hidden_size):
        super(NeuralODE, self).__init__()
        self.tau = nn.Parameter(tau)
        self.theta = nn.Parameter(theta)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.W = nn.Parameter(torch.randn(hidden_size, hidden_size))
    
    def activation(self, x):
        return torch.sigmoid(x + self.theta)  # Example activation function
    
    def forward(self, x, u):
        return -x / self.tau + torch.matmul(self.W, self.activation(x)) + u
    
class EEGModel(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(EEGModel, self).__init__()
        self.fc1 = nn.Linear(hidden_size, 20)
        self.fc2 = nn.Linear(20, 20)
        self.fc3 = nn.Linear(20, output_size)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Initialize the model
input_size = 3
hidden_size = 3
tau = torch.tensor(0.1, requires_grad=True)
theta = torch.tensor(0.1, requires_grad=True)
neural_ode = NeuralODE(tau, theta, input_size, hidden_size)

def compute_membrane_potential(neural_ode, u, x0, dt, timesteps):
    x = torch.zeros(timesteps, neural_ode.hidden_size)
    x[0] = x0
    for t in range(1, timesteps):
        x[t] = x[t-1] + dt * neural_ode(x[t-1], u[t-1])
    return x

# Initialize the EEG model
eeg_model = EEGModel(hidden_size, input_size)
criterion = nn.MSELoss()
optimizer = optim.Adam(list(eeg_model.parameters()), lr=0.01)

# Initialize Kalman filter parameters
n_params = hidden_size * hidden_size + 2  # Number of parameters: W (NxN), theta, tau
P = np.eye(n_params) * 1e-2  # Covariance matrix
Q = np.eye(n_params) * 1e-5  # Process noise covariance matrix
R = np.eye(input_size) * 1e-2  # Measurement noise covariance matrix


# Initial parameters
params = flatten_params(neural_ode)

# Training loop


inquiries, timings, labels = read_subject_data("/Users/aliag/Desktop/BciPy/bcipy/EEG_generation",[1,4,5])
total_stimulis = compute_stimuli(timings, labels, 396)
timings = np.array(timings)
labels = np.array(labels)
stimuli_per_inquiry = []
stimuli_per_inquiry_2 = []

for i in range(100):
    stimuli = np.zeros(396 * 2 * 300)
    stimuli_2 = np.zeros(396)
    for j in range(10):
        if labels[i][j] == 1:
            stimuli[timings[i][j] * 2 * 300 : (timings[i][j] + 10) * 2 * 300] = 1
            stimuli_2[timings[i][j] : (timings[i][j] + 10)] = 1
        else:
            stimuli[timings[i][j] * 2 * 300 : (timings[i][j] + 10) * 2 * 300] = 0.5
            stimuli_2[timings[i][j]: (timings[i][j] + 10)] = 0.5
    stimuli_per_inquiry.append(stimuli)
    stimuli_per_inquiry_2.append(stimuli_2)
total_stimulis = []
total_stimulis_2 = []
for i in range(0,100):
    list=[]
    list_2 = []
    for j in range (0,3):
        list.append(stimuli_per_inquiry[i])
        list_2.append(stimuli_per_inquiry_2[i])
    total_stimulis.append(list)
    total_stimulis_2.append(list_2)
total_stimulis = np.array(total_stimulis)
total_stimulis_2 = np.array(total_stimulis_2)


input_signals = torch.tensor(total_stimulis_2, dtype=torch.float32)
eeg_data = torch.tensor(inquiries, dtype=torch.float32).reshape(100, 3, 396)
timesteps = 396
dt = 1/3000
x0=torch.randn(3)
x0_hat=torch.randn(3)
epochs = 10
batch_size = 10
for epoch in range(epochs):
    epoch_loss = 0
    for i in range(0, len(input_signals[0:10]), batch_size):
        batch_inputs = input_signals[i:i + batch_size]
        batch_eeg = eeg_data[i:i + batch_size]
        
        batch_membrane_potentials = []
        for inp in batch_inputs:
            u = inp.T  # Transpose to match (timesteps, input_size)
            x = compute_membrane_potential(neural_ode, u, x0, dt, timesteps)
            batch_membrane_potentials.append(x)
        
        batch_membrane_potentials = torch.stack(batch_membrane_potentials)
        batch_membrane_potentials = batch_membrane_potentials.view(-1, hidden_size)
        batch_eeg = batch_eeg.view(-1, input_size)
        
        # Train the EEG model
        optimizer.zero_grad()
        outputs = eeg_model(batch_membrane_potentials)
        loss = criterion(outputs, batch_eeg)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        
        for inp in batch_inputs:
        # Update Kalman filter
            u = inp.T
            for t in range(timesteps):
                # Predict step
                x_hat = compute_membrane_potential(neural_ode, u[:t+1], x0, dt, t+1)[-1].detach().numpy() 

                H = jacobian(lambda p: compute_membrane_potential(neural_ode, u[:t+1], x0, dt, t+1)[-1].detach().numpy())(params)
                print(H)
                P = P + Q  

                # Update step
                y = batch_eeg[t].detach().numpy() - eeg_model(torch.tensor(x_hat).float().view(1, -1)).detach().numpy() 
                S = np.dot(H, np.dot(P, H.T)) + R
                K = np.dot(P, np.dot(H.T, np.linalg.inv(S)))  
                params = params + np.dot(K, y.flatten())  
                P = P - np.dot(K, np.dot(S, K.T))  
                
                # Set updated parameters back to the neural ODE model
                unflatten_params(neural_ode, params)
            print("first finished")
    
    print(f'Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss / (len(input_signals) / batch_size)}')



input_signal = total_stimulis_2[0]  # Replace with actual input signal
eeg_output, membrane_potential = evaluate_model(neural_ode, eeg_model, input_signal, dt, timesteps)

# Print the EEG output shape to verify
print(eeg_output.shape)  # Should be (3, 396)
print(membrane_potential.shape)  # Should be (396, hidden_size)

# Plot the EEG output and membrane potential
for i in range(3):
    plt.figure()
    plt.plot(eeg_output[i], label='EEG Output')
    plt.plot(eeg_data[0][i], label='EEG Ground Truth')
    plt.title(f'EEG Channel {i+1}')
    plt.xlabel('Time')
    plt.ylabel('EEG Signal')
    plt.legend()
    plt.show()

# Plot the membrane potential
for i in range(hidden_size):
    plt.figure()
    plt.plot(membrane_potential[:, i], label='Membrane Potential')
    plt.title(f'Membrane Potential Channel {i+1}')
    plt.xlabel('Time')
    plt.ylabel('Membrane Potential')
    plt.legend()
    plt.show()
