import sys
sys.path.insert(0, '../')
import os
import numpy as np
import matplotlib.pyplot as plt
import csv 

def plot_losses_from_csv(file_path, save_path, model):
    # Initialize lists to store the data
    iterations = []
    training_mse_list = []
    testing_mse_list = []
    states_training_mse_list = []
    states_testing_mse_list = []

    # Read the CSV file
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip the header
        for row in reader:
            iterations.append(int(row[0]))  # First column: iteration number
            states_training_mse_list.append(float(row[1]))  # Second column: training MSE
            states_testing_mse_list.append(float(row[2]))  # Third column: testing MSE
            training_mse_list.append(float(row[3]))
            testing_mse_list.append(float(row[4]))

    # Plot the data
    plt.figure(figsize=(10, 6))
    plt.plot(iterations, training_mse_list, label='Training Loss', color='blue')
    plt.plot(iterations, testing_mse_list, label='Testing Loss', color='red')
    plt.xlabel('Iteration')
    plt.ylabel('MSE')
    plt.title('Training and Testing Loss Over Iterations')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_path, f'train_vs_test_loss_{model}.pdf'))
    plt.show(block = False)
    plt.figure(figsize=(10, 6))
    plt.plot(iterations, states_training_mse_list, label='Training Loss', color='blue')
    plt.plot(iterations, states_testing_mse_list, label='Testing Loss', color='red')
    plt.xlabel('Iteration')
    plt.ylabel('MSE')
    plt.title('Training and Testing Loss Over Iterations')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_path, f'state_train_vs_test_loss_{model}.pdf'))
    plt.show(block = False)

def get_norm_squared_error(x, x_hat, regularization_term=1e-4):

    squared_error = get_squared_error(x, x_hat)

    norm_sq_err = squared_error / (x + regularization_term)**2
    return norm_sq_err

def get_squared_error(x, x_hat):

    return (x - x_hat)**2

def plot_states_predicted_vs_real(states_predicted, real_states, save_path, model):
    plt.figure()
    plt.title("Predicted vs real states")
    plt.plot(states_predicted[:,0,0], c='#1f77b4', label = 'prediction region 0')
    plt.plot(states_predicted[:,1,0], c = 'orange', label = 'prediction region 1')
    # plt.plot(real_states[:,0], c='#1f77b4', label = 'ground truth region 0', linestyle = '--', alpha = 0.5)
    # plt.plot(real_states[:,1], c='orange', label = 'ground truth region 1', linestyle = '--', alpha = 0.5)
    plt.plot(real_states[0,0,:], c='#1f77b4', label = 'real state region 0', linestyle = '--', alpha = 0.5)
    plt.plot(real_states[0,1,:], c='orange', label = 'real state region 1', linestyle = '--', alpha = 0.5)
    plt.xlabel("time")
    plt.ylabel("signal value")
    plt.legend()
    plt.savefig(os.path.join(save_path, f'states_predicted_vs_real_{model}.pdf'))
    plt.show(block = False)

def plot_measurements_predicted_vs_real(measurements_predicted, real_measurements, save_path, model):
    plt.figure()
    plt.title("Predicted vs ground truth measurements")
    plt.plot(measurements_predicted[:,0], c='#1f77b4', label = 'prediction region 0')
    plt.plot(measurements_predicted[:,1], c='orange', label = 'prediction region 1')
    plt.plot(real_measurements[:,0], c='#1f77b4', label = 'ground truth region 0', linestyle = '--', alpha = 0.5)
    plt.plot(real_measurements[:,1], c='orange', label = 'ground truth region 1', linestyle = '--', alpha = 0.5)
    plt.xlabel("time")
    plt.ylabel("signal value")
    plt.legend()
    plt.savefig(os.path.join(save_path, f'measurements_predicted_vs_real_{model}.pdf'))
    plt.show(block = False)

model = 'dcm'
model_data = 'dcm'
states_file=f"/Users/aliag/Desktop/TFG/Figures/Results/synthetic_data/unknown_H/{model}/{model_data}_data/states_predicted.npy"
measurements_file = f"/Users/aliag/Desktop/TFG/Figures/Results/synthetic_data/unknown_H/{model}/{model_data}_data/measurements_predicted.npy"
data_file = f'/Users/aliag/Desktop/EEG_Generation/data/synthetic_data/synthetic_data_{model_data}.npy'
loss_file = f'/Users/aliag/Desktop/TFG/Figures/Results/synthetic_data/unknown_H/{model}/{model_data}_data/mse_csv.csv'
states_predicted = np.load(states_file, allow_pickle=True)
measurements_predicted = np.load(measurements_file, allow_pickle=True)
real_data = np.load(data_file, allow_pickle=True).item()
real_states = real_data['states']
real_states = np.array(real_states)
real_measurements = real_data['measurements_noisy']
save_path = f'/Users/aliag/Desktop/TFG/Figures/Results/synthetic_data/unknown_H/{model}/{model_data}_data'
norm_squared_error = get_norm_squared_error(real_measurements, measurements_predicted)
# plt.figure()
# plt.plot(norm_squared_error)
# plt.show()
plot_states_predicted_vs_real(states_predicted, real_states, save_path, model)
plot_measurements_predicted_vs_real(measurements_predicted, real_measurements, save_path, model)
plot_losses_from_csv(loss_file, save_path, model)
print('hello')