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
    colors = ['tab:blue', 'tab:orange', 'tab:green']

    # Create a figure with three subplots in a row
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Loop through each source and plot the predicted vs real states in each subplot
    for source in [0, 1, 2]:
        ax = axes[source]
        ax.set_title(f"Source {source}")
        ax.plot(states_predicted[:, 0, source], c=colors[source], label='prediction')
        ax.plot(real_states[0, source, :], c=colors[source], label='real', linestyle='--', alpha=0.5, linewidth = 3)
        ax.set_xlabel("time")
        ax.set_ylabel("signal value")
        ax.legend(loc="lower right")

    # Adjust layout to avoid overlap and save the figure
    fig.suptitle("Predicted vs Real States Across All Sources", fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f'states_predicted_vs_real_{model}.pdf'))
    plt.show(block=False)

def plot_measurements_predicted_vs_real(measurements_predicted, real_measurements, save_path, model):
    colors = ['tab:blue', 'tab:orange', 'tab:green']

    # Create a figure with three subplots in a row
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Loop through each source and plot the predicted vs real states in each subplot
    for source in [0, 1, 2]:
        ax = axes[source]
        ax.set_title(f"Source {source}")
        ax.plot(measurements_predicted[:, source], c=colors[source], label='prediction')
        ax.plot(real_measurements[:, source], c=colors[source], label='real', linestyle='--', alpha=0.5, linewidth = 3)
        ax.set_xlabel("time")
        ax.set_ylabel("signal value")
        ax.legend(loc="lower right")

    fig.suptitle("Predicted vs Real Measurements Across All Sources", fontsize=16)
    # Adjust layout to avoid overlap and save the figure
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f'measurements_predicted_vs_real_{model}.pdf'))
    plt.show(block=False)

# model = 'dcm'
# model_data = 'dcm'
# states_file=f"/Users/aliag/Desktop/TFG/Figures/Results/synthetic_data/known_H/states_predicted.npy"
# measurements_file = f"/Users/aliag/Desktop/TFG/Figures/Results/synthetic_data/known_H/measurements_predicted.npy"
# data_file = f'/Users/aliag/Desktop/EEG_Generation/data/synthetic_data/synthetic_data_{model_data}.npy'
# loss_file = f'/Users/aliag/Desktop/TFG/Figures/Results/synthetic_data/known_H/mse_csv.csv'
# states_predicted = np.load(states_file, allow_pickle=True)
# measurements_predicted = np.load(measurements_file, allow_pickle=True)
# real_data = np.load(data_file, allow_pickle=True).item()
# real_states = real_data['states']
# real_states = np.array(real_states)
# real_measurements = real_data['measurements_noisy']
# save_path = f'/Users/aliag/Desktop/TFG/Figures/Results/synthetic_data/known_H/'
# norm_squared_error = get_norm_squared_error(real_measurements, measurements_predicted)
# # plt.figure()
# # plt.plot(norm_squared_error)
# # plt.show()
# plot_states_predicted_vs_real(states_predicted, real_states, save_path, model)
# plot_measurements_predicted_vs_real(measurements_predicted, real_measurements, save_path, model)
# plot_losses_from_csv(loss_file, save_path, model)
# print('hello')