import sys
sys.path.insert(0, '../')
import numpy.random as rnd
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.optimize import minimize

def plot_stimuli(stimuli, save_path):
    plt.figure()
    plt.title('Synthetic input signal')
    plt.plot(stimuli[:,0], label = 'Input signal')
    plt.xlabel("Time")
    plt.ylabel("Signal value")
    plt.legend()
    plt.savefig(os.path.join(save_path, 'synthetic_input_signal.pdf'))
    plt.show(block = False)

def plot_states(states, model, save_path):
    plt.figure()
    plt.title("Synhtetic state signals")
    if model =='linea' or model == 'sigmoid':
        plt.plot(states[:,0], label = f'state signal for region 0')
        plt.plot(states[:,1], label = 'state signal for region 1')
    elif model == 'dcm':
        for state in range(len(states)):
            plt.plot(states[state,:,0], label = f'state signal for region {state}')
            # plt.plot(states[:,1], label = 'state signal for region 1')
    plt.xlabel("Time")
    plt.ylabel("Signal value")
    plt.legend()
    plt.savefig(os.path.join(save_path, 'synthetic_state_signals.pdf'))
    plt.show(block = False)

def plot_dcm_states(states, model, save_path):
    for state in [0,4,8]:
        plt.figure()
        plt.title(f"Synhtetic state signal for state {state}")
        plt.plot(states[state,0,:], label = f'state {state}')
        plt.xlabel("Time")
        plt.ylabel("Signal value")
        plt.legend()
        plt.savefig(os.path.join(save_path, f'synthetic_state_{state}_signals.pdf'))
        plt.show(block = False)

def plot_measurements(measurements, title, label_1, label_2, save_path, name):
    plt.figure()
    plt.title(title)
    plt.plot(measurements[:,0], label = label_1)
    plt.plot(measurements[:,1], label = label_2)
    plt.xlabel("Time")
    plt.ylabel("Signal value")
    plt.legend()
    plt.savefig(os.path.join(save_path, f'synthetic_{name}_signals.pdf'))
    plt.show(block = False)

model = 'dcm'
data_file=f'/Users/aliag/Desktop/EEG_Generation/data/synthetic_data/synthetic_data_{model}.npy'
data = np.load(data_file, allow_pickle=True).item()
stimuli = data['stimuli']
states = data['states']
print(states.shape)
states = np.array(states)
measurements = data['measurements']
measurements_noisy = data['measurements_noisy']
save_path = f'/Users/aliag/Desktop/TFG/Figures/Synthetic_data/{model}'
# plot_stimuli(stimuli, save_path)

# plot_dcm_states(states, model, save_path)
# plot_measurements(measurements, "Sinthetic measurement signals", "measurement signal for region 0", "measurement signal for region 1", save_path, "measurements")
# plot_measurements(measurements_noisy, "Sinthetic measurements", "measurement signal for region 0", "measurement signal for region 1", save_path, "measurements")

plt.figure()
plt.title("Synhtetic state signal for state 4 and 8")
# plt.plot(states[0,0,:], label = f'state 0')
plt.plot(states[4,0,:], label = f'state 4')
plt.plot(states[8,0,:], label = f'state 8')
# plt.plot(stimuli[:,0], label = 'stimuli', alpha = 0.5)
plt.xlabel("Time")
plt.ylabel("Signal value")
plt.legend()
plt.savefig(os.path.join(save_path, f'states_4_8.pdf'))
plt.show(block = False)
