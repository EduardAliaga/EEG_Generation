import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def compute_triggers_relative_time(file_path):
    triggers = []
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) == 3:
                triggers.append((parts[0], parts[1], float(parts[2])))

    initial_timestamp = min(timestamp for _, _, timestamp in triggers)*-1
    triggers_with_relative_time = [(letter, label, timestamp - initial_timestamp) for letter, label, timestamp in triggers]

    return triggers_with_relative_time

def get_real_eeg_signals(eeg_signals_file_path, triggers_with_relative_time):
    eeg_data = pd.read_csv(eeg_signals_file_path, skiprows=2)
    initial_timestamp_csv = eeg_data['lsl_timestamp'].min()
    eeg_data['relative_time'] = eeg_data['lsl_timestamp'] - initial_timestamp_csv
    
    first_trigger_relative_time = triggers_with_relative_time[1][2]
    last_trigger_relative_time = triggers_with_relative_time[-2][2]
    eeg_data['time_diff'] = (eeg_data['relative_time'] - first_trigger_relative_time).abs()
    eeg_data['time_diff_2'] = (eeg_data['relative_time'] - last_trigger_relative_time).abs()
    closest_index = eeg_data['time_diff'].idxmin()
    closest_index_last = eeg_data['time_diff_2'].idxmin()

    # Get the closest row
    closest_row = eeg_data.loc[closest_index]
    closest_row_last = eeg_data.loc[closest_index_last]
    # Print the closest row
    initial_time_experiment = int(closest_row.name)
    end_time_experiment = int(closest_row_last.name)
    experiment_eeg_data = eeg_data[initial_time_experiment:end_time_experiment]
    return experiment_eeg_data

def get_stimuli(experiment_eeg_data, triggers_with_relative_time):
    stimuli = np.zeros(len(experiment_eeg_data))
    initial_time_experiment = experiment_eeg_data.index[0]

    for label, _, trigger_time in triggers_with_relative_time:
        time_diff = (experiment_eeg_data['relative_time'] - trigger_time).abs()
        closest_index = time_diff.idxmin()
        position = closest_index - initial_time_experiment

        if label == 'nontarget':
            stimuli[position-30:position] = 0.5
        elif label == 'target':
            stimuli[position-30:position] = 1

    return stimuli

def get_real_data(eeg_signals_file_path, triggers_file_path, sensors):
    triggers_relative_time = compute_triggers_relative_time(triggers_file_path)
    real_eeg_signals = get_real_eeg_signals(eeg_signals_file_path, triggers_relative_time, sensors)
    stimuli = np.array([get_stimuli(real_eeg_signals, triggers_relative_time) for _ in sensors])

    return real_eeg_signals, stimuli.T


if __name__ == "__main__":
    sensors = ['Fz', 'Cz']
    eeg_signals_path = './src/models_src/raw_data.csv'
    triggers_path = './src/models_src/triggers.txt'
    real_eeg_signals, stimuli = get_real_data(eeg_signals_path, triggers_path, sensors)
    plt.plot(real_eeg_signals[sensors])
    plt.show()
