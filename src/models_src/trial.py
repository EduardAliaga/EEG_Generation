import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dcm_model import DCM


eeg_data = pd.read_csv('/Users/aliag/Desktop/Data/S001/S001_Matrix_Calibration_Mon_15_May_2023_10hr39min42sec_-0400/raw_data.csv', skiprows=2)


triggers = []
with open('/Users/aliag/Desktop/Data/S001/S001_Matrix_Calibration_Mon_15_May_2023_10hr39min42sec_-0400/triggers.txt', 'r') as file:
    for line in file:
        parts = line.strip().split()
        if len(parts) == 3:
            triggers.append((parts[0], parts[1], float(parts[2])))

trigger_timestamps = [timestamp for _, _, timestamp in triggers]


initial_timestamp_triggers = min(trigger_timestamps) *-1
initial_timestamp_csv = min(eeg_data['lsl_timestamp'])


eeg_data['relative_time'] = eeg_data['lsl_timestamp'] - initial_timestamp_csv

relative_triggers = [(letter, label, timestamp - initial_timestamp_triggers) for letter, label, timestamp in triggers]


first_trigger_relative_time = relative_triggers[1][2]
last_trigger_relative_time = relative_triggers[-2][2]
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

new_eeg_data = eeg_data[initial_time_experiment:end_time_experiment]

stimuli = np.zeros(end_time_experiment-initial_time_experiment)

for i in range(len(relative_triggers)):
    new_eeg_data['time_diff'] = (new_eeg_data['relative_time'] - relative_triggers[i][2]).abs()
    closest_index = new_eeg_data['time_diff'].idxmin()
    closest_row = new_eeg_data.loc[closest_index]
    position = int(closest_row['timestamp'] - initial_time_experiment)
    if relative_triggers[i][1] == 'nontarget':
        stimuli[position-30:position] = 0.5
    elif relative_triggers[i][1] == 'target':
        stimuli[position-30:position] = 1

initial_time = new_eeg_data['relative_time'].iloc[0]
new_eeg_data['relative_time'] = new_eeg_data['relative_time'] - initial_time

sensors_data = np.array(new_eeg_data[['Fz', 'Cz']])
time_axis = np.array(new_eeg_data['relative_time'])
# plt.plot(time_axis, sensors_data[:,0])
# plt.show()

# new_stimuli = np.array([stimuli, stimuli]).reshape(260152, 2)
# new_stimuli = new_stimuli[0:10000]
# sensors_data = sensors_data[0:10000]

# aug_state_dim_flattened = 22
# n_params = 23
# covariance_value = 1e-6
# sources = 2
# dt = 1e-3

# params_dict = {
#                 'theta': 0.5,
#                 'H_e': 0.4,
#                 'tau_e': 15.0,
#                 'H_i': 0.2,
#                 'tau_i': 12.0,
#                 'gamma_1': 4.0,
#                 'gamma_2': 4/20,
#                 'gamma_3': 1/7,
#                 'gamma_4': 1/7,
#                 'C_f': np.random.randn(sources, sources),
#                 'C_l': np.random.randn(sources, sources), 
#                 'C_u': np.random.randn(sources),
#                 'C_b': np.random.randn(sources, sources)
#             }
# np.save("initial_params.npy", params_dict)
# Q_x = np.eye(aug_state_dim_flattened) * covariance_value
# R_y = np.eye(sources) * covariance_value
# P_x_ = np.eye(aug_state_dim_flattened) * covariance_value
# P_x = np.eye(aug_state_dim_flattened) * covariance_value
# P_params_ = np.eye(n_params) * covariance_value
# P_params = np.eye(n_params) * covariance_value
# Q_params = np.eye(n_params) * covariance_value
# aug_state_dim = 11
# initial_x = np.zeros((aug_state_dim, sources))
# initial_H = np.eye(sources)
# state_dim = 9
# model = DCM(state_dim, aug_state_dim, sources, dt, initial_x, initial_H, params_dict, Q_x, R_y, P_x_, P_x, P_params_, P_params, Q_params)
# states_predicted, measurements_predicted = model.fit(new_stimuli, sensors_data)

# np.save("states_predicted_real_with_stimuli.npy", states_predicted)
# np.save("measurements_predicted_real_with_stimuli.npy", measurements_predicted)


data_measurements = "/Users/aliag/Desktop/EEG_Generation/src/models_src/measurements_predicted_real_with_stimuli.npy"
data_states = "/Users/aliag/Desktop/EEG_Generation/src/models_src/states_predicted_real_with_stimuli.npy"
data_measurements_no_stimuli = "/Users/aliag/Desktop/EEG_Generation/src/models_src/measurements_predicted_real.npy"
data_states_no_stimuli = "/Users/aliag/Desktop/EEG_Generation/src/models_src/states_predicted_real.npy"

states_predicted = np.load(data_states, allow_pickle = True)
measurements_predicted = np.load(data_measurements, allow_pickle = True)
states_predicted_no_stimuli = np.load(data_states_no_stimuli, allow_pickle = True)
measurements_predicted_no_stimuli = np.load(data_measurements_no_stimuli, allow_pickle = True)

# plot ground truth vs predictions with x axis as seconds
plt.figure()
plt.plot(time_axis[0:10000], sensors_data[0:10000,0],alpha = 0.5, linestyle = '--', label = 'ground truth source 0')
plt.plot(time_axis[0:10000], measurements_predicted[:,0], label = 'prediction source 0')
plt.xlabel("Time (s)")
plt.ylabel("Potential (mV)")
plt.title("Predictions vs ground truth measurements with no stimuli for source 0")
plt.legend()
plt.show()

plt.figure()
plt.plot(time_axis[0:10000], sensors_data[0:10000,1], alpha = 0.5, linestyle = '--', label = 'ground truth source 1')
plt.plot(time_axis[0:10000], measurements_predicted[:,1], label = 'prediction source 1')
plt.xlabel("Time (s)")
plt.ylabel("Potential (mV)")
plt.title("Predictions vs ground truth measurements with no stimuli for source 1")
plt.legend()
plt.show()

#plot predicted states with time axis
plt.figure()
plt.plot(time_axis[0:10000], states_predicted[:,0,0], label = 'Predicted states for source 0')
plt.plot(time_axis[0:10000], states_predicted[:,0,1], label = 'Predicted states for source 1')
plt.xlabel("Time (s)")
plt.ylabel("Potential (mV)")
plt.title("Predicted states with no stimuli")
plt.legend()
plt.show()
# plt.figure()
# plt.plot(sensors_data[:,1])
# plt.plot(measurements_predicted[:,1])

# plt.figure()
# plt.plot(states_predicted[:,4,:])
# plt.plot(stimuli[0:10000])
# plt.show()