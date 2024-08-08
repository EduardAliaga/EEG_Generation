import numpy as np
from generate_data import generate_synthetic_data
from visualize import plot_all
from linear_model import LinearModel
from sigmoid_model import SigmoidModel
from dcm_model import DCM
from utils import *
import jax.numpy as jnp
from jax import grad
from jax import jacobian
import matplotlib.pyplot as plt

def main():
    data_file='/Users/aliag/Desktop/EEG_Generation/src/models_src/synthetic_data.npy'
    data = np.load(data_file, allow_pickle=True).item()
    stimuli = data['stimuli']
    states = data['states']
    print(states.shape)
    states = np.array(states)
    measurements = data['measurements']
    measurements_noisy = data['measurements_noisy']
    aug_state_dim_flattened = 22
    n_params = 23
    covariance_value = 1e-6
    sources = 2
    dt = 1e-2

    params_dict = {
                    'theta': 0.5,
                    'H_e': 0.4,
                    'tau_e': 15.0,
                    'H_i': 0.2,
                    'tau_i': 12.0,
                    'gamma_1': 4.0,
                    'gamma_2': 4/20,
                    'gamma_3': 1/7,
                    'gamma_4': 1/7,
                    'C_f': np.random.randn(sources, sources),
                    'C_l': np.random.randn(sources, sources), 
                    'C_u': np.random.randn(sources),
                    'C_b': np.random.randn(sources, sources)
                }
    np.save("initial_params.npy", params_dict)
    Q_x = np.eye(aug_state_dim_flattened) * covariance_value
    R_y = np.eye(sources) * covariance_value
    P_x_ = np.eye(aug_state_dim_flattened) * covariance_value
    P_x = np.eye(aug_state_dim_flattened) * covariance_value
    P_params_ = np.eye(n_params) * covariance_value
    P_params = np.eye(n_params) * covariance_value
    Q_params = np.eye(n_params) * covariance_value
    aug_state_dim = 11
    initial_x = np.zeros((aug_state_dim, sources))
    initial_H = np.eye(sources)
    state_dim = 9
    model = DCM(state_dim, aug_state_dim, sources, dt, initial_x, initial_H, params_dict, Q_x, R_y, P_x_, P_x, P_params_, P_params, Q_params)
    states_predicted, measurements_predicted = model.fit(stimuli, measurements_noisy)
    #test_states_predicted, test_measurements_predicted = model.test(stimuli[2500:3000])

    H = states_predicted[-1, 9:11, :]
    y = []
    for t in range(0,len(states_predicted)):
        x0 = states_predicted[t, 2, :] - states_predicted[t, 3, :]
        y.append(H @ x0)
    nsqe_measurements = get_norm_squared_error(measurements, y)


    #norm_squared_errors =  get_norm_squared_errors(states_predicted, measurements_predicted, states, measurements, model.params_dict, real_params, model.state_dim)

   #save_results(model.params_dict, states_predicted, norm_squared_errors)

    np.save("states_predicted_synthetic_data.npy", states_predicted)
    np.save("measurements_predicted_synthetic_data.npy", measurements_predicted)
    np.save("params_vec_synthetic_data.npy", model.params_vec)
    np.save("H_synthetic_data.npy", H)

    data_file ='synthetic_data.npy'
    data_file_2 ='states_predicted.npy'
    data_file_3 = 'measurements_predicted.npy' 
    data_file_4 = 'params_vec.npy'
    data_file_5 = 'H.npy'
    #stimuli, states, measurements, measurements_noisy, real_params = load_synthetic_data(data_file, f)
    data = np.load(data_file, allow_pickle=True).item()
    states_predicted = np.load(data_file_2, allow_pickle = True)
    measurements_predicted = np.load(data_file_3, allow_pickle = True)
    params_vec = np.load(data_file_4, allow_pickle = True)
    H = np.load(data_file_5, allow_pickle = True)
    y = []
    for t in range(0,len(states_predicted)):
        x0 = states_predicted[t, 2, :] - states_predicted[t, 3, :]
        y.append(H @ x0)
    y = np.array(y)
    print(params_vec)
    print(H)
    stimuli = data['stimuli']
    states = data['states']
    states = np.array(states)
    measurements = data['measurements']
    measurements_noisy = data['measurements_noisy']
    print(states_predicted.shape)

    nsqe_measurements = []

    # for t in range(0,3000):
    #     nsqe_measurements.append(get_norm_squared_error(measurements_noisy[t,0], measurements_predicted[t,0]))
    # nsqe_measurements = np.array(nsqe_measurements)

    plt.figure()
    plt.plot(states_predicted[:,0,0], c = 'blue', label = "predicted state x0 for source 0")
    plt.plot(states_predicted[:,0,1], c = 'orange', label = "predicted state x0 for source 1")
    plt.plot(states[0,0,:], c = 'blue', linestyle = '--', alpha = 0.5, label = "ground truth for source 0")
    plt.plot(states[0,1,:], c = 'orange', linestyle = '--', alpha = 0.5, label = "ground truth for source 1")
    plt.xlabel("sample")
    plt.ylabel("state value")
    plt.title("Predicted vs ground truth for state x0")
    plt.legend()

    plt.figure()
    plt.plot(states_predicted[:,4,0], c = 'blue', label = "predicted state x4 for source 0")
    plt.plot(states_predicted[:,4,1], c = 'orange', label = "predicted state x4 for source 1")
    plt.plot(states[4,0,:], c = 'blue', linestyle = '--', alpha = 0.5, label = "ground truth for source 0")
    plt.plot(states[4,1,:], c = 'orange', linestyle = '--', alpha = 0.5, label = "ground truth for source 1")
    plt.xlabel("sample")
    plt.ylabel("state value")
    plt.title("Predicted vs ground truth for state x4")
    plt.legend()

    plt.figure()
    plt.plot(states_predicted[:,9,0], c = 'blue', label = "predicted state x4 for source 0")
    plt.plot(states_predicted[:,9,1], c = 'orange', label = "predicted state x4 for source 1")
    plt.plot(states[9,0,:], c = 'blue', linestyle = '--', alpha = 0.5, label = "ground truth for source 0")
    plt.plot(states[9,1,:], c = 'orange', linestyle = '--', alpha = 0.5, label = "ground truth for source 1")
    plt.xlabel("sample")
    plt.ylabel("state value")
    plt.title("Predicted vs ground truth for state x9")
    plt.legend()

    plt.figure()
    plt.plot(states_predicted[:,10,0], c = 'blue', label = "predicted state x4 for source 0")
    plt.plot(states_predicted[:,10,1], c = 'orange', label = "predicted state x4 for source 1")
    plt.plot(states[10,0,:], c = 'blue', linestyle = '--', alpha = 0.5, label = "ground truth for source 0")
    plt.plot(states[10,1,:], c = 'orange', linestyle = '--', alpha = 0.5, label = "ground truth for source 1")
    plt.xlabel("sample")
    plt.ylabel("state value")
    plt.title("Predicted vs ground truth for state x10")
    plt.legend()

    plt.figure()
    plt.plot(measurements_predicted[:,0], c = 'blue', label = "predicted measurement for source 0")
    plt.plot(measurements_predicted[:,1], c = 'orange', label = "predicted measurement for source 1")
    plt.plot(measurements[:,0], c = 'blue', linestyle = '--', alpha = 0.5, label = "ground truth for source 0")
    plt.plot(measurements[:,1], c = 'orange', linestyle = '--', alpha = 0.5, label = "ground truth for source 1")
    plt.xlabel("sample")
    plt.ylabel("measurement value")
    plt.title("Predicted vs ground truth measurements")
    plt.legend()

    plt.figure()
    plt.plot(measurements_predicted[:,0], c = 'blue', label = "predicted measurement for source 0")
    plt.plot(measurements_predicted[:,1], c = 'orange', label = "predicted measurement for source 1")
    plt.plot(measurements_noisy[:,0], c = 'blue', linestyle = '--', alpha = 0.5, label = "ground truth for source 0")
    plt.plot(measurements_noisy[:,1], c = 'orange', linestyle = '--', alpha = 0.5, label = "ground truth for source 1")
    plt.xlabel("sample")
    plt.ylabel("measurement value")
    plt.title("Predicted vs ground truth noisy measurements")
    plt.legend()

    plt.show()

    # for i in range(0,9):
    #     plt.figure()
    #     plt.plot(states[i, 0,:])
    #     plt.plot(states_predicted[:,i, 0])
    # plt.show()
    #plot_all()

if __name__ == "__main__":
    main()
