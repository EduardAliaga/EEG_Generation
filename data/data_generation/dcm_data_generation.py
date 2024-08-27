from generate_data import *

if __name__ == "__main__":
    total_time = 30
    dt = 1e-2
    n_time_points = int(total_time / dt)
    period_square = 0.3
    sources = 2
    model = 'dcm'

    params_x = {
        'state_dim': 9,
        'aug_state_dim': 11,
        'sources' : 2,
        'H': np.array([[1, 0.7], [0.5, 0.8]]),
        'dt': dt,
        'theta': 0.56,
        'H_e': 0.1,
        'tau_e': 10.0,
        'H_i': 16.0,
        'tau_i': 32.0,
        'gamma_1': 1.0,
        'gamma_2': 4/5,
        'gamma_3': 1/4,
        'gamma_4': 1/4,  # gamma_3 value
        'sources': 2,
        'C_f': np.eye(sources),
        'C_l': np.eye(sources), 
        'C_u': np.ones(sources),
        'C_b': np.eye(sources), 
    }
    params_y = {
        'H': params_x['H'],
        'state_dim': params_x['state_dim'],
        'aug_state_dim': params_x['aug_state_dim'],
        'sources': params_x['sources']
    }
    generate_synthetic_data(period_square, total_time, n_time_points, params_x, params_y, model)