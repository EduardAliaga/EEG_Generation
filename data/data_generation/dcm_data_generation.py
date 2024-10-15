from generate_data import *

if __name__ == "__main__":
    total_time = 300
    dt = 1e-1
    n_time_points = int(total_time / dt)
    period_square = 0.3
    sources = 3
    model = 'dcm'

    params_x = {
        'state_dim': 9,
        'aug_state_dim': 11,
        'sources' : sources,
        # 'H': np.array([[1, 1], [0.5, 0.4]]),
        # 'H': np.array([[-0.38, -0.14, 0.32], [-0.13, -0.05, -0.11], [-0.43, -0.16, -0.37]]),
        'H': np.array([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]]),
        'dt': dt,
        'theta': .56,
        'H_e': 2.08,
        'tau_e': 1.39,
        'H_i': 16.0,
        'tau_i': 32.0,
        'gamma_1': 1.0,
        'gamma_2': 4/5,
        'gamma_3': 1/4,
        'gamma_4': 1/4,  # gamma_3 value
        # 'C_f': np.array([[0, 0, 0], [25, 0, 0], [30, 0, 0]]), #connectivity matrix already transposed
        # 'C_l': np.array([[0, 0, 0], [0, 0, 2], [0, 10, 0]]), #connectivity matrix already transposed
        # 'C_u': np.array([5,0,0]),
        # 'C_b': np.array([[0, 0, 16], [0, 0, 0], [0, 0, 0]]), #connectivity matrix already transposed
        'C_f': np.array([[0, 0, 0], [-0.17, 0, 0], [-0.26, 0, 0]]), #connectivity matrix already transposed
        'C_l': np.array([[0, 0, 0], [0, 0, -0.22], [0, -0.19, 0]]), #connectivity matrix already transposed
        'C_u': np.array([-0.12,0,0]),
        'C_b': np.array([[0, 0, 0.33], [0, 0, 0], [0, 0, 0]]), #connectivity matrix already transposed
    }
    params_y = {
        'H': params_x['H'],
        'state_dim': params_x['state_dim'],
        'aug_state_dim': params_x['aug_state_dim'],
        'sources': params_x['sources']
    }
    generate_synthetic_data(period_square, total_time, n_time_points, params_x, params_y, model)