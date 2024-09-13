import sys
sys.path.insert(0, '../')
from generate_data import *

if __name__ == "__main__":
    total_time = 300
    dt = 1e-1
    n_time_points = int(total_time / dt)
    period_square = 3
    sources = 1
    model = 'sigmoid'
    tau = 100.0
    M = 100.0
    theta = 1.0
    W = np.zeros((2,2))
    W[0,1] = 1e-1
    W[1,0] = -1e-1
    H = np.array([[1, 1], [0.5, 0.4]])
    params_x = {
        'state_dim': 2,
        'aug_state_dim': 2,
        'sources' : 1,
        'dt': dt,
        'theta' : 0.56,
        'tau' : tau,
        'M': M,
        'W': W,
        'H': H,
    }
    params_y = {
        'H': params_x['H'],
        'state_dim': params_x['state_dim'],
        'aug_state_dim': params_x['aug_state_dim'],
        'sources': params_x['sources']
    }
    generate_synthetic_data(period_square, total_time, n_time_points, params_x, params_y, model)