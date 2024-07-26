from neural_model import *
import matplotlib.pyplot as plt
import state_functions_dcm as sf_dcm

def measurement_function(x):
    return x[9:11] @ (x[2] - x[3])

class DCM(NeuralModel):
    def __init__(self, C_f, C_l, C_u, C_b, state_dim = 9, aug_state_dim = 11, n_iterations = int(1e3), sources = 2, dt = 1e-2):
        super().__init__()
        #TODO: put constant variables out of the class and as a parameter in the super().__init__()
        self.dt = dt
        self.state_dim = state_dim
        self.aug_state_dim = aug_state_dim
        self.n_iterations = n_iterations
        self.sources = sources
        self.aug_state_dim_flattened = self.aug_state_dim * self.sources
        self.x = np.zeros((self.aug_state_dim, sources))
        self.H = np.eye(self.sources)
        self.x[self.state_dim:self.state_dim + self.sources] = self.H
        # Compute the exponentials to get C values
        self.C_f = C_f
        self.C_b = C_b
        self.C_l = C_l
        self.C_u= C_u
        self.H_e = 0.1
        self.H_i = 0.2
        self.tau_e = 10.0
        self.tau_i = 12.0
        self.gamma_1 = 1.0
        self.gamma_2 = 4/5 
        self.gamma_3 = 1/4
        self.gamma_4 = self.gamma_3
        self.params_f_o = [self.x, 0, self.dt, self.theta, self.H_e, self.tau_e, self.H_i, self.tau_i, self.gamma_1, self.gamma_2, self.gamma_3, self.gamma_4, self.C_f, self.C_l, self.C_u, self.C_b]
        self.params_vec = np.hstack((self.theta, self.H_e, self.tau_e, self.H_i, self.tau_i, self.gamma_1, self.gamma_2, self.gamma_3, self.gamma_4, self.C_f.flatten(), self.C_l.flatten(), self.C_u.flatten(), self.C_b.flatten()))
        self.n_params = len(self.params_vec)
        self.jac_f_dx = jax.jit(jax.jacobian(sf_dcm.f_o, argnums = (0)))
        self.F_x = np.zeros((self.aug_state_dim_flattened, self.aug_state_dim_flattened))
        self.jac_measurement_f_dH = jax.jit(jax.jacobian(measurement_function,argnums=(0)))
        self.jacobian_params_funcs = [
            (jax.jit(jax.jacobian(sf_dcm.f_o, argnums=3)), 'F_theta', -1),
            (jax.jit(jax.jacobian(sf_dcm.f_o, argnums=4)), 'F_H_e', -1),
            (jax.jit(jax.jacobian(sf_dcm.f_o, argnums=5)), 'F_tau_e', -1),
            (jax.jit(jax.jacobian(sf_dcm.f_o, argnums=6)), 'F_H_i', -1),
            (jax.jit(jax.jacobian(sf_dcm.f_o, argnums=7)), 'F_tau_i', -1),
            (jax.jit(jax.jacobian(sf_dcm.f_o, argnums=8)), 'F_gamma_1', -1),
            (jax.jit(jax.jacobian(sf_dcm.f_o, argnums=9)), 'F_gamma_2', -1),
            (jax.jit(jax.jacobian(sf_dcm.f_o, argnums=10)), 'F_gamma_3', -1),
            (jax.jit(jax.jacobian(sf_dcm.f_o, argnums=11)), 'F_gamma_4', -1),
            (jax.jit(jax.jacobian(sf_dcm.f_o, argnums=12)), 'F_C_f', 1),
            (jax.jit(jax.jacobian(sf_dcm.f_o, argnums=13)), 'F_C_l', 1),
            (jax.jit(jax.jacobian(sf_dcm.f_o, argnums=14)), 'F_C_u', 0),
            (jax.jit(jax.jacobian(sf_dcm.f_o, argnums=15)), 'F_C_b', 1)
        ]
    
    def f_o(self,u):
        self.params_f_o[1] = u
        self.x = np.array(sf_dcm.f_o(*self.params_f_o))
        return self.x.flatten()
    
    def jacobian_f_o_x(self, u):
        self.params_f_o[1] = u
        F_x = np.array(self.jac_f_dx(*self.params_f_o)).reshape(self.aug_state_dim_flattened, self.aug_state_dim_flattened)
        return F_x

    def compute_params_jacobian(self, jac_func, *params, reshape):
        jac = np.array(jac_func(*params))
        if reshape == 1:
            jac = jac.reshape(self.aug_state_dim_flattened, self.sources * 2)
        elif reshape == 0:
            jac = jac.reshape(self.aug_state_dim_flattened, self.sources)
        elif reshape == -1:
            jac = jac.reshape(self.aug_state_dim_flattened, 1)
        return jac
    
    def jacobian_f_o(self, u):
        params_f_o = [self.x, u, self.dt, self.theta, self.H_e, self.tau_e, self.H_i, self.tau_i, self.gamma_1, self.gamma_2, self.gamma_3, self.gamma_4, self.C_f, self.C_l, self.C_u, self.C_b]
        
        jac_u = self.jacobian_params_funcs[11][0](self.x, u, self.dt, self.theta, self.H_e, self.tau_e, self.H_i, self.tau_i, self.gamma_1, self.gamma_2, self.gamma_3, self.gamma_4, self.C_f, self.C_l, self.C_u, self.C_b)
        for jac_func, attr, *reshape in self.jacobian_params_funcs:
            setattr(self, attr, self.compute_params_jacobian(jac_func, *params_f_o, reshape=int(reshape[0])))
        
        F_params = np.hstack([getattr(self, attr) for _, attr, *reshape in self.jacobian_params_funcs])
        return F_params
    
    def jacobian_h(self, x):
        dH = np.array(self.jac_measurement_f_dH(x)).reshape(2,22)
        return dH
    
