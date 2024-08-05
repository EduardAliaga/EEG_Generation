from neural_model import *
import matplotlib.pyplot as plt
import state_functions_dcm as sf_dcm

def measurement_function(x):
    return x[9:11] @ (x[2] - x[3])

class DCM(NeuralModel):
    def __init__(self, state_dim, aug_state_dim, sources, dt, initial_x, initial_H, params_dict, Q_x, R_y, P_x_, P_x, P_params_, P_params, Q_params):
        super().__init__(state_dim, aug_state_dim, sources, dt, initial_x, initial_H, params_dict, Q_x, R_y, P_x_, P_x, P_params_, P_params, Q_params)
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
        self.x = np.array(sf_dcm.f_o(self.x, u, self.dt, *list(self.params_dict.values()))).flatten()
        return self.x
    
    def jacobian_f_o_x(self, u):
        F_x = np.array(jax.jit(jax.jacobian(sf_dcm.f_o, argnums = (0)))(self.x, u, self.dt, *list(self.params_dict.values()))).reshape(22,22)
        return F_x

    def compute_params_jacobian(self, jac_func, u, reshape, *params):
        jac = np.array(jac_func(self.x, u, self.dt, *params))
        if reshape == 1:
            jac = jac.reshape(self.aug_state_dim_flattened, self.sources * 2)
        elif reshape == 0:
            jac = jac.reshape(self.aug_state_dim_flattened, self.sources)
        elif reshape == -1:
            jac = jac.reshape(self.aug_state_dim_flattened, 1)
        return jac
    
    def jacobian_f_o(self,u):
            for jac_func, attr, reshape in self.jacobian_params_funcs:
                setattr(self, attr, self.compute_params_jacobian(jac_func, u, reshape,  *list(self.params_dict.values())))
            F_params = np.hstack([getattr(self, attr) for _, attr, reshape in self.jacobian_params_funcs])
            return F_params

    def jacobian_h(self, x):
        dH = np.array(self.jac_measurement_f_dH(x)).reshape(2,22)
        return dH
    
