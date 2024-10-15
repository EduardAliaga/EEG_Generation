from neural_model import *
import matplotlib.pyplot as plt
import state_functions_dcm as sf_dcm

def measurement_function(x,H):
    return H @ (x[2] - x[3])

class DCM(NeuralModel):
    def __init__(self, state_dim, aug_state_dim, sources, dt, initial_x, initial_H, params_dict, Q_x, R_y, P_x_, P_x, P_params_, P_params, Q_params):
        super().__init__(state_dim, aug_state_dim, sources, dt, initial_x, initial_H, params_dict, Q_x, R_y, P_x_, P_x, P_params_, P_params, Q_params)
        self.jac_f_dx = jax.jit(jax.jacobian(sf_dcm.f_o, argnums = (0)))
        self.F_x = np.zeros((self.aug_state_dim_flattened, self.aug_state_dim_flattened))
        self.jac_measurement_f_dH = jax.jit(jax.jacobian(measurement_function,argnums=(0)))
        self.jac_x_func = jax.jit(jax.jacobian(sf_dcm.f_o, argnums=0))
        self.jacobian_params_funcs = [
            # (jax.jit(jax.jacobian(sf_dcm.f_o, argnums=3)), 'F_theta', -1),
            # (jax.jit(jax.jacobian(sf_dcm.f_o, argnums=4)), 'F_H_e', -1),
            # (jax.jit(jax.jacobian(sf_dcm.f_o, argnums=5)), 'F_tau_e', -1),
            # (jax.jit(jax.jacobian(sf_dcm.f_o, argnums=6)), 'F_H_i', -1),
            # (jax.jit(jax.jacobian(sf_dcm.f_o, argnums=7)), 'F_tau_i', -1),
            # (jax.jit(jax.jacobian(sf_dcm.f_o, argnums=8)), 'F_gamma_1', -1),
            # (jax.jit(jax.jacobian(sf_dcm.f_o, argnums=9)), 'F_gamma_2', -1),
            # (jax.jit(jax.jacobian(sf_dcm.f_o, argnums=10)), 'F_gamma_3', -1),
            # (jax.jit(jax.jacobian(sf_dcm.f_o, argnums=11)), 'F_gamma_4', -1),
            (jax.jit(jax.jacobian(sf_dcm.f_o, argnums=12)), 'F_C_f', 1),
            (jax.jit(jax.jacobian(sf_dcm.f_o, argnums=13)), 'F_C_l', 1),
            (jax.jit(jax.jacobian(sf_dcm.f_o, argnums=14)), 'F_C_u', 0),
            (jax.jit(jax.jacobian(sf_dcm.f_o, argnums=15)), 'F_C_b', 1)
        ]
    
    def f_o(self,u):
        new_params = self.params_dict.copy()
        C_f = np.zeros((3,3))
        C_b = np.zeros((3,3))
        C_l = np.zeros((3,3))
        C_u = np.zeros(3)
        C_f[1,0] = self.params_dict["C_f"][0]
        C_f[2,0] = self.params_dict["C_f"][1]
        C_u[0] = self.params_dict['C_u']
        C_l[1,2] = self.params_dict['C_l'][0]
        C_l[2,1] = self.params_dict['C_l'][1]
        C_b[0,2] = self.params_dict['C_b']
        new_params['C_f'] = C_f
        new_params['C_l'] = C_l
        new_params['C_b'] = C_b
        new_params['C_u'] = C_u
        #x, u, dt, theta, H_e, tau_e, H_i, tau_i, gamma_1, gamma_2, gamma_3, gamma_4, C_f, C_l, C_u, C_b
        self.x = np.array(sf_dcm.f_o(self.x, u, self.dt,self.theta, self.H_e, self.tau_e, self.H_i, self.tau_i, self.gamma_1, self.gamma_2, self.gamma_3, self.gamma_4, *list(new_params.values()))).flatten()
        return self.x
    
    def jacobian_f_o_x(self, u):
        new_params = self.params_dict.copy()
        C_f = np.zeros((3,3))
        C_b = np.zeros((3,3))
        C_l = np.zeros((3,3))
        C_u = np.zeros(3)
        C_f[1,0] = self.params_dict["C_f"][0]
        C_f[2,0] = self.params_dict["C_f"][1]
        C_u[0] = self.params_dict['C_u']
        C_l[1,2] = self.params_dict['C_l'][0]
        C_l[2,1] = self.params_dict['C_l'][1]
        C_b[0,2] = self.params_dict['C_b']
        new_params['C_f'] = C_f
        new_params['C_l'] = C_l
        new_params['C_b'] = C_b
        new_params['C_u'] = C_u
        F_x = np.array(self.jac_x_func(self.x, u, self.dt,self.theta, self.H_e, self.tau_e, self.H_i, self.tau_i, self.gamma_1, self.gamma_2, self.gamma_3, self.gamma_4, *list(new_params.values()))).reshape(27,27)
        return F_x

    def compute_params_jacobian(self, jac_func, u, reshape, *params):
        jac = np.array(jac_func(self.x, u, self.dt, self.theta, self.H_e, self.tau_e, self.H_i, self.tau_i, self.gamma_1, self.gamma_2, self.gamma_3, self.gamma_4, *params))
        if reshape == 1:
            jac = jac.reshape(self.aug_state_dim_flattened, self.sources ** 2)
        elif reshape == 0:
            jac = jac.reshape(self.aug_state_dim_flattened, self.sources)
        elif reshape == -1:
            jac = jac.reshape(self.aug_state_dim_flattened, 1)
        return jac
    
    def jacobian_f_o(self,u):
            new_params = self.params_dict.copy()
            C_f = np.zeros((3,3))
            C_b = np.zeros((3,3))
            C_l = np.zeros((3,3))
            C_u = np.zeros(3)
            C_f[1,0] = self.params_dict["C_f"][0]
            C_f[2,0] = self.params_dict["C_f"][1]
            C_u[0] = self.params_dict['C_u']
            C_l[1,2] = self.params_dict['C_l'][0]
            C_l[2,1] = self.params_dict['C_l'][1]
            C_b[0,2] = self.params_dict['C_b']
            new_params['C_f'] = C_f
            new_params['C_l'] = C_l
            new_params['C_b'] = C_b
            new_params['C_u'] = C_u
            for jac_func, attr, reshape in self.jacobian_params_funcs:
                setattr(self, attr, self.compute_params_jacobian(jac_func, u, reshape,  *list(new_params.values())))
            F_params = np.hstack([getattr(self, attr) for _, attr, reshape in self.jacobian_params_funcs])
            new_F_params = np.zeros((27,6))
            # new_F_params[:,0:9] = F_params[:,0:9]
            new_F_params[:,0] = F_params[:,3]
            new_F_params[:,1] = F_params[:,6]
            new_F_params[:,2] = F_params[:,14]
            new_F_params[:,3] = F_params[:,16]
            new_F_params[:,4] = F_params[:,18]
            new_F_params[:,5] = F_params[:,23]
            return new_F_params

    def jacobian_h(self, x, H):
        dH = np.array(self.jac_measurement_f_dH(x, H)).reshape(2,18)
        return dH
