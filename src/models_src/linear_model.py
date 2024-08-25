from neural_model import *


def f(x, u, W, tau, M, dt):
     return x + dt * (-x / tau + W @ x + M * u)


class LinearModel(NeuralModel):
    def __init__(self, state_dim, aug_state_dim, sources, dt, initial_x, initial_H, params_dict, Q_x, R_y, P_x_, P_x, P_params_, P_params, Q_params):
        super().__init__(state_dim, aug_state_dim, sources, dt, initial_x, initial_H, params_dict, Q_x, R_y, P_x_, P_x, P_params_, P_params, Q_params)
        self.jac_f_dx = jax.jit(jax.jacobian(f,argnums=(0)))
        self.jac_f_dW = jax.jit(jax.jacobian(f,argnums=(2)))
        self.jac_f_dtau = jax.jit(jax.jacobian(f,argnums=(3)))
        self.jac_f_dM = jax.jit(jax.jacobian(f,argnums=(4)))

    def f_o(self, u):
        self.x[0:self.state_dim] += self.dt * (-self.x[0:self.state_dim] / self.params_dict['tau'] + self.params_dict['W'][0:self.state_dim] @ self.x + self.params_dict['M'] * u)
        return self.x

    def jacobian_f_o_x(self, u):
        F_x = np.array(self.jac_f_dx(self.x, u, self.params_dict['W'], self.params_dict['tau'], self.params_dict['M'], self.dt))
        return F_x
    
    def jacobian_f_o(self, u):

        F_W = np.array(self.jac_f_dW(self.x, u, self.params_dict['W'], self.params_dict['tau'], self.params_dict['M'], self.dt)).reshape(self.aug_state_dim, -1)
        F_M = np.array(self.jac_f_dM(self.x, u, self.params_dict['W'], self.params_dict['tau'], self.params_dict['M'], self.dt)).reshape(self.aug_state_dim, -1) 
        F_tau = np.array(self.jac_f_dtau(self.x, u, self.params_dict['W'], self.params_dict['tau'], self.params_dict['M'], self.dt)).reshape(self.aug_state_dim, -1) 
        J_combined = np.hstack((F_W, F_M, F_tau))
        return J_combined
