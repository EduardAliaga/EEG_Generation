import jax
import jax.numpy as np
from neural_for_linear import *

def f(x, u, W, tau, M, theta, dt):
     return x + dt * (-x / tau + W @ jax.nn.sigmoid(x * theta) + M * u)

class SigmoidModel(NeuralModel):
    def __init__(self, state_dim, aug_state_dim, sources, dt, initial_x, initial_H, params_dict, Q_x, R_y, P_x_, P_x, P_params_, P_params, Q_params):
        super().__init__(state_dim, aug_state_dim, sources, dt, initial_x, initial_H, params_dict, Q_x, R_y, P_x_, P_x, P_params_, P_params, Q_params)
        
        # Jacobians for each parameter
        self.jac_function_x = jax.jit(jax.jacobian(f, argnums=(0)))
        self.jacobian_functions = [
            jax.jit(jax.jacobian(f, argnums=(2))),  # W
            jax.jit(jax.jacobian(f, argnums=(3))),  # tau
            jax.jit(jax.jacobian(f, argnums=(4))),  # M
            # jax.jit(jax.jacobian(f, argnums=(5)))   # theta
        ]
        self.theta = 0.56
    def f_o(self, u):
        self.x = np.array(f(self.x[:, 0], u, *list(self.params_dict.values()),self.theta, self.dt))
        return self.x

    def jacobian_f_o_x(self, u):
        F_x = np.array(self.jac_function_x(self.x[:, 0], u, *list(self.params_dict.values()),self.theta, self.dt)).reshape(self.state_dim, self.state_dim)
        return F_x

    def jacobian_f_o(self, u):
        jacobians = []
        for jac_func in self.jacobian_functions:
            jacobian = jac_func(self.x[:, 0], u, *list(self.params_dict.values()),self.theta, self.dt).reshape(self.aug_state_dim, -1)
            jacobians.append(jacobian)
        stacked_jacobians = np.hstack(jacobians)
        return stacked_jacobians
