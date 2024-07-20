from neural_model import *

def f(x, u, W, tau, M, theta, dt):
     return x + dt * (-x / tau + W @ jax.nn.sigmoid(x*theta) + M * u)

class SigmoidModel(NeuralModel):
    def __init__(self):
        super().__init__()

        self.jac_f_dx = jax.jit(jax.jacobian(f,argnums=(0)))
        self.jac_f_dW = jax.jit(jax.jacobian(f,argnums=(2)))
        self.jac_f_dtau = jax.jit(jax.jacobian(f,argnums=(3)))
        self.jac_f_dM = jax.jit(jax.jacobian(f,argnums=(4)))
        self.jac_f_dtheta = jax.jit(jax.jacobian(f,argnums=(5)))

        self.params_dict = {
            'W': self.W,
            'M': self.M,
            'tau': self.tau,
            'theta' : self.theta
        }
        self.params_vec = np.hstack((self.W.flatten(), self.M, self.tau, self.theta))

    def f_o(self, u):
        self.x[0:self.state_dim] += self.dt * (-self.x[0:self.state_dim] / self.params_dict['tau'] + self.params_dict['W'][0:self.state_dim] @ np.array(jax.nn.sigmoid(self.x * self.params_dict['theta'])) + self.params_dict['M'] * u)
        return self.x
    
    def jacobian_f_o_x(self, u):
        F_x = np.array(self.jac_f_dx(self.x, u, self.params_dict['W'], self.params_dict['tau'], self.params_dict['M'], self.dt, self.params_dict['theta']))
        return F_x

    
    def jacobian_f_o(self, u):
        F_W = np.array(self.jac_f_dW(self.x, u, self.params_dict['W'], self.params_dict['tau'], self.params_dict['M'], self.params_dict['theta'], self.dt)).reshape(self.aug_state_dim, -1)
        F_M = np.array(self.jac_f_dM(self.x, u, self.params_dict['W'], self.params_dict['tau'], self.params_dict['M'], self.params_dict['theta'], self.dt)).reshape(self.aug_state_dim, -1) 
        F_tau = np.array(self.jac_f_dtau(self.x, u, self.params_dict['W'], self.params_dict['tau'], self.params_dict['M'], self.params_dict['theta'], self.dt)).reshape(self.aug_state_dim, -1)
        F_theta = np.array(self.jac_f_dtheta(self.x, u, self.params_dict['W'], self.params_dict['tau'], self.params_dict['M'], self.params_dict['theta'], self.dt)).reshape(self.aug_state_dim, -1)
        J_combined = np.hstack((F_W, F_M, F_tau, F_theta))
        return J_combined