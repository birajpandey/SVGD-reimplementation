from scipy.spatial.distance import pdist, squareform
import numpy as np
from tqdm import tqdm
import jax.numpy as jnp
import optax
class SVGDModel():

    def __init__(self):
        pass

    def svgd_kernel(self, theta, h=-1):
        sq_dist = pdist(theta)
        pairwise_dists = squareform(sq_dist) ** 2
        if h < 0:  # if h < 0, using median trick
            h = np.median(pairwise_dists)
            h = np.sqrt(0.5 * h / np.log(theta.shape[0] + 1))
            # h = 0.5

        # print(f'Kernel length scale: {h}')

        # compute the rbf kernel
        Kxy = np.exp(-pairwise_dists / h ** 2 / 2)

        dxkxy = -np.matmul(Kxy, theta)
        sumkxy = np.sum(Kxy, axis=1)
        for i in range(theta.shape[1]):
            dxkxy[:, i] = dxkxy[:, i] + np.multiply(theta[:, i], sumkxy)
        dxkxy = dxkxy / (h ** 2)
        return (Kxy, dxkxy)

    def calculate_gradients(self, theta, lnprob, h=-1):
        Kxy, dxkxy = self.svgd_kernel(theta, h)
        lngrad = lnprob(theta)
        grad_theta = (np.matmul(Kxy, lngrad) + dxkxy) / theta.shape[0]
        return grad_theta

    def update(self, x0, lnprob, n_iter=1000, stepsize=1e-3, bandwidth=-1,
               alpha=0.9, debug=False, adagrad=True):
        # Check input
        if x0 is None or lnprob is None:
            raise ValueError('x0 or lnprob cannot be None!')

        theta = np.copy(x0)

        # adagrad with momentum
        fudge_factor = 1e-6
        historical_grad = 0
        for iter in tqdm(range(n_iter)):
            # calculate gradient
            grad_theta = self.calculate_gradients(theta, lnprob, bandwidth)

            if adagrad:
                if iter == 0:
                    historical_grad = historical_grad + grad_theta ** 2
                else:
                    historical_grad = alpha * historical_grad + (1 - alpha) * (
                                grad_theta ** 2)
                adj_grad = np.divide(grad_theta,
                                     fudge_factor + np.sqrt(historical_grad))
                theta = theta + stepsize * adj_grad
            else:
                theta = theta + stepsize * grad_theta

        return theta

    def update_optimizer(self, x0, lnprob, n_iter=1000, stepsize=1e-3,
                         bandwidth=-1, alpha=0.9, debug=False, adagrad=True):
        # Check input
        if x0 is None or lnprob is None:
            raise ValueError('x0 or lnprob cannot be None!')

        theta = np.copy(x0)

        # Initialize the Optax Adagrad optimizer
        optimizer = optax.adagrad(learning_rate=stepsize,
                                  initial_accumulator_value=alpha,
                                  eps=1e-6)
        optimizer_state = optimizer.init(theta)

        for iter in tqdm(range(n_iter)):
            if debug and (iter + 1) % 1000 == 0:
                print('iter ' + str(iter + 1))

            # calcualte gradient
            grad_theta = self.calculate_gradients(theta, lnprob, h=bandwidth)

            # Optax update
            updates, optimizer_state = optimizer.update(grad_theta,
                                                        optimizer_state, theta)
            theta = optax.apply_updates(theta, updates)

        return theta

    def test_optimizers(self):
        def manual_adagrad_update(theta, grad_theta, historical_grad, stepsize,
                                  alpha=0.9, fudge_factor=1e-6):
            if historical_grad is None:
                historical_grad = np.zeros_like(grad_theta)
            historical_grad = alpha * historical_grad + (1 - alpha) * (
                        grad_theta ** 2)
            adj_grad = grad_theta / (fudge_factor + np.sqrt(historical_grad))
            updated_theta = theta + stepsize * adj_grad
            return updated_theta, historical_grad

        def custom_adagrad(learning_rate, initial_accumulator_value=0.0,
                           eps=1e-6, alpha=0.9):
            def init_fn(params):
                return params, jnp.full_like(params, initial_accumulator_value)

            def update_fn(updates, state, params=None):
                params, historical_grad = state
                historical_grad = alpha * historical_grad + (1 - alpha) * (
                            updates ** 2)
                scaled_updates = updates / (eps + jnp.sqrt(historical_grad))
                new_params = params + learning_rate * scaled_updates
                return new_params, (new_params, historical_grad)

            return optax.GradientTransformation(init_fn, update_fn)

        def optax_adagrad_update(theta, grad_theta, optimizer_state,
                                 optimizer):
            updates, optimizer_state = optimizer.update(grad_theta,
                                                        optimizer_state, theta)
            updated_theta = optax.apply_updates(theta, updates)
            return updated_theta, optimizer_state

        np.random.seed(0)
        theta = np.random.randn(10, 5)
        grad_theta = np.random.randn(10, 5)
        stepsize = 1e-2
        alpha = 0.9
        fudge_factor = 1e-6

        # Manual Adagrad
        historical_grad = np.zeros_like(grad_theta)
        updated_theta_manual, historical_grad = manual_adagrad_update(theta,
                                                                      grad_theta,
                                                                      historical_grad,
                                                                      stepsize,
                                                                      alpha,
                                                                      fudge_factor)

        # Optax Adagrad
        optimizer = optax.adagrad(learning_rate=stepsize,
                                  initial_accumulator_value=alpha,
                                  eps=fudge_factor)
        optimizer_state = optimizer.init(theta)
        updated_theta_optax, optimizer_state = optax_adagrad_update(theta,
                                                                    grad_theta,
                                                                    optimizer_state,
                                                                    optimizer)

        # Compare the updates
        np.testing.assert_allclose(updated_theta_manual, updated_theta_optax,
                                   rtol=1e-5, atol=1e-8)
        print("Test passed. Both optimizers produce similar updates.")
