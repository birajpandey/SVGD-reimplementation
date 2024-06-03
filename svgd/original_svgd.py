from scipy.spatial.distance import pdist, squareform
import numpy as np
from tqdm import tqdm
import jax.numpy as jnp
 
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
            print(f'Gradient: {np.linalg.norm(grad_theta)}')

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
