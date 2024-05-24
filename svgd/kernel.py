import jax
import jax.numpy as jnp
import equinox as eqx


class Kernel(eqx.Module):
    """ Base Kernel class.
    """
    kernel_fun: callable
    params: dict

    def __init__(self, kernel_fun, params):
        """
        :param kernel_fun: A kernel function with the signature f(X, Y, parameters)
        :type kernel_fun: function
        :param params: Kernel specific parameters.
        :type params: dict
        """
        self.params = params
        self.kernel_fun = kernel_fun

    def __call__(self, x, y, params=None):
        """
        Compute the gram matrix in blocks of 5000 for memory efficiency.
        :param x: Input 1
        :type x: array
        :param y: Input 2
        :type y: array
        :return: Gram matrix between x and y with the specified kernel
        :rtype: array
        """
        if params is None:
            params = self.params
        K = jax.vmap(
            lambda x1: jax.vmap(lambda y1: self.kernel_fun(x1, y1, params))(
                y))(x)
        return K

    def gradient_wrt_first_arg(self, x, y, params=None):
        """
        Compute the gradient of the kernel with respect to the first argument.
        :param x: Input 1
        :type x: array
        :param y: Input 2
        :type y: array
        :return: Gradient of the kernel with respect to the first argument
        :rtype: array
        """
        if params is None:
            params = self.params
        grad_kernel_fun = jax.grad(self.kernel_fun, argnums=0)
        grad_K = jax.vmap(
            lambda x1: jax.vmap(lambda y1: grad_kernel_fun(x1, y1, params))(
                y))(x)
        return grad_K


def sqeuclidean_distances(x, y):
    """
    Compute squared euclidean distance between two vectors

    :param x: Input 1
    :type x: array
    :param y: Input 2
    :type y: array
    :return: squared euclidean distance
    :rtype: float
    """
    return jnp.sum((x - y) ** 2)


# Example kernel function
def rbf_kernel(x, y, params):
    # print(params['length_scale'])
    length_scale = params['length_scale']
    sqdist = jnp.sum((x - y) ** 2)
    return jnp.exp(-sqdist / (2 * length_scale ** 2))