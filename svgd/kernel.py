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

    def __call__(self, x, y):
        """
        Compute the gram matrix in blocks of 5000 for memory efficiency.
        :param x: Input 1
        :type x: array
        :param y: Input 2
        :type y: array
        :return: Gram matrix between x and y with the specified kernel
        :rtype: array
        """
        K = jax.vmap(lambda x1: jax.vmap(lambda y1: self.kernel_fun(x1, y1, self.params))(y))(x)
        return K
    
    def gradient_wrt_first_arg(self, x, y):
        """
        Compute the gradient of the kernel with respect to the first argument.
        :param x: Input 1
        :type x: array
        :param y: Input 2
        :type y: array
        :return: Gradient of the kernel with respect to the first argument
        :rtype: array
        """
        grad_kernel_fun = jax.grad(self.kernel_fun, argnums=0)
        grad_K = jax.vmap(lambda x1: jax.vmap(lambda y1: grad_kernel_fun(x1, y1, self.params))(y))(x)
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

def rbf_kernel(x, y, params):
    length_scale = params['length_scale']
    if isinstance(length_scale, float):
        length_scale = [length_scale]
    dist = sqeuclidean_distances(x, y)
    kvals = jnp.array([jnp.exp(-dist / (2 * l ** 2)) for l in length_scale])
    return jnp.sum(kvals)