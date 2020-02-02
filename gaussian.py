import numpy as np


def gaussian_function_2d_particular(x, sigma=1.0):
    """It's a simplified formula from 
    [Multivariate normal distribution](https://en.wikipedia.org/wiki/Multivariate_normal_distribution) 
    for our special case here, where mu is fixed to 0 and the sigmas from both dimensions are the same."""
    return np.exp(-0.5 * (1 / sigma) * (x * x).sum()) / (2 * sigma * np.pi)


def generate_gaussian_kernel(kernel_size, sigma=1.0):
    side = (kernel_size - 1) / 2
    x, y = np.meshgrid(np.arange(kernel_size) - side, np.arange(kernel_size) - side)
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    for i in range(kernel_size):
        for j in range(kernel_size):
            kernel[i, j] = gaussian_function_2d_particular(
                np.array([x[i, j], y[i, j]]), sigma=sigma
            )
    return kernel
