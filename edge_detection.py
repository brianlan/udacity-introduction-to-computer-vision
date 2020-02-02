import numpy as np
from scipy.signal import correlate2d


def get_sobel_grad_2d(im):
    """calculate sobel grad for 2d array
    
    Arguments:
        im {np.ndarray} -- assume it's been normalized to [0, 1], dtype is float
    
    Returns:
        np.ndarray -- gradient magnitude
        np.ndarray -- gradient direction, in rad.
    """
    sobel_x = np.array([[-1, 0, 1], 
                        [-2, 0, 2], 
                        [-1, 0, 1]], dtype=np.float32)
    sobel_y = np.array([[-1, -2, -1], 
                        [0, 0, 0], 
                        [1, 2, 1]], dtype=np.float32)
    grad_x = correlate2d(im, sobel_x, mode="same")
    grad_y = correlate2d(im, sobel_y, mode="same")
    grad_magnitude = np.sqrt(np.square(grad_x) + np.square(grad_y)) / (4 * np.sqrt(2))
    grad_direction = np.arctan2(grad_y, grad_x)
    return grad_magnitude, grad_direction
