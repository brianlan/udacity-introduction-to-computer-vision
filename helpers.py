from PIL import Image
import numpy as np


def read_image(path, as_ndarray=True, convert_grayscale=False):
    im = Image.open(path)
    if convert_grayscale:
        im = im.convert("L")
    if as_ndarray:
        im = np.array(im)
    return im


def plot_image(im, ax, title=None, **kwargs):
    ax.set_axis_off()
    ax.imshow(im, **kwargs)
    if title is not None:
        ax.set_title(title)
