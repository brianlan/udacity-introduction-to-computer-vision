def plot_image(im, ax, title=None, **kwargs):
    ax.set_axis_off()
    ax.imshow(im, **kwargs)
    if title is not None:
        ax.set_title(title)
