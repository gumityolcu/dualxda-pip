import torch
import matplotlib.pyplot as plt
import matplotlib.transforms
import matplotlib.path
from matplotlib.collections import LineCollection
import matplotlib.colors as colors
import numpy as np
from zennit.core import BasicHook


class InitRelevanceRule(BasicHook):
    def __init__(self, init_grad):
        super().__init__()
        self.init_grad = init_grad

    def backward(self, module, grad_input, grad_output):
        init = self.init_grad.to(grad_output[0].device).type_as(grad_output[0])

        if isinstance(grad_input, (list, tuple)):
            # fill with None except for the first relevant slot
            outs = [None] * len(grad_input)
            outs[0] = init
            return tuple(outs)
        else:
            return (init,)  # single tensor input

    def copy(self):
        return InitRelevanceRule(self.init_grad)


def truncate_colormap(cmapIn="jet", minval=0.0, maxval=1.0, n=100):
    """truncate_colormap(cmapIn='jet', minval=0.0, maxval=1.0, n=100)"""
    cmapIn = plt.get_cmap(cmapIn)

    new_cmap = colors.LinearSegmentedColormap.from_list(
        "trunc({n},{a:.2f},{b:.2f})".format(n=cmapIn.name, a=minval, b=maxval),
        cmapIn(np.linspace(minval, maxval, n)),
    )
    return new_cmap


# from https://stackoverflow.com/questions/47163796/using-colormap-with-annotate-arrow-in-matplotlib
def colourgradarrow(ax, start, end, cmap="viridis", n=50, lw=3):
    # cmap = plt.get_cmap(cmap, n)
    cmap = truncate_colormap(cmap, 0.0, 0.7, n)
    # Arrow shaft: LineCollection
    x = np.linspace(start[0], end[0], n)
    y = np.linspace(start[1], end[1], n)
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(segments, cmap=cmap, linewidth=lw)
    lc.set_array(np.linspace(0, 1, n))
    ax.add_collection(lc)
    # Arrow head: Triangle
    tricoords = [(0, -0.9), (0.9, 0), (0, 0.9), (0, -0.9)]
    angle = np.arctan2(end[1] - start[1], end[0] - start[0])
    rot = matplotlib.transforms.Affine2D().rotate(angle)
    tricoords2 = rot.transform(tricoords)
    tri = matplotlib.path.Path(tricoords2, closed=True)
    ax.scatter(end[0], end[1], c=1, s=(2 * lw) ** 2, marker=tri, cmap=cmap, vmin=0)
    ax.autoscale_view()


def display_img(ax, input, inverse_transform):
    img = torch.clip(
        inverse_transform(input.clone().detach()), min=0.0, max=1.0
    ).squeeze()
    channels = img.shape[0]
    if channels == 3:
        img.permute(1, 2, 0)
        ax.imshow(img)
    else:
        ax.imshow(img, cmap="gray_r")
