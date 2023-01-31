#%%
import torch
import numpy as np
from matplotlib import pyplot as plt # type: ignore

# %%
x0_means = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
x1_means = torch.tensor([[-1.0, 0.0], [-1.0, -1.0]])
means_nb = x0_means.shape[0]
assert means_nb == x1_means.shape[0]
# add random points around those means
n = 10
std = 0.1
D = 2
bias = 0
x0 = (torch.randn(n, means_nb, D) * std + x0_means).reshape(-1, D)
x1 = (torch.randn(n, means_nb, D) * std + x1_means).reshape(-1, D)

intervals = 100
amplitude = 5
linespace = torch.linspace(-amplitude, amplitude, intervals)

probe_dirs = torch.empty(intervals, intervals, D)
probe_dirs[:, :, 0] = linespace.repeat(intervals, 1)
probe_dirs[:, :, 1] = linespace.repeat(intervals, 1).T
biases = torch.zeros(probe_dirs.shape[:-1]) + 1


def probe(x, probe_dirs, biases):
    z = torch.einsum("ijh,nh->ijn", probe_dirs, x) + biases[..., None]
    return torch.sigmoid(z)


p0s = probe(x0, probe_dirs, biases)
p1s = probe(x1, probe_dirs, biases)


def loss(p0, p1):
    inf, cons = (torch.min(p0, p1) ** 2).mean(-1), ((p0 - (1 - p1)) ** 2).mean(-1)
    return inf, cons, inf + cons


inf, cons, losses = loss(p0s, p1s)

fig, axs = plt.subplots(1, 3, figsize=(15, 5))

for ax, l, title in zip(axs, [inf, cons, losses], ["informative", "consistent", "total"]):

    m = ax.imshow(l, origin="lower", extent=(-amplitude, amplitude, -amplitude, amplitude), vmin=0, vmax=0.5)

    for i in range(x0.shape[0]):
        ax.plot([x0[i, 0], x1[i, 0]], [x0[i, 1], x1[i, 1]], "orange")
    ax.set_title(title)
    ax.plot([0], [0], "o", color="red")
fig.colorbar(m, ax=ax)
# %%
