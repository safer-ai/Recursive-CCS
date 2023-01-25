#%%
import matplotlib.pyplot as plt
import numpy as np
import json
import torch
from utils import get_parser, load_all_generations, CCS, assert_orthonormal, LinearWithConstraints
from pathlib import Path
from utils_generation.state_load_utils import getNegPosLabel
from tqdm import trange, tqdm

#%%
# model_name = "gpt-neo-2.7B"
model_name = "unifiedqa-t5-11b"
# dataset_list = ["imdb","amazon-polarity","copa","ag-news","dbpedia-14","rte","boolq","qnli","piqa"]
dataset_list = ["copa"]
num_examples = 1000
layer = None # None for unifiedqa

css_path = "uqa_copa_30_w01_"
    
layer_suffix = f"/layer{layer}" if layer is not None else ""

assert Path(f"ccs_dirs/{css_path}0{layer_suffix}/ccs0.pt").exists()

layer_ = layer if layer is not None else -1
neg_hs_train, pos_hs_train, y_train = getNegPosLabel(model_name, dataset_list, split="train", data_num=num_examples, layer=layer_)
neg_hs_test, pos_hs_test, y_test = getNegPosLabel(model_name, dataset_list, split="test", data_num=num_examples, layer=layer_)
# %%
device = "cuda"
d = neg_hs_train.shape[1]
constraints = torch.empty((0, d)).to(device)
nb_dirs = 30

def get_dir(ccs_path):
    ccs = CCS(neg_hs_train, pos_hs_train, constraints=constraints, device=device)
    ccs.load(Path(ccs_path))
    raw_acc = ccs.get_acc(neg_hs_test, pos_hs_test, y_test, raw=True)
    return ccs.get_direction() if raw_acc > 0.5 else -ccs.get_direction()

dirs = [get_dir(f"ccs_dirs/{css_path}0{layer_suffix}/ccs{i}.pt") for i in range(nb_dirs)]

ccs_mix = CCS(neg_hs_train, pos_hs_train, along=dirs[0], device=device, lbfgs=True, ntries=1, weight_decay=0)
loss, test_loss, test_acc = ccs_mix.repeated_train(neg_hs_test, pos_hs_test, y_test, verbose=False)
print(f"Loss: {loss: 4f}, Test loss: {test_loss:.4f}, Test acc: {test_acc:.4f}")
# %%

ccs_mix = CCS(neg_hs_train, pos_hs_train, along=dirs[10], device=device, lbfgs=True, ntries=1, weight_decay=0)
loss, test_loss, test_acc = ccs_mix.repeated_train(neg_hs_test, pos_hs_test, y_test, verbose=False)
print(f"Loss: {loss: 4f}, Test loss: {test_loss:.4f}, Test acc: {test_acc:.4f}")
#%%
ccs_mix = CCS(neg_hs_train, pos_hs_train, along=dirs[0] + dirs[10], device=device, lbfgs=True, ntries=1, weight_decay=0)
loss, test_loss, test_acc = ccs_mix.repeated_train(neg_hs_test, pos_hs_test, y_test, verbose=False)
print(f"Loss: {loss: 4f}, Test loss: {test_loss:.4f}, Test acc: {test_acc:.4f}")
# %%
# best_so_far = 0
# for a in range(3,11):
#     for b in range(a+1,11):
#         ccs_mix = CCS(neg_hs_train, pos_hs_train, along=2 * dirs[b] + dirs[a], device=device, lbfgs=True, ntries=1, weight_decay=0)
#         lossx, _, _ = ccs_mix.repeated_train(neg_hs_test, pos_hs_test, y_test, verbose=False)
#         ccs_mix = CCS(neg_hs_train, pos_hs_train, along=dirs[0] + 2 * dirs[a], device=device, lbfgs=True, ntries=1, weight_decay=0)
#         lossy, _, _ = ccs_mix.repeated_train(neg_hs_test, pos_hs_test, y_test, verbose=False)
#         ccs_mix = CCS(neg_hs_train, pos_hs_train, along=dirs[0] + 2 * dirs[b], device=device, lbfgs=True, ntries=1, weight_decay=0)
#         lossz, _, _ = ccs_mix.repeated_train(neg_hs_test, pos_hs_test, y_test, verbose=False)
#         if min(lossx, lossy, lossz) > best_so_far:
#             best_so_far = min(lossx, lossy, lossz)
#             print(a,b, best_so_far)
#             print()
#%%
# %%
# increase plt size
from matplotlib import rcParams
rcParams['figure.figsize'] = 10, 10

start = 0
colors = plt.cm.get_cmap('viridis', 15)
for end in trange(15):
    losses = []
    for alpha in np.linspace(0, 1, 10):
        dir_mix = (1 - alpha) * dirs[start] + alpha * dirs[end]
        loss, _, _ = CCS(neg_hs_train, pos_hs_train, along=dir_mix, device=device, lbfgs=True, ntries=1, weight_decay=0).repeated_train(neg_hs_test, pos_hs_test, y_test, verbose=False)
        losses.append(loss)
    plt.plot(np.linspace(0, 1, 10), losses, label=f"{start} -> {end}", color=colors(end))
plt.title(f"Losses when you mix directions {start} with another one")
plt.xlabel("alpha")
plt.ylabel("loss")
plt.legend()
# %%
a = 0
b = 3
c = 10
intervals = 10
losses = np.zeros((intervals, intervals))
for i, alpha in tqdm(enumerate(np.linspace(0, 1, intervals)), total=intervals):
    for j, beta in enumerate(np.linspace(0, 1, intervals)):
        dir_mix = ((1 - alpha) * dirs[a] + alpha * dirs[b]) * (1 - beta) + beta * dirs[c]
        loss, _, _ = CCS(neg_hs_train, pos_hs_train, along=dir_mix, device=device, lbfgs=True, ntries=1, weight_decay=0).repeated_train(neg_hs_test, pos_hs_test, y_test, verbose=False)
        losses[i, j] = loss
#%%
plt.imshow(losses, origin="lower", extent=(0, 1, 0, 1))
plt.colorbar()
plt.title(f"Losses when you mix directions {a}, {b} and {c}")
plt.xlabel(f"how much {c} is mixed")
plt.ylabel(f"how much {a} vs {b} are mixed")

# %%
a = 0
b = 3
c = 10
intervals = 20
losses = np.zeros((intervals, intervals))
for i, thetha in tqdm(enumerate(np.linspace(0, np.pi, intervals)), total=intervals):
    for j, phi in enumerate(np.linspace(0, 2 * np.pi, intervals)):
        dir_mix = np.cos(thetha) * dirs[a] + np.sin(thetha) * np.cos(phi) * dirs[b] + np.sin(thetha) * np.sin(phi) * dirs[c]
        loss, _, _ = CCS(neg_hs_train, pos_hs_train, along=dir_mix, device=device, lbfgs=True, ntries=1, weight_decay=0).repeated_train(neg_hs_test, pos_hs_test, y_test, verbose=False)
        losses[i, j] = loss
#%%
rcParams['figure.figsize'] = 10,6
plt.imshow(losses, origin="lower", extent=[0, 2 * np.pi, 0, np.pi], aspect="auto")
plt.colorbar()
plt.title(f"Losses when you mix directions {a}, {b} and {c}")
plt.xlabel(f"how much {b} vs {c} are mixed (phi)")
plt.ylabel(f"how much {a} is mixed (thetha)")
# %%
