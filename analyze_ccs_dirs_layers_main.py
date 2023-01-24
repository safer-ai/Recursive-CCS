#%%
# %load_ext autoreload
# %autoreload 2
#%%
import matplotlib.pyplot as plt
import numpy as np
import json
import torch
from utils import get_parser, load_all_generations, CCS, assert_orthonormal, LinearWithConstraints
from pathlib import Path
from utils_generation.state_load_utils import getNegPosLabel
from tqdm import tqdm
# large plots
from matplotlib import rcParams
rcParams['figure.figsize'] = 10, 8

#%%
# model_name = "gpt-neo-2.7B"
model_name = "gpt-neo-2.7B"
# model_name = "gpt-j-6B"
css_path = "neo27_imdb_30_"
dataset_list = ["imdb"]
num_examples = 1000
nb_dirs = 30
dirs_displayed = [0, 1, 2, 10, 20, 29]
layers = [l for l in range(1,100) if Path(f"ccs_dirs/{css_path}0/layer{l}/ccs0.pt").exists()]
# css_no_train_path = "notrain_" + css_path
css_no_train_path = "notrain_" + css_path.replace("_w01", "")
#%%
datasets = {}
for layer in layers:
    neg_hs_train, pos_hs_train, y_train = getNegPosLabel(model_name, dataset_list, split="train", data_num=num_examples, layer=layer)
    neg_hs_test, pos_hs_test, y_test = getNegPosLabel(model_name, dataset_list, split="test", data_num=num_examples, layer=layer)
    datasets[layer] = (neg_hs_train, pos_hs_train, y_train, neg_hs_test, pos_hs_test, y_test)
# %%
device = "cuda"
# layer, train/test, seed, dir
ccs_perfss = {}
ccs_consistency_losss = {}
ccs_informative_losss = {}
ccs_losss = {}
d = neg_hs_train.shape[1]
constraints = torch.empty((0, d)).to(device)
rand_mins, rand_maxs = {}, {}

# css_no_train_path = None
for layer in tqdm(layers):
    layer_suffix = f"/layer{layer}"
    
    neg_hs_train, pos_hs_train, y_train, neg_hs_test, pos_hs_test, y_test = datasets[layer]
    
    ccs_perfs = ([], [])
    ccs_consistency_loss = ([], [])
    ccs_informative_loss = ([], [])
    ccs_loss = ([], [])
    
    rdm_accs = []
    for i in range(100):
        for j in dirs_displayed:
            path = Path(f"ccs_dirs/{css_no_train_path}{i}{layer_suffix}/ccs{j}.pt")
            if path.exists():
                ccs = CCS(neg_hs_train[0:1], pos_hs_train[0:1], constraints=constraints, device=device)
                ccs.load(path)
                acc = ccs.get_acc(neg_hs_test, pos_hs_test, y_test)
                rdm_accs.append(acc)
    rand_min, rand_max = np.mean(rdm_accs) - 2*np.std(rdm_accs), np.mean(rdm_accs) + 2*np.std(rdm_accs)
    rand_mins[layer], rand_maxs[layer] = rand_min, rand_max

    for k,use_train in enumerate([False, True]):
        for i in range(10):
            path = Path(f"ccs_dirs/{css_path}{i}{layer_suffix}/ccs{dirs_displayed[-1]}.pt")
            if path.exists():
                ccs_perfs[k].append([])
                ccs_consistency_loss[k].append([])
                ccs_informative_loss[k].append([])
                ccs_loss[k].append([])
                for j in dirs_displayed:
                    path = Path(f"ccs_dirs/{css_path}{i}{layer_suffix}/ccs{j}.pt")
                    ccs = CCS(neg_hs_train[0:1], pos_hs_train[0:1], constraints=constraints, device=device)
                    ccs.load(path)
                    if use_train:
                        perf = ccs.get_acc(neg_hs_train, pos_hs_train, y_train)
                        c, I, l = ccs.eval(neg_hs_train, pos_hs_train)
                    else:
                        perf = ccs.get_acc(neg_hs_test, pos_hs_test, y_test)
                        c, I, l = ccs.eval(neg_hs_test, pos_hs_test)

                    ccs_perfs[k][-1].append(perf)
                    ccs_consistency_loss[k][-1].append(c)
                    ccs_informative_loss[k][-1].append(I)
                    ccs_loss[k][-1].append(l)
    ccs_perfss[layer] = ccs_perfs
    ccs_consistency_losss[layer] = ccs_consistency_loss
    ccs_informative_losss[layer] = ccs_informative_loss
    ccs_losss[layer] = ccs_loss
# %%
plt.fill_between(layers, list(rand_mins.values()), list(rand_maxs.values()), color="gray", alpha=0.2, label="random +2/-2 std")
colors = plt.cm.viridis(np.linspace(0, 1, len(dirs_displayed)))
for j, dir in enumerate(dirs_displayed):
    # layer, seed
    test_accs_per_layer = np.array([ccs_perfss[l][0] for l in layers])[..., j]
    train_accs_per_layer = np.array([ccs_perfss[l][1] for l in layers])[..., j]
    # layer
    test_accs_means = np.mean(test_accs_per_layer, axis=1)
    train_accs_means = np.mean(train_accs_per_layer, axis=1)
    
    for i in range(test_accs_per_layer.shape[1]):
        plt.plot(layers, test_accs_per_layer[:, i], color=colors[j], alpha=0.2)
        plt.plot(layers, train_accs_per_layer[:, i], color=colors[j], alpha=0.2, linestyle="dotted")
    plt.plot(layers, test_accs_means, color=colors[j], label=f"test it {dir}")
    plt.plot(layers, train_accs_means, color=colors[j], label=f"train it {dir}", linestyle="dotted")

plt.title(f"Accuracy on {model_name} - {dataset_list}")
plt.ylabel("Accuracy")
plt.xlabel("Layer")
plt.xticks(layers, layers)
plt.legend()
# %%
for j, dir in enumerate(dirs_displayed):
    # layer, seed
    test_loss_per_layer = np.array([ccs_losss[l][0] for l in layers])[..., j]
    train_loss_per_layer = np.array([ccs_losss[l][1] for l in layers])[..., j]
    # layer
    test_loss_means = np.mean(test_loss_per_layer, axis=1)
    train_loss_means = np.mean(train_loss_per_layer, axis=1)
    
    for i in range(test_loss_per_layer.shape[1]):
        plt.plot(layers, test_loss_per_layer[:, i], color=colors[j], alpha=0.2)
        plt.plot(layers, train_loss_per_layer[:, i], color=colors[j], alpha=0.2, linestyle="dotted")
    plt.plot(layers, test_loss_means, color=colors[j], label=f"test it {dir}")
    plt.plot(layers, train_loss_means, color=colors[j], label=f"train it {dir}", linestyle="dotted")

plt.title(f"Loss on {model_name} - {dataset_list}")
plt.ylabel("Loss")
plt.xlabel("Layer")
plt.xticks(layers, layers)
plt.legend()
# %%
