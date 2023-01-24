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

#%%
# model_name = "gpt-neo-2.7B"
model_name = "unifiedqa-t5-11b"
dataset_list = ["imdb"]
num_examples = 1000
layer = None # None for unifiedqa

css_path = "uqa_imdb_30_w0001_"

css_no_train_path = "notrain_" + css_path
for p in ["orig_", "w01_", "w03_" "w1_", "w001_", "w003_", "w0001_"]:
    css_no_train_path = css_no_train_path.replace(p, "")
layer_suffix = f"/layer{layer}" if layer is not None else ""

layer_ = layer if layer is not None else -1
neg_hs_train, pos_hs_train, y_train = getNegPosLabel(model_name, dataset_list, split="train", data_num=num_examples, layer=layer_)
neg_hs_test, pos_hs_test, y_test = getNegPosLabel(model_name, dataset_list, split="test", data_num=num_examples, layer=layer_)
# %%
device = "cuda"
ccs_perfs = ([], [])
ccs_consistency_loss = ([], [])
ccs_informative_loss = ([], [])
ccs_loss = ([], [])
d = neg_hs_train.shape[1]
constraints = torch.empty((0, d)).to(device)
nb_dirs = 30


rdm_accs = []
for i in list(map(str,range(100))) + [""]:
    for j in range(100):
        path = Path(f"ccs_dirs/{css_no_train_path}{i}{layer_suffix}/ccs{j}.pt")
        if path.exists():
            ccs = CCS(neg_hs_train[0:1], pos_hs_train[0:1], constraints=constraints, device=device)
            ccs.load(path)
            acc = ccs.get_acc(neg_hs_test, pos_hs_test, y_test)
            rdm_accs.append(acc)
rand_min, rand_max = np.mean(rdm_accs) - 2*np.std(rdm_accs), np.mean(rdm_accs) + 2*np.std(rdm_accs)

for k,use_train in enumerate([False, True]):
    for i in list(map(str,range(100))) + [""]:
        path = Path(f"ccs_dirs/{css_path}{i}{layer_suffix}/ccs{nb_dirs-1}.pt")
        if path.exists():
            ccs_perfs[k].append([])
            ccs_consistency_loss[k].append([])
            ccs_informative_loss[k].append([])
            ccs_loss[k].append([])
            for j in range(nb_dirs):
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
# %%
for perfs in ccs_perfs[0]:
    plt.plot(perfs, color="green", alpha=0.2)
avg_perfs = np.mean(ccs_perfs[0], axis=0)
plt.plot(avg_perfs, color="green", label="test mean", marker="o")

for perfs in ccs_perfs[1]:
    plt.plot(perfs, color="blue", alpha=0.2)
avg_perfs = np.mean(ccs_perfs[1], axis=0)
plt.plot(avg_perfs, color="blue", label="train mean", marker="o")

plt.title(f"Accuracy on {model_name}{layer_suffix} - {dataset_list}")
plt.axhspan(rand_min, rand_max, label="chance +/- 2std", color="gray", alpha=0.2)
plt.ylabel("Accuracy")
plt.xlabel("Iteration")
plt.legend()
# %%
for perfs in ccs_loss[0]:
    plt.plot(perfs, color="green", alpha=0.2)
avg_perfs = np.mean(ccs_loss[0], axis=0)
plt.plot(avg_perfs, color="green", label="test mean", marker="o")

for perfs in ccs_loss[1]:
    plt.plot(perfs, color="blue", alpha=0.2)
avg_perfs = np.mean(ccs_loss[1], axis=0)
plt.plot(avg_perfs, color="blue", label="train mean", marker="o")
plt.legend()
plt.ylabel("Loss")
plt.xlabel("Iteration")
# %%
avg_perfs = np.mean(ccs_consistency_loss[0], axis=0)
for perfs in ccs_consistency_loss[0]:
    plt.plot(perfs, color="green", alpha=0.2)
plt.plot(avg_perfs, color="green", label="consistency loss test", marker="x", linestyle="dashed")
avg_perfs = np.mean(ccs_informative_loss[0], axis=0)
for perfs in ccs_informative_loss[0]:
    plt.plot(perfs, color="green", alpha=0.2)
plt.plot(avg_perfs, color="green", label="informative loss test", marker="o")

avg_perfs = np.mean(ccs_consistency_loss[1], axis=0)
for perfs in ccs_consistency_loss[1]:
    plt.plot(perfs, color="blue", alpha=0.2)
plt.plot(avg_perfs, color="blue", label="consistency loss train", marker="x", linestyle="dashed")
avg_perfs = np.mean(ccs_informative_loss[1], axis=0)
for perfs in ccs_informative_loss[1]:
    plt.plot(perfs, color="blue", alpha=0.2)
plt.plot(avg_perfs, color="blue", label="informative loss train", marker="o")

plt.legend()
plt.ylabel("Loss")
plt.xlabel("Iteration")
# %%
use_train = True
# subplots
gaps = 4
dir_nbs = np.linspace(0, nb_dirs - 1, gaps, dtype=int)
fig, axs = plt.subplots(gaps, 2, figsize=(4, 6), sharex=True, sharey=True)
fig.tight_layout()
for i, use_train in enumerate([False, True]):
    for j, dir_nb in enumerate(dir_nbs):
        ax = axs[j,i]
        path = Path(f"ccs_dirs/{css_path}0{layer_suffix}/ccs{dir_nb}.pt")
        ccs = CCS(neg_hs_train, pos_hs_train, constraints=constraints, device=device)
        ccs.load(path)
        neg, pos, y =( *ccs.prepare(neg_hs_train, pos_hs_train), y_train) if use_train else (*ccs.prepare(neg_hs_test, pos_hs_test), y_test)
        with torch.no_grad():
            neg_activations = ccs.best_probe(neg)
            pos_activations = ccs.best_probe(pos)
        good_activation = (0.5 * (pos_activations + (1 - neg_activations)))[y == 1]
        bad_activation = (0.5 * (pos_activations + (1 - neg_activations)))[y == 0]
        ax.hist(good_activation.cpu().numpy(), bins=50, range=(0,1), alpha=0.5, label="True")
        ax.hist(bad_activation.cpu().numpy(), bins=50, range=(0,1), alpha=0.5, label="False")
for i, use_train in enumerate([False, True]):
    name = "train" if use_train else "test"
    axs[-1, i].set_xlabel(f"{name} activation")
for j, dir_nb in enumerate(dir_nbs):
    axs[j, 0].set_ylabel(f"dir {dir_nb}")

plt.suptitle(f"Activation distribution on {model_name}{layer_suffix} - {dataset_list}")
plt.legend()
#%%
# Single test
path = Path(f"ccs_dirs/{css_path}0{layer_suffix}/ccs0.pt")
ccs = CCS(neg_hs_train, pos_hs_train, constraints=constraints, device=device)
ccs.load(path)
neg, pos = ccs.prepare(neg_hs_train, pos_hs_train)
with torch.no_grad():
    neg_activations = ccs.best_probe(neg)
    pos_activations = ccs.best_probe(pos)
good_activation = (0.5 * (pos_activations + (1 - neg_activations)))[y_train == 1]
bad_activation = (0.5 * (pos_activations + (1 - neg_activations)))[y_train == 0]
plt.hist(good_activation.cpu().numpy(), bins=50, range=(0,1), alpha=0.5, label="True")
plt.hist(bad_activation.cpu().numpy(), bins=50, range=(0,1), alpha=0.5, label="False")
plt.title(f"Activation distribution on {model_name} - {dataset_list}")
plt.legend()

# %%
