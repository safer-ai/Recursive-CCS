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
from utils_generation.state_load_utils import getActsLabel

#%%
# model_name = "gpt-neo-2.7B"
model_name = "unifiedqa-t5-11b"
dataset_list = ["ag-news", "dbpedia-14", "tweet-eval-emotion", "tweet-eval-sentiment", "amazon-polarity"]
# dataset_list = ["amazon-polarity"]
num_examples = 4000
layer = None  # None for unifiedqa

raw=True
css_path = "uqa_n_all_30_w001_i0__"
# css_path = "uqa_n_all_30_w003_"
informative_strength = 1

css_no_train_path = "notrain_" + css_path
for p in ["orig_", "w01_", "w03_" "w1_", "w001_", "w003_", "w0001_", "w0_"]:
    css_no_train_path = css_no_train_path.replace(p, "")
for p in ["i10__", "i1__", "i0.1__", "i0.01__", "i0__"]:
    if p in css_no_train_path:
        informative_strength = float(p[1:-2])
        css_no_train_path = css_no_train_path.replace(p, "")
layer_suffix = f"/layer{layer}" if layer is not None else ""

assert Path(f"ccs_dirs/{css_path}0{layer_suffix}/ccs0.pt").exists()
assert Path(f"ccs_dirs/{css_no_train_path}0{layer_suffix}/ccs0.pt").exists(), css_no_train_path

nlabels = "_n_" in css_path
layer_ = layer if layer is not None else -1
train_ds = getActsLabel(model_name, dataset_list, split="train", data_num=num_examples, layer=layer_,nlabels=nlabels)
test_ds = getActsLabel(model_name, dataset_list, split="test", data_num=num_examples, layer=layer_,nlabels=nlabels)
# %%
device = "cuda"
ccs_perfs = ([], [])
ccs_consistency_loss = ([], [])
ccs_informative_loss = ([], [])
ccs_loss = ([], [])
d = train_ds[0][0].shape[-1]
constraints = torch.empty((0, d)).to(device)
nb_dirs = 30

constant_guess_loss = 0
along = torch.randn((1, d)).to(device)
ccs = CCS(train_ds, along=along, device=device, lbfgs=True, weight_decay=0, ntries=1,informative_strength=informative_strength)
loss, test_loss, test_acc = ccs.repeated_train(test_ds)


rdm_accs = []
for i in list(map(str, range(100))) + [""]:
    for j in range(100):
        path = Path(f"ccs_dirs/{css_no_train_path}{i}{layer_suffix}/ccs{j}.pt")
        if path.exists():
            ccs = CCS(train_ds, constraints=constraints, device=device)
            ccs.load(path)
            acc = ccs.get_acc(test_ds, raw=raw)
            rdm_accs.append(acc)
rand_min, rand_max = np.mean(rdm_accs) - 2 * np.std(rdm_accs), np.mean(rdm_accs) + 2 * np.std(rdm_accs)

for k, use_train in enumerate([False, True]):
    for i in list(map(str, range(100))) + [""]:
        path = Path(f"ccs_dirs/{css_path}{i}{layer_suffix}/ccs{nb_dirs-1}.pt")
        if path.exists():
            ccs_perfs[k].append([])
            ccs_consistency_loss[k].append([])
            ccs_informative_loss[k].append([])
            ccs_loss[k].append([])
            for j in range(nb_dirs):
                path = Path(f"ccs_dirs/{css_path}{i}{layer_suffix}/ccs{j}.pt")
                ccs = CCS(train_ds, constraints=constraints, device=device,informative_strength=informative_strength)
                ccs.load(path)
                if use_train:
                    perf = ccs.get_acc(train_ds, raw=raw)
                    c, I, l = ccs.eval([x for x,y in train_ds])
                else:
                    perf = ccs.get_acc(test_ds, raw=raw)
                    c, I, l = ccs.eval([x for x,y in test_ds])

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

plt.title(f"Accuracy on {css_path[:-1]}")
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
plt.axhline(test_loss, color="black", label="blind loss")
plt.axhline(0, color="black", label="perfect loss", linestyle="dashed")
plt.legend()
plt.ylabel(f"Loss")
plt.title(f"Loss on {css_path[:-1]}")
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
plt.ylabel(f"Loss")
plt.title(f"Loss on {css_path[:-1]}")
plt.xlabel("Iteration")
# %%
use_train = False
studied_ds = 0
# subplots
gaps = 3
dir_nbs = np.linspace(0, nb_dirs - 1, gaps, dtype=int)
fig, axs = plt.subplots(gaps, 2, figsize=(6, 6), sharex=True, sharey=True)
fig.tight_layout()
for i, use_train in enumerate([False, True]):
    for j, dir_nb in enumerate(dir_nbs):
        ax = axs[j, i]
        path = Path(f"ccs_dirs/{css_path}0{layer_suffix}/ccs{dir_nb}.pt")
        
        x,y = (train_ds if use_train else test_ds)[studied_ds]
        ccs = CCS([(x,y)], constraints=constraints, device=device)
        random_subset = np.random.choice(len(x), 500)
        x, y = x[random_subset], y[random_subset]
        ccs.load(path)
        with torch.no_grad():
            activations = ccs.best_probe(ccs.prepare([x])[0])[:, :, 0]
            good_activations = []
            for k in range(activations.shape[0]):
                for l in range(activations.shape[1]):
                    if l == y[k]:
                        good_activations.append(activations[k, l])
            good_activations = torch.stack(good_activations)
            bad_activations = []
            for k in range(activations.shape[0]):
                for l in range(activations.shape[1]):
                    if l != y[k]:
                        bad_activations.append(activations[k, l])
            bad_activations = torch.stack(bad_activations)
        ax.hist(good_activations.cpu().numpy(), bins=50, range=(0, 1), alpha=0.5, label="True")
        ax.hist(bad_activations.cpu().numpy(), bins=50, range=(0, 1), alpha=0.5, label="False")
for i, use_train in enumerate([False, True]):
    name = "train" if use_train else "test"
    axs[-1, i].set_xlabel(f"{name} activation")
for j, dir_nb in enumerate(dir_nbs):
    axs[j, 0].set_ylabel(f"dir {dir_nb}")

plt.suptitle(f"Activation distribution on {css_path[:-1]}, {dataset_list[studied_ds]}")
plt.legend()

# %%
