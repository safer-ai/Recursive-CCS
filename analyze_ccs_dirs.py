#%%
import matplotlib.pyplot as plt
import numpy as np
import json
import torch
from utils import get_parser, load_all_generations, CCS, assert_orthonormal, LinearWithConstraints
from pathlib import Path
#%%
data_name = "__model_name_deberta__parallelize_False__dataset_name_imdb__split_test__prompt_idx_0__batch_size_40__num_examples_2000__use_decoder_False__layer_-1__all_layers_False__token_idx_-1.npy"
# data_name = "__model_name_deberta__parallelize_False__dataset_name_imdb__split_test__prompt_idx_0__batch_size_40__num_examples_400__use_decoder_False__layer_-1__all_layers_False__token_idx_-1.npy"
folder = "generated_hidden_states"
neg_hs, pos_hs, y = np.load(f"./{folder}/negative_hidden_states{data_name}"), np.load(f"{folder}/positive_hidden_states{data_name}"), np.load(f"{folder}/labels{data_name}")
# Make sure the shape is correct
assert neg_hs.shape == pos_hs.shape
neg_hs, pos_hs = neg_hs[..., -1], pos_hs[..., -1]  # take the last layer
if neg_hs.shape[1] == 1:  # T5 may have an extra dimension; if so, get rid of it
    neg_hs = neg_hs.squeeze(1)
    pos_hs = pos_hs.squeeze(1)

# Very simple train/test split (using the fact that the data is already shuffled)
neg_hs_train, neg_hs_test = neg_hs[:len(neg_hs) // 2], neg_hs[len(neg_hs) // 2:]
pos_hs_train, pos_hs_test = pos_hs[:len(pos_hs) // 2], pos_hs[len(pos_hs) // 2:]
y_train, y_test = y[:len(y) // 2], y[len(y) // 2:]
# %%
device = "cuda"
ccs_perfs = []
ccs_consistency_loss = []
ccs_informative_loss = []
ccs_loss = []
d = neg_hs.shape[1]
constraints = torch.empty((0, d)).to(device)
nb_dirs = 10
use_train = True
for i in range(10):
    path = Path(f"ccs_dirs/rccs_10_xl_{i}/ccs{nb_dirs-1}.pt")
    if path.exists():
        ccs_perfs.append([])
        ccs_consistency_loss.append([])
        ccs_informative_loss.append([])
        ccs_loss.append([])
        for j in range(nb_dirs):
            path = Path(f"ccs_dirs/rccs_10_xl_{i}/ccs{j}.pt")
            ccs = CCS(neg_hs_train, pos_hs, constraints=constraints, device=device)
            ccs.load(path)
            if use_train:
                perf = ccs.get_acc(neg_hs_train, pos_hs_train, y_train)
                c, I, l = ccs.eval(neg_hs_train, pos_hs_train)
            else:
                perf = ccs.get_acc(neg_hs_test, pos_hs_test, y_test)
                c, I, l = ccs.eval(neg_hs_test, pos_hs_test)
                
            ccs_perfs[-1].append(perf)
            ccs_consistency_loss[-1].append(c)
            ccs_informative_loss[-1].append(I)
            ccs_loss[-1].append(l)
# %%
avg_perfs = np.mean(ccs_perfs, axis=0)
for perfs in ccs_perfs:
    plt.plot(perfs, color="blue", alpha=0.2)
plt.plot(avg_perfs, color="blue", label="mean", marker="o")
plt.legend()
plt.ylabel("Accuracy")
plt.xlabel("Iteration")
# %%
avg_perfs = np.mean(ccs_loss, axis=0)
for perfs in ccs_loss:
    plt.plot(perfs, color="red", alpha=0.2)
plt.plot(avg_perfs, color="red", label="mean")
plt.legend()
plt.ylabel("Loss")
plt.xlabel("Iteration")
# %%
avg_perfs = np.mean(ccs_consistency_loss, axis=0)
for perfs in ccs_consistency_loss:
    plt.plot(perfs, color="orange", alpha=0.2)
plt.plot(avg_perfs, color="orange", label="consistency loss")
avg_perfs = np.mean(ccs_informative_loss, axis=0)
for perfs in ccs_informative_loss:
    plt.plot(perfs, color="darkviolet", alpha=0.2)
plt.plot(avg_perfs, color="darkviolet", label="informative loss")
plt.legend()
plt.ylabel("Loss")
plt.xlabel("Iteration")
# %%
path = Path(f"ccs_dirs/rccs_10_xl_0/ccs9.pt")
ccs = CCS(neg_hs_train, pos_hs_train, constraints=constraints, device=device)
ccs.load(path)
neg, pos = ccs.prepare(neg_hs_train, pos_hs_train)
neg_activations = ccs.best_probe(neg)
pos_activations = ccs.best_probe(pos)
m = torch.minimum(neg_activations, pos_activations)
M = torch.maximum(neg_activations, pos_activations)
plt.hist(m.data.cpu().numpy(), bins=100, range=(-1,2), alpha=0.5, label="min")
plt.hist(M.data.cpu().numpy(), bins=100, range=(-1,2), alpha=0.5, label="max")
plt.ylabel("Count")
plt.xlabel("Activation")
plt.legend()
#%%