#%%
import matplotlib.pyplot as plt
import numpy as np
import json
import torch
from utils import get_parser, load_all_generations, CCS, assert_orthonormal
from pathlib import Path
#%%
data_name = "__model_name_deberta__parallelize_False__dataset_name_imdb__split_test__prompt_idx_0__batch_size_40__num_examples_400__use_decoder_False__layer_-1__all_layers_False__token_idx_-1.npy"
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
d = neg_hs.shape[1]
constraints = torch.empty((0, d)).to(device)
for i in range(10):
    path = Path(f"css_dirs/rcss_20_dirs_{i}/ccs19.pt")
    if path.exists():
        ccs_perfs.append([])
        for j in range(20):
            path = Path(f"css_dirs/rcss_20_dirs_{i}/ccs{j}.pt")
            ccs = CCS(neg_hs_train, pos_hs, constraints=constraints, device=device)
            ccs.load(path)
            perf = ccs.get_acc(neg_hs_test, pos_hs_test, y_test)
            ccs_perfs[-1].append(perf)
# %%
avg_perfs = np.mean(ccs_perfs, axis=0)
for perfs in ccs_perfs:
    plt.plot(perfs, color="blue", alpha=0.2)
plt.plot(avg_perfs, color="blue", label="mean")
plt.legend()
plt.ylabel("Accuracy")
plt.xlabel("Iteration")
# %%
