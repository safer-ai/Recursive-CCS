import os
from pathlib import Path
import numpy as np
import pandas as pd
import json

######## JSON Load ########
json_dir = "./registration"

with open("{}.json".format(json_dir), "r") as f:
    global_dict = json.load(f)
registered_dataset_list = global_dict["dataset_list"]
registered_models = global_dict["registered_models"]
registered_prefix = global_dict["registered_prefix"]
models_layer_num = global_dict["models_layer_num"]


load_dir = "./generation_results"
"""------ Func: set_load_dir ------
## Input = (path) ##
    path: set the dir that loads hidden states to path.
[ATTENTION]: Only load the first 4 prompts for speed.
"""


def set_load_dir(path):
    global load_dir
    load_dir = path


def getDirList(mdl, set_name, load_dir, data_num, confusion, place, prompt_idx, nlabels=False):
    length = len(mdl)
    filter = [
        w
        for w in os.listdir(load_dir)
        if (
            mdl == w[:length]
            and mdl + "_" in w
            and set_name + "_" in w
            and str(data_num) + "_" in w
            and confusion + "_" in w
            and place in w
            and (nlabels ^ ("nlabels" not in w))
        )
    ]
    if prompt_idx is not None:
        filter = [w for w in filter if int(w.split("_")[3][6:]) in prompt_idx]
    return [os.path.join(load_dir, w) for w in filter]


def getNumberedDirList(root: str, pattern: str, max_int: int = 100) -> list[Path]:
    existing_paths = []
    for i in range(max_int):
        path = Path(root) / (pattern.format(i))
        if path.exists():
            existing_paths.append(path)
    assert existing_paths, f"No files found in {root} matching {pattern}"
    return existing_paths


def loadHiddenStates(
    mdl,
    set_name,
    load_dir,
    promtpt_idx,
    location="encoder",
    layer=-1,
    data_num=1000,
    confusion="normal",
    place="last",
    verbose=True,
    nlabels=False,
):
    """
    Load generated hidden states, return a dict where key is the dataset name and values is a list. Each tuple in the list is the (x,y) pair of one prompt.
    if mode == minus, then get h - h'
    if mode == concat, then get np.concatenate([h,h'])
    elif mode == 0 or 1, then get h or h'
    """

    dir_list = getDirList(mdl, set_name, load_dir, data_num, confusion, place, promtpt_idx, nlabels=nlabels)

    has_all_layers = Path(dir_list[0] + "/0.npy").exists()
    if has_all_layers:
        hidden_states = [
            np.stack(
                [
                    np.load(d)[
                        :,
                        layer:,
                    ]
                    for d in getNumberedDirList(w, "{}.npy")
                ], axis=1
            )
            for w in dir_list
        ]
    else:
        append_list = ["_" + location + str(layer) for _ in dir_list]
        hidden_states = [
            np.stack([np.load(d) for d in getNumberedDirList(w, f"{{}}{app}.npy")], axis=1) for w, app in zip(dir_list, append_list)
        ]
    
    if not nlabels:
        for a in hidden_states:
            assert a.shape[1] == 2, "Only binary classification is supported for nlabels=False."

    # normalize
    if verbose:
        print("{} prompts for {}, with shape {}".format(len(hidden_states), set_name, hidden_states[0].shape))
    labels = [np.array(pd.read_csv(os.path.join(w, "frame.csv"))["label"].to_list()) for w in dir_list]

    return [(u, v) for u, v in zip(hidden_states, labels)]


def getPermutation(data_list, rate=0.6, seed=0):
    """Deterministic random permutation"""
    length = len(data_list[0][1])
    np.random.seed(seed)
    permutation = np.random.permutation(range(length)).reshape(-1)
    return [permutation[: int(length * rate)], permutation[int(length * rate) :]]


"""------ Func: getDic ------
## Input = (mdl_name, dataset_list, prefix = \"normal\", location=\"encoder\", layer=-1, scale = True, demean = True, mode = \"minus\", verbose = True) ##
    mdl_name: name of the model
    dataset_list: list of all datasets
    prefix: the prefix used for the hidden states
    location: Either 'encoder' or 'decoder'. Determine which hidden states to load.
    layer: An index representing which layer in `location` should we load the hidden state from.
    prompt_dict: dict of prompts to consider. Default is taking all prompts (empty dict). Key is the set name and value is an index list. Only return hiiden states from corresponding prompts.
    data_num: population of the dataset. Default is 1000, and it depends on generation process.
    scale: whether to rescale the whole dataset
    demean: whether to subtract the mean
    mode: how to generate hidden states from h and h'
    verbose: Whether to print more
## Output = [data_dict, permutation_dict] ##
    data_dict: a dict with key equals to set name, and value is a list. Each element in the list is a tuple (state, label). state has shape (#data * #dim), and label has shape (#data).
    permutation_dict: [train_idx, test_idx], where train_idx is the subset of [#data] that corresponds to the training set, and test_idx is the subset that corresponds to the test set.
"""


def getDic(
    mdl_name,
    dataset_list,
    prefix="normal",
    location="auto",
    layer=-1,
    prompt_dict=None,
    data_num=1000,
    verbose=True,
    nlabels=False,
):
    global load_dir
    if location == "auto":
        location = "decoder" if "gpt" in mdl_name else "encoder"
    if location == "decoder" and layer < 0:
        layer += models_layer_num[mdl_name]
    print(
        "start loading {} hidden states {} for {} with {} prefix. Prompt_dict: {}".format(
            location, layer, mdl_name, prefix, prompt_dict if prompt_dict is not None else "ALL"
        )
    )
    prompt_dict = prompt_dict if prompt_dict is not None else {key: None for key in dataset_list}
    data_dict = {
        set_name: loadHiddenStates(
            mdl_name,
            set_name,
            load_dir,
            prompt_dict[set_name],
            location,
            layer,
            data_num=data_num,
            confusion=prefix,
            verbose=verbose,
            nlabels=nlabels,
        )
        for set_name in dataset_list
    }
    permutation_dict = {set_name: getPermutation(data_dict[set_name]) for set_name in dataset_list}
    return data_dict, permutation_dict


def getConcat(data_list, axis=0):
    sub_list = [w for w in data_list if w is not None]
    if sub_list == []:
        return None
    return np.concatenate(sub_list, axis=axis)


def getPair(target_dict, data_dict, permutation_dict, split="train"):
    split_idx = 0 if split == "train" else 1
    lis = []
    for key, prompt_lis in target_dict.items():
        for idx in prompt_lis:
            lis.append(
                [
                    data_dict[key][idx][0][permutation_dict[key][split_idx]],
                    data_dict[key][idx][1][permutation_dict[key][split_idx]],
                ]
            )  # each is a data & label paird, selecting the corresponding split

    data, label = getConcat([w[0] for w in lis]), getConcat([w[1] for w in lis])

    return data, label

def getPairs(target_dict, data_dict, permutation_dict, split="train"):
    split_idx = 0 if split == "train" else 1
    liss = []
    for key, prompt_lis in target_dict.items():
        lis = []
        for idx in prompt_lis:
            lis.append(
                [
                    data_dict[key][idx][0][permutation_dict[key][split_idx]],
                    data_dict[key][idx][1][permutation_dict[key][split_idx]],
                ]
            )  # each is a data & label paird, selecting the corresponding split
        data, label = getConcat([w[0] for w in lis]), getConcat([w[1] for w in lis])
        liss.append([data, label])

    return liss

def getNegPosLabel(
    mdl_name,
    dataset_list,
    prefix="normal",
    location="auto",
    split="train",
    layer=-1,
    prompt_dict=None,
    data_num=1000,
    verbose=True,
    nlabels=False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data_dict, permutation_dict = getDic(
        mdl_name, dataset_list, prefix, location, layer, prompt_dict, data_num, verbose, nlabels=nlabels,
    )
    projection_dict = {key: range(len(data_dict[key])) for key in dataset_list}
    data, labels = getPair(projection_dict, data_dict, permutation_dict, split)

    neg = data[:, 0, :]
    pos = data[:, 1, :]

    return neg, pos, labels

def getActsLabel(
    mdl_name,
    dataset_list,
    prefix="normal",
    location="auto",
    split="train",
    layer=-1,
    prompt_dict=None,
    data_num=1000,
    verbose=True,
    nlabels=True,
) -> list[tuple[np.ndarray, np.ndarray]]:
    data_dict, permutation_dict = getDic(
        mdl_name, dataset_list, prefix, location, layer, prompt_dict, data_num, verbose, nlabels=nlabels,
    )
    projection_dict = {key: range(len(data_dict[key])) for key in dataset_list}
    return getPairs(projection_dict, data_dict, permutation_dict, split)


"""------ Func: get_zeros_acc ------
## Input = csv_name, mdl_name, dataset_list, prefix, prompt_dict = None, avg = False
    csv_name: The name of csv we get accuracy from.
    mdl_name: The name of the model.
    dataset_list: List of dataset you want the accuracy from.
    prefix: The name of prefix.
    prompt_dict: Same as in getDir(). You can specify which prompt to get using this variable. Default is None, i.e. get all prompts.
    avg: Whether to average upon return. If True, will return a numbers, otherwise a dict with key from dataset_list and values being a list of accuracy.
## Output = number / dict, depending on `avg`
"""


def get_zeros_acc(csv_name, mdl_name, dataset_list, prefix, prompt_dict=None, avg=False):
    zeros = pd.read_csv(os.path.join(load_dir, csv_name + ".csv"))
    zeros.dropna(subset=["calibrated"], inplace=True)
    subzeros = zeros.loc[(zeros["model"] == mdl_name) & (zeros["prefix"] == prefix)]

    # Extend prompt_dict to ALL dict if it is None
    if prompt_dict is None:
        prompt_dict = {key: range(1000) for key in dataset_list}

    # Extract accuracy, each key is a set name and value is a list of acc
    acc_dict = {}
    for dataset in dataset_list:
        filtered_csv = subzeros.loc[
            (subzeros["dataset"] == dataset) & (subzeros["prompt_idx"].isin(prompt_dict[dataset]))
        ]
        acc_dict[dataset] = filtered_csv["calibrated"].to_list()

    if not avg:
        return acc_dict
    else:
        # get the dataset avg, and finally the global level avg
        return np.mean([np.mean(values) for values in acc_dict.values()])
