import argparse
import copy
import functools
import os
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# make sure to install promptsource, transformers, and datasets!
from tqdm import tqdm # type: ignore
from transformers import (AutoModelForCausalLM, AutoModelForMaskedLM,
                          AutoModelForSeq2SeqLM, AutoTokenizer)

from datasets import load_dataset
from utils_generation.state_load_utils import Dataset # type: ignore

############# Model loading and result saving #############

# Map each model name to its full Huggingface name; this is just for convenience for common models. You can run whatever model you'd like.
model_mapping = {
    "gpt-j": "EleutherAI/gpt-j-6B",
    "T0pp": "bigscience/T0pp",
    "unifiedqa": "allenai/unifiedqa-t5-11b",
    "T5": "t5-11b",
    "deberta-mnli": "microsoft/deberta-xxlarge-v2-mnli",
    "deberta": "microsoft/deberta-xxlarge-v2",
    "roberta-mnli": "roberta-large-mnli",
}


def get_parser():
    """
    Returns the parser we will use for generate.py and evaluate.py
    (We include it here so that we can use the same parser for both scripts)
    """
    parser = argparse.ArgumentParser()
    # setting up model
    parser.add_argument("--model_name", type=str, default="T5", help="Name of the model to use")
    parser.add_argument("--cache_dir", type=str, default=None, help="Cache directory for the model and tokenizer")
    parser.add_argument("--parallelize", action="store_true", help="Whether to parallelize the model")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use for the model")
    # setting up data
    parser.add_argument("--dataset_name", nargs="+", default=["imdb"], help="Name of the datasets to use")
    parser.add_argument("--split", type=str, default="test", help="Which split of the dataset to use")
    parser.add_argument("--prompt_idx", type=int, default=0, help="Which prompt to use")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size to use")
    parser.add_argument("--num_examples", type=int, default=1000, help="Number of examples to generate")
    # which hidden states we extract
    parser.add_argument(
        "--use_decoder",
        action="store_true",
        help="Whether to use the decoder; only relevant if model_type is encoder-decoder. Uses encoder by default (which usually -- but not always -- works better)",
    )
    parser.add_argument("--layer", type=int, default=-1, help="Which layer to use (if not all layers)")
    parser.add_argument("--all_layers", action="store_true", help="Whether to use all layers or not")
    parser.add_argument("--token_idx", type=int, default=-1, help="Which token to use (by default the last token)")
    # saving the hidden states
    parser.add_argument(
        "--save_dir", type=str, default="generated_hidden_states", help="Directory to save the hidden states"
    )
    parser.add_argument("--nlabels", action="store_true")

    return parser


def load_model(model_name: str, cache_dir=None, parallelize=False, device="cuda"):
    """
    Loads a model and its corresponding tokenizer, either parallelized across GPUs (if the model permits that; usually just use this for T5-based models) or on a single GPU
    """
    if model_name in model_mapping:
        # use a nickname for our models
        full_model_name = model_mapping[model_name]
    else:
        # if you're trying a new model, make sure it's the full name
        full_model_name = model_name

    # use the right automodel, and get the corresponding model type
    try:
        model = AutoModelForSeq2SeqLM.from_pretrained(full_model_name, cache_dir=cache_dir)
        model_type = "encoder_decoder"
    except:
        try:
            model = AutoModelForMaskedLM.from_pretrained(full_model_name, cache_dir=cache_dir)
            model_type = "encoder"
        except:
            model = AutoModelForCausalLM.from_pretrained(full_model_name, cache_dir=cache_dir)
            model_type = "decoder"

    # specify model_max_length (the max token length) to be 512 to ensure that padding works
    # (it's not set by default for e.g. DeBERTa, but it's necessary for padding to work properly)
    tokenizer = AutoTokenizer.from_pretrained(full_model_name, cache_dir=cache_dir, model_max_length=512)
    model.eval()

    # put on the correct device
    if parallelize:
        model.parallelize()
    else:
        model = model.to(device)

    return model, tokenizer, model_type



############# CCS #############
class MLPProbe(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.linear1 = nn.Linear(d, 100)
        self.linear2 = nn.Linear(100, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.linear1(x))
        o = self.linear2(h)
        return torch.sigmoid(o)


def project(x: torch.Tensor, constraints: torch.Tensor) -> torch.Tensor:
    """Projects on the hyperplane defined by the constraints"""
    inner_products = torch.einsum("...h,nh->...n", x, constraints)
    return x - torch.einsum("...n,nh->...h", inner_products, constraints)


def normalize(x: torch.Tensor) -> torch.Tensor:
    return x / torch.norm(x, dim=-1, keepdim=True)


def assert_orthonormal(x: torch.Tensor):
    max_diff = torch.max(torch.abs(torch.einsum("nh,mh->nm", x, x) - torch.eye(x.size(0)).to(x.device)))
    if max_diff > 1e-4:
        print("Warning max_diff =", max_diff)
    # assert torch.allclose(torch.einsum("nh,mh->nm", x, x), torch.eye(x.size(0)).to(x.device), atol=1e-4, rtol=1e-4)


class LinearWithConstraints(nn.Module):
    def __init__(self, d: int, constraints: torch.Tensor):
        super().__init__()
        self.linear = nn.Linear(d, 1)
        self.constraints = constraints

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = project(self.linear.weight, self.constraints)
        y = F.linear(x, w, self.linear.bias)
        return torch.sigmoid(y)

    def project(self):
        with torch.no_grad():
            self.linear.weight[:] = project(self.linear.weight, self.constraints)


class LinearAlong(nn.Module):
    def __init__(self, along: torch.Tensor):
        super().__init__()
        n, d = along.shape
        self.coeffs = nn.Linear(n, 1)
        self.coeffs.weight.data = torch.ones_like(self.coeffs.weight.data) / n
        self.along = along

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = torch.einsum("nd,nh->hd", self.along, self.coeffs.weight)
        y = F.linear(x, w, self.coeffs.bias)
        return torch.sigmoid(y)


class Linear(nn.Module):
    def __init__(self, d: int) -> None:
        super().__init__()
        self.linear = nn.Linear(d, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.linear(x))

NormData = tuple[np.ndarray, np.ndarray] # mean, std
Probe = Union[Linear, LinearAlong, LinearWithConstraints, MLPProbe]
class CCS(object):
    def __init__(
        self,
        train_ds: Dataset,
        nepochs: int=1000,
        ntries: int=10,
        lr: float=1e-3,
        batch_size: int=-1,
        verbose: bool=False,
        device: str="cuda",
        linear:bool=True,
        weight_decay:float=0.01,
        var_normalize: bool=True,
        constraints: Optional[torch.Tensor]=None,
        along: Optional[torch.Tensor]=None,
        lbfgs: bool=False,
        informative_strength: float=1.,
    ):
        # data
        self.var_normalize = var_normalize
        # list of (batch, answer, hidden)
        # answer dims may vary
        self.xs = [x for x,y in train_ds]
        self.train_ds = train_ds
        self.norm_datas: list[NormData] = [(x.mean(0, keepdims=True), x.std(0, keepdims=True)) for x,y in train_ds]
        self.d = self.xs[0].shape[-1]

        # training
        self.nepochs = nepochs
        self.ntries = ntries
        self.lr = lr
        self.verbose = verbose
        self.device = device
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.lbfgs = lbfgs
        self.informative_strength = informative_strength

        # probe
        self.linear = linear
        assert not (constraints is not None and along is not None)
        self.constraints = constraints
        self.along = along
        self.initialize_probe()
        self.best_probe = copy.deepcopy(self.probe)

    def initialize_probe(self):
        if self.constraints is not None:
            self.probe = LinearWithConstraints(self.d, self.constraints)
        elif self.along is not None:
            self.probe = LinearAlong(self.along)
        elif self.linear:
            self.probe = Linear(self.d)
        else:
            self.probe = MLPProbe(self.d)
        self.probe.to(self.device)

    def normalize(self, x: np.ndarray, norm_data: Optional[NormData]=None) -> np.ndarray:
        """
        Mean-normalizes the data x (of shape (n, a, d))
        If self.var_normalize, also divides by the standard deviation
        """
        if norm_data is None:
            mean: np.ndarray = x.mean(axis=0, keepdims=True)
            std: np.ndarray = x.std(axis=0, keepdims=True)
        else:
            mean, std = norm_data
        
        normalized_x: np.ndarray = x - mean
        if self.var_normalize:
            normalized_x /= std

        return normalized_x

    def get_tensor_data(self) -> list[torch.Tensor]:
        """
        Returns self.xs as appropriate normalized tensors (rather than np arrays)
        """
        return self.prepare(self.xs)

    def prepare(self, xs: list[np.ndarray]) -> list[torch.Tensor]:
        """
        Returns xs as appropriate normalized tensors (rather than np arrays)
        """
        return [torch.tensor(self.normalize(x, norm_data), dtype=torch.float, requires_grad=False, device=self.device) for x, norm_data in zip(xs, self.norm_datas)]

    def get_loss(self, ps: list[torch.Tensor]) -> torch.Tensor:
        """
        Returns the CCS loss the probabilities on each dataset, each of shape (n, a) or (n, a, 1)
        """
        return self.get_consistent_loss(ps) + self.informative_strength * self.get_informative_loss(ps)

    def get_consistent_loss(self, ps: list[torch.Tensor]) -> torch.Tensor:
        # return ((p0 - (1 - p1)) ** 2).mean(0)
        num_samples = sum(p.shape[0] for p in ps)
        total_loss = sum(((1 - p.sum(1)) ** 2).sum() for p in ps)
        return total_loss / num_samples

    def get_informative_loss(self, ps: list[torch.Tensor]) -> torch.Tensor:
        # return (torch.min(p0, p1) ** 2).mean(0)
        num_samples = sum(p.shape[0] for p in ps)
        total_loss = sum(((1 - p.max(1)[0]) ** 2).sum() for p in ps)
        return total_loss / num_samples

    def get_acc(self, test_ds: Dataset, raw: bool=False) -> float:
        """
        Computes accuracy for the current parameters on the given test inputs
        """
        return self.get_probe_acc(self.best_probe, test_ds, raw=raw)

    def get_probe_acc(self, probe: Probe, test_ds: Dataset, raw: bool=False):
        xs = [x for x,y in test_ds]
        xs = self.prepare(xs)
        with torch.no_grad():
            ps: list[torch.Tensor] = [probe(x) for x in xs]
        preds: list[np.ndarray] = [p.argmax(1).detach().cpu().numpy().astype(int)[:, 0] for p in ps] # same as below according to math
        # 0.5 * (p0 + (1 - p1)) < 0.5 <=> p0 + 1 - p1 < 1 <=> p0 < p1
        # print(ps[0][:10],"preds", preds[0][:10], "corrects", [y for x,y in test_ds][:10])
        
        # avg_confidence = 0.5 * (p0 + (1 - p1))
        # print("avg_confidence", avg_confidence[:10])
        # predictions = (avg_confidence.detach().cpu().numpy() < 0.5).astype(int)[:, 0]
        corrects: list[np.ndarray] = [(pred == y) for pred, y in zip(preds, [y for x,y in test_ds])]
        num_samples: int = sum(p.shape[0] for p in ps)
        # print("corrects", corrects[:10], "num_samples", num_samples, "sum", sum(c.sum() for c in corrects))
        acc = sum(c.sum() for c in corrects) / num_samples
        # for i in [10,100,200, 500, 700, 800, 1000]:
        #     print(f"acc@{i}", (predictions[:i] == y_test[:i]).mean())
        if not raw:
            acc = max(acc, 1 - acc)

        return acc

    def train(self):
        """
        Does a single training run of nepochs epochs

        constraints is a tensor of shape (n, d) where n is the number of constraints
        and constraints are <x, d_i> = 0 for each i
        """
        raise NotImplementedError("this branch does not support training other than lbfgs yet")
        
        x0, x1 = self.get_tensor_data()

        # set up optimizer
        optimizer = torch.optim.AdamW(self.probe.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        if isinstance(self.probe, LinearWithConstraints):
            self.probe.project()

        batch_size = len(x0) if self.batch_size == -1 else self.batch_size
        nbatches = len(x0) // batch_size

        if self.nepochs == 0:
            # comptue loss on first batch
            loss = self.get_loss(self.probe(x0[:batch_size]), self.probe(x1[:batch_size]))

        # Start training (full batch)
        for epoch in range(self.nepochs):
            permutation = torch.randperm(len(x0))
            x0, x1 = x0[permutation], x1[permutation]
            for j in range(nbatches):
                x0_batch = x0[j * batch_size : (j + 1) * batch_size]
                x1_batch = x1[j * batch_size : (j + 1) * batch_size]

                # probe
                p0, p1 = self.probe(x0_batch), self.probe(x1_batch)

                # get the corresponding loss
                loss = self.get_loss(p0, p1)

                # update the parameters
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if isinstance(self.probe, LinearWithConstraints):
                    self.probe.project()

        return loss.detach().cpu().item()

    def train_with_lbfgs(self, debug:bool=False):
        """
        Does a single training run of nepochs epochs

        constraints is a tensor of shape (n, d) where n is the number of constraints
        and constraints are <x, d_i> = 0 for each i
        """
        xs = self.get_tensor_data()

        # l2 with SGD is equivalent to weight decay
        # with l2, w = w - lr * (grad + l2 * w)
        # with weight decay, w = w - w * weight_decay - lr * grad
        # therefore weight_decay = l2 * lr
        # but this doesn't work in practice so we just use weight decay
        l2 = self.weight_decay

        # set up optimizer
        optimizer = torch.optim.LBFGS(
            self.probe.parameters(),
            line_search_fn="strong_wolfe",
            max_iter=self.nepochs,
            tolerance_change=torch.finfo(xs[0].dtype).eps,
            tolerance_grad=torch.finfo(xs[0].dtype).eps,
        )
        if isinstance(self.probe, LinearWithConstraints):
            self.probe.project()

        def closure(debug:bool=False):
            if isinstance(self.probe, LinearWithConstraints):
                self.probe.project()
            optimizer.zero_grad()
            ps = [self.probe(x) for x in xs]
            loss = self.get_loss(ps)

            if debug:
                print("loss", loss.item())

            if isinstance(self.probe, LinearWithConstraints):
                loss += l2 * self.probe.linear.weight[0].norm() ** 2 / 2

            if debug:
                print("loss", loss.item(), self.probe.linear.weight[0].shape)

            loss.backward()
            # if isinstance(self.probe, LinearWithConstraints):
            #     self.probe.project()
            if not loss.isfinite():
                print("Loss is not finite")
                loss = torch.tensor(0.0, device=loss.device)
                optimizer.zero_grad()
            return loss

        if debug:
            closure(debug=True)

        optimizer.step(closure)

        loss = closure(debug=debug)
        if isinstance(self.probe, LinearWithConstraints):
            self.probe.project()
        return loss.detach().cpu().item()

    def repeated_train(self, test_ds:Dataset, additional_info:str="", verbose:bool=True):
        best_loss = np.inf
        best_test_loss = np.inf
        best_test_acc = 0
        for train_num in range(self.ntries):
            self.initialize_probe()
            loss = self.train_with_lbfgs() if self.lbfgs else self.train()
            test_loss = self.eval_probe(self.probe, [x for x,y in test_ds])[2]
            test_acc = self.get_probe_acc(self.probe, test_ds, raw=True)
            train_acc = self.get_probe_acc(self.probe, self.train_ds, raw=True)
            if verbose:
                print(
                    f"{additional_info}try {train_num}: train_loss={loss:.5f} test_loss={test_loss:.5f} train_acc={train_acc:.5f} test_acc={test_acc:.5f} "
                )
            if loss < best_loss:
                self.best_probe = copy.deepcopy(self.probe)
                best_loss = loss
                best_test_loss = test_loss
                best_test_acc = test_acc

        return best_loss, best_test_loss, best_test_acc

    def eval(self, xs: list[np.ndarray]) -> tuple[float, float, float]:
        """
        return consistent loss, informative loss, and loss on the test set
        """
        return self.eval_probe(self.best_probe, xs)

    def eval_probe(self, probe: Probe, xs: list[np.ndarray]) -> tuple[float, float, float]:
        xst = self.prepare(xs)
        with torch.no_grad():
            # probe
            ps = [probe(x) for x in xst]

            # get the corresponding loss
            consistent_loss = self.get_consistent_loss(ps).item()
            informative_loss = self.get_informative_loss(ps).item()
            # print(f"consistent_loss={consistent_loss:.5f} informative_loss={informative_loss:.5f} ")
            return consistent_loss, informative_loss, consistent_loss + self.informative_strength * informative_loss

    def save(self, path: Union[str, Path]):
        torch.save(self.best_probe.state_dict(), path)

    def load(self, path: Union[str, Path]):
        self.best_probe.load_state_dict(torch.load(path))
        self.best_probe = self.best_probe.to(self.device)

    def get_direction(self) -> torch.Tensor:
        """
        Returns the direction of the probe
        """

        def get_dir():
            if isinstance(self.best_probe, nn.Linear):
                return self.best_probe.weight
            elif isinstance(self.best_probe, LinearWithConstraints):
                return self.best_probe.linear.weight
            else:
                raise NotImplementedError("Can't get direction for this probe")

        return normalize(get_dir().detach())
