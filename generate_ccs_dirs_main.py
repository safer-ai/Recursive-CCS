import sys
from sklearn.linear_model import LogisticRegression
from utils import get_parser, load_all_generations, CCS, assert_orthonormal
import torch
from pathlib import Path
import json
from utils_generation.state_load_utils import getNegPosLabel, models_layer_num
from time import time
from datetime import datetime

def main(args, generation_args):
    # load hidden states and labels
    neg_hs_train, pos_hs_train, y_train = getNegPosLabel(generation_args.model_name, generation_args.dataset_name, split="train", data_num=generation_args.num_examples, layer=generation_args.layer)
    neg_hs_test, pos_hs_test, y_test = getNegPosLabel(generation_args.model_name, generation_args.dataset_name, split="test", data_num=generation_args.num_examples, layer=generation_args.layer)
    
    # Set up CCS. Note that you can usually just use the default args by simply doing ccs = CCS(neg_hs, pos_hs, y)
    print(neg_hs_train.shape)
    d = neg_hs_train.shape[1]
    constraints = torch.empty((0, d)).to(args.ccs_device)

    arg_dict = vars(args)
    exclude_keys = ["save_dir", "cache_dir", "device"]
    infos = "__".join(["{}_{}".format(k, v) for k, v in arg_dict.items() if k not in exclude_keys])
    folder_name = args.run_name or str(hash(infos))[:20]
    path = Path("./ccs_dirs") / folder_name
    path.mkdir(parents=True, exist_ok=True)
    
    st = time()
    perfs = []

    for it in range(args.reciters):
        ccs = CCS(
            neg_hs_train,
            pos_hs_train,
            nepochs=args.nepochs,
            ntries=args.ntries,
            lr=args.lr,
            batch_size=args.ccs_batch_size,
            verbose=args.verbose,
            device=args.ccs_device,
            weight_decay=args.weight_decay,
            var_normalize=args.var_normalize,
            lbfgs=args.lbfgs,
            constraints=constraints,
        )
        loss, test_loss, test_acc = ccs.repeated_train(neg_hs_test, pos_hs_test, y_test, additional_info=f"it {it} ")
        perfs.append((loss, test_loss, test_acc))
        constraints = torch.cat([constraints, ccs.get_direction()], dim=0)
        assert_orthonormal(constraints)
        ccs.best_probe.constraints = torch.empty((0, d)).to(args.ccs_device) # empty the constraints before save
        ccs.save((path / f"ccs{it}.pt").open("wb"))
    
    runtime = time() - st
    
    json.dump({"args": arg_dict, "runtime": runtime, "timestamp": datetime.now().strftime("%m/%d/%Y, %H:%M:%S"), "loss_testloss_testacc": perfs}, (path / "args.json").open("w"))


if __name__ == "__main__":
    all_args = sys.argv[1:]
    try:
        spliter = all_args.index("--")
        generation_argv = all_args[:spliter]
        evaluation_argv = all_args[spliter + 1 :]
    except:
        generation_argv = all_args
        evaluation_argv = []

    parser = get_parser()
    generation_args = parser.parse_args(generation_argv)  # we'll use this to load the correct hidden states + labels
    # We'll also add some additional args for evaluation
    parser.add_argument("--reciters", type=int, default=3)
    parser.add_argument("--nepochs", type=int, default=1000)
    parser.add_argument("--ntries", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--ccs_batch_size", type=int, default=-1)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--ccs_device", type=str, default="cuda")
    parser.add_argument("--linear", action="store_true")
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--var_normalize", action="store_true")
    parser.add_argument("--lbfgs", action="store_true")
    parser.add_argument("--run_name", type=str, default="")
    args = parser.parse_args(generation_argv + evaluation_argv)
    print(args)
    
    main(args, generation_args)
