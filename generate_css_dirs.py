import sys
from sklearn.linear_model import LogisticRegression
from utils import get_parser, load_all_generations, CCS, assert_orthonormal
import torch
from pathlib import Path
import json

def main(args, generation_args):
    # load hidden states and labels
    neg_hs, pos_hs, y = load_all_generations(generation_args)

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

    # Set up CCS. Note that you can usually just use the default args by simply doing ccs = CCS(neg_hs, pos_hs, y)
    print(neg_hs.shape)
    d = neg_hs.shape[1]
    constraints = torch.empty((0, d)).to(args.ccs_device)
    
    arg_dict = vars(args)
    exclude_keys = ["save_dir", "cache_dir", "device"]
    infos = "__".join(['{}_{}'.format(k, v) for k, v in arg_dict.items() if k not in exclude_keys])
    folder_name = str(hash(infos))[:20]
    path = Path("./css_dirs") / folder_name
    path.mkdir(parents=True, exist_ok=True)
    json.dump(arg_dict, (path / "args.json").open("w"))
    
    for it in range(args.reciters):
        ccs = CCS(neg_hs_train, pos_hs_train, nepochs=args.nepochs, ntries=args.ntries, lr=args.lr, batch_size=args.ccs_batch_size, 
                    verbose=args.verbose, device=args.ccs_device, weight_decay=args.weight_decay, 
                    var_normalize=args.var_normalize, constraints=constraints)
        ccs.repeated_train(neg_hs_test, pos_hs_test, y_test)
        ccs.save((path / f"ccs{it}.pt").open("wb"))
        constraints = torch.cat([constraints, ccs.get_direction()], dim=0)
        assert_orthonormal(constraints)
        ccs_acc = ccs.get_acc(neg_hs_test, pos_hs_test, y_test)
        print("CCS accuracy: {}".format(ccs_acc))

if __name__ == "__main__":
    all_args = sys.argv[1:]
    try:
        spliter = all_args.index("--")
        generation_argv = all_args[:spliter]
        evaluation_argv = all_args[spliter + 1:]
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
    args = parser.parse_args(evaluation_argv)
    print(args)
    main(args, generation_args)
