import argparse
import json
from utils_generation.construct_prompts import confusion_prefix

######## JSON Load ########
json_dir = "./registration"

with open("{}.json".format(json_dir), "r") as f:
    global_dict = json.load(f)
dataset_list = global_dict["dataset_list"]
registered_models = global_dict["registered_models"]
registered_prefix = global_dict["registered_prefix"]
models_layer_num = global_dict["models_layer_num"]


def getArgs():
    parser = argparse.ArgumentParser()

    # datasets loading
    parser.add_argument(
        "--data_base_dir",
        type=str,
        default="datasets/complete_ten",
        help="The base dir of all datasets (csv files) you wnat to generate hidden states.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        help='List of name of datasets you want to use. Please make sure that the path of file is like `os.path.join(data_base_dir, name + ".csv"` for all name in datasets',
    )

    # models loading
    parser.add_argument(
        "--model",
        type=str,
        help="The model you want to use. Please use the model in huggingface and only leave the final path, i.e. for `allenai/unifiedqa-t5-11b`, only input `unifiedqa-t5-11b`.",
    )
    parser.add_argument(
        "--parallelize",
        action="store_true",
        help="Whether to parallelize models in multiple gpus. Please notice that at least one gpu must be provided by `CUDA_VISIBLE_DEVICES` or `os.environ`. Using this args will help split the model equally in all gpus you provide.",
    )
    parser.add_argument("--cache_dir", type=str, default="models", help="The path to save and load pretrained model.")

    # datasets processing
    parser.add_argument(
        "--prefix",
        type=str,
        nargs="+",
        default=["normal"],
        help="The name of prefix added before the question. normal means no index. You can go to `utils/prompts.json` to add new prompt.",
    )
    parser.add_argument(
        "--num_data",
        nargs="+",
        default=[1000],
        help="number of data points you want to use in each datasets. If one integer is provide, if will be extended to a list with the same length as `datasets`. If the size of datasets are no enough, will use all the data points.",
    )
    parser.add_argument(
        "--reload_data",
        action="store_true",
        help="Whether to use the old version of datasets if there exists one. Using `reload_data` will let the program reselect data points from the datasets.",
    )
    parser.add_argument(
        "--swipe",
        action="store_true",
        help="Whether to swipe all prompts. If this is true, then `prompt_idx` will be ignored, and for each dataset in `datasets`, all prompts will be executed.",
    )
    parser.add_argument("--prompt_idx", nargs="+", default=[0], help="The indexs of prompt you want to use.")

    # generation & zero-shot accuracy calculation
    parser.add_argument("--cal_zeroshot", type=int, default=1, help="Whether to calculate the zero-shot accuracy.")
    parser.add_argument("--cal_hiddenstates", type=int, default=1, help="Whether to extract the hidden states.")
    parser.add_argument(
        "--cal_logits",
        type=int,
        default=0,
        help="Whether to extract the logits of the token in which the prediction firstly differs.",
    )
    parser.add_argument(
        "--token_place",
        type=str,
        default="last",
        help="Determine which token's hidden states will be generated. Can be `first` or `last` or `average`.",
    )
    parser.add_argument(
        "--states_location",
        type=str,
        default="null",
        choices=["encoder", "decoder", "null"],
        help="Whether to generate encoder hidden states or decoder hidden states. Default is null, which will be extended to decoder when the model is gpt or encoder otherwise.",
    )
    parser.add_argument(
        "--states_index",
        nargs="+",
        default=[-1],
        help="List of layer hidden states index to generate. -1 means the last layer. For encoder, we will transform positive index into negative. For example, T0pp has 25 layer, indexed by 0, ..., 24. Index 20 will be transformed into -5. For decoder, index will instead be transform into non-negative value. For example, the last decoder layer will be 24 (rather than -1). The choice between encoder and decoder is specified by `states_location`. For decoder, answer will be padded into token rather than into the input.",
    )
    parser.add_argument("--tag", type=str, default="", help="Tag added as the suffix of the directory.")
    parser.add_argument(
        "--save_base_dir",
        type=str,
        default="generation_results",
        help="The base dir where you want to save the directories of hidden states.",
    )
    parser.add_argument(
        "--save_csv_name", type=str, default="results", help="Name of csv that store all running records."
    )
    parser.add_argument(
        "--save_all_layers",
        action="store_true",
        help="Whether to save the hidden states of all layers. Notice that this will increase the disk load significantly.",
    )
    parser.add_argument("--print_more", action="store_true", help="Whether to print more.")

    args = parser.parse_args()

    if args.datasets == ["all"]:
        args.datasets = dataset_list
    else:
        for w in args.datasets:
            assert w in dataset_list, NotImplementedError(
                "Dataset {} not registered in {}.json. Please check the name of the dataset!".format(w, json_dir)
            )

    # if (args.cal_zeroshot or args.cal_logits) and "bert" in args.model:
    # Add features. Only forbid cal_logits for bert type model now
    if args.cal_logits and "bert" in args.model:
        raise NotImplementedError(
            "You use {}, but bert type models do not have standard logits. Please set cal_logits to 0.".format(
                args.model
            )
        )

    assert args.model in registered_models, NotImplementedError(
        "You use model {}, but it's not registered. For any new model, please make sure you implement the code in `load_utils` and `generation`, and then register it in `parser.py`".format(
            args.model
        )
    )

    for prefix in args.prefix:
        assert prefix in registered_prefix, NotImplementedError(
            "Invalid prefix name {}. Please check your prefix name. To add new prefix, please mofidy `utils/prompts.json` and register new prefix in {}.json.".format(
                prefix, json_dir
            )
        )

    # Set default states_location according to model type
    if args.states_location == "null":
        args.states_location = "decoder" if "gpt" in args.model else "encoder"

    if args.states_location == "encoder" and args.cal_hiddenstates:
        assert "gpt" not in args.model, ValueError(
            "GPT type model does not have encoder. Please set `states_location` to `decoder`."
        )
    if args.states_location == "decoder" and args.cal_hiddenstates:
        assert "bert" not in args.model, ValueError(
            "BERT type model does not have decoder. Please set `states_location` to `encoder`."
        )
    # Set index into int.
    for i in range(len(args.states_index)):
        pos_index = int(args.states_index[i]) % models_layer_num[args.model]
        # For decoder, the index lies in [0,layer_num)
        # For encoder, the index lies in [-layer_num, -1]
        args.states_index[i] = (
            pos_index if args.states_location == "decoder" else pos_index - models_layer_num[args.model]
        )

    print("-------- args --------")
    for key in list(vars(args).keys()):
        print("{}: {}".format(key, vars(args)[key]))
    print("-------- args --------")

    return args
