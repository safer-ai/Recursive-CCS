#!/bin/bash

python generate_ccs_dirs_main.py --model_name unifiedqa-t5-11b --dataset_name imdb -- --ntries 1 --reciters 4 --nepochs 4000 --lbfgs --weight_decay 0.001 --run_name lbfgs_attempts_4000

# python generation_main.py --model gpt-neo-125M --swipe --num_data 4000 --print_more --cal_zeroshot 0 --datasets ag-news dbpedia-14 tweet-eval-emotion tweet-eval-sentiment amazon-polarity

# models=("unifiedqa-t5-11b" "deberta-v2-xxlarge-mnli")

# for model in "${models[@]}"; do
#     python generation_main.py --model $model --swipe --num_data 4000 --print_more --cal_zeroshot 0 --datasets ag-news dbpedia-14 tweet-eval-emotion tweet-eval-sentiment amazon-polarity
# done

# gpt_models=("gpt-neo-125M" "gpt-neo-2.7B" "gpt-j-6B")

# for model in "${gpt_models[@]}"; do
#     python generation_main.py --model $model --swipe --num_data 4000 --print_more --cal_zeroshot 0 --datasets ag-news dbpedia-14 tweet-eval-emotion tweet-eval-sentiment amazon-polarity --save_all_layers
# done

# for i in {0..3}; do
#     python generate_ccs_dirs_main.py --model_name unifiedqa-t5-11b --dataset_name imdb amazon-polarity copa ag-news dbpedia-14 rte boolq qnli piqa -- --ntries 1 --reciters 30 --nepochs 0 --run_name notrain_uqa_all_30_$i;
# done

# list=(0.1 0.03 0.01 0.003 0.001 0.0003 0)
# for w in "${list[@]}"; do
#     for i in {0..3}; do
#         echo $w ${w#*.}
#         python generate_ccs_dirs_main.py --model_name unifiedqa-t5-11b --dataset_name imdb amazon-polarity copa ag-news dbpedia-14 rte boolq qnli piqa -- --ntries 1 --reciters 30 --nepochs 4000  --lbfgs --weight_decay $w --run_name uqa_all_30_w${w#*.}_$i;
#     done
# done

# list=(0.1 0.03 0.01 0.003 0.001)
# for w in "${list[@]}"; do
#     for i in {0..3}; do
#         echo $w ${w#*.}
#         python generate_ccs_dirs_main.py --model_name unifiedqa-t5-11b --dataset_name copa -- --ntries 1 --reciters 30 --nepochs 4000  --lbfgs --weight_decay $w --run_name uqa_copa_30_w${w#*.}_$i;
#         python generate_ccs_dirs_main.py --model_name unifiedqa-t5-11b --dataset_name imdb -- --ntries 1 --reciters 30 --nepochs 4000  --lbfgs --weight_decay $w --run_name uqa_imdb_30_w${w#*.}_$i;
#     done
# done

# for i in {0..2}; do
# python generate_ccs_dirs_main.py --model_name unifiedqa-t5-11b --dataset_name copa -- --ntries 10 --reciters 30 --run_name uqa_copa_orig_1000_$i
# done

# for i in {0..3}; do
# for l in $(seq 0 4 12)
# do
#     python generate_ccs_dirs_main.py --model_name gpt-neo-125M --layer $l -- --ntries 1 --reciters 30 --nepochs 4000  --lbfgs --weight_decay 0.01 --run_name neo125_imdb_w01_30_${i}/layer$l;
# done
# done

# for i in {0..3}; do
# for l in $(seq 0 4 32)
# do
#     python generate_ccs_dirs_main.py --model_name gpt-neo-2.7B --layer $l -- --ntries 1 --reciters 30 --nepochs 4000  --lbfgs --weight_decay 0.01 --run_name neo27_imdb_w01_30_${i}/layer$l;
# done
# done

# for i in {0..3}; do
# for l in $(seq 0 4 28)
# do
#     python generate_ccs_dirs_main.py --model_name gpt-j-6B --layer $l -- --ntries 1 --reciters 30 --nepochs 4000  --lbfgs --weight_decay 0.01 --run_name gptj_imdb_w01_30_${i}/layer$l;
# done
# done

# for i in {0..3}; do
# for l in $(seq 0 4 12)
# do
#     python generate_ccs_dirs_main.py --model_name gpt-neo-125M --layer $l -- --ntries 1 --reciters 30 --nepochs 0 --run_name notrain_neo125_imdb_30_${i}/layer$l;
# done
# done

# for i in {0..3}; do
# for l in $(seq 0 4 12)
# do
#     python generate_ccs_dirs_main.py --model_name gpt-neo-125M --layer $l -- --ntries 1 --reciters 30 --nepochs 4000  --lbfgs --weight_decay 0 --run_name neo125_imdb_30_${i}/layer$l;
# done
# done

# for i in {0..3}; do
# for l in $(seq 0 4 32)
# do
#     python generate_ccs_dirs_main.py --model_name gpt-neo-2.7B --layer $l -- --ntries 1 --reciters 30 --nepochs 0 --run_name notrain_neo27_imdb_30_${i}/layer$l;
# done
# done

# for i in {0..3}; do
# for l in $(seq 0 4 32)
# do
#     python generate_ccs_dirs_main.py --model_name gpt-neo-2.7B --layer $l -- --ntries 1 --reciters 30 --nepochs 4000  --lbfgs --weight_decay 0 --run_name neo27_imdb_30_${i}/layer$l;
# done
# done

# for i in {0..3}; do
# for l in $(seq 0 4 28)
# do
#     python generate_ccs_dirs_main.py --model_name gpt-j-6B --layer $l -- --ntries 1 --reciters 30 --nepochs 0 --run_name notrain_gptj_imdb_30_${i}/layer$l;
# done
# done

# for i in {0..3}; do
# for l in $(seq 0 4 28)
# do
#     python generate_ccs_dirs_main.py --model_name gpt-j-6B --layer $l -- --ntries 1 --reciters 30 --nepochs 4000  --lbfgs --weight_decay 0 --run_name gptj_imdb_30_${i}/layer$l;
# done
# done


# for i in {0..20}
# do
#     python generate_ccs_dirs_main.py --model_name unifiedqa-t5-11b -- --ntries 1 --reciters 30 --nepochs 0 --run_name notrain_uqa_imdb_30_$i;
# done

# for i in {0..4}
# do
#     python generate_ccs_dirs_main.py --model_name unifiedqa-t5-11b -- --ntries 1 --reciters 30 --nepochs 4000  --lbfgs --run_name uqa_imdb_30_w01_$i;
# done

# for i in {0..4}
# do
#     python generate_ccs_dirs_main.py --model_name unifiedqa-t5-11b -- --ntries 1 --reciters 30 --nepochs 4000  --lbfgs --weight_decay 0 --run_name uqa_imdb_30_$i;
# done

# for i in {0..20}
# do
#     python generate_ccs_dirs_main.py --model_name unifiedqa-t5-11b --dataset_name copa -- --ntries 1 --reciters 30 --nepochs 0 --run_name notrain_uqa_copa_30_$i;
# done

# for i in {0..4}
# do
#     python generate_ccs_dirs_main.py --model_name unifiedqa-t5-11b --dataset_name copa -- --ntries 1 --reciters 30 --nepochs 4000  --lbfgs --run_name uqa_copa_30_w01_$i;
# done

# for i in {0..4}
# do
#     python generate_ccs_dirs_main.py --model_name unifiedqa-t5-11b --dataset_name copa  -- --ntries 1 --reciters 30 --nepochs 4000  --lbfgs --weight_decay 0 --run_name uqa_copa_30_$i;
# done

# for i in {0..3}
# do
#     python generate_ccs_dirs_main.py --model_name unifiedqa-t5-11b --dataset_name copa --  --ntries 10 --reciters 30 --nepochs 1000  --run_name uqa_copa_30_orig_$i;
# done

# for i in {0..3}
# do
#     python generate_ccs_dirs_main.py --model_name unifiedqa-t5-11b --dataset_name imdb --  --ntries 10 --reciters 30 --nepochs 1000  --run_name uqa_imdb_30_orig_$i;
# done

# for l in $(seq 0 4 12)
# do
#     python generate_ccs_dirs_main.py --model_name gpt-neo-125M --layer $l -- --ntries 10 --reciters 30 --nepochs 1000  --run_name neo125_imdb_orig_30_0/layer$l;
# done

# for l in $(seq 0 4 32)
# do
#     python generate_ccs_dirs_main.py --model_name gpt-neo-2.7B --layer $l -- --ntries 10 --reciters 30 --nepochs 1000  --run_name neo27_imdb_orig_30_0/layer$l;
# done

# for l in $(seq 0 4 28)
# do
#     python generate_ccs_dirs_main.py --model_name gpt-j-6B --layer $l -- --ntries 10 --reciters 30 --nepochs 1000  --run_name gptj_imdb_orig_30_0/layer$l;
# done

# for i in {0..20}
# do
#     python generate_ccs_dirs.py --model_name deberta  --num_examples 2000 --batch_size 40 -- --ntries 1 --reciters 30 --nepochs 0 --run_name no_train_30_xl_$i;
# done

# for i in {0..4}
# do
#     python generate_ccs_dirs.py --model_name deberta  --num_examples 2000 --batch_size 40 -- --ntries 1 --reciters 30 --nepochs 4000  --lbfgs --run_name rccs_30_xl2_lbfgs_$i;
# done


# for i in {0..8}
# do
#     python generate_ccs_dirs.py --model_name deberta  --num_examples 400 --batch_size 40 -- --ntries 1 --reciters 20 --nepochs 0 --run_name no_train_$i;
# done

# nepochss=(100 400 1000 4000 10000 40000)

# for nepochs in "${nepochss[@]}"
# do
#     python generate_ccs_dirs.py --model_name deberta  --num_examples 2000 --batch_size 40 -- --ntries 2 --reciters 2 --nepochs $nepochs --lbfgs --run_name lbfgs_attempts_$nepochs;
# done

# python generate_ccs_dirs_main.py --model_name unifiedqa-t5-11b --dataset_name copa -- --ntries 3 --reciters 2 --nepochs 10000 --lbfgs --weight_decay 0 --run_name lbfgs_attempts_10000;
# python generate_ccs_dirs_main.py --model_name unifiedqa-t5-11b --dataset_name copa -- --ntries 3 --reciters 2 --nepochs 10000 --weight_decay 0 --run_name adamw_attempts_10000;

# for i in {0..20}
# do
#     mv css_dirs/rccs_10_xl_$i ccs_dirs/rccs_10_xl_$i;
# done

# python generate.py --model_name unifiedqa --num_examples 4000 --batch_size 1 --dataset_name copa --split train;

# python generation_main.py --model unifiedqa-t5-11b --swipe --print_more --cal_zeroshot 0 --datasets imdb amazon-polarity
# python generation_main.py --model gpt-j-6B --swipe --print_more --cal_zeroshot 0 --datasets imdb amazon-polarity --save_all_layers
# python generation_main.py --model unifiedqa-t5-11b --swipe --print_more --cal_zeroshot 0 --datasets copa ag-news dbpedia-14 rte boolq qnli piqa
# python generation_main.py --model gpt-neo-125M --swipe --print_more --cal_zeroshot 0 --datasets imdb amazon-polarity --save_all_layers
# python generation_main.py --model gpt-neo-2.7B --swipe --print_more --cal_zeroshot 0 --datasets imdb amazon-polarity --save_all_layers
