#!/bin/bash

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


# for i in {0..20}
# do
#     mv css_dirs/rccs_10_xl_$i ccs_dirs/rccs_10_xl_$i;
# done

# python generate.py --model_name unifiedqa --num_examples 4000 --batch_size 1 --dataset_name copa --split train;

# python generation_main.py --model unifiedqa-t5-11b --swipe --print_more --cal_zeroshot 0 --datasets imdb amazon-polarity
python generation_main.py --model gpt-j-6B --swipe --print_more --cal_zeroshot 0 --datasets imdb amazon-polarity --save_all_layers
python generation_main.py --model unifiedqa-t5-11b --swipe --print_more --cal_zeroshot 0 --datasets copa ag-news dbpedia-14 rte boolq qnli piqa