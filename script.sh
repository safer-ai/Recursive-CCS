#!/bin/bash

# python generate_css_dirs.py --model_name deberta  --num_examples 400 --batch_size 40 -- --ntries 5 --reciters 20 --run_name rcss_20_dirs_1;

for i in {0..8}
do
    python generate_css_dirs.py --model_name deberta  --num_examples 400 --batch_size 40 -- --ntries 10 --reciters 20 --nepochs 10000 --run_name rcss_20_dirs_long_$i;
done