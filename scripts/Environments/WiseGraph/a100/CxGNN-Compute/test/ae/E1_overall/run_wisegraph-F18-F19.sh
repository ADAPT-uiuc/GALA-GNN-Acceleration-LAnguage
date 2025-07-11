#!/bin/bash

# This script is used to run the experiments for WiseGraph benchmark for the paper "X"

# If log files already exist, scrapping them

#python fig_parser.py --log_file results_fig16_17.log --hardware h100 --output results_fig16_17.csv
#python fig_parser.py --log_file results_fig18_19.log --hardware h100 --output results_fig18_19.csv
#python fig_parser.py --log_file results_table5.log --hardware h100 --output results_table5.csv

# ----------- Figure 18 & 19 -----------
echo "Running experiments for Figure 18 & 19..."
dsets=(reddit)
models=(GCN)
graph_types=(CSR_Layer)
num_layers=(2 3 4 8)
hidden_feats=(32 64 128 256 512 1024)

for graph_type in "${graph_types[@]}"; do
    for dset in "${dsets[@]}"; do
        for model in "${models[@]}"; do
            for num_layer in "${num_layers[@]}"; do
                for hidden_feat in "${hidden_feats[@]}"; do
                    echo "Running: dataset=${dset}, model=${model}, graph_type=${graph_type}, hidden_feat=${hidden_feat}, num_layer=${num_layer}"
                    python3 test_model.py --dataset "$dset" --model "$model" --graph_type "$graph_type" --hidden_feat "$hidden_feat" --num_layer "$num_layer" \
                        >> results_fig18_19.log 2>&1
                done
            done
        done
    done
done

python fig_parser.py --log_file results_fig18_19.log --hardware h100 --output results_fig18_19.csv

echo "Task completed."