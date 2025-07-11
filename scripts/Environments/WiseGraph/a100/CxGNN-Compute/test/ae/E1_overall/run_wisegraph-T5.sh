#!/bin/bash

# This script is used to run the experiments for WiseGraph benchmark for the paper "X"

# If log files already exist, scrapping them

#python fig_parser.py --log_file results_fig16_17.log --hardware h100 --output results_fig16_17.csv
#python fig_parser.py --log_file results_fig18_19.log --hardware h100 --output results_fig18_19.csv
#python fig_parser.py --log_file results_table5.log --hardware h100 --output results_table5.csv

# ----------- Table 5 -----------
echo "Running experiments for Table 5..."
dsets=(ogbn-papers100M_1 ogbn-papers100M_2 ogbn-papers100M_5 ogbn-papers100M_10)
models=(GCN)
graph_types=(CSR_Layer)
num_layers=(2)
hidden_feats=(32)

for graph_type in "${graph_types[@]}"; do
    for dset in "${dsets[@]}"; do
        for model in "${models[@]}"; do
            for num_layer in "${num_layers[@]}"; do
                for hidden_feat in "${hidden_feats[@]}"; do
                    echo "Running: dataset=${dset}, model=${model}, graph_type=${graph_type}, hidden_feat=${hidden_feat}, num_layer=${num_layer}"
                    python3 test_model.py --dataset "$dset" --model "$model" --graph_type "$graph_type" --hidden_feat "$hidden_feat" --num_layer "$num_layer" \
                        >> results_table5.log 2>&1
                done
            done
        done
    done
done

python fig_parser.py --log_file results_table5.log --hardware h100 --output results_table5.csv

echo "Task completed."