#dsets=(arxiv products reddit papers100M-sample-1000 friendster-sample-1000)
dsets=(arxiv products cora corafull pubmed reddit)
# models=(LSTM)
models=(GAT SAGE GCN GIN)
# graph_types=(CSR_Layer)
graph_types=(CSR_Layer)
#num_layers=(2 3 4)
#hidden_feats=(32 64 256 1024)
num_layers=(2)
hidden_feats=(32)

for graph_type in ${graph_types[@]}; do
    for dset in ${dsets[@]}; do
        for model in ${models[@]}; do
            for num_layer in ${num_layers[@]}; do
                for hidden_feat in ${hidden_feats[@]}; do
                    echo "python3 test_model.py --dataset ${dset} --model ${model} --graph_type ${graph_type} --hidden_feat ${hidden_feat} --num_layer ${num_layer}"
                    python3 test_model.py --dataset ${dset} --model ${model} --graph_type ${graph_type} --hidden_feat ${hidden_feat} --num_layer ${num_layer}
                done
            done
        done
    done
done

