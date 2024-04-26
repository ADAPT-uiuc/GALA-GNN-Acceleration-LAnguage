########## Configurations ###################
# GNN configurations
#declare -a input_Array=(32 \
#64 \
#128 \
#256 \
#512 \
#1024)
declare -a input_Array=(32)

#declare -a graph_Array=("RedditDataset" \
#"ogbn-products")
#declare -a graph_Array=("RedditDataset")
declare -a graph_Array=("CoraGraphDataset")

##################################
echo "" > timing_explore
##################################
for ie in "${input_Array[@]}"; do
  echo "*****************" >> timing_explore
  echo "$graph" >> timing_explore
  for graph in "${graph_Array[@]}"; do
    echo "++++++++++" >> timing_explore
    echo "$ie" >> timing_explore
    python test_gather.py --dataset "$graph" --n_input "$ie" >> timing_explore
  done
done

