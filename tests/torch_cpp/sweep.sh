echo "" > sweep.txt
########## Actual ###################
# Graph configurations
declare -a layers_Array=(1 \
2 \
3)
declare -a hdim_Array=(32 \
64 \
256 \
1024)
# declare -a hdim_Array=(256 \
# 1024)
# declare -a hdim_Array=(1024)

for layers in "${layers_Array[@]}"; do
  echo "layers: $layers" >> sweep.txt
  for hid in "${hdim_Array[@]}"; do
    echo "hidden: $hid" >> sweep.txt
    # echo "CoraGraphDataset" >> sweep.txt
    # build/gala_sweep_op /shared/damitha2/gala_npy/CoraGraphDataset/ 1000 3000 $hid $layers >> sweep.txt
    # echo "PubmedGraphDataset" >> sweep.txt
    # build/gala_sweep_op /shared/damitha2/gala_npy/PubmedGraphDataset/ 1000 10000 $hid $layers >> sweep.txt
    # echo "CoraFullDataset" >> sweep.txt
    # build/gala_sweep_op /shared/damitha2/gala_npy/CoraFullDataset/ 1000 100000 $hid $layers >> sweep.txt
    echo "RedditDataset" >> sweep.txt
    build/gala_sweep_op /shared/damitha2/gala_npy/RedditDataset/ 100 37000 $hid $layers >> sweep.txt
    # echo "ogbn-arxiv" >> sweep.txt
    # build/gala_sweep_op /shared/damitha2/gala_npy/ogbn-arxiv/ 1000 1000000 $hid $layers >> sweep.txt
    # echo "ogbn-products" >> sweep.txt
    # build/gala_sweep_op /shared/damitha2/gala_npy/ogbn-products/ 100 1400000 $hid $layers >> sweep.txt
  done
done

