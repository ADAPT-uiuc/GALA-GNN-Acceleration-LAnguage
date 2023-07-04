# Name of the matrix to be downloaded from suite sparse
data_path="/home/damitha2/GNN-Acceleration-Language/data_schedule/"
test_path="../../build/tests/spade_graph_data"

export OMP_NUM_THREADS=56

echo "" > transf_data
for d in "$data_path"*/; do
  folder_name=$(basename $d)
  {
    echo "**********"
    echo "$folder_name"
    echo "**********"
  } >>transf_data
  numactl --physcpubind=0-55 --interleave=all $test_path "$d" 56 100 >>transf_data
done
