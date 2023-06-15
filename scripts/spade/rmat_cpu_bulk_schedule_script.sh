# Name of the matrix to be downloaded from suite sparse
data_path="/home/damitha2/GNN-Acceleration-Language/data_schedule/"
test_path="../../build/tests/spade_cpu_spmm_impl"

if [ -d "../../build" ]; then
  echo "build folder exists"
else
  mkdir "../../build"
fi

{
  echo "**********"
  echo "graph"
  echo "**********"
  echo "emb_size,col_tile,row_tile,loop_order,slice_size,barrier,work_div,reorder,prefetch"
} >transf_times
(cd ../../build && rm -r * && cmake .. && make)

for d in "$data_path"*/; do
  folder_name=$(basename $d)

  {
    echo "**********"
    echo "$folder_name"
    echo "**********"
  } >>transf_times
  numactl --physcpubind=0-55 --interleave=all $test_path "$d" >>transf_times
done
