# Name of the matrix to be downloaded from suite sparse
data_path="/home/damitha2/GNN-Acceleration-Language/data_schedule/"
test_path="../../build/tests/spade_cpu_spmm_impl_bulk2"

if [ -d "../../build" ]; then
  echo "build folder exists"
else
  mkdir "../../build"
fi

{
  echo "**********"
  echo "graph"
  echo "**********"
  echo "emb_size,col_tile,row_tile,slice_size,barrier,work_div,reorder,prefetch,time,time_std"
} >transf_times
#(cd ../../build && rm -r * && cmake .. && make)

export OMP_NUM_THREADS=56

for d in "$data_path"*/; do
  folder_name=$(basename $d)
  IFS='_' read -ra name_Array <<< "$folder_name"

  if [[ $((name_Array[0])) -le 250000 ]]
  then
    {
      echo "**********"
      echo "$folder_name"
      echo "**********"
    } >>transf_times
    numactl --physcpubind=0-55 --interleave=all $test_path "$d" 1 >>transf_times
  fi
done

for d in "$data_path"*/; do
  folder_name=$(basename $d)
  IFS='_' read -ra name_Array <<< "$folder_name"

  if [[ $((name_Array[0])) -gt 250000 ]]
  then
    {
      echo "**********"
      echo "$folder_name"
      echo "**********"
    } >>transf_times
    numactl --physcpubind=0-55 --interleave=all $test_path "$d" 1 >>transf_times
  fi
done
