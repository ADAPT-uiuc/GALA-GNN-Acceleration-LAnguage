# Name of the matrix to be downloaded from suite sparse
matrix_gen_path="/home/damitha/PycharmProjects/generate/"

row_tile_size=2
column_tile_size=20000
schedule_output="../../acc_out_1/"
nPEs=5
feat_size=128
have_empty=0
reord_mtx=0

# Graph configurations
declare -a nodes_Array=(10000 \
100000 \
1000000)
declare -a edge_mul_Array=(2 \
10 \
100)
declare -a power_law_Array=("1000" \
"1100" \
"1110")
declare -a pw_edge_ratio_Array=(1 \
2 \
10 \
100)

# Embedding configurations
declare -a emb_Array=(32 \
128 \
512 \
2048)

# Optimization options
declare -a col_tile_Array=(512 \
2048 \
8192 \
32768 \
131072 \
524228 \
2097152)
declare -a row_tile_Array=(1 \
4 \
16 \
64 \
256 \
1024 \
4096)
declare -a barriered_Array=("0" \
"1")
declare -a bypass_Array=("0" \
"1")
declare -a work_div_Array=("0") # TODO
declare -a reord_Array=("0" \
"1")
declare -a prefetch_Array=("0") # TODO

# Create a main data folder if there isn't one
#if [ -d "../../data_schedule" ]
#then
#    echo "data_schedule folder exists"
#else
#    mkdir "../../data_schedule"
#fi
#
## Create a sub data folder for the specific matrix
#if [ -d "../../data_schedule/$mtx_ss_name" ]
#then
#    echo "data_schedule/$mtx_ss_name folder exists"
#else
#    mkdir "../../data_schedule/$mtx_ss_name"
#fi

if [[ $nPEs -lt $feat_size ]]
then
 echo "Works"
else
  echo "still works"
fi

for node in "${nodes_Array[@]}"; do
  for edge_mul in "${edge_mul_Array[@]}"; do
    for pw in "${power_law_Array[@]}"; do
      for pw_ratio in "${pw_edge_ratio_Array[@]}"; do
        python ../data/rmat_generate.py --pw_abcd "$pw" --rat_pw "$pw_ratio" --nodes "$node" --mul_edges "$edge_mul" --outp "$matrix_gen_path"

      done
    done
  done
done

# Schedule the mtx_ss_name
#../build/tests/accel_spmm_pretiling_test ../../data_schedule/$mtx_ss_name/$mtx_ss_name.mtx 1 100 1 100 $row_tile_size $column_tile_size $schedule_output $nPEs 1 $feat_size $have_empty $reord_mtx

