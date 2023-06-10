# Name of the matrix to be downloaded from suite sparse
data_path="/home/damitha/GNN-Acceleration-Language/data_schedule/"
test_path="../../build/tests/spade_cpu_spmm_impl"
iters=100

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
# Slice size
declare -a slice_Array=(32 \
128 \
512 \
2048)
#declare -a barriered_Array=("0" \
#"1") # TODO
#declare -a work_div_Array=("0") # TODO
declare -a reord_Array=("0" \
"1")
#declare -a prefetch_Array=("0") # TODO
declare -a loop_order_Array=(2 \
3 \
5)


echo "path,emb_size,col_tile,row_tile,loop_order,slice_size,barrier,work_div,reorder,prefetch" > transf_times
for lo in "${loop_order_Array[@]}"; do
  (cd ../../build && rm -r * && cmake -DL_ORDER=$lo .. && make)

  # Select the Graph + embedding
  for d in "$data_path"*/ ; do
    for emb_size in "${emb_Array[@]}"; do
      folder_name=$(basename $d)
      IFS='_' read -ra name_Array <<< "$folder_name"

      # Tiling
      for col_tile in "${col_tile_Array[@]}"; do
        col_chk=$((4*$((name_Array[0]))))
        if [[ $col_tile -lt $col_chk ]]
        then
          for row_tile in "${row_tile_Array[@]}"; do

            # If loop order 2, then try the other optimizations. Else just slicing
            if [[ $lo == "2" ]]
            then

              for reord in "${reord_Array[@]}"; do
                echo "$d,$emb_size,$col_tile,$row_tile,$lo,no_slice,no_barr,no_wdiv,$reord,no_pref," >> transf_times
                numactl --physcpubind=0-55 --interleave=all $test_path $d $emb_size $col_tile $row_tile 0 0 0 $reord 0 $iters >> transf_times
              done

            else
              for slice in "${slice_Array[@]}"; do
                if [[ $slice -lt $emb_size ]]
                then
                  for reord in "${reord_Array[@]}"; do
                    echo "$d,$emb_size,$col_tile,$row_tile,$lo,$slice,no_barr,no_wdiv,$reord,no_pref," >> transf_times
                    numactl --physcpubind=0-55 --interleave=all $test_path $d $emb_size $col_tile $row_tile $slice 0 0 $reord 0 $iters >> transf_times
                  done
                fi
              done
            fi
          done
        fi
      done
    done
  done
done





