# Name of the matrix to be downloaded from suite sparse
matrix_gen_path="../../data_schedule/"

schedule_output="../../data_schedule/"
nPEs=5
generate_empty=0

############ Single test for testing pipeline
# Graph configurations
declare -a nodes_Array=(10000)
declare -a edge_mul_Array=(2)
declare -a power_law_Array=("1000")
declare -a pw_edge_ratio_Array=(1)

# Embedding configurations
declare -a emb_Array=(32)
# Optimization options
declare -a col_tile_Array=(2048)
declare -a row_tile_Array=(16)
declare -a barriered_Array=("0")
declare -a bypass_Array=("0")
declare -a work_div_Array=("0") # TODO
declare -a reord_Array=("1")
declare -a prefetch_Array=("0") # TODO
#################


########### Actual ###################
## Graph configurations
#declare -a nodes_Array=(10000 \
#100000 \
#1000000)
#declare -a edge_mul_Array=(2 \
#10 \
#100)
#declare -a power_law_Array=("1000" \
#"1100" \
#"1110")
#declare -a pw_edge_ratio_Array=(1 \
#2 \
#10 \
#99)
#
## Embedding configurations
#declare -a emb_Array=(32 \
#128 \
#512 \
#2048)
## Optimization options
#declare -a col_tile_Array=(512 \
#2048 \
#8192 \
#32768 \
#131072 \
#524228 \
#2097152)
#declare -a row_tile_Array=(1 \
#4 \
#16 \
#64 \
#256 \
#1024 \
#4096)
#declare -a barriered_Array=("0" \
#"1")
#declare -a bypass_Array=("0" \
#"1")
#declare -a work_div_Array=("0") # TODO
#declare -a reord_Array=("0" \
#"1")
#declare -a prefetch_Array=("0") # TODO
###################################

# Create a main data folder if there isn't one
if [ -d "../../data_schedule" ]
then
  echo "data_schedule folder exists"
else
  mkdir "../../data_schedule"
fi

# Make the PaRMAT file in not there
if [ -f "../../utils/third_party/parmat/Release/PaRMAT" ]
then
  echo "PaRMAT exists."
else
  (cd ../../utils/third_party/parmat/Release && make)
fi

# Make the PaRMAT file in not there
# Create a main data folder if there isn't one
if [ -d "../../build" ]
then
  echo "build folder exists"
else
  mkdir "../../build"
fi
(cd ../../build && rm -r * && cmake .. && make)
#if [ -f "../../build/tests/spade_codegen_spmm_test" ]
#then
#  echo "Code generation for SPADE exists."
#else
#  (cd ../../build && cmake .. && make)
#fi

#if [[ $nPEs -lt $feat_size ]]
#then
# a=$(( 4*$nPEs))
# echo $a
#else
#  echo "still works"
#fi

for node in "${nodes_Array[@]}"; do
  for edge_mul in "${edge_mul_Array[@]}"; do
    for pw in "${power_law_Array[@]}"; do
      for pw_ratio in "${pw_edge_ratio_Array[@]}"; do
        python ../data/rmat_generate.py --pw_abcd "$pw" --rat_pw "$pw_ratio" --nodes "$node" --mul_edges "$edge_mul" --outp "$matrix_gen_path"

        for emb_size in "${emb_Array[@]}"; do
          for col_tile in "${col_tile_Array[@]}"; do
            col_chk=$((4*$node))
            if [[ $col_tile -lt $col_chk ]]
            then
              for row_tile in "${row_tile_Array[@]}"; do
                for barr in "${barriered_Array[@]}"; do
                  for byp in "${bypass_Array[@]}"; do
                    for wdiv in "${work_div_Array[@]}"; do
                      for reord in "${reord_Array[@]}"; do
                        for pref in "${prefetch_Array[@]}"; do
                          ../../build/tests/spade_codegen_spmm_test $matrix_gen_path $emb_size $nPEs $col_tile $row_tile $barr $wdiv $reord $pref $byp $generate_empty $schedule_output
                        done
                      done
                    done
                  done
                done
              done
            fi
          done
        done

      done
    done
  done
done

