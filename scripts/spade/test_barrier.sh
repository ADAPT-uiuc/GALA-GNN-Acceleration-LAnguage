#!/bin/bash
#SBATCH --time=21-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=64
#SBATCH --mem-per-cpu=3800
#SBATCH --job-name="spmm_bless"
#SBATCH --partition=charithm
#SBATCH --exclusive
#SBATCH --output=out_bless.txt

export OMP_NUM_THREADS=64
emb_size=128
col_tile_size=32000

echo "mycielskian17"
numactl --interleave=all --physcpubind=0-63 ../../build/tests/spade_cpu_spmm_impl_test_barrierless "/home/damitha2/projects/SparseAcc/data_exp_npy/mycielskian17/" $emb_size $col_tile_size 2 32 0 0 0 0 1
echo "as-Skitter"
numactl --interleave=all --physcpubind=0-63 ../../build/tests/spade_cpu_spmm_impl_test_barrierless "/home/damitha2/projects/SparseAcc/data_exp_npy/as-Skitter/" $emb_size $col_tile_size 2 32 0 0 0 0 1
echo "asia_osm"
numactl --interleave=all --physcpubind=0-63 ../../build/tests/spade_cpu_spmm_impl_test_barrierless "/home/damitha2/projects/SparseAcc/data_exp_npy/asia_osm/" $emb_size $col_tile_size 2 32 0 0 0 0 1
echo "delaunay_n21"
numactl --interleave=all --physcpubind=0-63 ../../build/tests/spade_cpu_spmm_impl_test_barrierless "/home/damitha2/projects/SparseAcc/data_exp_npy/delaunay_n21/" $emb_size $col_tile_size 2 32 0 0 0 0 1
echo "com-LiveJournal"
numactl --interleave=all --physcpubind=0-63 ../../build/tests/spade_cpu_spmm_impl_test_barrierless "/home/damitha2/projects/SparseAcc/data_exp_npy/com-LiveJournal/" $emb_size $col_tile_size 2 32 0 0 0 0 1
