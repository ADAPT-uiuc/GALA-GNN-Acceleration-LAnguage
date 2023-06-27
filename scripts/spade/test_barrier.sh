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

numactl --interleave=all --physcpubind=0-63 ../../build/tests/spade_cpu_spmm_impl_test_barrierless "/home/damitha2/projects/SparseAcc/data_exp_npy/mycielskian17/" 128 80000 2 32 0 0 0 0 10
numactl --interleave=all --physcpubind=0-63 ../../build/tests/spade_cpu_spmm_impl_test_barrierless "/home/damitha2/projects/SparseAcc/data_exp_npy/as-Skitter/" 128 80000 2 32 0 0 0 0 10
numactl --interleave=all --physcpubind=0-63 ../../build/tests/spade_cpu_spmm_impl_test_barrierless "/home/damitha2/projects/SparseAcc/data_exp_npy/asia_osm/" 128 80000 2 32 0 0 0 0 10
numactl --interleave=all --physcpubind=0-63 ../../build/tests/spade_cpu_spmm_impl_test_barrierless "/home/damitha2/projects/SparseAcc/data_exp_npy/delaunay_n21/" 128 80000 2 32 0 0 0 0 10
numactl --interleave=all --physcpubind=0-63 ../../build/tests/spade_cpu_spmm_impl_test_barrierless "/home/damitha2/projects/SparseAcc/data_exp_npy/com-LiveJournal/" 128 80000 2 32 0 0 0 0 10
