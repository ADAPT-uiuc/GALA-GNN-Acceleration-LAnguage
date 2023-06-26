#!/bin/bash
#SBATCH --time=21-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=64
#SBATCH --mem-per-cpu=3800
#SBATCH --job-name="spmm_normal"
#SBATCH --partition=charithm
#SBATCH --exclusive
#SBATCH --output=out_normal.txt

export OMP_NUM_THREADS=63

numactl --interleave=all --physcpubind=0-63 tests/spade_cpu_spmm_impl "/home/damitha2/projects/SparseAcc/data_exp_npy/mycielskian17/" 256 1024 2 32 0 0 0 0 10
