export OMP_NUM_THREADS=56
emb_size=128
col_tile_size=32000

echo "mycielskian17"
numactl --interleave=all --physcpubind=0-55 ../../build/tests/spade_cpu_spmm_impl_test_barrierless "/home/damitha2/SparseAcc/data_exp_npy/mycielskian17/" $emb_size $col_tile_size 2 32 0 0 0 0 10
echo "as-Skitter"
numactl --interleave=all --physcpubind=0-55 ../../build/tests/spade_cpu_spmm_impl_test_barrierless "/home/damitha2/SparseAcc/data_exp_npy/as-Skitter/" $emb_size $col_tile_size 2 32 0 0 0 0 10
echo "asia_osm"
numactl --interleave=all --physcpubind=0-55 ../../build/tests/spade_cpu_spmm_impl_test_barrierless "/home/damitha2/SparseAcc/data_exp_npy/asia_osm/" $emb_size $col_tile_size 2 32 0 0 0 0 10
echo "delaunay_n21"
numactl --interleave=all --physcpubind=0-55 ../../build/tests/spade_cpu_spmm_impl_test_barrierless "/home/damitha2/SparseAcc/data_exp_npy/delaunay_n21/" $emb_size $col_tile_size 2 32 0 0 0 0 10
echo "com-LiveJournal"
numactl --interleave=all --physcpubind=0-55 ../../build/tests/spade_cpu_spmm_impl_test_barrierless "/home/damitha2/SparseAcc/data_exp_npy/com-LiveJournal/" $emb_size $col_tile_size 2 32 0 0 0 0 10
