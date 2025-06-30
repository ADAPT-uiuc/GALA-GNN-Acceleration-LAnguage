for i in {1..16}

do
    OMP_PROC_BIND=true OMP_NUM_THREADS=$i numactl -C 72-100  ./a.out | tee -a output.txt
done
