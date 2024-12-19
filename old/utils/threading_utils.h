#ifndef _THREADING_UTILS_H
#define _THREADING_UTILS_H
#include <omp.h>

double get_time() { return omp_get_wtime(); }

#ifndef EMU
#define pragma_parallel() _Pragma("omp parallel")

#define pragma_parallel_for(C)\
    int sch = C;\
    _Pragma("omp parallel for schedule(dynamic, sch)")

#else
#define pragma_parallel()

#endif

int get_num_threads(){
#ifndef EMU
    return omp_get_num_threads();
#else
    return 1;
#endif
}

int get_thread_num(){
#ifndef EMU
    return omp_get_thread_num();
#else
    return 0;
#endif
}

double get_wtime(){
#ifndef EMU
    return omp_get_wtime();
#else
    return 0;
#endif
}
#endif