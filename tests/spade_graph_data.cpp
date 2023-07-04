#include "../src/utils/mtx_io.h"
#include "../src/utils/threading_utils.h"
#include "../src/ops/approx.h"
#include "../src/ops/tiling.h"

#include "common.h"
#include <iostream>

// definitions for data types
typedef uint32_t ind1_t;
typedef uint64_t ind2_t;
typedef float val_t;
typedef int val_int_t;

#define MKL_INT ind1_t

typedef CSRCMatrix<ind1_t, ind2_t, val_t> SM_t;
typedef DenseMatrix<ind1_t, ind2_t, val_t> DMd_t;

int main(int argc, char** argv) {
    auto start = get_time();

    typedef typename SM_t::itype iT; // Node IDs
    typedef typename SM_t::ntype nT; // Edge IDs
    typedef typename SM_t::vtype vT; // Value of node

    using namespace std;

    std::string path = argv[1];
    int cores = stoi(argv[2]);
    int prec = stoi(argv[3]);

    std::string filename = path;
    SM_t adj;
#ifdef RNPY
    readSM_npy(path, &adj);
#else
    readSM<SM_t>(filename, &adj);
#endif
    adj.set_all(1);

//    int prec = 100;
    // TODO select the present to process based on the number of rows
//    if (nrows <= 1000){
//        prec = 80;
//    } else if (nrows <= 10000){
//        prec = 50;
//    } else if (nrows <= 100000){
//        prec = 30;
//    } else if (nrows <= 1000000){
//        prec = 10;
//    } else {
//        prec = 1;
//    }

    auto start2 = get_time();
    auto nrows = adj.nrows();
    auto nvals = adj.nvals();
    iT min_max[2] = {0};

    auto edge_num = adj.nvals();

    // TODO Have a common limit generation
    iT rows_per_thread = adj.nrows() / cores;
    double percen = (double) prec / 100.0;

    std::vector<GNOpTile<SM_t, DMd_t>> tile_infos;
    for (iT i = 0; i < cores; i++) {
        iT u_rows;
        if ((prec == 100) && (i == cores - 1)){
            u_rows = nrows;
        } else {
            u_rows = rows_per_thread * i + (iT) (rows_per_thread * percen);
        }

        GNOpTile<SM_t, DMd_t> tile_info;
        tile_info.srows_start = rows_per_thread * i;
        tile_info.srows_end = u_rows;
        tile_infos.push_back(tile_info);
    }

    std::cout << nrows << "," << nvals << ",";
    approx_range<SM_t>(&adj, min_max, tile_infos);
    approx_reord_met<SM_t>(&adj, tile_infos);
    approx_vert_entr<SM_t>(&adj, prec, min_max[1]);

    auto stop = get_time();
    auto duration = stop - start;
    //std::cout << "," << duration;
    duration = stop - start2;
    std::cout << "," << duration << std::endl;
    return 0;
}