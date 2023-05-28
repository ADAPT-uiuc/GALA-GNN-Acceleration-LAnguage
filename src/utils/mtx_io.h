#ifndef _MTX_IO_H
#define _MTX_IO_H

#include <cassert>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <parallel/algorithm>
#include <parallel/numeric>
#include <string>

#include "info_error.h"
#include "threading_utils.h"

/*
 * @todo extend reader with other formats
 * @body only Matrix Market format is implemented
 */

#ifdef SparseLibDBG
#define MTX_IO_DBG
#endif

#ifdef MTX_IO_DBG
#define MTXIO_DBG_MSG std::cout << "MTXIO_" << __LINE__ << ": "
#else
#define MTXIO_DBG_MSG \
  if (0) std::cout << "MTXIO_" << __LINE__ << ": "
#endif

template<typename I, typename N>
bool compSortingPair4(const std::pair<I, N> &a, const std::pair<I, N> &b) {
    return a.second > b.second;
}

template<typename I_, typename N_, typename V_, template<class A> class Alloc = std::allocator>
class MtxIO {
public:
    MtxIO() : nrows_(0), ncols_(0), nvals_(0), rows_(nullptr), cols_(nullptr), vals_(nullptr) {}

    ~MtxIO() { /*nothing to do here*/
        clear();
    }

    enum class MtxValType {
        PATTERN, INTEGER, FLOAT, DOUBLE, COMPLEX, UNKNOWN
    };
    enum class MtxSymmetry {
        SYMMETRIC, SKEW_SYMMETRIC, HERMITIAN, GENERAL, UNKNOWN
    };
    enum class MtxFormat {
        COORDINATE, ARRAY, UNKNOWN
    };

    std::string mtx_name() { return mtx_name_; }

    IO_INFO readMtx(std::string filename) {
        filename_ = filename;
        mtx_name_ = getMtxName(filename);
        if (getFileSuffix(filename) == "mtx") {
            return readMM();
        }
        // TODO: other formats
        return IO_INFO::SUCCESS;
    }

    // TODO Sande
    IO_INFO readMtxFast(std::string filename) {
        filename_ = filename;
        mtx_name_ = getMtxName(filename);
        if (getFileSuffix(filename) == "mtx") {
            return readMM();
        }
        // TODO: other formats
        return IO_INFO::SUCCESS;
    }

    void getData(I_ &nrows, I_ &ncols, N_ &nvals, N_ &size, I_ *&rows, I_ *&cols, V_ *&vals) {
        nrows = nrows_;
        ncols = ncols_;
        nvals = nvals_;
        rows = rows_;
        cols = cols_;
        vals = vals_;
        size = size_;
        // rows_ = nullptr; cols_ = nullptr;
        // vals_ = nullptr;
        // nrows_ = 0; ncols_=0; nvals_=0;
        // rows_ = nullptr; cols_ = nullptr; vals_=nullptr;
    }

    void relabel_cfs_rfs() {
        N_ *counts;
        I_ *iperm_col;
        I_ *iperm_row;

        iperm_col = new I_[ncols_];
        iperm_row = new I_[nrows_];

        counts = new N_[ncols_];
        count_atomic(cols_, counts, ncols_, nvals_);

        std::vector<std::pair<I_, N_>> list(std::max(nrows_, ncols_));

#pragma omp parallel for
        for (I_ r = 0; r < ncols_; r++) {
            list[r].first = r;
            list[r].second = counts[r];
        }
        __gnu_parallel::sort(list.begin(), list.end(), compSortingPair4<I_, N_>);

#pragma omp parallel for
        for (I_ r = 0; r < ncols_; r++) {
            iperm_col[list[r].first] = r;
        }

#pragma omp parallel for
        for (N_ n = 0; n < nvals_; n++) {
            cols_[n] = iperm_col[cols_[n]];
        }

        for (int i = 0; i < 10; i++) {
            std::cout << list[i].first << " " << list[i].second << std::endl;
        }

        delete[] counts;

        counts = new N_[nrows_];
        count_atomic(rows_, counts, nrows_, nvals_);

        // std::vector<std::pair<I_, N_>> list(nrows_);

#pragma omp parallel for
        for (I_ r = 0; r < nrows_; r++) {
            list[r].first = r;
            list[r].second = counts[r];
        }
        __gnu_parallel::sort(list.begin(), list.end(), compSortingPair4<I_, N_>);

#pragma omp parallel for
        for (I_ r = 0; r < nrows_; r++) {
            iperm_row[list[r].first] = r;
        }

#pragma omp parallel for
        for (N_ n = 0; n < nvals_; n++) {
            rows_[n] = iperm_row[rows_[n]];
        }

        for (int i = 0; i < 10; i++) {
            std::cout << list[i].first << " " << list[i].second << std::endl;
        }

        delete[] counts;
        delete[] iperm_col;
        delete[] iperm_row;
    }

    void relabel_cfs() {
        N_ *counts;
        I_ *iperm_col;
        I_ *iperm_row;

        iperm_col = new I_[ncols_];
        iperm_row = new I_[nrows_];

        counts = new N_[ncols_];
        count_atomic(cols_, counts, ncols_, nvals_);

        std::vector<std::pair<I_, N_>> list(std::max(nrows_, ncols_));

#pragma omp parallel for
        for (I_ r = 0; r < ncols_; r++) {
            list[r].first = r;
            list[r].second = counts[r];
        }
        __gnu_parallel::sort(list.begin(), list.end(), compSortingPair4<I_, N_>);

#pragma omp parallel for
        for (I_ r = 0; r < ncols_; r++) {
            iperm_col[list[r].first] = r;
        }

#pragma omp parallel for
        for (N_ n = 0; n < nvals_; n++) {
            cols_[n] = iperm_col[cols_[n]];
        }

        for (int i = 0; i < 10; i++) {
            std::cout << list[i].first << " " << list[i].second << std::endl;
        }

        delete[] counts;
        delete[] iperm_col;
    }

private:
    IO_INFO readHeaderMM(std::ifstream &mfs, std::string &headerStr) {
        if (getline(mfs, headerStr).eof()) {
            std::cerr << "Error: file " << filename_ << " does not store a matrix" << std::endl;
            return IO_INFO::FILE_ERROR;
        }
        char id[FIELD_LENGTH];
        char object[FIELD_LENGTH];
        char format[FIELD_LENGTH];
        char field[FIELD_LENGTH];
        char symmetry[FIELD_LENGTH];

        sscanf(headerStr.c_str(), "%s %s %s %s %s", id, object, format, field, symmetry);

        if (strcmp(object, "matrix") != 0) {
            std::cerr << "Error: file " << filename_ << " does not store a matrix" << std::endl;
            return IO_INFO::NOT_A_MATRIX;
        }

        if (strcmp(format, "coordinate") == 0) {
            format_ = MtxFormat::COORDINATE;
        } else if (strcmp(format, "array") == 0) {
            format_ = MtxFormat::ARRAY;
            //return IO_INFO::DENSE_NOT_SUPPORTED;
        } else {
            format_ = MtxFormat::UNKNOWN;
            return IO_INFO::UNKNOWN_FORMAT;
        }

        if (strcmp(field, "pattern") == 0) {
            val_type_ = MtxValType::PATTERN;
        } else if (strcmp(field, "real") == 0) {
            val_type_ = MtxValType::FLOAT;
        } else if (strcmp(field, "integer") == 0) {
            val_type_ = MtxValType::INTEGER;
        } else if (strcmp(field, "double") == 0) {
            val_type_ = MtxValType::DOUBLE;
        } else if (strcmp(field, "complex") == 0) {
            val_type_ = MtxValType::COMPLEX;
            return IO_INFO::COMPLEX_NOT_SUPPORTED;
        } else {
            std::cerr << "Unknown field" << std::endl;
            val_type_ = MtxValType::UNKNOWN;
            return IO_INFO::UNKNOWN_FIELD_TYPE;
        }

        if (strcmp(symmetry, "symmetric") == 0) {
            symmetry_ = MtxSymmetry::SYMMETRIC;
        } else if (strcmp(symmetry, "skew-symmetric") == 0) {
            symmetry_ = MtxSymmetry::SKEW_SYMMETRIC;
        } else if (strcmp(symmetry, "hermitian") == 0) {
            symmetry_ = MtxSymmetry::HERMITIAN;
            return IO_INFO::HERMITIAN_NOT_SUPPORTED;
        } else if (strcmp(symmetry, "general") == 0) {
            symmetry_ = MtxSymmetry::GENERAL;
        } else {
            symmetry_ = MtxSymmetry::UNKNOWN;
            return IO_INFO::UNKNOWN_SYMMETRY;
        }
        return IO_INFO::SUCCESS;
    }

    void skipCommentsMM(std::ifstream &mfs, std::string &line) {
        while (!getline(mfs, line).eof()) {
            if (line[0] != '%') {
                break;
            }
        }
    }

    void readCooSizeMM(std::ifstream &mfs, std::string &line) {
        uint64_t nr, nc, nz;
        // read the matrix size and number of non-zero elements
        sscanf(line.c_str(), "%lu %lu %lu", &nr, &nc, &nz);
        nrows_ = nr;
        ncols_ = nc;
        nvals_ = nz;
    }

    void readArrSizeMM(std::ifstream &mfs, std::string &line) {
        uint64_t nr, nc;
        // read the matrix size and number of non-zero elements
        sscanf(line.c_str(), "%lu %lu", &nr, &nc);
        nrows_ = nr;
        ncols_ = nc;
        nvals_ = nr * nc;
    }

    IO_INFO readMM() {
        std::string line;
        //std::cout<<filename_<<std::endl;
        std::ifstream mfs(filename_);
        if (!mfs.good()) {
            std::cerr << "Error: unable to open matrix file " << filename_ << std::endl;
            return IO_INFO::FILE_ERROR;
        }


        IO_INFO header_err = readHeaderMM(mfs, line);
        if (header_err != IO_INFO::SUCCESS) {
            return header_err;
        }
        skipCommentsMM(mfs, line);

        if (format_ == MtxFormat::COORDINATE) {
            readCooSizeMM(mfs, line);
        }
        if (format_ == MtxFormat::ARRAY) {
            readArrSizeMM(mfs, line);
        }

        MTXIO_DBG_MSG << "MtxName: " << mtx_name_ << std::endl;
        MTXIO_DBG_MSG << "FileName: " << filename_ << std::endl;
        MTXIO_DBG_MSG << "Format: " << format2str(format_) << " Symmetry: " << symmetry2str(symmetry_) << " FieldType: "
                      << valType2str(val_type_) << std::endl;
        MTXIO_DBG_MSG << "NumRows: " << nrows_ << " NumCols: " << ncols_ << " NumVals: " << nvals_ << std::endl;


        double s;
        uint64_t line_index = 0;
        N_ index = 0;

        //Added support for MtxFormat::ARRAY.
        //Pattern does not make sense as an arithmetic field for dense matrices so it is not supported.
        //Symmetry is not supported yet (and is not expected to be needed for dense matrices in GNN frameworks).
        if (format_ == MtxFormat::ARRAY) {

            size_ = nvals_;
            if ((symmetry_ != MtxSymmetry::UNKNOWN) && (symmetry_ != MtxSymmetry::GENERAL)) {
                return IO_INFO::DENSE_SYMMETRY_NOT_SUPPORTED;
            }

            if (val_type_ == MtxValType::PATTERN) {
                return IO_INFO::DENSE_PATTERN_NOT_SUPPORTED;
            }


            try {
                vals_ = val_alloc_.allocate(size_);
            } catch (const std::bad_array_new_length &e) {
                return IO_INFO::MEM_ERROR;
            } catch (const std::bad_alloc &e) {
                return IO_INFO::MEM_ERROR;
            }

            double dd = 0;
            int64_t di = 0;
            s = get_wtime();



            //Matrix Market does listing by columns in array format.
            //E.g. [1 2]
            //     [3 4]
            //will be listed as :
            //  1
            //  3
            //  2
            //  4

            while (line_index < nvals_ && (!getline(mfs, line).eof())) {
                line_index++;
                if (val_type_ == MtxValType::INTEGER) {
                    sscanf(line.c_str(), "%ld", &di);
                    //index=
                    vals_[(index % nrows_) * ncols_ + (index / nrows_)] = (V_) (di);
                } else if (val_type_ == MtxValType::FLOAT) {
                    sscanf(line.c_str(), "%lf", &dd);
                    vals_[(index % nrows_) * ncols_ + (index / nrows_)] = (V_) (dd);
                } else if (val_type_ == MtxValType::DOUBLE) {
                    sscanf(line.c_str(), "%lf", &dd);
                    vals_[(index % nrows_) * ncols_ + (index / nrows_)] = (V_) (dd);
                }
                index++;
            }
        }

        //MtxFormat::COORDINATE case
        if (format_ == MtxFormat::COORDINATE) {
            size_ = nvals_;
            if (symmetry_ == MtxSymmetry::SYMMETRIC || symmetry_ == MtxSymmetry::SKEW_SYMMETRIC) {  // if symmetric
                size_ = size_ * 2;
            }

            try {
                rows_ = id_alloc_.allocate(size_);
            } catch (const std::bad_array_new_length &e) {
                return IO_INFO::MEM_ERROR;
            } catch (const std::bad_alloc &e) {
                return IO_INFO::MEM_ERROR;
            }
            try {
                cols_ = id_alloc_.allocate(size_);
            } catch (const std::bad_array_new_length &e) {
                return IO_INFO::MEM_ERROR;
            } catch (const std::bad_alloc &e) {
                return IO_INFO::MEM_ERROR;
            }
            if (val_type_ != MtxValType::PATTERN) {
                try {
                    vals_ = val_alloc_.allocate(size_);
                } catch (const std::bad_array_new_length &e) {
                    return IO_INFO::MEM_ERROR;
                } catch (const std::bad_alloc &e) {
                    return IO_INFO::MEM_ERROR;
                }
            }


            s = get_wtime();
            double dd = 0;
            int64_t di = 0;
            while (line_index < nvals_ && (!getline(mfs, line).eof())) {
                line_index++;
                uint64_t row_id, col_id;
                if (val_type_ == MtxValType::PATTERN) {
                    sscanf(line.c_str(), "%ld %ld", &row_id, &col_id);
                    rows_[index] = (I_) row_id - (I_) (1);
                    cols_[index] = (I_) col_id - (I_) (1);
                } else if (val_type_ == MtxValType::INTEGER) {
                    sscanf(line.c_str(), "%ld %ld %ld", &row_id, &col_id, &di);
                    rows_[index] = (I_) row_id - (I_) (1);
                    cols_[index] = (I_) col_id - (I_) (1);
                    vals_[index] = (V_) (di);
                } else if (val_type_ == MtxValType::FLOAT) {
                    sscanf(line.c_str(), "%ld %ld %lf", &row_id, &col_id, &dd);
                    rows_[index] = (I_) row_id - (I_) (1);
                    cols_[index] = (I_) col_id - (I_) (1);
                    vals_[index] = (V_) (dd);
                } else if (val_type_ == MtxValType::DOUBLE) {
                    sscanf(line.c_str(), "%ld %ld %lf", &row_id, &col_id, &dd);
                    rows_[index] = (I_) row_id - (I_) (1);
                    cols_[index] = (I_) col_id - (I_) (1);
                    vals_[index] = (V_) (dd);
                }
                index++;

                /* @todo: add hermitian*/
                if (symmetry_ == MtxSymmetry::SYMMETRIC || symmetry_ == MtxSymmetry::SKEW_SYMMETRIC) {  // if symmetric
                    if (rows_[index - 1] !=
                        cols_[index - 1]) {                                           // not on the diagonal
                        rows_[index] = cols_[index - 1];
                        cols_[index] = rows_[index - 1];
                        if (val_type_ != MtxValType::PATTERN) {
                            vals_[index] = vals_[index - 1];
                        }
                        index++;
                    }
                }
            }


//            index = 0;
//#pragma omp parallel for schedule(static)
//            for (line_index = 0; line_index < nvals_; line_index++) {
//                std::string local_line;
//                uint64_t row_id, col_id;
//                int64_t di = 0;
//                double dd = 0;
//                N_ local_index;
//#pragma omp critical
//                {
//                    std::getline(mfs, local_line);
//                    local_index = index++;
//                }
//
//                if (val_type_ == MtxValType::PATTERN) {
//                    sscanf(local_line.c_str(), "%ld %ld", &row_id, &col_id);
//                    rows_[local_index] = (I_) row_id - (I_) (1);
//                    cols_[local_index] = (I_) col_id - (I_) (1);
//                } else if (val_type_ == MtxValType::INTEGER) {
//                    sscanf(local_line.c_str(), "%ld %ld %ld", &row_id, &col_id, &di);
//                    rows_[local_index] = (I_) row_id - (I_) (1);
//                    cols_[local_index] = (I_) col_id - (I_) (1);
//                    vals_[local_index] = (V_) (di);
//                } else if (val_type_ == MtxValType::FLOAT) {
//                    sscanf(local_line.c_str(), "%ld %ld %lf", &row_id, &col_id, &dd);
//                    rows_[local_index] = (I_) row_id - (I_) (1);
//                    cols_[local_index] = (I_) col_id - (I_) (1);
//                    vals_[local_index] = (V_) (dd);
//                } else if (val_type_ == MtxValType::DOUBLE) {
//                    sscanf(local_line.c_str(), "%ld %ld %lf", &row_id, &col_id, &dd);
//                    rows_[local_index] = (I_) row_id - (I_) (1);
//                    cols_[local_index] = (I_) col_id - (I_) (1);
//                    vals_[local_index] = (V_) (dd);
//                }
//
//                //std::cout << std::to_string(row_id) + " " +  std::to_string(col_id) + "\n" << std::endl;
//
//                /* @todo: add hermitian*/
//                if (symmetry_ == MtxSymmetry::SYMMETRIC || symmetry_ == MtxSymmetry::SKEW_SYMMETRIC) {  // if symmetric
//                    if (row_id != col_id) {                                           // not on the diagonal
//#pragma omp critical
//                        {
//                            local_index = index++;
//                        }
//                        rows_[local_index] = (I_) col_id - (I_) (1);
//                        cols_[local_index] = (I_) row_id - (I_) (1);
//                        if (val_type_ != MtxValType::PATTERN) {
//                            vals_[local_index] = dd;
//                        }
//                    }
//                }
//            }
        }

        nvals_ = index;
        double e = get_wtime();
        MTXIO_DBG_MSG << "File reading took: " << (e - s) << " seconds." << std::endl;
        MTXIO_DBG_MSG << "Read " << line_index << " lines, " << index << " nnzs." << std::endl;

//        std::cout << "File reading took: " << (e - s) << " seconds." << std::endl;
//        std::cout << "Read " << line_index << " lines, " << index << " nnzs." << std::endl;
        // nvals_ = index;
        assert(size_ >= index);
        mfs.close();
        return IO_INFO::SUCCESS;
    }

    /***
     * Fast matrix reader made using boost
     * @return
     */
    IO_INFO readMMFast() {
        std::string line;
        std::ifstream mfs(filename_);
        if (!mfs.good()) {
            std::cerr << "Error: unable to open matrix file " << filename_ << std::endl;
            return IO_INFO::FILE_ERROR;
        }


        IO_INFO header_err = readHeaderMM(mfs, line);
        if (header_err != IO_INFO::SUCCESS) {
            return header_err;
        }
        skipCommentsMM(mfs, line);

        if (format_ == MtxFormat::COORDINATE) {
            readCooSizeMM(mfs, line);
        }
        if (format_ == MtxFormat::ARRAY) {
            readArrSizeMM(mfs, line);
        }

        MTXIO_DBG_MSG << "MtxName: " << mtx_name_ << std::endl;
        MTXIO_DBG_MSG << "FileName: " << filename_ << std::endl;
        MTXIO_DBG_MSG << "Format: " << format2str(format_) << " Symmetry: " << symmetry2str(symmetry_) << " FieldType: "
                      << valType2str(val_type_) << std::endl;
        MTXIO_DBG_MSG << "NumRows: " << nrows_ << " NumCols: " << ncols_ << " NumVals: " << nvals_ << std::endl;


        double s;
        uint64_t line_index = 0;
        N_ index = 0;

        //Added support for MtxFormat::ARRAY.
        //Pattern does not make sense as an arithmetic field for dense matrices so it is not supported.
        //Symmetry is not supported yet (and is not expected to be needed for dense matrices in GNN frameworks).
        if (format_ == MtxFormat::ARRAY) {

            size_ = nvals_;
            if ((symmetry_ != MtxSymmetry::UNKNOWN) && (symmetry_ != MtxSymmetry::GENERAL)) {
                return IO_INFO::DENSE_SYMMETRY_NOT_SUPPORTED;
            }

            if (val_type_ == MtxValType::PATTERN) {
                return IO_INFO::DENSE_PATTERN_NOT_SUPPORTED;
            }


            try {
                vals_ = val_alloc_.allocate(size_);
            } catch (const std::bad_array_new_length &e) {
                return IO_INFO::MEM_ERROR;
            } catch (const std::bad_alloc &e) {
                return IO_INFO::MEM_ERROR;
            }

            double dd = 0;
            int64_t di = 0;
            s = get_wtime();



            //Matrix Market does listing by columns in array format.
            //E.g. [1 2]
            //     [3 4]
            //will be listed as :
            //  1
            //  3
            //  2
            //  4

            while (line_index < nvals_ && (!getline(mfs, line).eof())) {
                line_index++;
                if (val_type_ == MtxValType::INTEGER) {
                    sscanf(line.c_str(), "%ld", &di);
                    //index=
                    vals_[(index % nrows_) * ncols_ + (index / nrows_)] = (V_) (di);
                } else if (val_type_ == MtxValType::FLOAT) {
                    sscanf(line.c_str(), "%lf", &dd);
                    vals_[(index % nrows_) * ncols_ + (index / nrows_)] = (V_) (dd);
                } else if (val_type_ == MtxValType::DOUBLE) {
                    sscanf(line.c_str(), "%lf", &dd);
                    vals_[(index % nrows_) * ncols_ + (index / nrows_)] = (V_) (dd);
                }
                index++;
            }
        }

        //MtxFormat::COORDINATE case
        if (format_ == MtxFormat::COORDINATE) {
            size_ = nvals_;
            if (symmetry_ == MtxSymmetry::SYMMETRIC || symmetry_ == MtxSymmetry::SKEW_SYMMETRIC) {  // if symmetric
                size_ = size_ * 2;
            }

            try {
                rows_ = id_alloc_.allocate(size_);
            } catch (const std::bad_array_new_length &e) {
                return IO_INFO::MEM_ERROR;
            } catch (const std::bad_alloc &e) {
                return IO_INFO::MEM_ERROR;
            }
            try {
                cols_ = id_alloc_.allocate(size_);
            } catch (const std::bad_array_new_length &e) {
                return IO_INFO::MEM_ERROR;
            } catch (const std::bad_alloc &e) {
                return IO_INFO::MEM_ERROR;
            }
            if (val_type_ != MtxValType::PATTERN) {
                try {
                    vals_ = val_alloc_.allocate(size_);
                } catch (const std::bad_array_new_length &e) {
                    return IO_INFO::MEM_ERROR;
                } catch (const std::bad_alloc &e) {
                    return IO_INFO::MEM_ERROR;
                }
            }


            s = get_wtime();
            index = 0;
#pragma omp parallel for schedule(static)
            for (line_index = 0; line_index < nvals_; line_index++) {
                std::string local_line;
                uint64_t row_id, col_id;
                int64_t di = 0;
                double dd = 0;
                N_ local_index;
#pragma omp critical
                {
                    std::getline(mfs, local_line);
                    local_index = index++;
                }

                if (val_type_ == MtxValType::PATTERN) {
                    sscanf(local_line.c_str(), "%ld %ld", &row_id, &col_id);
                    rows_[local_index] = (I_) row_id - (I_) (1);
                    cols_[local_index] = (I_) col_id - (I_) (1);
                } else if (val_type_ == MtxValType::INTEGER) {
                    sscanf(local_line.c_str(), "%ld %ld %ld", &row_id, &col_id, &di);
                    rows_[local_index] = (I_) row_id - (I_) (1);
                    cols_[local_index] = (I_) col_id - (I_) (1);
                    vals_[local_index] = (V_) (di);
                } else if (val_type_ == MtxValType::FLOAT) {
                    sscanf(local_line.c_str(), "%ld %ld %lf", &row_id, &col_id, &dd);
                    rows_[local_index] = (I_) row_id - (I_) (1);
                    cols_[local_index] = (I_) col_id - (I_) (1);
                    vals_[local_index] = (V_) (dd);
                } else if (val_type_ == MtxValType::DOUBLE) {
                    sscanf(local_line.c_str(), "%ld %ld %lf", &row_id, &col_id, &dd);
                    rows_[local_index] = (I_) row_id - (I_) (1);
                    cols_[local_index] = (I_) col_id - (I_) (1);
                    vals_[local_index] = (V_) (dd);
                }

                //std::cout << std::to_string(row_id) + " " +  std::to_string(col_id) + "\n" << std::endl;

                /* @todo: add hermitian*/
                if (symmetry_ == MtxSymmetry::SYMMETRIC || symmetry_ == MtxSymmetry::SKEW_SYMMETRIC) {  // if symmetric
                    if (row_id != col_id) {                                           // not on the diagonal
#pragma omp critical
                        {
                            local_index = index++;
                        }
                        rows_[local_index] = (I_) col_id - (I_) (1);
                        cols_[local_index] = (I_) row_id - (I_) (1);
                        if (val_type_ != MtxValType::PATTERN) {
                            vals_[local_index] = dd;
                        }
                    }
                }
            }
        }

        nvals_ = index;
        double e = get_wtime();
        MTXIO_DBG_MSG << "File reading took: " << (e - s) << " seconds." << std::endl;
        MTXIO_DBG_MSG << "Read " << line_index << " lines, " << index << " nnzs." << std::endl;

//        std::cout << "File reading took: " << (e - s) << " seconds." << std::endl;
//        std::cout << "Read " << line_index << " lines, " << index << " nnzs." << std::endl;
        // nvals_ = index;
        assert(size_ >= index);
        mfs.close();
        return IO_INFO::SUCCESS;
    }

    std::string getFileName(std::string filePath, bool withExtension = false, char seperator = '/') {
        std::size_t dotPos = filePath.rfind('.');
        std::size_t sepPos = filePath.rfind(seperator);
        if (sepPos != std::string::npos) {
            return filePath.substr(sepPos + 1);
        }
        return "";
    }

    std::string getMtxName(std::string filePath, char seperator = '/') {
        std::size_t dotPos = filePath.rfind('.');
        std::size_t sepPos = filePath.rfind(seperator);
        std::size_t len = dotPos - sepPos;
        return filePath.substr(sepPos + 1, len - 1);
    }

    std::string getFileSuffix(std::string filename) { return filename.substr(filename.find_last_of(".") + 1); }

    void clear() {
        if (rows_ != nullptr) {
            id_alloc_.deallocate(rows_, size_);
            rows_ = nullptr;
        }
        if (cols_ != nullptr) {
            id_alloc_.deallocate(cols_, size_);
            cols_ = nullptr;
        }
        if (vals_ != nullptr) {
            val_alloc_.deallocate(vals_, size_);
            vals_ = nullptr;
        }
    }

    std::string format2str(MtxFormat format) {
        // COORDINATE, ARRAY, UNKNOWN
        switch (format) {
            case MtxFormat::COORDINATE:
                return "COORDINATE";
                break;
            case MtxFormat::ARRAY:
                return "ARRAY";
                break;
            default:
                return "UNKNOWN";
                break;
        }
    }

    std::string symmetry2str(MtxSymmetry sym) {
        // SYMMETRIC, SKEW_SYMMETRIC, HERMITIAN, GENERAL, UNKNOWN
        switch (sym) {
            case MtxSymmetry::SYMMETRIC:
                return "SYMMETRIC";
                break;
            case MtxSymmetry::SKEW_SYMMETRIC:
                return "SKEW_SYMMETRIC";
                break;
            case MtxSymmetry::HERMITIAN:
                return "HERMITIAN";
                break;
            case MtxSymmetry::GENERAL:
                return "GENERAL";
                break;
            default:
                return "UNKNOWN";
                break;
        }
    }

    std::string valType2str(MtxValType valT) {
        //  PATTERN, INTEGER, FLOAT, DOUBLE, COMPLEX, UNKNOWN
        switch (valT) {
            case MtxValType::PATTERN:
                return "PATTERN";
                break;
            case MtxValType::INTEGER:
                return "INTEGER";
                break;
            case MtxValType::FLOAT:
                return "FLOAT";
                break;
            case MtxValType::DOUBLE:
                return "DOUBLE";
                break;
            case MtxValType::COMPLEX:
                return "COMPLEX";
                break;
            default:
                return "UNKNOWN";
                break;
        }
    }

    std::string filename_;
    std::string mtx_name_;
    I_ *rows_;
    I_ *cols_;
    V_ *vals_;
    I_ nrows_;
    I_ ncols_;
    N_ nvals_;
    static const uint32_t FIELD_LENGTH = 4096;

    N_ size_;
    MtxSymmetry symmetry_;
    MtxValType val_type_;
    MtxFormat format_;
    Alloc<I_> id_alloc_;
    Alloc<V_> val_alloc_;
};

#endif