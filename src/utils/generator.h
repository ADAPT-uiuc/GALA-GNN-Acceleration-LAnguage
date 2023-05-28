
#ifndef _GENERATOR_H
#define _GENERATOR_H

#include <algorithm>
#include <cassert>
#include <cinttypes>
#include <iostream>
#include <parallel/algorithm>
#include <parallel/numeric>
#include <random>
#include <vector>

template <typename I_>
struct nonzero {
  I_ r;
  I_ c;
  inline bool operator==(const nonzero<I_> &struct2) { return (r == struct2.r && c == struct2.c); }
};

template <typename I_>
bool compare(const nonzero<I_> &a, const nonzero<I_> &b) {
  if (a.r < b.r)
    return true;
  else if (a.r > b.r)
    return false;
  else {
    if (a.c < b.c)
      return true;
    else
      return false;
    // else if(a.c>b.c) return false;
  }
}

template <typename I_, typename N_>
void generate_rmat(I_ nrows, I_ ncols, I_ deg, double a, double b, double c, bool directed, bool randomize, N_ &nnz, I_ *&row_ids, I_ *&col_ids) {
  N_ nnz_tmp = (N_)(nrows) * (N_)(deg);
  N_ list_size = nnz_tmp;

  std::cout << "Generating directed?: " << directed << " " << nrows << "x" << ncols << " with " << nnz_tmp << " nnz" << std::endl;
  std::cout << "Degree " << deg << std::endl;

  if (directed == false) list_size *= 2;

  std::vector<nonzero<I_>> list(list_size);

  I_ nr = nrows - 1;
  I_ nc = ncols - 1;
  int64_t kRandSeed = 27491095;
#pragma omp parallel
  {
    N_ block_size = nnz_tmp / (N_)(omp_get_num_threads()) + 1;
    std::mt19937 rng;
    std::uniform_real_distribution<double> udist(0, 1.0);
#pragma omp for
    for (N_ block = 0; block < nnz_tmp; block += block_size) {
      rng.seed(kRandSeed + block / block_size);
      for (N_ e = block; e < std::min(block + block_size, nnz_tmp); e++) {
        //   el[e] = Edge(udist(rng), udist(rng));
        I_ sr = 0;
        I_ er = nr;
        I_ sc = 0;
        I_ ec = nc;

        while (sr != er && sc != ec) {
          double rand_point = udist(rng);
          if (rand_point < a) {
            er = (sr + er) / 2;
            ec = (sc + ec) / 2;
          } else if (rand_point < a + b) {
            er = (sr + er) / 2;
            sc = (sc + ec) / 2;
          } else if (rand_point < a + b + c) {
            sr = (sr + er) / 2;
            ec = (sc + ec) / 2;
          } else {
            sr = (sr + er) / 2;
            sc = (sc + ec) / 2;
          }
        }

        assert(sr < nrows && sc < ncols);
        if (directed == false) {
          list[2 * e].r = sr;
          list[2 * e].c = sc;
          list[2 * e + 1].r = sc;
          list[2 * e + 1].c = sr;
        } else {
          list[e].r = sr;
          list[e].c = sc;
        }
      }
    }
  }

  __gnu_parallel::sort(list.begin(), list.end(), compare<I_>);
  list.erase(unique(list.begin(), list.end()), list.end());
  if (directed == false) {
    std::cout << "List size: " << list.size() << " nvals: " << 2 * nnz_tmp << std::endl;
    std::cout << "==== Rsize: " << (double)(list.size()) / (double)(2 * nnz_tmp) << std::endl;
  } else {
    std::cout << "List size: " << list.size() << " nvals: " << nnz_tmp << std::endl;
    std::cout << "==== Rsize: " << (double)(list.size()) / (double)(nnz_tmp) << std::endl;
  }

  nnz = list.size();
  row_ids = new I_[nnz];
  col_ids = new I_[nnz];
#pragma omp parallel for
  for (N_ e = 0; e < nnz; e++) {
    row_ids[e] = list[e].r;
    col_ids[e] = list[e].c;
  }

  // nnz = nnz_tmp;
  // for(int e=0; e<20; e++) std::cout<<row_ids[e]<<" "<<col_ids[e]<<std::endl;
}

template <typename I_>
struct vtx2d {
  double x;
  double y;
  I_ id;
};

template <typename I_>
struct cell {
  I_ xid;
  I_ yid;
};

template <typename I_>
double euclid_dist2d(const vtx2d<I_> &v1, const vtx2d<I_> &v2) {
  return sqrt(pow((v1.x - v2.x), 2) + pow((v1.y - v2.y), 2));
}

template <typename I_, typename N_>
void generate_rgg2D(I_ nrows, double radius, N_ &nnz, I_ *&row_ids, I_ *&col_ids) {
  double sq_nrows = sqrt(nrows);
  uint64_t dim = floor(sqrt(nrows)) + 1;
  double cellsize = 1.0 / (double)(dim);
  if (cellsize < radius) {
    cellsize = radius;
    dim = floor(1 / radius) + 1;
  }

  std::cout << "Dims: " << dim << "x" << dim << std::endl;
  std::cout << "Cell size: " << cellsize << " x " << cellsize << std::endl;
  std::cout << "Radius: " << radius << std::endl;
  assert(cellsize >= radius);

  int64_t kRandSeed1 = 27491095;
  int64_t kRandSeed2 = 2749105;
  // int64_t kRandSeed1 = 0;
  // int64_t kRandSeed2 = 0;

  vtx2d<I_> *vertices = new vtx2d<I_>[nrows];
  I_ *counts = new I_[dim * dim];
#pragma omp parallel for
  for (I_ i = 0; i < dim * dim; i++) counts[i] = 0;

  std::cout << "Done" << std::endl;
#pragma omp parallel
  {
    I_ block_size = nrows / (N_)(omp_get_num_threads()) + 1;
    std::mt19937 rngx;
    std::mt19937 rngy;
    std::uniform_real_distribution<double> udistx(0, 1.0);
    std::uniform_real_distribution<double> udisty(0, 1.0);
#pragma omp for
    for (I_ block = 0; block < nrows; block += block_size) {
      rngx.seed(kRandSeed1 + block / block_size);
      rngy.seed(kRandSeed2 + block / block_size);
      for (I_ e = block; e < std::min(block + block_size, nrows); e++) {
        double randx = udistx(rngx);
        double randy = udisty(rngy);
        vertices[e].x = randx;
        vertices[e].y = randy;
        vertices[e].id = e;
        assert(dim > (I_)floor(randx / cellsize));
        assert(dim > (I_)floor(randy / cellsize));

        I_ dimx = (I_)floor(randx / cellsize);
        I_ dimy = (I_)floor(randy / cellsize);
        I_ pos = dimx * dim + dimy;
        __sync_fetch_and_add(&counts[pos], 1);
      }
    }
  }

  std::cout << "Done2" << std::endl;
  I_ *offsets = new I_[dim * dim + 1];
  offsets[0] = 0;
  __gnu_parallel::partial_sum(&(counts[0]), &(counts[0]) + (dim * dim), &(offsets[1]));
  std::cout << "last " << offsets[dim * dim] << " " << nrows << std::endl;

  auto sorter = [&cellsize](const vtx2d<I_> &a, const vtx2d<I_> &b) {
    if ((I_)floor(a.x / cellsize) < (I_)floor(b.x / cellsize))
      return true;
    else if ((I_)floor(a.x / cellsize) > (I_)floor(b.x / cellsize))
      return false;
    else {
      if ((I_)floor(a.y / cellsize) < (I_)floor(b.y / cellsize))
        return true;
      else if ((I_)floor(a.y / cellsize) > (I_)floor(b.y / cellsize))
        return false;
      else {
        if (a.x < b.x)
          return true;
        else if (a.x > b.x)
          return false;
        else {
          if (a.y < b.y)
            return true;
          else if (a.y >= b.y)
            return false;
        }
      }
    }
  };

  __gnu_parallel::sort(&vertices[0], &vertices[nrows], sorter);
  //   std::cout<<"offset_0 "<<offsets[0]<<" "<<offsets[1]<<std::endl;
  //  for(I_ i=0; i<10;i++){
  //    std::cout<<i<<" "<<vertices[i].x<<" "<<vertices[i].y<<" : "<<(I_)floor(vertices[i].x/cellsize)<<" "<<(I_)floor(vertices[i].y/cellsize)<<std::endl;
  //  }

  //  for(I_ i=offsets[0]; i<offsets[1];i++){
  //    std::cout<<i<<" "<<vertices[i].x<<" "<<vertices[i].y<<" : "<<(I_)floor(vertices[i].x/cellsize)<<" "<<(I_)floor(vertices[i].y/cellsize)<<std::endl;
  //  }
  I_ num_cells = dim * dim;
  N_ num_edges = 0;
  N_ num_matches = 0;

  N_ *edge_counts = new N_[dim * dim];
#pragma omp parallel for
  for (I_ i = 0; i < dim * dim; i++) edge_counts[i] = 0;

#pragma omp parallel for reduction(+ : num_edges) reduction(+ : num_matches)
  for (I_ b = 0; b < num_cells; b++) {
    I_ ob = offsets[b];
    I_ oe = offsets[b + 1];
    if (oe - ob > 0) {  // nonempty cell
      I_ xd = b / dim;
      I_ yd = b % dim;
      // itself
      N_ local_edges = 0;
      for (int xx = -1; xx <= 1; xx++) {
        for (int yy = -1; yy <= 1; yy++) {
          // if(xx == -1 && xd ==0){ /*do nothing*/}
          // else if(yy == -1 && yd ==0){ /*do nothing*/}

          // check first/last column/rows of the cells
          if ((xx == -1 && xd == 0) == false && (yy == -1 && yd == 0) == false && (xx == 1 && xd == dim - 1) == false && (yy == 1 && yd == dim - 1) == false) {
            I_ bb = (xd + xx) * dim + (yd + yy);
            I_ oob = offsets[bb];
            I_ ooe = offsets[bb + 1];
            if (ooe - oob > 0) {
              for (I_ e1 = ob; e1 < oe; e1++) {
                for (I_ e2 = oob; e2 < ooe; e2++) {
                  if (e1 != e2) {
                    if (euclid_dist2d<I_>(vertices[e1], vertices[e2]) < radius) {
                      num_edges++;
                      local_edges++;
                    }
                    num_matches++;
                  }
                }
              }
            }
          }
        }
      }

      edge_counts[b] = local_edges;
    }
  }
  N_ *edge_offsets = new N_[dim * dim + 1];
  edge_offsets[0] = 0;
  __gnu_parallel::partial_sum(&(edge_counts[0]), &(edge_counts[0]) + (dim * dim), &(edge_offsets[1]));
  std::cout << "last " << edge_offsets[dim * dim] << " " << num_edges << std::endl;

  nnz = num_edges;

  std::cout << "Edges: " << num_edges << std::endl;
  std::cout << "Num_matches: " << num_matches << std::endl;
  std::cout << "Ratio: " << (double)(num_edges) / (double)(num_matches) << std::endl;

  // row_ids = new I_[num_edges];
  // col_ids = new I_[num_edges];
  num_edges = 0;
  num_matches = 0;

  std::vector<nonzero<I_>> list(nnz);

#pragma omp parallel for reduction(+ : num_edges) reduction(+ : num_matches)
  for (I_ b = 0; b < num_cells; b++) {
    I_ ob = offsets[b];
    I_ oe = offsets[b + 1];
    if (oe - ob > 0) {  // nonempty cell
      I_ xd = b / dim;
      I_ yd = b % dim;
      // itself
      N_ local_edges = 0;
      N_ edge_off = edge_offsets[b];
      for (int xx = -1; xx <= 1; xx++) {
        for (int yy = -1; yy <= 1; yy++) {
          // if(xx == -1 && xd ==0){ /*do nothing*/}
          // else if(yy == -1 && yd ==0){ /*do nothing*/}

          // check first/last column/rows of the cells
          if ((xx == -1 && xd == 0) == false && (yy == -1 && yd == 0) == false && (xx == 1 && xd == dim - 1) == false && (yy == 1 && yd == dim - 1) == false) {
            I_ bb = (xd + xx) * dim + (yd + yy);
            I_ oob = offsets[bb];
            I_ ooe = offsets[bb + 1];
            if (ooe - oob > 0) {
              for (I_ e1 = ob; e1 < oe; e1++) {
                for (I_ e2 = oob; e2 < ooe; e2++) {
                  if (e1 != e2) {
                    if (euclid_dist2d<I_>(vertices[e1], vertices[e2]) < radius) {
                      // row_ids[edge_off+local_edges] = vertices[e1].id;
                      // col_ids[edge_off+local_edges] = vertices[e2].id;
                      N_ pos = edge_off + local_edges;
                      list[pos].r = e1;
                      // vertices[e1].id;
                      list[pos].c = e2;
                      // vertices[e2].id;

                      num_edges++;
                      local_edges++;
                    }
                    num_matches++;
                  }
                }
              }
            }
          }
        }
      }
    }
  }

  // __gnu_parallel::sort(list.begin(), list.end(), compare<I_>);
  // list.erase(unique(list.begin(), list.end()), list.end());
  // std::cout << "List size: " << list.size() << " nvals: " << num_edges << std::endl;
  // std::cout << "==== Rsize: " << (double)(list.size()) / (double)(num_edges) << std::endl;

  nnz = list.size();
  row_ids = new I_[nnz];
  col_ids = new I_[nnz];
#pragma omp parallel for
  for (N_ e = 0; e < nnz; e++) {
    row_ids[e] = list[e].r;
    col_ids[e] = list[e].c;
  }
  double pi = 3.14159265358979323846;
  std::cout << "Ave_deg: " << (double)(nnz) / (double)(nrows) << " Expected: " << (pi * pow(radius, 2) / tgamma(2)) << " Expected: " << (pi * nrows * pow(radius, 2)) << std::endl;

  delete[] offsets;
  delete[] counts;
  delete[] vertices;
  delete[] edge_offsets;
  delete[] edge_counts;
}
#endif