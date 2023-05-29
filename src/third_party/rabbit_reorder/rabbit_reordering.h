//
// Created by damitha on 8/16/22.
//

#ifndef SPARSE_ACCELERATOR_REORDERIN_H
#define SPARSE_ACCELERATOR_REORDERIN_H
//
//
// A demo program of reordering using Rabbit Order.
//
// Author: ARAI Junya <arai.junya@lab.ntt.co.jp> <araijn@gmail.com>
//
#include <boost/range/adaptor/transformed.hpp>
#include <boost/range/algorithm/count.hpp>
#include "rabbit_order.hpp"
#include "edge_list.hpp"
#include "../../matrix/csrc_matrix.h"

using rabbit_order::vint;
typedef std::vector<std::vector<std::pair<vint, float> > > adjacency_list;

// TODO use the commons there instead of defining in each test
int PTHHREADS = 56;

vint count_unused_id(const vint n, const std::vector<edge_list::edge> &edges) {
    std::vector<char> appears(n);
    for (size_t i = 0; i < edges.size(); ++i) {
        appears[std::get<0>(edges[i])] = true;
        appears[std::get<1>(edges[i])] = true;
    }
    return static_cast<vint>(boost::count(appears, false));
}

template<typename RandomAccessRange>
adjacency_list make_adj_list(const vint n, const RandomAccessRange &es) {
    using std::get;

#ifdef DBG_RO
    double start_time, start_total;
    double time_ss_loop, time_sort, time_adj_parallel, time_total;
    start_time = get_time();
    start_total = get_time();
#endif

    std::vector<edge_list::edge> ss(boost::size(es) * 2);
#pragma omp parallel for
    for (size_t i = 0; i < boost::size(es); ++i) {
        auto &e = es[i];
        if (get<0>(e) != get<1>(e)) {
            ss[i * 2] = std::make_tuple(get<0>(e), get<1>(e), get<2>(e));
            ss[i * 2 + 1] = std::make_tuple(get<1>(e), get<0>(e), get<2>(e));
        } else {
            // Insert zero-weight edges instead of loops; they are ignored in making
            // an adjacency list
            ss[i * 2] = std::make_tuple(0, 0, 0.0f);
            ss[i * 2 + 1] = std::make_tuple(0, 0, 0.0f);
        }
    }

#ifdef DBG_RO
    time_ss_loop = (get_time() - start_time);
    start_time = get_time();
#endif
    // Sort the edges
    __gnu_parallel::sort(ss.begin(), ss.end()); // CHANGED FROM
//    std::sort(ss.begin(), ss.end());

#ifdef DBG_RO
    time_sort = (get_time() - start_time);
    start_time = get_time();
#endif
    // Convert to an adjacency list
    adjacency_list adj(n);
#pragma omp parallel
    {
        // Advance iterators to a boundary of a source vertex
        const auto adv = [](auto it, const auto first, const auto last) {
            while (first != it && it != last && get<0>(*(it - 1)) == get<0>(*it))
                ++it;
            return it;
        };

        // Compute an iterator range assigned to this thread
        const int p = omp_get_max_threads();
        const size_t t = static_cast<size_t>(omp_get_thread_num());
        const size_t ifirst = ss.size() / p * (t) + std::min(t, ss.size() % p);
        const size_t ilast = ss.size() / p * (t + 1) + std::min(t + 1, ss.size() % p);
        auto it = adv(ss.begin() + ifirst, ss.begin(), ss.end());
        const auto last = adv(ss.begin() + ilast, ss.begin(), ss.end());

        // Reduce edges and store them in std::vector
        while (it != last) {
            const vint s = get<0>(*it);

            // Obtain an upper bound of degree and reserve memory
            const auto maxdeg =
                    std::find_if(it, last, [s](auto &x) { return get<0>(x) != s; }) - it;
            adj[s].reserve(maxdeg);

            while (it != last && get<0>(*it) == s) {
                const vint t = get<1>(*it);
                float w = 0.0;
                while (it != last && get<0>(*it) == s && get<1>(*it) == t)
                    w += get<2>(*it++);
                if (w > 0.0)
                    adj[s].push_back({t, w});
            }

            // The actual degree can be smaller than the upper bound
            adj[s].shrink_to_fit();
        }
    }

#ifdef DBG_RO
    time_adj_parallel = (get_time() - start_time);
    time_total = (get_time() - start_total);

    std::cout << "--------------------------------------" << std::endl;
    std::cout << "Debug times of reorder make adj" << std::endl;
    std::cout << "SS parallel: " << time_ss_loop << " , percen: " << time_ss_loop * 100 / time_total
              << std::endl;
    std::cout << "Sort: " << time_sort << " , percen: " << time_sort * 100 / time_total
              << std::endl;
    std::cout << "Adj parallel: " << time_adj_parallel << " , percen: " << time_adj_parallel * 100 / time_total
              << std::endl;
#endif

    return adj;
}

template<class SM>
adjacency_list read_graph(SM* src) {
    typedef typename SM::itype iT;
    typedef typename SM::ntype nT;
    typedef typename SM::vtype vT;

    iT src_ncols = src->ncols();
    iT src_nrows = src->nrows();

    nT *src_offset_ptr = src->offset_ptr();
    iT *src_ids_ptr = src->ids_ptr();
    vT *src_vals_ptr = src->vals_ptr();

    adjacency_list adj(src_nrows);

#pragma omp parallel for schedule(dynamic, 4)
    for (iT i_i = 0; i_i < src_nrows; i_i += 1) {
        nT first_node_edge = src_offset_ptr[i_i];
        nT last_node_edge = src_offset_ptr[i_i + 1];

        nT deg = last_node_edge - first_node_edge;

        // TODO always assume that self edge is there
        adj[i_i].reserve(deg - 1);
        for (nT e = first_node_edge; e < last_node_edge; e++) {
            iT u = src_ids_ptr[e];
            if (u == i_i){
                continue;
            }
            vT val = src_vals_ptr[e];
            adj[i_i].push_back({u, val});
        }
    }
    return adj;
}

template<class SM>
adjacency_list read_graph(typename SM::itype nrows,
                          typename SM::ntype nvals,
                          typename SM::itype *row_ids,
                          typename SM::itype *col_ids,
                          typename SM::vtype *vals) {
    typedef typename SM::itype iT;
    typedef typename SM::ntype nT;
    typedef typename SM::vtype vT;

    std::vector<edge_list::aux::edge> edges;
    edges.resize(nvals);

#ifdef DBG_RO
    double start_time, start_total;
    double time_work_loop, time_accum, time_adj_list, time_total;
    start_total = get_time();
#endif

    double work_thread = (double) nvals / PTHHREADS;
    if (work_thread < 1) {
        work_thread = 1;
    }

#ifdef DBG_RO
    start_time = get_time();
#endif
    auto edges_start = edges.begin();
#pragma omp parallel for schedule(static)
    for (int i = 0; i < PTHHREADS; i++) {
        for (nT j = std::ceil(work_thread * i); j < std::ceil(work_thread * (i + 1)); j++) {

            edges[j] = std::make_tuple((rabbit_order::vint) row_ids[j], (rabbit_order::vint) col_ids[j],
                                       (float) vals[j]);
        }
    }
#ifdef DBG_RO
    time_work_loop = (get_time() - start_time);
    start_time = get_time();
#endif
    // The number of vertices = max vertex ID + 1 (assuming IDs start from zero)
//    const auto n =
//            boost::accumulate(edges, static_cast<vint>(0), [](vint s, auto &e) {
//                return std::max(s, std::max(std::get<0>(e), std::get<1>(e)) + 1);
//            });

    const auto n = nrows;


#ifdef DBG_RO
    time_accum = (get_time() - start_time);
#endif

    if (const size_t c = count_unused_id(n, edges)) {
        std::cerr << "WARNING: " << c << "/" << n << " vertex IDs are unused"
                  << " (zero-degree vertices or noncontiguous IDs?)\n";
    }

#ifdef DBG_RO
    start_time = get_time();
#endif
    auto res = make_adj_list(n, edges);
#ifdef DBG_RO
    time_adj_list = (get_time() - start_time);
    time_total = (get_time() - start_total);
    std::cout << "--------------------------------------" << std::endl;
    std::cout << "Debug times of reorder read graph" << std::endl;
    std::cout << "Work loop: " << time_work_loop << " , percen: " << time_work_loop * 100 / time_total
              << std::endl;
    std::cout << "Accum: " << time_accum << " , percen: " << time_accum * 100 / time_total
              << std::endl;
    std::cout << "Make adj: " << time_adj_list << " , percen: " << time_adj_list * 100 / time_total
              << std::endl;
#endif

    return res;
}


void reorder(adjacency_list adj, std::unique_ptr<vint[]> &perm) {
    std::cerr << "Generating a permutation...\n";
    const double tstart = rabbit_order::now_sec();
    //--------------------------------------------
    const auto g = rabbit_order::aggregate(std::move(adj));
    perm = rabbit_order::compute_perm(g);
    //--------------------------------------------
    std::cerr << "Runtime for permutation generation [sec]: "
              << rabbit_order::now_sec() - tstart << std::endl;
}

template<class SM>
void get_perm_graph(SM* src,
                    std::unique_ptr<vint[]> &perm) {
    using boost::adaptors::transformed;


    std::cerr << "Number of threads: " << omp_get_max_threads() << std::endl;

    auto adj = read_graph<SM>(src);
    std::cerr << "Number of vertices: " << adj.size() << std::endl;
    std::cerr << "Number of edges: " << src->nvals() << std::endl;

    reorder(std::move(adj), perm);
    // TODO rabbit re-ordering also has communities.
    //  this had more code but removed for readability. If you want the earlier code, check the backup file.
}

template<class SM>
void get_perm_graph(typename SM::itype nrows,
                    typename SM::ntype nvals,
                    typename SM::itype *row_ids,
                    typename SM::itype *col_ids,
                    typename SM::vtype *vals,
                    std::unique_ptr<vint[]> &perm) {
    using boost::adaptors::transformed;


    std::cerr << "Number of threads: " << omp_get_max_threads() << std::endl;

    // TODO add the new graph read here
    auto adj = read_graph<SM>(nrows, nvals, row_ids, col_ids, vals);

    const auto m =
            boost::accumulate(adj | transformed([](auto &es) { return es.size(); }),
                              static_cast<size_t>(0));
    std::cerr << "Number of vertices: " << adj.size() << std::endl;
    std::cerr << "Number of edges: " << m << std::endl;

    reorder(std::move(adj), perm);
}


#endif //SPARSE_ACCELERATOR_REORDERIN_H
