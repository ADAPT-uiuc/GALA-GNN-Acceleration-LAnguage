//
// Created by damitha on 4/9/24.
//
#include "common.h"
#include <iostream>

#ifdef TMKL
typedef long long int ind1_t;
#else
typedef uint32_t ind1_t;
#endif

#ifdef TMKL
typedef long long int ind2_t;
#else
typedef uint64_t ind2_t;
#endif
typedef float val_t;
typedef int val_int_t;

// IR classes
#include "../src/codegen/gala/ir/data.h"
#include "../src/codegen/gala/ir/compute.h"

// Matrix classes
//#include "../src/utils/mtx_io.h"
#include "../src/matrix/dense_matrix.h"
#include "../src/matrix/csrc_matrix.h"


//Dense matrix with double values.
typedef DenseMatrix<ind1_t, ind2_t, val_t> DMd_t;
//Dense matrix with integer values.
typedef DenseMatrix<ind1_t, ind2_t, val_int_t> DMi_t;
// Sparse matrix (graph)
typedef CSRCMatrix<ind1_t, ind2_t, val_t> SM_t;


int main(int argc, char **argv) {
    std::cout << "Hello" << std::endl;

    auto g_feats = DataNode<DMd_t>("G.feats", 0, 10);
    std::cout << "Test IR--1: " << g_feats.getName() << std::endl;

    auto g_graph = DataNode<SM_t>("G.graph", 0, 10);
    auto graph_data = DataList<SM_t>(true);
    g_graph.setData(&graph_data);
    std::cout << "Test IR--2: " << g_graph.getName() << std::endl;

    auto g_relation = RelationEdge<DMd_t, SM_t>(&g_feats, ROW, &g_graph, ROW);
    std::cout << "Test relation nodes: " << g_relation.getNode1()->getName() << " " << g_relation.getNode2()->getName()
              << std::endl;
    std::cout << "Test relation edge: " << g_relation.getRelation1() << " " << g_relation.getRelation2()
              << std::endl;

    auto g_graph2 = g_graph.cloneData();
    g_graph2.getData().setIndependence(false);
    auto g_transform = TransformEdge<SM_t>(&g_graph, &g_graph2);

    auto col_tile_trans = TransformData(COL_TILE);
    col_tile_trans.addParam("65849");
    g_transform.addTransformation(&col_tile_trans);

    auto new_compute = StatementNode(AGGREGATE, 0);

    std::cout << "Test transform node: " << g_transform.getNode1()->getName() << std::endl;

}