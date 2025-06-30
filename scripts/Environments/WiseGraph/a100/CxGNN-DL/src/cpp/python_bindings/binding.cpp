#include <memory.h>
#include <pybind11/pybind11.h>

#include "batch.h"
#include "common.h"
#include "fastgraph.h"
#include "loader.h"
#include "memory_access.h"
#include "sample_kernel.h"
namespace py = pybind11;
using namespace pybind11::literals;

void init_batch(py::module &m) {
  py::class_<Batch, std::shared_ptr<Batch>>(m, "Batch")
      .def_readwrite("edge_index", &Batch::edge_index)
      .def_readwrite("x", &Batch::x)
      .def_readwrite("y", &Batch::y)
      .def_readwrite("mask", &Batch::mask)
      .def_readwrite("ptr", &Batch::ptr)
      .def_readwrite("idx", &Batch::idx)
      .def_readwrite("num_node_in_layer", &Batch::num_node_in_layer)
      .def_readwrite("num_edge_in_layer", &Batch::num_edge_in_layer)
      .def_readwrite("num_etype_in_layer", &Batch::num_etype_in_layer)
      .def_readwrite("sub_to_full", &Batch::sub_to_full)
      .def_readwrite("edge_type", &Batch::edge_type)
      .def_readwrite("edge_indexs", &Batch::edge_indexs)
      .def_readwrite("xs", &Batch::xs)
      .def_readwrite("ys", &Batch::ys)
      .def_readwrite("masks", &Batch::masks)
      .def_readwrite("ptrs", &Batch::ptrs)
      .def_readwrite("idxs", &Batch::idxs)
      .def_readwrite("sub_to_fulls", &Batch::sub_to_fulls)
      .def_readwrite("etype_partition", &Batch::etype_partition)
      .def_readwrite("typed_num_node_in_layer",
                     &Batch::typed_num_node_in_layer);
}

void init_fastgraph(py::module &m) {
  py::class_<FastGraph>(m, "CXGDL")
      .def(py::init<const std::string &>())
      .def("get_batch", &FastGraph::get_batch)
      .def("start", &FastGraph::start)
      .def("num_iters", &FastGraph::num_iters)
      .def_readwrite("train_loader", &FastGraph::train_loader)
      .def_readwrite("val_loader", &FastGraph::val_loader)
      .def_readwrite("test_loader", &FastGraph::test_loader);
}

void init_memory_access(py::module &m) {
  m.def("uvm_select", &uvm_select, "");
  m.def("uvm_select_masked", &uvm_select_masked, "");
  m.def("uvm_select_half", &uvm_select_half, "");
  m.def("uvm_select_masked_half", &uvm_select_masked_half, "");
  m.def("gen_mmap", &gen_mmap, "");
  m.def("mmap_select", &mmap_select, "");
  m.def("single_thread_mmap_load", &mmap_select_st, "");
  m.def("read_to_ptr", &read_to_ptr, "");
  m.def("graph_analysis", &graph_analysis, "");
}

void init_util(py::module &m) {
  m.def("show_mem_curr_proc", &showCpuMemCurrProc,
        "Show memory usage of current proc");
}

void init_sample_kernel(py::module &m) {
  m.def("neighbor_sample", &neighbor_sample, "sample kernel");
}

PYBIND11_MODULE(cxgnndl_backend, m) {
  m.doc() = "A Supa Fast Graph Loader";
  init_batch(m);
  init_fastgraph(m);
  init_util(m);
  init_memory_access(m);
  init_sample_kernel(m);
}
