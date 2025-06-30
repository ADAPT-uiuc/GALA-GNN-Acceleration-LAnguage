#include "compute_schedule.h"

ComputeScheduler::ComputeScheduler(Yaml::Node &config) {
  // model = new XGB(reader);
  iter_limit = config["compute_schedule"]["iter_limit"].As<int>(1);
  graph2mm = config["compute_schedule"]["graph2mm"].As<float>(0.17);
  degree_thres = config["compute_schedule"]["degree_thres"].As<int>(1);
  score_thres = config["compute_schedule"]["score_thres"].As<float>(1.f);
  feat_in = config["dataset"]["feature_dim"].As<int>(0);
  ASSERT(feat_in != 0);
  feat_in = 768;
  SPDLOG_INFO("Schedule config {} {} {}", iter_limit, graph2mm, degree_thres);
}

void showGraphDegreeDist(shared_ptr<Graph> graph) {
  std::vector<int> in_degree(graph->getNumNode());
  std::vector<int> out_degree(graph->getNumUnique(), 0);
  for (int i = 0; i < graph->csr_ptr.size() - 1; ++i) {
    in_degree[i] = graph->csr_ptr[i + 1] - graph->csr_ptr[i];

    for (int j = graph->csr_ptr[i]; j < graph->csr_ptr[i + 1]; ++j) {
      Index nodeid = graph->csr_idx[j];
      // if (out_degree.size() < nodeid + 1) out_degree.resize(nodeid + 1);
      out_degree[nodeid] += 1;
    }
  }
  SPDLOG_INFO("size {} {} {} {}", in_degree.size(), out_degree.size(),
              graph->csr_ptr.back(), graph->getNumEdge());
  std::vector<float> ranges = {0.f, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.f};
  for (auto i : ranges) {
    auto v = in_degree;
    int pos = i * v.size();
    if (i == 1.f) pos = v.size() - 1;
    SPDLOG_INFO("{} {}", pos, v.size());
    std::nth_element(v.begin(), v.begin() + pos, v.end());
    int in = v[pos];
    v = out_degree;
    pos = i * v.size();
    if (i == 1.f) pos = v.size() - 1;
    SPDLOG_INFO("{} {}", pos, v.size());
    std::nth_element(v.begin(), v.begin() + pos, v.end());
    int out = v[pos];
    SPDLOG_INFO("degree dist {} {} {}", in, out, i);
  }
}

vector<shared_ptr<Graph>> ComputeScheduler::schedule(shared_ptr<Graph> graph,
                                                     int layer_id,
                                                     torch::Tensor cache) {
  showGraphDegreeDist(graph);
  ASSERTWITH(graph->hasCSR(), "Needs CSR format for scheduling");
  ASSERTWITH(graph->layered(), "should be layered CSR");
  // TODO: num used v <-> num unique node
  const Index num_unique = graph->getNumUnique();
  vector<bool> good_vertex(num_unique, 0);
  Index good_cnt = 0;
  vector<Index> pre_transform_nodes;
  vector<float> score(num_unique, 0);
  int feat_out = 256;
  float save_graph =
      graph2mm * (1 - 1.f * (feat_out * 2) / feat_in);  // predicted
  save_graph = std::max(save_graph, 0.f);

  // cache
  int *cache_ptr = nullptr;
  Index *sub_to_full = nullptr;
  if (cache.sizes()[0] != 0) {
    cache_ptr = cache.data<int>();
    sub_to_full = graph->subgraph_index.sub_to_full.data();
    save_graph = 0.f;  // there is no influence on graph OP
  }

  vector<float> score_save_graph_op(num_unique, 0);
  // ASSERT(graph->num_node_in_layer[layer_id] == graph->csr_ptr.size() - 1);
  for (Index i = 0; i < graph->csr_ptr.size() - 1; ++i)
    for (Index j = graph->csr_ptr[i]; j < graph->csr_ptr[i + 1]; ++j)
      score_save_graph_op[graph->csr_idx[j]] += save_graph;
  int iter;
  for (iter = 0; iter < iter_limit; ++iter) {
    bool end_flag = 1;
    memcpy(score.data(), score_save_graph_op.data(),
           sizeof(float) * score.size());
    for (Index i = 0; i < graph->csr_ptr.size() - 1; ++i) {
      Index degree = 0;
      for (Index j = graph->csr_ptr[i]; j < graph->csr_ptr[i + 1]; ++j) {
        Index nodeid = graph->csr_idx[j];
        if (!good_vertex[nodeid] &&
            (cache_ptr == nullptr || !cache_ptr[sub_to_full[nodeid]]))
          ++degree;
      }
      // less than degree_thres: the math expectation is larger than 0
      if (degree <= degree_thres && degree > 0) {
        float score_to_add = 1.f / degree;
        if (degree_thres == 999999) score_to_add = 1.f;
        for (Index j = graph->csr_ptr[i]; j < graph->csr_ptr[i + 1]; ++j) {
          Index nodeid = graph->csr_idx[j];
          if (!good_vertex[nodeid] &&
              (cache_ptr == nullptr || !cache_ptr[sub_to_full[nodeid]])) {
            score[nodeid] += score_to_add;
          }
        }
      }
    }
    for (Index i = 0; i < score.size(); ++i) {
      if (score[i] >= score_thres && !good_vertex[i] &&
          (cache_ptr == nullptr || !cache_ptr[sub_to_full[i]])) {
        good_vertex[i] = 1;
        ++good_cnt;
        // node i is shared by multiple dst nodes, pre transform it
        pre_transform_nodes.push_back(i);
        end_flag = 0;
      }
    }
    if (end_flag || iter == iter_limit - 1) {
      break;
    }
  }
  // pre_transform_nodes = vector<Index>();
  auto vec_output_graph = breakGraph(graph, pre_transform_nodes);
  for (auto item : vec_output_graph) item->display();
  Index num_workload = 0;
  Index num_post = 0;
  if (cache_ptr == nullptr) {
    num_post = vec_output_graph[1]->csr_ptr.size() - 1;
  } else {
    for (Index i = 0; i < graph->csr_ptr.size() - 1; ++i) {
      for (Index j = graph->csr_ptr[i]; j < graph->csr_ptr[i + 1]; ++j) {
        int nodeid = graph->csr_idx[j];
        if (!good_vertex[nodeid] &&
            !cache_ptr[sub_to_full[nodeid]])  // not cached, and not
                                              // communicated
        {
          ++num_post;
          break;
        }
      }
    }
  }
  num_workload = pre_transform_nodes.size() + num_post;
  SPDLOG_INFO("num-origin {} num new {}\nratio {} pre {} post {} iter {}\n",
              graph->csr_ptr.size() - 1, num_workload,
              num_workload * 1.f / (graph->csr_ptr.size() - 1),
              pre_transform_nodes.size(), num_post, iter);
  if (save_graph != 0.f) {
    float new_graph_workload =
        1.f * (feat_out * 2) / feat_in * vec_output_graph[0]->getNumEdge() +
        vec_output_graph[1]->getNumEdge();
    float new_nn_workload = pre_transform_nodes.size() + num_post;
    float new_sum = new_graph_workload * graph2mm + new_nn_workload;
    float graph_ratio = new_graph_workload * graph2mm / new_sum;

    float old_graph_workload = graph->getNumEdge();
    float old_nn_workload = graph->csr_ptr.size() - 1;
    float old_sum = old_graph_workload * graph2mm + old_nn_workload;
    float old_graph_ratio = old_graph_workload * graph2mm / old_sum;

    float old_graph_workload2 =
        graph->getNumEdge() * 1.f * (feat_out * 2) / feat_in;
    float old_nn_workload2 = graph->getNumUnique();
    float old_sum2 = old_graph_workload2 * graph2mm + old_nn_workload2;
    float old_graph_ratio2 = old_graph_workload2 * graph2mm / old_sum2;
    SPDLOG_INFO(
        "Save-graph old-graph {} old-nn {} old-graph2 {} old-nn2 {} new-graph "
        "{} new-nn {} "
        "ratio {} {} graph-in-all {} {} {}",
        old_graph_workload * graph2mm, old_nn_workload,
        old_graph_workload2 * graph2mm, old_nn_workload,
        new_graph_workload * graph2mm, new_nn_workload, new_sum / old_sum,
        new_sum / old_sum2, old_graph_ratio, old_graph_ratio2, graph_ratio);
  }
  graph->display();
  return vec_output_graph;
}

vector<shared_ptr<Graph>> ComputeScheduler::breakGraph(
    shared_ptr<Graph> graph, const vector<Index> &nodes) {
  timestamp(t0);
  Index num_unique = graph->getNumUnique();
  //   vector<bool> included(num_unique, 0);
  SubgraphIndex subgraph_index;
  for (auto &item : nodes) {
    // included[item] = 1;
    subgraph_index.addNode(item);
  }
  vector<Index> include_csr_ptr;
  vector<Index> include_csr_idx;
  vector<Index> exclude_csr_ptr;
  vector<Index> exclude_csr_idx;
  vector<Index> include_target;
  vector<Index> exclude_target;
  vector<Index> to_origin_csr_ptr;
  vector<Index> to_origin_csr_idx;
  include_csr_ptr.push_back(0);
  exclude_csr_ptr.push_back(0);
  to_origin_csr_ptr.push_back(0);
  int owned_1 = 0, owned_2 = 0, owned_hyb = 0;
  for (Index i = 0; i < graph->getNumNode();
       ++i) {  // TODO: config it with layer
    Index cnt = 0;
    Index num_neighbor = graph->csr_ptr[i + 1] - graph->csr_ptr[i];
    ASSERT(num_neighbor != 0);
    int origin_cnt = 0;
    for (Index j = graph->csr_ptr[i]; j < graph->csr_ptr[i + 1]; ++j) {
      auto sub = subgraph_index.findNode(graph->csr_idx[j]);
      if (sub != -1) {
        ++cnt;
        include_csr_idx.push_back(
            sub);  // include_graph read from shrinked graph
      } else
        exclude_csr_idx.push_back(
            graph->csr_idx[j]);  // exclude_graph read from origin
    }
    if (cnt != 0) {
      include_csr_ptr.push_back(include_csr_ptr.back() + cnt);
      include_target.push_back(i);
      ++origin_cnt;
      to_origin_csr_idx.push_back(include_target.size() - 1);
    }
    if (cnt != num_neighbor) {
      exclude_csr_ptr.push_back(exclude_csr_ptr.back() + num_neighbor - cnt);
      exclude_target.push_back(i);
      ++origin_cnt;
      to_origin_csr_idx.push_back(-1 * (int)exclude_target.size());
    }
    owned_1 += cnt == num_neighbor;
    owned_2 += cnt == 0;
    owned_hyb += (cnt != 0 && cnt != num_neighbor);
    to_origin_csr_ptr.push_back(to_origin_csr_ptr.back() + origin_cnt);
  }
  const int base = include_target.size();
  for (int i = 0; i < to_origin_csr_idx.size(); ++i) {
    if (to_origin_csr_idx[i] < 0) {
      to_origin_csr_idx[i] = base - to_origin_csr_idx[i] - 1;
    }
  }
  timestamp(t1);
  SPDLOG_INFO("OWN {} {} {}", owned_1, owned_2, owned_hyb);
  SPDLOG_INFO("EXECTIME graph partition {}", getDuration(t0, t1));
  SPDLOG_INFO("num node {} {}", include_csr_ptr.size() - 1,
              exclude_csr_ptr.size() - 1);
  auto graph_include = make_shared<Graph>(std::move(include_csr_ptr),
                                          std::move(include_csr_idx));
  auto graph_exclude = make_shared<Graph>(std::move(exclude_csr_ptr),
                                          std::move(exclude_csr_idx));
  auto graph_origin = make_shared<Graph>(std::move(to_origin_csr_ptr),
                                         std::move(to_origin_csr_idx));
  graph_include->setParentGraph(graph);
  graph_exclude->setParentGraph(graph);
  graph_include->setTarget(std::move(include_target));
  graph_exclude->setTarget(std::move(exclude_target));
  graph_include->setSubgraphIdx(std::move(subgraph_index));
  vector<shared_ptr<Graph>> output = {graph_include, graph_exclude,
                                      graph_origin};
  return output;
}

std::vector<torch::Tensor> rel_schedule(shared_ptr<Graph> subgraph,
                                        const std::vector<EtypeIndex> &rel) {
  int num_rel = 7;
  std::vector<torch::Tensor> output;
  int num_layer = subgraph->num_node_in_layer.size() - 1;
  ASSERT(num_layer == 3);
  auto rel_num_node_in_layer_tensor =
      torch::empty({num_rel * num_layer}, int64_option);
  Index *p_num = rel_num_node_in_layer_tensor.data<Index>();
  for (int rel_iter = 0; rel_iter < num_rel; ++rel_iter) {
    int layer_id = 0;
    std::vector<Index> sub_ptr;
    sub_ptr.push_back(0);
    std::vector<Index> sub_idx;
    std::vector<Index> sub_target;
    for (Index i = 0; i < subgraph->csr_ptr.size() - 1; ++i) {
      Index cnt = 0;
      for (Index j = subgraph->csr_ptr[i]; j < subgraph->csr_ptr[i + 1]; ++j) {
        if (rel_iter == rel[j]) {
          ++cnt;
          sub_idx.push_back(subgraph->csr_idx[j]);
        }
      }
      if (cnt > 0) {
        sub_target.push_back(i);
        sub_ptr.push_back(sub_ptr.back() + cnt);
      }
      if (i == subgraph->num_node_in_layer[layer_id] - 1) {
        p_num[layer_id + num_layer * rel_iter] = sub_target.size();
        ++layer_id;
      }
    }

    auto ptr_tensor = torch::empty({(int)sub_ptr.size()}, int64_option);
    memcpy(ptr_tensor.data<Index>(), sub_ptr.data(),
           sub_ptr.size() * sizeof(Index));
    auto idx_tensor = torch::empty({(int)sub_idx.size()}, int64_option);
    memcpy(idx_tensor.data<Index>(), sub_idx.data(),
           sub_idx.size() * sizeof(Index));
    auto target_tensor = torch::empty({(int)sub_target.size()}, int64_option);
    memcpy(target_tensor.data<Index>(), sub_target.data(),
           sub_target.size() * sizeof(Index));
    output.push_back(ptr_tensor);
    output.push_back(idx_tensor);
    output.push_back(target_tensor);
  }
  output.push_back(rel_num_node_in_layer_tensor);
  return output;
}