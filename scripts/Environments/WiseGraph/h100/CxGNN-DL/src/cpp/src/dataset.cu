#include <string>

#include "dataset.h"
void Dataset::fetchData(const std::vector<Index> &index, int64_t feature_len,
                        Tensor &buf) {
  ASSERTWITH(false, "Not implemented");
}

void Dataset::fetchAllData(Tensor &buf) {
  ASSERTWITH(false, "Not implemented");
}

Tensor Dataset::index2D(const Tensor &tensor, const std::vector<Index> &index) {
  ASSERTWITH(false, "Not implemented");
}

shared_ptr<Dataset> Dataset::create(Yaml::Node &config) {
  std::string dtype = config["dataset"]["dtype"].As<std::string>("float32");
  shared_ptr<Dataset> dataset;
  if (dtype == "float32") {
    dataset = make_shared<DatasetTemplate<float>>(config);
    dataset->feature_option = float32_option;
  } else if (dtype == "float16") {
    dataset = make_shared<DatasetTemplate<at::Half>>(config);
    dataset->feature_option = float16_option;
  } else {
    ASSERTWITH(false, "Unsupported dtype");
  }
  return dataset;
}