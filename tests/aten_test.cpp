//
// Created by damitha on 12/6/23.
//
//#include <torch/extension.h>
#include <torch/torch.h>
#include <iostream>
#include <vector>

void convertTensorToArray(torch::Tensor tensor) {
    // Check if the tensor is contiguous
    if (!tensor.is_contiguous()) {
        tensor = tensor.contiguous();
    }

    // Get the data pointer from the tensor
    float* data_ptr = tensor.data_ptr<float>();

    // Get the size of the tensor
    int tensor_size = tensor.numel();

    // Convert the tensor data to a C++ array
    std::vector<float> array_data(data_ptr, data_ptr + tensor_size);

    // Print the elements of the array
    std::cout << "Converted Array: ";
    for (const auto& value : array_data) {
        std::cout << value << " ";
    }
    std::cout << std::endl;
}

int main() {
    torch::Tensor tensor = torch::rand({2, 3});
    convertTensorToArray(tensor);
    std::cout << tensor << std::endl;
}
