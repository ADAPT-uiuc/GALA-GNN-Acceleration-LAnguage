#!/bin/bash

source ~/miniforge3/bin/activate gala
cd ../../build
tests/gala_inference ../tests/GALA-DSL/gcn/Reddit/h100.txt ../codegen/
cd ../codegen/
mkdir build
cd build
make -j6
./gala_model # should print two numbers (inference time and training time per epoch)