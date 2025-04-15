# GNN-Acceleration-Language

## Prerequisites
* Flex
* Bison
### Installation (Linux/Ubuntu)
```angular2html
sudo apt-get update
sudo apt-get install flex
sudo apt-get install bison
```

## Running code generation tests for GNNs
```angular2html
mkdir build # create a build directory 
cd build
cmake .. # run cmake
make -j8 # make the tests
tests/gcn_ir # run the codegen for gcn
```
Others models are GIN and GAT.
This should generate the final executable code in the `test-codegen` folder.

The test files are in the `tests` folder, with the name `gala_<model>_IR.cpp`.
Currently, the manual IR in uncommented, and the IR generation from the front-end language is commented (needs to fix some bugs).  

## Data
Scripts necessary for downloading data can be found in `scripts/data`.

There is also a notebook to visualize two arrays of src, and dst npy files (Graph represented in COO format) to help get a visual idea of the NNZ distribution in a graph.

## Running for SPADE code generation
The scripts for SPADE are located in `scripts/spade`.
The script with the base pipeline for the code generation should be in `rmat_schedule_script/sh`

(P.S. I managed to run it with only GCC (Worked with 7.2.0 and 12.2.0, both in ICCP), BUT not sure if some dependency was already in my environment at the time.)