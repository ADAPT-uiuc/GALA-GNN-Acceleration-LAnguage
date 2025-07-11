#!/bin/bash

conda create -n gala python=3.11 --yes
source ~/miniforge3/bin/activate gala # Please give the correct path to activate the env if different
conda install nvidia/label/cuda-12.4.0::cuda-toolkit --yes
pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu124
conda install -c dglteam/label/th24_cu124 dgl --yes
pip install ogb
conda install packaging --yes
conda install conda-forge::bison --yes
conda install conda-forge::seaborn --yes
conda install anaconda::pandas --yes
conda install anaconda::scipy --yes
cd ../../Environments
mkdir libtorch
cd libtorch
wget https://download.pytorch.org/libtorch/cu126/libtorch-cxx11-abi-shared-with-deps-2.7.1%2Bcu126.zip
unzip libtorch-cxx11-abi-shared-with-deps-2.7.1+cu126.zip
cd ../../..
mkdir build
cd build
cmake ..
make -j5
cd ../codegen
mkdir build
cd build
cmake -DCMAKE_PREFIX_PATH="$PWD/../../scripts/Environments/libtorch" ..