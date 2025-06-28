#!/bin/bash

#module load cmake
#module load cuda-toolkit/12.2

mkdir ~/.venv
python3 -m venv ~/.venv/cxgnnz
source ~/.venv/cxgnnz/bin/activate

git clone --recurse-submodules -j8 https://github.com/chamikasudusinghe/CxGNN-Compute.git ../../../CxGNN-Compute
git clone --recurse-submodules -j8 https://github.com/chamikasudusinghe/CxGNN-DL.git ../../../CxGNN-DL
git clone --recurse-submodules -j8 https://github.com/xxcclong/triton.git ../../../triton

cd ../../../CxGNN-Compute

git checkout h100
bash install.sh