#!/bin/bash

# module load cmake
# module load cuda-toolkit/11.8

mkdir ~/.venv
python3 -m venv ~/.venv/cxgnn
source ~/.venv/cxgnn/bin/activate

git clone https://github.com/chamikasudusinghe/CxGNN-Compute.git --recurse-submodules
git clone https://github.com/xxcclong/CxGNN-DL.git --recurse-submodules
git clone --recurse-submodules https://github.com/xxcclong/triton.git triton

cd CxGNN-Compute
bash install.sh