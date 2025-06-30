#!/bin/bash

module load cmake
module load cuda-toolkit/12.2

mkdir ~/.venv
python3 -m venv ~/.venv/cxgnn
source ~/.venv/cxgnn/bin/activate

git clone --recurse-submodules -j8 https://github.com/xxcclong/triton.git triton

cd CxGNN-Compute
bash install.sh