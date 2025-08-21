#!/bin/bash

cd scripts/Environments/SeaStar
conda env create --name seastar-gala-ae --file=.yml
source ~/miniforge3/bin/activate seastar-gala-ae
git clone --recursive https://github.com/ydwu4/dgl-hack # Might need to fix compilation errors
cd dgl-hack && mkdir build && cd build && cmake .. && cd .. && ./compile.sh
cd ..
git clone https://github.com/nithinmanoj10/Seastar-Documentation.git # Might need to fix compilation errors
cd Seastar-Documentation
cd Seastar/python
python setup.py install