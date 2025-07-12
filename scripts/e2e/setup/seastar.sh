#!/bin/bash

cd scripts/Environments/SeaStar
conda env create --name seastar-gala-ae --file=.yml
source ~/miniforge3/bin/activate seastar-gala-ae
tar -xvzf seastar-source.tar.gz
cd seastar
cd dgl-hack && mkdir build && cd build && cmake .. && cd .. && ./compile.sh
cd ..
cd Seastar-Documentation
cd Seastar/python
python setup.py install