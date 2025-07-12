#!/bin/bash

source ~/miniforge3/bin/activate gala
cd ../Evaluations
python Figures-16-17.py
python Figures-16-17.py --hw a100 --train
python Figures-16-17.py --hw a100 --job dgl
source ~/miniforge3/bin/activate seastar-gala-ae
python Figures-16-17.py --job sea
source ~/miniforge3/bin/activate stir-gala-ae
export PREV_LIBRARY_PATH="$LIBRARY_PATH"
export PREV_LD_LIBRARY_PATH="$LD_LIBRARY_PATH"
export LIBRARY_PATH="$CONDA_PREFIX/lib:$CONDA_PREFIX/lib64:$LIBRARY_PATH"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$CONDA_PREFIX/lib64:$LD_LIBRARY_PATH"
python Figures-16-17.py --job stir
export LIBRARY_PATH="$PREV_LIBRARY_PATH"
export LD_LIBRARY_PATH="$PREV_LD_LIBRARY_PATH"
source ../Environments/WiseGraph/a100/.venv/cxgnn2/bin/activate
python WiseGraph.py --job F16n17 --a100
source ~/miniforge3/bin/activate gala
python Figure-18.py --hw a100 --job stat
python Figure-18.py --hw a100 --job stat --train