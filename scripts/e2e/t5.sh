#!/bin/bash

source ~/miniforge3/bin/activate gala
cd ../Evaluations
python Table-5.py
python Table-5.py --job dgl
source ../Environments/WiseGraph/h100/.venv/cxgnn2/bin/activate
python WiseGraph.py --job T5 --h100
source ~/miniforge3/bin/activate gala
python Table-5.py --job stat