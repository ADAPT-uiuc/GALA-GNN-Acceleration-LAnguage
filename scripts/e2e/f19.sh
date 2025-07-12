#!/bin/bash

source ~/miniforge3/bin/activate gala
cd ../Evaluations
python Figure-19.py
source ../Environments/WiseGraph/h100/.venv/cxgnn2/bin/activate
python WiseGraph.py --job F19 --h100
source ~/miniforge3/bin/activate gala
python Figure-19.py --job stat